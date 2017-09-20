package com.socialgraze.spark.corenlp;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * @author cristian
 * Test class for Stanford CoreNLP wrapper.
 */
public class CoreNlpFunctionsTest {
	public SparkConf conf;
	public JavaSparkContext jsc;
	public SQLContext sqlContext;
	public DataFrame documents_df;
	public DataFrame sentenceSplit_df;
	public CoreNlpFunctions nlpFunctions;
	public RegexFunctions regexFunctions;
	public CoreNlpFunctionsTest() {}

	/**
	 * Initialize Spark, register UDF, create a test DataFrame and explode it into sentences.
	 */
	@Before
	public void init(){
		conf = new SparkConf()
				.setMaster("local[*]")
				.setAppName("CoreNLP Wrapper Test");
		jsc = new JavaSparkContext(conf);  
		sqlContext = new SQLContext(jsc);
		nlpFunctions = new CoreNlpFunctions();
		regexFunctions = new RegexFunctions();
		sqlContext.udf().register("ssplit", nlpFunctions.ssplit, DataTypes.createArrayType(DataTypes.StringType));
		sqlContext.udf().register("tokenize", nlpFunctions.tokenize, DataTypes.createArrayType(DataTypes.StringType));
		sqlContext.udf().register("pos", nlpFunctions.pos, DataTypes.createArrayType(DataTypes.StringType));
		sqlContext.udf().register("lemma", nlpFunctions.lemma, DataTypes.createArrayType(DataTypes.StringType));
		sqlContext.udf().register("ner", nlpFunctions.ner, DataTypes.createArrayType(DataTypes.StringType));
		sqlContext.udf().register("sentiment", nlpFunctions.sentiment, DataTypes.IntegerType);
		sqlContext.udf().register("score", nlpFunctions.score, DataTypes.StringType);
		this.sqlContext.udf().register("sentenceRegex", regexFunctions.sentenceRegex, DataTypes.StringType);
		
		JavaRDD<DocumentBeans> documents_rdd = jsc.parallelize(Arrays.asList(
				new DocumentBeans(1, "Stanford University is located in California. It is a great university."),
				new DocumentBeans(2, "Bucharest is located in Romania. It has a great Polytechnic university.")
				));
		documents_df = sqlContext.createDataFrame(documents_rdd, DocumentBeans.class).select("doc_id", "text");
		sentenceSplit_df = documents_df
				.withColumn("sentences", callUDF("ssplit", col("text")))
				.select(functions.col("doc_id"), functions.explode(functions.col("sentences")).as("sen"));
	}

	@Test 
	public void testPOSTags(){
		DataFrame pos_df = sentenceSplit_df
				.withColumn("pos_tags", callUDF("pos", col("sen")));
		pos_df.show(false);	
		String[] pos_tags = new String[]{"NNP", "NNP", "VBZ", "JJ", "IN", "NNP", "."};
		assertArrayEquals(pos_tags, pos_df.head().getList(2).toArray(new String[0]));
	}

	@Test
	public void testTokenizer(){
		DataFrame tokens_df = sentenceSplit_df
				.withColumn("words", callUDF("tokenize", col("sen")));
		tokens_df.show(false);
		assertEquals(7, tokens_df.head().getList(2).size());
	}

	@Test
	public void testLemmas(){
		String[] lemmas = new String[]{"Stanford", "University", "be", "located", "in", "California", "."};
		DataFrame lemmas_df = sentenceSplit_df
				.withColumn("lemmas", callUDF("lemma", col("sen")));
		lemmas_df.show(false);
		assertArrayEquals(lemmas, lemmas_df.head().getList(2).toArray(new String[0]));
	}

	@Test
	public void testNER(){
		DataFrame ners_df = sentenceSplit_df
				.withColumn("ners", callUDF("ner", col("sen")));
		ners_df.show(false);
		assertEquals("LOCATION", ners_df.head().getList(2).get(5));
	}

	@Test
	public void sentimentBySentence(){
		Integer[] expected = new Integer[]{1,4,1,3}; 
		DataFrame sentimentBySentence_df = sentenceSplit_df
				.withColumn("sentiment", callUDF("sentiment", col("sen")));
		sentimentBySentence_df.show(false);
		Integer[] sentiment = sentimentBySentence_df
				.collectAsList()
				.stream()
				.map(r -> r.getInt(2))
				.collect(Collectors.toList())
				.toArray(new Integer[0]);		
		assertArrayEquals(expected, sentiment);
	}

	@Test 
	public void sentimentByDocument(){
		String[] expected = new String[]{"pos", "neu"};
		DataFrame sentimentBySentence_df = sentenceSplit_df
				.withColumn("sentiment", callUDF("sentiment", col("sen")));

		DataFrame sentimentDocument_df = sentimentBySentence_df
				.groupBy("doc_id")
				.agg(functions.avg(functions.col("sentiment")).as("sentiment_avg"))
				.withColumnRenamed("doc_id", "id");

		DataFrame result_df = documents_df
				.join(sentimentDocument_df, documents_df.col("doc_id").equalTo(sentimentDocument_df.col("id")), "left_outer")
				.select("id", "text", "sentiment_avg")
				.withColumn("sentiment", callUDF("score", col("sentiment_avg")));
		result_df.show(false);
		String[] sentiment =  result_df
				.collectAsList()
				.stream()
				.map(r -> r.getString(3))
				.collect(Collectors.toList())
				.toArray(new String[0]);
		assertArrayEquals(expected, sentiment);
	}


	/**
	 * Loads a prepared parquet file containing 32,000 tweets, apply regex, explode into sentences,
	 * calculates sentimant for each sentence, average sentiment value for each tweet and add labels
	 * for each tweet.
	 */
	@Test
	public void testTwitterSentimentWithDataFrame(){
		String path = ClassLoader.getSystemResource("twitter_test.parquet").toString();
		int cores = Runtime.getRuntime().availableProcessors();
		int multiplicationFactor = 3;
		DataFrame loaded = sqlContext
				.read()
				.parquet(path)
				.withColumn("regexed", callUDF("sentenceRegex", col("text")))
				.select("id", "regexed").where(col("regexed").isNotNull())
				.repartition(cores*multiplicationFactor)
				.cache();
		
		DataFrame sentences = loaded
				.withColumn("sentences", callUDF("ssplit", col("regexed")))
				.select(functions.col("id"), functions.explode(functions.col("sentences")).as("sen"))
				.where(functions.col("sen").isNotNull());
		
		DataFrame sentimentSentence = sentences
				.withColumn("sentiment", callUDF("sentiment", col("sen")));
		
		DataFrame sentimentDocument = sentimentSentence
				.groupBy("id")
				.agg(functions.avg(functions.col("sentiment")))
				.withColumnRenamed("id", "doc_id")
				.withColumnRenamed("avg(sentiment)", "avg_sentiment")
				.repartition(cores*multiplicationFactor)
				.cache();
		
		DataFrame result = loaded.join(sentimentDocument, loaded.col("id").equalTo(sentimentDocument.col("doc_id")), "left_outer")
				.select("id", "regexed", "avg_sentiment")
				.withColumn("sentiment", callUDF("score", col("avg_sentiment")));
		result.show(false);
		assertEquals(1.5d, result.head().getDouble(2), 0.01d);
		assertEquals("neg", result.head().getString(3));
	}
	
	@Test
	public void testSsplit(){
		DataFrame sentenceSplit_df = documents_df
				.withColumn("sentences", callUDF("ssplit", col("text")))
				.select(functions.col("doc_id"), functions.explode(functions.col("sentences")).as("sen"));
		sentenceSplit_df.show(false);
		assertEquals(4, sentenceSplit_df.count());
		assertEquals("Stanford University is located in California.", sentenceSplit_df.head().getString(1));
	}
	

	@After
	public void stop(){
		sqlContext.clearCache();
		jsc.sc().stop();
		jsc.sc().stop();
	}
}
