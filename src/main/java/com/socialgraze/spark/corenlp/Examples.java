package com.socialgraze.spark.corenlp;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;

import org.apache.spark.sql.functions;


public class Examples {
	private SparkConf conf;
	public JavaSparkContext jsc;
	public SQLContext sqlContext;
	private CoreNlpFunctions nlpFunctions;

	/**
	 * Constructor
	 * Initialize Spark and register UDF functions with current SQLContext.
	 */
	public Examples() {
		this.conf = new SparkConf().setAppName("Spark-CoreNlp-Java").setMaster("local[*]");
		this.jsc = new JavaSparkContext(this.conf);
		this.sqlContext = new SQLContext(this.jsc);
		this.nlpFunctions = new CoreNlpFunctions();
		this.sqlContext.udf().register("ssplit", nlpFunctions.ssplit, DataTypes.createArrayType(DataTypes.StringType));
		this.sqlContext.udf().register("tokenize", nlpFunctions.tokenize, DataTypes.createArrayType(DataTypes.StringType));
		this.sqlContext.udf().register("pos", nlpFunctions.pos, DataTypes.createArrayType(DataTypes.StringType));
		this.sqlContext.udf().register("lemma", nlpFunctions.lemma, DataTypes.createArrayType(DataTypes.StringType));
		this.sqlContext.udf().register("ner", nlpFunctions.ner, DataTypes.createArrayType(DataTypes.StringType));
		this.sqlContext.udf().register("sentiment", nlpFunctions.sentiment, DataTypes.IntegerType);
		this.sqlContext.udf().register("score", nlpFunctions.score, DataTypes.StringType);
	}

	public static void main(String[] args){
		Examples processor = new Examples();
		JavaRDD<DocumentBeans> documents_rdd = processor.jsc.parallelize(Arrays.asList(
				new DocumentBeans(1, "Stanford University is located in California. It is a great university."),
				new DocumentBeans(2, "Bucharest is located in Romania. It has a great Polytechnic university.")
				));
		
		//Initialize a test DataFrame.
		DataFrame documents_df = processor.sqlContext.createDataFrame(documents_rdd, DocumentBeans.class).select("doc_id", "text");
		documents_df.show(false);
		
		//Split into sentences via explode sql function. 
		DataFrame sentenceSplit_df = documents_df
				.withColumn("sentences", callUDF("ssplit", col("text")))
				.select(functions.col("doc_id"), functions.explode(functions.col("sentences")).as("sen"));
		sentenceSplit_df.show(false);
		
		//Tokenize exploded sentences.
		DataFrame tokens_df = sentenceSplit_df
				.withColumn("words", callUDF("tokenize", col("sen")));
		tokens_df.show(false);
		
		//Calculate posTags on sentences.
		DataFrame pos_df = sentenceSplit_df
				.withColumn("pos_tags", callUDF("pos", col("sen")));
		pos_df.show(false);	
		
		//Calculate lemmas on sentences.
		DataFrame lemmas_df = sentenceSplit_df
				.withColumn("lemmas", callUDF("lemma", col("sen")));
		lemmas_df.show(false);
		
		//Calculate NER's on sentences.
		DataFrame ners_df = sentenceSplit_df
				.withColumn("ners", callUDF("ner", col("sen")));
		ners_df.show(false);

		//Calculate sentiment for each exploded sentence.
		DataFrame sentimentBySentence_df = sentenceSplit_df
				.withColumn("sentiment", callUDF("sentiment", col("sen")));
		sentimentBySentence_df.show(false);
		
		//Rebuild initial document via groupBy and average sentiment for contained sentences.
		DataFrame sentimentDocument_df = sentimentBySentence_df
				.groupBy("doc_id")
				.agg(functions.avg(functions.col("sentiment")).as("sentiment_avg"))
				.withColumnRenamed("doc_id", "id");
		sentimentDocument_df.show(false);
		
		//Put labels on sentiment for documents (i.e tweet texts)
		DataFrame result = documents_df
				.join(sentimentDocument_df, documents_df.col("doc_id").equalTo(sentimentDocument_df.col("id")), "left_outer")
				.select("id", "text", "sentiment_avg")
				.withColumn("sentiment", callUDF("score", col("sentiment_avg")));
		result.show(false);
	}
}
