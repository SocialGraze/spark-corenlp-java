package com.socialgraze.spark.corenlp;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Properties;

import org.apache.spark.sql.api.java.UDF1;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.Tree;

/**
 * @author cristian
 * A collection of Spark SQL UDFs that wrap CoreNLP annotators and simple functions.
 * Inspired from https://github.com/databricks/spark-corenlp
 * @see [[edu.stanford.nlp.simple]]
 */
public class CoreNlpFunctions implements Serializable{
	private static final long serialVersionUID = -461308288666175052L;
	private transient StanfordCoreNLP pipeline;
	
	public CoreNlpFunctions() {}
	
	/**
	 * @return pipeline
	 * For large DataFrames uncomment lines.
	 * Running this method with parse model significantly increase the speed (about a factor of 16)
	 */
	public StanfordCoreNLP getOrCreateSentimentPipeline(){
		if (pipeline == null){
			Properties props = new Properties();
			props.put("annotators", "tokenize, ssplit, parse, sentiment");
			//Alternatively run with parse model -> increase speed by a fator of about 16 -> see Readme.md 
//			props.put("annotators", "tokenize, ssplit, pos, parse, sentiment");
//			props.put("parse.model", "edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz");
			pipeline = new StanfordCoreNLP(props);
		}
		return pipeline;
	}
	
	/**
	 * Splits a document into sentences.
	 *  @see [[Document#sentences]]
	 */
	public UDF1<String, String[]> ssplit = new UDF1<String, String[]>(){
		private static final long serialVersionUID = -5825088868451214287L;
		@Override
		public String[] call(String document) throws Exception {
			return Arrays.stream((new Document(document).sentences())
					.toArray(new Sentence[0]))
					.map(s -> s.text())
					.toArray(String[]::new);
		}
	};
	
	/**
	 * Tokenizes a sentence into words.
	 * @see [[Sentence#words]]
	 */
	public UDF1<String, String[]> tokenize = new UDF1<String, String[]>(){
		private static final long serialVersionUID = -5758868540212921375L;
		@Override
		public String[] call(String sentence) throws Exception {
			return new Sentence(sentence).words().toArray(new String[0]);
		}		
	};
	
	/**
	 * Generates the part of speech tags of the sentence.
	 *  @see [[Sentence#posTags]]
	 */
	public UDF1<String, String[]> pos = new UDF1<String, String[]>(){
		private static final long serialVersionUID = -7014917125588179986L;
		@Override
		public String[] call(String sentence) throws Exception {
			return new Sentence(sentence).posTags().toArray(new String[0]);
		}	
	};
	
	/**
	 * Generates the word lemmas of the sentence.
	 * @see [[Sentence#lemmas]]
	 */
	public UDF1<String, String[]> lemma = new UDF1<String, String[]>(){
		private static final long serialVersionUID = -6539169480615197346L;
		@Override
		public String[] call(String sentence) throws Exception {
			return new Sentence(sentence).lemmas().toArray(new String[0]);
		}
	};
	
	/**
	 *  Generates the named entity tags of the sentence.
	 *  @see [[Sentence#nerTags]]
	 */
	public UDF1<String, String[]> ner = new UDF1<String, String[]>(){
		private static final long serialVersionUID = 7287338868170935091L;
		@Override
		public String[] call(String sentence) throws Exception {
			return new Sentence(sentence).nerTags().toArray(new String[0]);
		}
	};
	
	/**
	 * Measures the sentiment of an input sentence on a scale of 0 (strong negative) to 4 (strong positive)
	 * If the input contains multiple sentences, only the first one is used.
	 * @see [[Sentiment]]
	 */
	public UDF1<String, Integer> sentiment =  new UDF1<String, Integer>(){
		private static final long serialVersionUID = 6241979898487114757L;
		@Override
		public Integer call(String sentence) throws Exception {
			StanfordCoreNLP pipeline = getOrCreateSentimentPipeline();
			Annotation annotation = pipeline.process(sentence);
			Tree tree = annotation
					.get(CoreAnnotations.SentencesAnnotation.class)
					.get(0)
					.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);			
			return RNNCoreAnnotations.getPredictedClass(tree);
		}
	};	
	
	/**
	 * Map tree labels for average sentiment calculated for a paragraph (multiple sentences -> @see {@link Examples})
	 * If avg sentiment is positive and less than 2.0 than sentiment labelk is neg,
	 * if is equal to 2.0 sentiment lable is neu (neutral) if id greater than 2 and less or equal to 4,
	 * than label is pos.
	 * Sentiment label unk means that sentences tree contains no sentence {@link sentiment}
	 */
	public UDF1<Double, String> score = new UDF1<Double, String>(){
		private static final long serialVersionUID = -6662908667643087463L;
		@Override
		public String call(Double d) throws Exception {
			String sentiment = new String();
			if (d < 2.0 && d >= 0.0) sentiment = "neg";
			if (d>2 && d <= 4.0) sentiment = "pos";
			if(d==2) sentiment = "neu";
			if (d==-1) sentiment = "unk";
			return sentiment;
		}
	};
}
