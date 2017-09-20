package com.socialgraze.spark.corenlp;

import java.io.Serializable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.spark.sql.api.java.UDF1;

/**
 * @author cristian
 * A collection of regex functions to prepare real world Twitter texts for Stanford CoreNLP sentiment analysis.
 */
public class RegexFunctions implements Serializable{
	private static final long serialVersionUID = -6695975321301773264L;
	
	public RegexFunctions() {}
	
	/**
	 * Convenience method to transform Twitter texts into clean sentences.
	 * Also for getting some accuracy from sentiment calculation, sentences with less than 5 words are eliminated.
	 */
	public UDF1<String, String> sentenceRegex = new UDF1<String, String>(){
		private static final long serialVersionUID = 8536541840126871826L;
		@Override
		public String call(String s) throws Exception {	
			String regexed = sentenceRegexFilter(s);
			if (regexed!=null && !regexed.isEmpty()){
				String[] splits = regexed.split("\\s+");
				if(splits.length>=5){
					return regexed;
				}else {
					return null;
				}
			}else{
				return null;
			}
		}
	};
	
	/**
	 * @param text
	 * @return
	 * Fix second, third, etc sentences in a Twitter text.
	 * Used in {@link RegexFunctions#sentenceRegex}
	 */
	private String secondSentence (String text){
		Pattern patt = Pattern.compile("(\\w+)([\\?\\.\\!]+)(\\s+)(\\w)");
		Matcher matcher = patt.matcher(text);
		StringBuffer sb = new StringBuffer(text.length());

		while(matcher.find()){
			String s = matcher.group();	
			String[] splits = s.split("\\s+");
			splits[1] = splits[1].substring(0, 1).toUpperCase() + splits[1].substring(1); 
			s = splits[0] + " " + splits[1];
			matcher.appendReplacement(sb, s);			
		}
		text = matcher.appendTail(sb).toString();

		patt = Pattern.compile("([A-Z][a-zA-Z]*\\s*)([\\?\\.\\!]+)(\\s+)(\\w+)");
		matcher = patt.matcher(text);
		sb = new StringBuffer(text.length());
		while(matcher.find()){
			String s = matcher.group();
			String[] splits = s.split("\\s+");

			if(splits[1].substring(0, 1).equals("'") || splits[1].substring(0, 1).equals("\"")){
				splits[1] = splits[1].substring(0,1) + splits[1].substring(1, 2).toUpperCase() + splits[1].substring(2);
			}else{
				splits[1] = splits[1].substring(0, 1).toUpperCase() + splits[1].substring(1);
			}
			s = splits[0] + " " + splits[1];
			matcher.appendReplacement(sb, s);			
		}
		text = matcher.appendTail(sb).toString();
		return text;
	}
	
	/**
	 * @param text
	 * @return 
	 * Returns clean sentences from tweets text.
	 */
	public String sentenceRegexFilter(String text){
		Pattern emptyLines = Pattern.compile("[\\t\\n\\r]+");
		Pattern url = Pattern.compile("((www\\.[^\\s]+)|(https?://[^\\s]+))");
		Pattern allHTML = Pattern.compile("\\<.*?>");
		Pattern email = Pattern.compile("[a-zA-Z0-9]+@[a-zA-Z]+\\.[a-zA-Z]+");
		Pattern entitiesHTML = Pattern.compile("\\&.*?;");
		Pattern hashtags = Pattern.compile("(\\s+(via)\\s+)?#[^\\s]*");
		Pattern capitalWords = Pattern.compile("\\([A-Z\\s]+\\)");
		Pattern mentions = Pattern.compile("(\\s+(via)\\s+)?@[^\\s]*");
		Pattern nonAlphanumeric = Pattern.compile("[\\\"\\~\\*\\(\\)\\+\\-\\_\\{\\}\\=]+");
		Pattern specialChars = Pattern.compile("[^\\x20-\\x7e]");
		Pattern emptySpaces = Pattern.compile("\\s+");
		Pattern emptySpacesStartEnd = Pattern.compile("(^\\s+)|(\\s+$)");	
		Pattern marksInsideWords = Pattern.compile("(\\S)([\\.\\?\\!,;])(\\S)");//removed : /
		Pattern alternativeMark = Pattern.compile("(\\w+)(/)");
		Pattern isolatedMarks = Pattern.compile("(\\w)(\\s+)([\\.\\?\\!,:;]+)(\\s+)");
		Pattern multipleIsolatedMarks = Pattern.compile("(\\s+)([\\.\\?\\!;:,']+)(\\s+)");
		Pattern isolatedMarksEOL = Pattern.compile("(\\S)(\\s+)([(\\.\\?\\!)])+$");
		Pattern wordWithMultipleMarks = Pattern.compile("(\\S)([\\?\\.\\!.;]+)((\\s+)|(\\S))");//removed :
		Pattern marksAtWordBeginning = Pattern.compile("(\\s+)([\\?\\.\\!.;:]+)([a-z])");
		Pattern endLine = Pattern.compile("(\\S)([\\?\\.\\!])(\\s+)([\\?\\.\\!]+)$");
		Pattern endLineNoMark = Pattern.compile("([a-zA-Z0-9'$\\]])$");
		Pattern misplacedPunct = Pattern.compile("(\\w)(\\s+)([\\.\\?\\!,:;])(\\S)");
		Pattern asciiFaces = Pattern.compile("(\\S?;P)|(\\S?;p)|(\\S?;D)|(\\S?;3)|(\\S?:-\\|)|(\\S?:\\[)|"
				+ "(\\S?:/)|(\\S?;\\))|(\\S?\\^\\^)|(\\S?\\^.\\^)|(\\S?o_O)|(\\S?o_0)|(\\S?:\\|)|(\\S?:')|"
				+ "(\\S?\\-_\\-\\S?)|(;;)|(re:)|( : )|(\\\\)");
		Pattern numberFormat = Pattern.compile("(\\d),(\\s?)(\\d)");
		Pattern decimalFormat = Pattern.compile("(\\d.)(\\s)(\\d)");
		Pattern wrongCommaPlace = Pattern.compile("(\\.')");
		Pattern wrongColonPlace = Pattern.compile("(\\S?),(\\s+)([\\!\\?\\.])");
		Pattern wrongMarkPlace = Pattern.compile("(\\S?)(\\s+)([\\!\\?\\.\\$])(\\S)");

		text = emptyLines.matcher(text).replaceAll(" ");
		text = wrongCommaPlace.matcher(text).replaceAll("'.");
		text = url.matcher(text).replaceAll("*");
		text = allHTML.matcher(text).replaceAll("");
		text = email.matcher(text).replaceAll("");
		text = hashtags.matcher(text).replaceAll(" ");
		text = capitalWords.matcher(text).replaceAll(" ");
		text = mentions.matcher(text).replaceAll(" ");
		text = wrongColonPlace.matcher(text).replaceAll("$1$3");
		text = wrongMarkPlace.matcher(text).replaceAll("$1$2$4");	
		text = asciiFaces.matcher(text).replaceAll("");
		text = nonAlphanumeric.matcher(text).replaceAll(" ");
		text = specialChars.matcher(text).replaceAll("");
		text = entitiesHTML.matcher(text).replaceAll("");
		text = misplacedPunct.matcher(text).replaceAll("$1$3 $4");
		text = multipleIsolatedMarks.matcher(text).replaceAll("$2");
		text = isolatedMarks.matcher(text).replaceAll("$1$3 ");
		text = isolatedMarksEOL.matcher(text).replaceAll("$1$3");
		text = marksAtWordBeginning.matcher(text).replaceAll("$1 $3");
		text = wordWithMultipleMarks.matcher(text).replaceAll("$1" + ". " + "$3");
		text = endLine.matcher(text).replaceAll("$1$2");
		text = endLineNoMark.matcher(text).replaceAll("$1.");
		text = marksInsideWords.matcher(text).replaceAll("$1$2 $3");
		text = alternativeMark.matcher(text).replaceAll("$1 $2 ");// "/"
		text = numberFormat.matcher(text).replaceAll("$1,$3");
		text = decimalFormat.matcher(text).replaceAll("$1$3");
		text = multipleIsolatedMarks.matcher(text).replaceAll(" ");//Redundant required
		text = decimalFormat.matcher(text).replaceAll("$1$3");//Redundant required	
		text = emptySpaces.matcher(text).replaceAll(" ");
		text = emptySpacesStartEnd.matcher(text).replaceAll("");	

		//Upercase first letter
		if (text.length()>1){	
			if( text.substring(0, 1).equals("'")|| text.substring(0, 1).equals("\"")|| text.substring(0, 1).equals("/") ){
				text = text.substring(0,1) + text.substring(1, 2).toUpperCase() + text.substring(2); 
			}else{
				text = text.substring(0,1).toUpperCase() + text.substring(1);
			}
		}

		text = secondSentence(text);
		text = endLineNoMark.matcher(text).replaceAll("$1.");//Redundant required
		return text.trim();
	}
}
