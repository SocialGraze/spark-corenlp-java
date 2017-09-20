package com.socialgraze.spark.corenlp;

import java.io.Serializable;

public class DocumentBeans implements Serializable{
	private static final long serialVersionUID = -8315449885809501494L;
	public int doc_id;
	public String text;
	
	public DocumentBeans(int doc_id, String text) {
		this.doc_id = doc_id;
		this.text = text;
	}
	
	public int getDoc_id() {
		return doc_id;
	}
	
	public void setDoc_id(int doc_id) {
		this.doc_id = doc_id;
	}
	
	public String getText() {
		return text;
	}
	
	public void setText(String text) {
		this.text = text;
	}
	
	@Override
	public String toString() {
		return "Documents [doc_id=" + doc_id + ", text=" + text + "]";
	}
	
}