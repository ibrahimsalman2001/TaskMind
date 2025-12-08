Artifacts:
- embeddings.npy : SBERT embeddings cache (optional)
- labels.csv : labels used during training
- label_classes.json : mapping index -> class name
- video_classifier.pkl : sklearn classifier trained on embeddings
- sbert_model/ : SentenceTransformer folder (for inference)
- label_encoder.pkl : LabelEncoder for labels

Inference steps:
1) Load sbert_model with SentenceTransformer
2) Encode new text (title_cleaned + description_cleaned + tags_cleaned)
3) clf.predict(embedding) -> class index; map with label_encoder
