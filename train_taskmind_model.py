import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import joblib

# ---------------------------------------------------------------------------
# train_taskmind_model.py
# ---------------------------------------------------------------------------
# This script trains a simple text classifier for YouTube-style video metadata.
# Workflow summary:
#  1. Load labeled CSV (title, description, tags, label).
#  2. Concatenate text fields to form one input string per video.
#  3. Convert labels to integer ids using LabelEncoder.
#  4. Split into train/test sets (stratified by label to preserve class ratios).
#  5. Use a pre-trained Sentence-BERT model to produce fixed-size embeddings
#     for each text (these are dense vectors, typically 768 dims for mpnet).
#  6. Train a scikit-learn LogisticRegression on these embeddings.
#  7. Evaluate and save the trained artifacts (classifier, label encoder,
#     and SBERT model for future inference).
#
# Important notes:
# - SBERT encoding is the slowest and most GPU/CPU intensive part. Consider
#   precomputing embeddings and saving them (np.save) if you iterate often.
# - LogisticRegression here is a lightweight classifier that trains quickly on
#   embeddings; it's not trained by epochs like deep nets — it uses an
#   iterative numerical optimizer (LBFGS by default).
# - If you run into memory/GPU OOM during encoding, reduce batch_size or
#   switch to a smaller SBERT model such as 'all-MiniLM-L6-v2'.
# ---------------------------------------------------------------------------

# === 1. Load dataset ===
# Expect a CSV with at least these columns: title, description, tags, label
# If the file isn't in the current working directory, use an absolute path.
df = pd.read_csv("trending_labeled.csv")

# Combine text fields into a single string per sample.
# Using astype(str) prevents NaN from causing errors when concatenating.
# Result: df['text'] is a pandas Series of length N (number of samples).
df["text"] = df["title"].astype(str) + " " + df["description"].astype(str) + " " + df["tags"].astype(str)

# Remove rows that don't have a label. Label must be present to train.
df = df.dropna(subset=["label"])

# === 2. Encode labels ===
# LabelEncoder maps class names (strings) to integers 0..K-1. Keep the
# fitted encoder so we can translate model outputs back to human-readable
# labels during inference.
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# === 3. Train-test split ===
# Use a stratified split so class proportions remain similar in train/test.
# test_size=0.2 -> 80% train, 20% test. random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_encoded"], test_size=0.2, random_state=42, stratify=df["label_encoded"]
)

# === 4. Generate SBERT embeddings ===
print("Loading SBERT model...")
# 'all-mpnet-base-v2' produces 768-d embeddings and has good accuracy.
# If you have limited VRAM or want much faster runs, consider
# 'all-MiniLM-L6-v2' (384-d) as a faster alternative.
model = SentenceTransformer("all-mpnet-base-v2")  # high-quality embeddings

print("Encoding training data...")
# The heavy work: model.encode maps a list of texts -> numpy array of shape
# (n_samples, embedding_dim). This step is GPU-accelerated when device='cuda'.
# batch_size controls memory vs throughput. If you encounter OOM, lower it.
X_train_emb = model.encode(X_train.tolist(), batch_size=16, show_progress_bar=True, device="cuda")

print("Encoding test data...")
X_test_emb = model.encode(X_test.tolist(), batch_size=16, show_progress_bar=True, device="cuda")

# === 5. Train Logistic Regression classifier ===
print("Training classifier...")
# LogisticRegression is a fast linear classifier. It expects feature vectors
# as a 2D array of shape (n_samples, n_features) — here n_features ==
# embedding dimension (e.g. 768 for mpnet). The solver is iterative and will
# run until convergence or until max_iter steps. class_weight='balanced'
# helps with class imbalance by weighting the loss inversely to class freq.
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_emb, y_train)

# === 6. Evaluate ===
# Predict on test embeddings and print classification metrics. If some
# classes are missing in the test split you may need to pass `labels=` to
# classification_report to avoid a mismatch between label ids and target_names.
y_pred = clf.predict(X_test_emb)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 7. Save models ===
# Persist the trained classifier, label encoder, and SBERT model so you can
# load them later for inference without retraining. The SBERT `.save()` will
# store the full preprocessor + weights in 'models/sbert_encoder'.
joblib.dump(clf, "models/taskmind_classifier.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
model.save("models/sbert_encoder")

print("\n✅ Model training complete! Saved in 'models/' folder.")