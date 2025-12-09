import os, re, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# ========= CONFIG =========
DATA_PATH = r"D:/FYP/TaskMind/cleaned_labeled_dataset.xlsx"   # <-- change if needed
TEXT_COLS = ["title_cleaned", "description_cleaned", "tags_cleaned"]  # auto-create if missing
LABEL_COL = "label"  # final class column (e.g. Entertainment, Educational, Music...)
MODEL_NAME = "all-mpnet-base-v2"   # SBERT variant: good accuracy, 768-d embeddings
DEVICE = "cuda"                    # 'cuda' uses GPU; use 'cpu' if no GPU available
BATCH_SIZE = 32                      # batch size for encoding; reduce if you run out of VRAM
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUT_DIR = Path(r"D:\FYP\TaskMind\models")  # where artifacts will be saved
# ==========================

OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(s: str) -> str:
    if not isinstance(s, str): s = ""  # guard
    s = re.sub(r"http\S+|www\.\S+", " ", s)         # remove URLs
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)           # keep letters/numbers/space
    s = re.sub(r"\s+", " ", s).strip()              # collapse spaces
    return s

print("Loading dataset:", DATA_PATH)
# Load spreadsheet with cleaned columns if available. If not, we will
# fall back to the raw columns and apply basic cleaning below.
df = pd.read_excel(DATA_PATH)

# Ensure LABEL_COL exists and is non-empty
if LABEL_COL not in df.columns:
    # The script depends on a final label column. Stop early with a clear message
    # rather than producing unclear errors later.
    raise ValueError(f"'{LABEL_COL}' column not found. Please ensure your dataset has a final label column.")

# If cleaned text columns don’t exist, create them from raw ones
fallbacks = {
    "title_cleaned": "title",
    "description_cleaned": "description",
    "tags_cleaned": "tags",
}
for c in TEXT_COLS:
    # For each expected cleaned text column, try to use it; otherwise fall
    # back to the raw column name and apply `clean_text` to generate it.
    if c not in df.columns:
        raw = fallbacks.get(c)
        if raw not in df.columns:
            raise ValueError(f"Missing both '{c}' and fallback '{raw}'. Please include text columns.")
        # fillna + astype(str) ensures no NaN and consistent string inputs
        df[c] = df[raw].fillna("").astype(str).map(clean_text)
    else:
        # If the cleaned column exists, just fill missing and ensure string type
        df[c] = df[c].fillna("").astype(str)

# Combine title, description, tags into one input string per sample. This is
# what we feed to SBERT to obtain a single vector representation per sample.
df["text_data_final"] = (df[TEXT_COLS[0]] + " " + df[TEXT_COLS[1]] + " " + df[TEXT_COLS[2]]).str.replace(r"\s+", " ", regex=True).str.strip()

# Remove samples that have no label or no text after cleaning. This keeps
# downstream encoding and training free of empty examples.
df = df[ df[LABEL_COL].notna() & df["text_data_final"].str.len().gt(0) ].copy()
df.reset_index(drop=True, inplace=True)

print("Label distribution:\n", df[LABEL_COL].value_counts())

# Encode labels
le = LabelEncoder()
# y is an integer array of shape (N,) with values in 0..K-1 where K is number
# of unique classes. le.classes_ maps indices -> original class names.
y = le.fit_transform(df[LABEL_COL])

# SBERT embeddings on GPU
print(f"Loading SBERT model: {MODEL_NAME} on {DEVICE}")
# Instantiate SentenceTransformer. This internally loads a tokenizer and model
# weights. On first run it will download model files (large) to the HF cache.
sbert = SentenceTransformer(MODEL_NAME, device=DEVICE)
text_list = df["text_data_final"].tolist()

print("Encoding texts to embeddings (this uses your GPU)…")
# encode(...) returns an array of shape (N, D) where D is embedding dim
# (768 for mpnet). This is the main heavy step. If you have VRAM issues,
# lower BATCH_SIZE or set device='cpu' (slower).
embeddings = sbert.encode(text_list, show_progress_bar=True, batch_size=BATCH_SIZE, convert_to_numpy=True)

# Save embeddings and label metadata so we can skip encoding on future runs.
# This is highly recommended if you iterate often — encoding is the slow part.
np.save(OUT_DIR / "embeddings.npy", embeddings)
df[[LABEL_COL]].to_csv(OUT_DIR / "labels.csv", index=False)
with open(OUT_DIR / "label_classes.json", "w", encoding="utf-8") as f:
    json.dump(list(le.classes_), f, ensure_ascii=False, indent=2)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Simple, reliable baseline on embeddings
clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
# Fit a simple linear classifier on top of the fixed embeddings. Logistic
# Regression is fast and often strong when embeddings are informative.
clf.fit(X_train, y_train)

# Evaluate

# Predict on the held-out test set and print evaluation metrics. We compute
# `present_labels` to avoid a potential ValueError when some classes are
# absent from the test set (common for small or imbalanced datasets).
y_pred = clf.predict(X_test)
present_labels = np.unique(np.concatenate((y_test, y_pred)))
present_target_names = [le.classes_[i] for i in present_labels]
print("Classes present in test/pred (label ids):", present_labels)
print("Corresponding class names:", present_target_names)
report = classification_report(
    y_test,
    y_pred,
    labels=present_labels.tolist(),
    target_names=present_target_names,
    digits=4,
)
print("\n=== Classification Report ===\n", report)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)

# Save artifacts for inference
import joblib
joblib.dump(clf, OUT_DIR / "video_classifier.pkl")
# Save the fine-tuned artifacts. The classifier is small and pickled with
# joblib. SBERT's .save() writes the tokenizer and model files for later use
# with SentenceTransformer(..., local_folder).
sbert.save(str(OUT_DIR / "sbert_model"))            # saves transformer & tokenizer
joblib.dump(le, OUT_DIR / "label_encoder.pkl")

# Also save a small README with usage
with open(OUT_DIR / "README.txt", "w", encoding="utf-8") as f:
    f.write(
        "Artifacts:\n"
        "- embeddings.npy : SBERT embeddings cache (optional)\n"
        "- labels.csv : labels used during training\n"
        "- label_classes.json : mapping index -> class name\n"
        "- video_classifier.pkl : sklearn classifier trained on embeddings\n"
        "- sbert_model/ : SentenceTransformer folder (for inference)\n"
        "- label_encoder.pkl : LabelEncoder for labels\n\n"
        "Inference steps:\n"
        "1) Load sbert_model with SentenceTransformer\n"
        "2) Encode new text (title_cleaned + description_cleaned + tags_cleaned)\n"
        "3) clf.predict(embedding) -> class index; map with label_encoder\n"
    )