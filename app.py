from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import io
import os
import torch

# === Load models ===
# Try to load local model, fallback to pre-trained model if not found
sbert_path = "models/sbert_encoder"
if os.path.exists(sbert_path):
    encoder = SentenceTransformer(sbert_path)
else:
    print("⚠️  Local SBERT model not found. Using pre-trained 'all-mpnet-base-v2' model.")
    encoder = SentenceTransformer("all-mpnet-base-v2")

# Determine device (use CPU if CUDA not available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

classifier = joblib.load("models/taskmind_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

app = FastAPI(
    title="TaskMind Video Classifier API",
    description="Endpoints for single and batch YouTube video classification",
    version="1.0.0"
)

# === Pydantic model for input ===
class VideoInput(BaseModel):
    title: str
    description: str = ""
    tags: str = ""

# === Single video classification ===
@app.post("/classify")
async def classify_video(video: VideoInput, min_conf: float = 0.45):
    text = f"{video.title} {video.description} {video.tags}"
    emb = encoder.encode([text], device=device)
    probs = classifier.predict_proba(emb)[0]
    idx = probs.argmax()
    label = label_encoder.inverse_transform([idx])[0]
    conf = float(probs[idx])
    if conf < min_conf:
        label = "Other"
    return {
        "label": label,
        "confidence": round(conf, 3)
    }

# === Batch classification from JSON array ===
@app.post("/classify-batch")
async def classify_batch(videos: List[VideoInput], min_conf: float = 0.45):
    texts = [f"{v.title} {v.description} {v.tags}" for v in videos]
    embs = encoder.encode(texts, batch_size=16, device=device)
    probas = classifier.predict_proba(embs)
    preds, confs = [], []
    for proba in probas:
        idx = proba.argmax()
        label = label_encoder.inverse_transform([idx])[0]
        conf = float(proba[idx])
        if conf < min_conf:
            label = "Other"
        preds.append(label)
        confs.append(conf)
    return [{"label": l, "confidence": round(c, 3)} for l, c in zip(preds, confs)]

'''
# === Optional: CSV upload endpoint ===
@app.post("/classify-csv")
async def classify_csv(file: UploadFile = File(...), min_conf: float = 0.45):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    for col in ["title", "description", "tags"]:
        if col not in df.columns:
            df[col] = ""
    texts = (df["title"].astype(str) + " " + df["description"].astype(str) + " " + df["tags"].astype(str)).tolist()
    embs = encoder.encode(texts, batch_size=16, device="cuda")
    probas = classifier.predict_proba(embs)
    labels = []
    confs = []
    for proba in probas:
        idx = proba.argmax()
        label = label_encoder.inverse_transform([idx])[0]
        conf = float(proba[idx])
        if conf < min_conf:
            label = "Other"
        labels.append(label)
        confs.append(conf)
    df["pred_label"] = labels
    df["pred_conf"] = confs
    # Return top 10 preview rows
    return df.head(10).to_dict(orient="records")
    '''