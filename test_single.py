from sentence_transformers import SentenceTransformer
import joblib

# Load saved artifacts
enc = SentenceTransformer("models/sbert_encoder")
clf = joblib.load("models/taskmind_classifier.pkl")
le  = joblib.load("models/label_encoder.pkl")

def predict(title, description="", tags="", min_conf=0.45):
    text = f"{title} {description} {tags}"
    emb = enc.encode([text], device="cuda")
    proba = clf.predict_proba(emb)[0]
    idx = proba.argmax()
    label = le.inverse_transform([idx])[0]
    conf = float(proba[idx])
    if conf < float(min_conf):     # <— make sure it's float
        label = "Other"
    return label, conf


# Try a few
print(predict(
    "Syrian Revolution l Islamic Nasheed - 'Take Our Blood' - English Lyrics",    "This is a Nasheed by Ahrar al Sham in the Syrian civil war",    "revolution islamic revival"))
#print(predict("MrBeast Challenge", "We buried a Lamborghini", "challenge, fun"))
#print(predict("Breaking News: Market Crash", "S&P 500 plunges 5%", "news, market"))
#print(predict("Valorant Ranked Match", "Duelist gameplay ranked", "valorant, gameplay"))