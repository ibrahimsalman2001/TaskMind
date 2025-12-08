# 🧠 TaskMind – AI-Powered YouTube Content Classifier
TaskMind is an intelligent productivity assistant that classifies YouTube videos into **seven content categories** using advanced Natural Language Processing (NLP).  
It helps users understand and optimize their content consumption patterns through semantic video analysis, dashboards, and smart insights.

---

## 🚀 Features

### 🎥 Real-Time Video Classification
Classifies any YouTube video into:
- Educational
- Entertainment
- Gaming
- Music
- News
- Vlogs
- Other

### 🧠 AI-Powered NLP Pipeline
- Zero-shot labeling using `facebook/bart-large-mnli`
- Semantic embeddings using Sentence-BERT (`all-mpnet-base-v2`)
- Supervised classifier using Logistic Regression
- Achieves ~70% accuracy on real-world YouTube data

### 🌐 FastAPI Backend
Exposes two API endpoints:
- `POST /classify` (single video)
- `POST /classify-batch` (multiple videos)

### 🔗 YouTube Metadata Integration
Uses YouTube titles, descriptions, and tags to generate semantic predictions.

### 📊 Smart Insights
Designed to work with a dashboard that visualizes:
- Viewing patterns
- Productivity scores
- Trends in content consumption

---

## 🧬 Model Architecture

```
YouTube Metadata (title, description, tags)
            │
            ▼
Zero-Shot Labeling (BART MNLI)
            │
            ▼
Labeled Dataset → SBERT Embeddings
            │
            ▼
Logistic Regression Classifier
            │
            ▼
FastAPI REST Endpoints
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/ibrahimsalman2001/TaskMind.git
cd TaskMind
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the API

Start the backend:
```bash
uvicorn app:app --reload --port 8000
```

Open the Swagger UI:  
👉 http://127.0.0.1:8000/docs

---

## 🧪 Example Request

**POST /classify**
```json
{
  "title": "How to Learn Python Fast",
  "description": "Complete crash course for beginners",
  "tags": "python, tutorial"
}
```

**Response:**
```json
{
  "label": "Educational",
  "confidence": 0.93
}
```

---

## 📁 Project Structure

```
TaskMind/
│── app.py                      # FastAPI backend
│── train_taskmind_model.py     # Model training pipeline
│── test_single.py              # Local model testing
│── requirements.txt
│── README.md
│── models/
│   ├── taskmind_classifier.pkl
│   ├── label_encoder.pkl
│   └── sbert_encoder/
│
└── data/ (optional)
```

---

## 📦 Model Files

If you want to include trained models:

Use Git LFS for `sbert_encoder/`:
```bash
git lfs install
git lfs track "models/sbert_encoder/*"
git add .gitattributes
git add models/
git commit -m "Add model files"
git push
```

---

## 📝 Contributions

Pull requests are welcome — this is an actively evolving FYP project.

---

## 🛡️ License

MIT License (or your university requirements)

---

## 🙋 Author

**Ibrahim Salman**  
FAST NUCES