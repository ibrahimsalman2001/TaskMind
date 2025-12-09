import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch

LABELS = [
    "Educational",
    "Entertainment",
    "Gaming",
    "Music",
    "News",
    "Vlogs",
    "Other",
]

HYPOTHESIS_TEMPLATE = "This YouTube video is about {}."

def _clip(txt, max_chars=2000):
    txt = str(txt or "")
    return txt if len(txt) <= max_chars else (txt[:max_chars] + " …")

def build_text(row):
    title = _clip(row.get("title", ""))
    desc  = _clip(row.get("description", ""))
    tags  = _clip(row.get("tags", ""))
    return f"{title}\n\nDescription: {desc}\n\nTags: {tags}"

def main(input_csv, output_csv, model_name, batch_size, min_conf=None):
    # Load data
    df = pd.read_csv(input_csv)
    required = {"title", "description", "tags"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV must contain columns: {required}. Missing: {missing}")

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1

    # Prepare classifier
    clf = pipeline(
        task="zero-shot-classification",
        model=model_name,                  # "facebook/bart-large-mnli"
        device=device,
        truncation=True
    )

    texts = [build_text(row) for _, row in df.iterrows()]

    labels_out, scores_out, full_scores_out = [], [], []
    n = len(texts)

    for i in tqdm(range(0, n, batch_size), desc="Classifying"):
        batch = texts[i:i+batch_size]
        results = clf(
            batch,
            candidate_labels=LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        if isinstance(results, dict):
            results = [results]

        for res in results:
            best_label = res["labels"][0]
            best_score = float(res["scores"][0])
            all_scores = {lab: float(scr) for lab, scr in zip(res["labels"], res["scores"])}

            # Optional confidence threshold: route low-confidence to "Other"
            if min_conf is not None and best_score < min_conf:
                best_label = "Other"

            labels_out.append(best_label)
            scores_out.append(best_score)
            # Keep scores for auditing in a fixed order
            ordered = {lab: all_scores.get(lab, 0.0) for lab in LABELS}
            full_scores_out.append(json.dumps(ordered))

    df["label"] = labels_out
    df["label_score"] = scores_out
    df["label_scores_json"] = full_scores_out

    df.to_csv(output_csv, index=False)
    print("\nSaved:", output_csv)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot label YouTube trending CSV.")
    parser.add_argument("--input", default="trending.csv", help="Path to input CSV.")
    parser.add_argument("--output", default="trending_labeled.csv", help="Path to output CSV.")
    parser.add_argument("--model", default="facebook/bart-large-mnli", help="HuggingFace model name.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--min_conf", type=float, default=None, help="Optional min confidence; below -> 'Other'. e.g. 0.45")
    args = parser.parse_args()

    main(args.input, args.output, args.model, args.batch_size, args.min_conf)
