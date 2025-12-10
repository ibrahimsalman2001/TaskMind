**TaskMind — Multi-Modal Video Classification: Detailed Pipeline Documentation**

Overview
--------
+ Project: TaskMind — a three-pipeline, multi-modal video classification system that combines Computer Vision (frames + OCR), Audio Transcription (ASR) and Metadata analysis to classify YouTube or local videos into a fixed set of categories defined in `keywords.json`.
+ Goal: Provide an explainable, extendable pipeline that produces per-modality category scores and a final aggregated probability distribution.

How to use this document
------------------------
This file is intended to be self-contained and used for:
- Developer onboarding: precise function-level details and commands to run the system locally.
- Project demonstration: step-by-step explanation and a suggested demo plan for a panel.
- Extension & research: clear places to improve, instrument, or benchmark.

Table of contents
-----------------
1. High-level architecture and data flow
2. Computer Vision pipeline (details + code references)
3. Audio (ASR) pipeline (details + code references)
4. Metadata pipeline (details + code references)
5. Scoring, normalization and aggregation (formulas and examples)
6. Output format and sample JSON
7. Implementation notes, parameters and commands
8. Failure modes, limitations and mitigation
9. Evaluation, metrics and test set suggestions
10. Demo script and final-year panel Q&A (technical + non-technical)
11. Next steps & recommended improvements

1 — High-level architecture and data flow
----------------------------------------
Inputs: a YouTube URL (downloaded using `yt-dlp`) or a local video file. The pipeline runs three independent modules and fuses their outputs.

High-level processing steps (detailed):
1. Video acquisition: download (yt-dlp) or use local file. Save to a temporary working directory.
2. Frame extraction: sample `N` frames from the video using `ffmpeg` / OpenCV utilities (`frame_extractor.py`). Default `N=6`.
3. CV pipeline: classify each frame with `vision_model.classify_frame()` (EfficientNet-B0) and extract text from each frame with EasyOCR (`ocr_module.extract_text_from_frames()`).
4. Audio pipeline: extract audio (`audio_transcriber.extract_audio()`), transcribe (Whisper `small`), chunk transcription, and map tokens to categories via `classify_text_chunks()`.
5. Metadata pipeline: tokenize title/description/tags and map tokens to categories (`metadata_classifier.classify_metadata()`).
6. Aggregation: normalize each modality to a probability vector and compute a weighted sum to produce the final distribution (`pipeline.aggregate_scores()`).

2 — Computer Vision pipeline (CV)
---------------------------------
Goal: produce visual evidence that supports category labels (objects, scene context, on-screen text).

Files: `frame_extractor.py`, `vision_model.py`, `ocr_module.py`, `metadata_classifier.py` (used for final scoring)

Frame extraction (what and why)
- Code: `frame_extractor.extract_sample_frames(video_path, num_frames, output_dir)`
- Method: sample `num_frames` evenly across video duration (default 6). Sampling fewer frames reduces compute but may miss context; more frames increase robustness at cost of slower runtime.
- Example ffmpeg command used by the module (conceptually):

```bash
ffmpeg -i video.mp4 -vf "select='not(mod(n,FRAME_INTERVAL))'" -vsync vfr -q:v 2 frames/out_%03d.jpg
```

Image classification (model + preprocessing)
- Code: `vision_model.classify_frame(image_path, top_k=3)`
- Model: `torchvision.models.efficientnet_b0(pretrained=True)` (ImageNet weights).
- Preprocessing: resize/crop to 224x224, ToTensor, ImageNet normalization.
- Output: top-k ImageNet labels with probabilities per frame.

Why EfficientNet-B0?
- Good accuracy/compute tradeoff for CPU inference.
- Low setup complexity — reliable baseline for student projects.

OCR (why included)
- Code: `ocr_module.extract_text_from_frames(frames)` (uses EasyOCR)
- Purpose: extract on-screen text (titles, captions, menus, logos) that can be strong category signals (e.g., "Tutorial", "Recipe").
- Limitations: OCR may be noisy on low-resolution frames or stylized fonts.

CV post-processing and mapping to categories
- The CV pipeline collects top labels from all frames plus OCR tokens.
- It uses `metadata_classifier.get_category_scores_from_text()` (shared keyword matcher) to count keyword hits per category.
- This count is normalized by `metadata_classifier.normalize_scores()` to produce `cv_scores`.

Implementation notes (function-level)
- `vision_model.classify_frame(image_path, top_k=3)`:
  - Loads image, applies `image_transform`, runs model, applies softmax, returns top-k labels and probabilities as floats.

3 — Audio (ASR) pipeline
-------------------------
Goal: transform speech and significant sung parts into text tokens that yield semantic signals for classification.

Files: `audio_transcriber.py`

Audio extraction
- Code: `extract_audio(video_path, output_dir)`
- Implementation: uses ffmpeg to produce `audio.wav` with parameters optimized for Whisper (mono, 16kHz):

```bash
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 -y audio.wav
```

Transcription (Whisper)
- Code: `transcribe_audio(audio_path)`
- Model: `whisper.load_model("small")` — chosen for the project as a balance of accuracy and cost.
- Why `small`:
  - `small` improves transcription on music and singing relative to `base`/`tiny`, without the memory/time requirements of `medium`/`large`.
  - Allows the project to run locally and still produce usable transcriptions for children’s songs / music-heavy content.

Parallelization and segmentation
- The repository includes `transcribe_audio_parallel()` which:
  - Splits audio into 30s segments (via ffmpeg), transcribes each segment in separate processes (each worker loads a Whisper model). This reduces end-to-end wall-clock time at the cost of higher memory use.
- Default, simpler path: sequential transcription using the already-loaded `whisper_model`.

Chunking and keyword classification
- `chunk_text(text, chunk_size=500, overlap=50)` splits the transcription into overlapping chunks. Each chunk is tokenized.
- `classify_text_chunks(chunks)` combines tokens across chunks and uses `get_category_scores_from_text()` to count keyword matches from `keywords.json`.
- If transcription is empty, very short (<5 tokens), or no keywords matched, the function returns a uniform distribution (this communicates low confidence instead of biasing to specific categories with zero evidence).

Practical considerations
- ASR quality degrades with heavy music, layered vocals, or strong background instrumentation. If transcription quality is critical, upgrade to Whisper `medium`/`large` and run on GPU.

4 — Metadata pipeline
----------------------
Goal: use video-provided text (title/description/tags) — often the single strongest signal.

Files: `metadata_classifier.py`

Process
- `classify_metadata(title, description, tags)` concatenates inputs, tokenizes with `re.findall(r"\\w+", text.lower())`, and counts keyword matches against `keywords.json`.
- `normalize_scores()` converts raw counts to a probability distribution (sum to 1). If nothing matched, it returns uniform distribution.

Why this simple approach?
- Metadata is usually short and contains direct cues. Keyword matching is explainable and fast.

5 — Scoring, normalization and aggregation (math + examples)
--------------------------------------------------------
Notation:
- Let C be the set of categories from `keywords.json` (|C| = M).
- For each pipeline p in {cv, audio, metadata}, let s_p(c) be the raw score for category c before normalization (counts or model outputs).

Normalization (per pipeline):
  cv_norm(c) = s_cv(c) / sum_c s_cv(c)  if sum_c s_cv(c) > 0
  otherwise cv_norm(c) = 1/M (uniform)

Aggregation (weighted sum):
  final(c) = w_cv * cv_norm(c) + w_audio * audio_norm(c) + w_meta * meta_norm(c)
  where w_cv + w_audio + w_meta = 1 (default each = 1/3)

Example (toy):
- Categories: [Kids, Music, Travel] (M=3)
- cv_norm = [0.6, 0.4, 0.0]
- audio_norm = [0.0, 1.0, 0.0]
- meta_norm = [1.0, 0.0, 0.0]
- weights = [1/3, 1/3, 1/3]

final(Kids) = (1/3)*0.6 + (1/3)*0.0 + (1/3)*1.0 = 0.5333
final(Music) = (1/3)*0.4 + (1/3)*1.0 + (1/3)*0.0 = 0.4667
final(Travel) = 0.0

The top category is Kids (53.33% confidence).

6 — Output format and sample JSON
---------------------------------
Output file: `backend/app/multimodal/classification_result.json`

Structure (fields of interest):
- `cv_predictions`: per-frame predictions (frame index → top labels + confidences)
- `ocr_text`: concatenated OCR text
- `cv_scores`: normalized CV category distribution (dict category → float)
- `transcription`: full ASR text
- `audio_scores`: normalized audio category distribution
- `metadata_scores`: normalized metadata category distribution
- `final_scores`: aggregated distribution across categories
- `top_category`: {"category": <str>, "confidence": <float>}

Example snippet (abridged JSON):

```json
{
  "cv_scores": {"Kids": 0.66, "Travel": 0.33},
  "audio_scores": {"Music": 1.0},
  "metadata_scores": {"Kids": 1.0},
  "final_scores": {"Kids": 0.66, "Music": 0.17, "Travel": 0.17},
  "top_category": {"category": "Kids", "confidence": 0.66}
}
```

7 — Implementation notes, parameters and commands
-------------------------------------------------
Key config points you may change for experiments:
- `--frames`: number of frames to extract (more frames → better CV coverage, slower).
- Whisper model: change `whisper.load_model("small")` to `"medium"`/`"large"` for higher ASR quality (requires GPU for practical speeds).
- Segment length for parallel ASR: `segment_length` in `transcribe_audio_parallel()` (default 30s).

Run examples

```bash
cd /workspaces/TaskMind/backend/app/multimodal
# Run with YouTube (download) and cookies
python main.py --url "https://www.youtube.com/watch?v=..." --title "Video Title" --cookies /path/to/cookies.txt

# Run with local file and 8 frames
python main.py --input-file /path/to/video.mp4 --title "Video Title" --frames 8
```

8 — Failure modes, limitations and mitigation
---------------------------------------------
ASR-specific
- Problem: low-quality transcription for music and singing.
- Mitigation: use Whisper `medium`/`large` or specialized music ASR models; run on GPU; add voice activity detection (VAD) to isolate speech segments.

CV-specific
- Problem: ImageNet labels do not cover stylized cartoon objects (domain gap).
- Mitigation: fine-tune on domain dataset; use CLIP zero-shot matching with category prompts; map common visual tokens (balloon, toy) to Kids category via keyword expansion.

Text-classification-specific
- Problem: keywords miss synonyms, abbreviations, or multi-word phrases.
- Mitigation: add stemming/lemmatization, synonyms, or train a transformer-based classifier on labeled samples.

Operational
- Problem: `yt-dlp` rate limits and cookies needed for age-restricted content.
- Mitigation: provide `--cookies`, add retries, increase backoff delays.

9 — Evaluation strategy and metrics
----------------------------------
Dataset creation
- Curate a balanced set of videos for each category. For each video include: URL, title, expected category label(s), and optional ground-truth timestamps.

Metrics
- Top-1 accuracy: percent of videos where `top_category` equals ground truth.
- Top-3 accuracy: whether ground-truth appears in the top-3 categories by final score.
- Per-category precision/recall/F1 — useful to detect weak categories.
- Per-modality ablation: run with only a single modality to estimate its importance.

10 — Demo script and final-year panel questions (with answers)
-------------------------------------------------------------
Demo script (2–4 minutes)
1. Show code layout and `keywords.json` (explain how categories are defined).
2. Run `python main.py --input-file samples/baby_shark.mp4 --title "Baby Shark"` (pre-downloaded sample to avoid network issues).
3. Show live output JSON and explain each field: transcription, per-frame CV labels, per-pipeline scores, final aggregated score.
4. Demonstrate an edge-case: run an animation video (Kids) where CV missed kids signals but metadata contains "Baby Shark" — explain how aggregation recovered correct label.

Panel Q&A (selected & expanded):
Technical
- Q: Why combine modalities? Aren’t titles enough?
  - A: Titles are powerful but not always reliable (clickbait, missing metadata). Combining CV and ASR reduces single-source failure and improves robustness across a wider range of content.

- Q: How are categories defined and updated?
  - A: Categories and their keyword lists live in `backend/app/multimodal/keywords.json`. To add a new category, add a key with representative keywords. For larger scale, replace the keyword matcher with a trained text classifier.

- Q: How do you choose modality weights?
  - A: Default is equal weights for simplicity. In production, you would tune weights on a labeled validation set or learn them using a meta-classifier.

Non-technical
- Q: How could a platform use this?
  - A: Automated tagging for recommendations, moderation (flagging inappropriate content), search indexing, or analytics.

11 — Next steps & recommended improvements (actionable)
-------------------------------------------------------
Short term (low effort, high impact)
- Expand keywords and add synonym sets for weak categories.
- Add simple lemmatization or casefolding in text pipelines.
- Add a small labeled validation set (100–300 videos) and tune modality weights.

Medium term (requires compute/data)
- Replace keyword matching for text with a fine-tuned transformer classifier (DistilBERT or RoBERTa) trained on metadata + ASR tokens.
- Integrate CLIP for better visual-text grounding and zero-shot classification.

Long term (research-level)
- Replace frame-level CV with short-clip video models (I3D, SlowFast) to capture motion cues.
- Build a learned fusion network that consumes embeddings from each modality and outputs calibrated probabilities.

Appendix — useful code references
- `main.py`: orchestrates the pipeline and arguments.
- `frame_extractor.py`: download + frame sampling.
- `vision_model.py`: EfficientNet-B0 inference.
- `ocr_module.py`: EasyOCR usage.
- `audio_transcriber.py`: ffmpeg + Whisper transcription + text chunking.
- `metadata_classifier.py`: keyword matching and normalization.
- `pipeline.py`: normalization + aggregation math.

Contact & credits
- Author: TaskMind project (repository owner)
- License & usage: see repository README for licensing and attribution notes.

-- End of document --

Generated: December 10, 2025
