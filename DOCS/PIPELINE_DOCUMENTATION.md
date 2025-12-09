**TaskMind — Multi-Modal Video Classification: Full Pipeline Documentation

**Overview**
- **Project**: TaskMind — a three-pipeline, multi-modal video classification system designed to classify YouTube or local videos into a fixed set of categories using visual, audio (transcription), and metadata signals.
- **Purpose**: Provide robust, explainable classification by fusing complementary signals: visual content, spoken words (ASR), and text metadata.
- **Output**: Per-pipeline category scores and a final aggregated probability distribution across categories.

**Contents of this document**
- System architecture and processing flow
- Detailed description of each pipeline: Computer Vision (CV), Audio Transcription (ASR) and Audio Classification, Metadata
- Models and libraries used and rationale
- Data flow and implementation details (how videos are processed end-to-end)
- Scoring, normalization, aggregation and final decision rules
- Failure modes and limitations
- Suggested improvements and future work
- Typical questions (technical & non-technical) and suggested answers for a final-year project panel
- How to run and test the pipeline locally

**System Architecture & Processing Flow**
- **High-level steps**:
  - Input: YouTube URL (via `yt-dlp`) or local video file
  - Frame extraction: sample N frames (default 6)
  - CV pipeline: classify frames with a pre-trained image classifier and run OCR on frames
  - Audio pipeline: extract audio, transcribe using Whisper, classify transcription via keyword matching
  - Metadata pipeline: analyze title/description/tags via keyword matching
  - Aggregation: normalize each pipeline’s scores, apply weights (default equal), compute final scores
  - Output: JSON containing `cv_scores`, `audio_scores`, `metadata_scores`, `final_scores`, per-frame predictions and the transcription

**Categories & Keywords**
- Categories are defined in `backend/app/multimodal/keywords.json` with keyword lists per category.
- Keyword matching is the primary approach for text-based classification (metadata + ASR + OCR tokens).
- This design allows quick extension of categories and keywords without retraining models.

**Pipeline 1 — Computer Vision (CV)**
**Purpose**: Extract visual cues from frames to infer content category signals (objects, scenes, text in frames).

**Implementation Summary**:
- **Frame extraction**: `backend/app/multimodal/frame_extractor.py`
  - Uses `ffmpeg` (or OpenCV) to extract `num_frames` sample frames from the video, saved to a temporary directory.
  - Default sampling: 6 frames (configurable via `--frames` argument to `main.py`).
- **Image classification**: `backend/app/multimodal/vision_model.py`
  - Model used: `torchvision.models.efficientnet_b0(pretrained=True)` (ImageNet-pretrained).
  - Preprocessing: resize to 224x224, normalize with ImageNet mean/std.
  - Inference: top-k predictions per frame (default k=3).
- **Optical Character Recognition (OCR)**: `backend/app/multimodal/ocr_module.py`
  - Library: `easyocr.Reader(['en','ur'], gpu=False)`
  - Extracts textual tokens from each frame; output concatenated and used as additional text tokens for CV scoring.

**Why EfficientNet-B0?**
- EfficientNet-B0 is a strong, lightweight ImageNet model offering a good accuracy/compute trade-off for inference on CPU (suitable for dev machines and small servers).
- Uses a pre-trained model avoids the need for domain-specific dataset or retraining for many categories.
- Practical for a student project: easy to use, widely documented, quick to run.

**CV Scoring Process**:
- Collect top labels from all frames (labels come from ImageNet taxonomy). Combine these labels (converted to tokens) with OCR tokens.
- Apply keyword matching against `keywords.json` (same logic used by metadata pipeline) to count keyword matches per category.
- Normalize raw counts to form a probability distribution (sum to 1). If no matches, return a uniform distribution (fallback).

**Pipeline 2 — Audio Transcription & Audio Classification (ASR)**
**Purpose**: Extract spoken words from audio track (ASR) and classify the resulting text into categories using keyword matching.

**Implementation Summary**:
- **Audio extraction**: `audio_transcriber.extract_audio()`
  - Uses `ffmpeg` to convert the video’s audio to mono, 16kHz WAV (`pcm_s16le`) — a format well-suited for Whisper.
- **ASR / Transcription**: `audio_transcriber.transcribe_audio()`
  - Model used: `openai/whisper` Python package (local whisper binding)
  - Model variant: **`small`** (default chosen in this repo)
  - Rationale for `small`:
    - Whisper model family: `tiny` → `base` → `small` → `medium` → `large`.
    - `small` was chosen as a trade-off: significantly better transcription quality on music / singing than `base`/`tiny`, while still feasible to run locally in reasonable time (compared to `medium`/`large`).
    - For challenging content (heavy music, children’s singing), larger models (`medium`/`large`) offer further quality improvements but require more memory/time.
  - Options & parallelization: the code contains utilities to split audio into segments and transcribe in parallel (with a worker-per-process model), but the default runs sequential transcription with the loaded `whisper_model` for simplicity.
- **Text chunking**: `chunk_text()` splits large transcriptions into overlapping chunks (defaults: 500 words per chunk with 50-word overlap) to limit keyword matching biases across long transcripts.
- **Classification**: `classify_text_chunks()` performs token extraction and keyword matching against `keywords.json`.
  - If transcription is empty or very short (less than 5 tokens) or if no keywords matched, the pipeline returns a uniform distribution across categories (fallback to indicate low confidence instead of returning zeros).
  - Otherwise, raw keyword match counts per category are normalized to probabilities.

**Why Whisper (small)?**
- Whisper is robust at noisy audio and has language detection plus strong generalization across accents and content types.
- Music and singing are particularly challenging for ASR; `small` provides a practical balance between quality and inference cost for local runs.
- Using Whisper allows us to extract richer textual signals for semantic classification via keywords without training a custom ASR.

**Pipeline 3 — Metadata Classification**
**Purpose**: Use the video’s title, description and tags as explicit textual signals (often the strongest single signal for many content categories).

**Implementation Summary**:
- **Extraction**: The `main.py` accepts `--title`, `--description` and `--tags`. For YouTube downloads, if title/description are provided they’re used; otherwise the user supplies them for classification.
- **Tokenization & Matching**: `metadata_classifier.classify_metadata()` tokenizes the text and counts keyword matches per category using `keywords.json`.
- **Normalization**: Raw counts are normalized to form a probability distribution. If no tokens or no matches, return uniform distribution.

**Why keyword matching for metadata?**
- Metadata is textual and often contains direct category signals (e.g., "Baby Shark", "tutorial", "vlog"). A keyword approach is simple, explainable, and effective without collecting labeled metadata for supervised training.
- This method is easy to audit and update (add keywords for edge cases) and very fast to run.

**Aggregation & Final Scoring**
**Module**: `pipeline.aggregate_scores()`

**Steps**:
- Normalize each pipeline’s output scores to sum to 1 (the `normalize_scores()` function ensures a valid probability distribution; if a pipeline produces all zeros it converts them to a uniform distribution).
- Default weights: equal weighting for each pipeline: `cv=1/3`, `audio=1/3`, `metadata=1/3`. Weights are configurable via the `aggregate_scores()` call.
- For each category, final_score = weight_cv * cv_norm[cat] + weight_audio * audio_norm[cat] + weight_metadata * metadata_norm[cat].
- Final scores are returned as a dictionary mapping categories to probabilities; the top category and its confidence are stored as `top_category`.

**Why equal weighting?**
- Simplicity and robustness: equal weights provide a neutral starting point when modality reliabilities vary by video.
- Metadata is often the most reliable signal; in practice teams may tune weights (e.g., give more weight to metadata) or learn weights using a small validation set.

**End-to-End Execution (What happens when you run `main.py`)**
1. Input validation (title required, either `--url` or `--input-file` required).
2. Download or use local file.
3. Extract frames and run CV classification + OCR → `cv_scores`.
4. Extract audio → run Whisper transcription → chunk & classify → `audio_scores`.
5. Classify metadata text → `metadata_scores`.
6. Aggregate the three normalized score vectors → `final_scores`.
7. Save output JSON to `backend/app/multimodal/classification_result.json` with per-pipeline outputs and final category.

**Failure Modes & Limitations**
- **ASR Errors on Music & Singing**: Even Whisper can mis-transcribe highly musical or overlapped-sung audio. Mitigation: use `small/medium/large` Whisper or specialized music-robust ASR.
- **CV Domain Gap**: EfficientNet-B0 is trained on ImageNet (natural images). It may not recognize stylized cartoons or animation well (affects kids’ content). Mitigation: fine-tune on domain-specific frames or use models trained on scene/gesture/thumbnail datasets.
- **Keyword Approach Limitations**: Keyword matching is brittle to phrasing and synonyms. It provides explainability but lower recall for paraphrases. Mitigation: expand keyword lists, use lemmatization/stemming, or switch to a supervised text classifier (fine-tuned transformer) for metadata/ASR tokens.
- **YouTube Download Errors**: `yt-dlp` may require cookies or encounter rate-limits. Provide `--cookies` and handle download retries.
- **Resource Constraints**: Whisper `small`/`medium`/`large` models are resource-hungry; CPU inference is slow. GPU recommended for medium/large.

**Suggested Improvements & Future Work**
- Replace keyword matching with a learned text classifier (fine-tune BERT/DistilBERT) for metadata and ASR tokens to improve robustness.
- Fine-tune a vision model (or use modern architectures like CLIP or ViT) on video thumbnails and frames for domain-specific performance.
- Use CLIP (openai/clip) to jointly embed images and text; CLIP can directly score categories with text prompts (zero-shot) and often works well on diverse images (cartoons, stylized thumbnails).
- Add a confidence calibration step and/or a meta-classifier to learn optimal modality weights from a labeled validation set.
- Implement ensemble strategies and temporal modeling (use more frames or short clips, apply video-level models e.g., I3D, SlowFast) to capture motion cues.

**Evaluation & Metrics**
- Prepare a labeled test set containing representative videos for each category.
- Compute precision/recall/F1 and accuracy for top-1 and top-3 predictions.
- Measure per-modality performance to guide weight tuning (e.g., metadata may have high precision but low recall on certain categories).

**Practical Notes & Commands**
- Run pipeline (example using YouTube URL):

```bash
cd /workspaces/TaskMind/backend/app/multimodal
python main.py --url "https://www.youtube.com/watch?v=..." --title "Video Title" --cookies /path/to/cookies.txt
```

- Run with local file:

```bash
python main.py --input-file /path/to/video.mp4 --title "Video Title" --frames 8
```

- Output JSON saved to: `backend/app/multimodal/classification_result.json`.

**Questions for Final-Year Project Panel (with concise suggested answers)**
**Technical**
- Q: Why three pipelines (CV, ASR, Metadata)?
  - A: Complementary signals reduce overall error — metadata often contains explicit category hints, ASR captures spoken content, and CV captures visual context; fusion yields more robust classification.
- Q: Why use keyword matching instead of ML classifiers for text? 
  - A: Simplicity, explainability, no labeled data required; allows rapid prototyping and easy extension. For higher performance, move to supervised models.
- Q: Why EfficientNet-B0 and Whisper-small? 
  - A: EfficientNet-B0 balances speed and accuracy on CPU; Whisper-small balances transcription quality vs compute for music heavy content. Both choices are pragmatic for a student project running on limited hardware.
- Q: How is scoring aggregated?
  - A: Normalize per-pipeline distributions to sum to 1, apply pipeline weights (default equal), and sum weighted probabilities per category to produce final distribution.
- Q: How can you improve results on cartoons/animated kids content? 
  - A: Use CLIP or fine-tune a vision model on a labeled dataset containing thumbnails/frames from animated content, or include keywords that map common visuals (balloons, toys) to Kids.

**Non-Technical / Presentation**
- Q: What real-world problem does TaskMind solve? 
  - A: It helps automatically tag and route user-generated videos (e.g., moderation, content recommendation, indexing) by combining audio, visual, and metadata signals.
- Q: How does the project consider privacy and ethics? 
  - A: The pipeline operates on public videos or those provided by the user; transcription and metadata handling should follow privacy policies and avoid storing sensitive data. Use opt-in data sources and consider redaction.
- Q: What are the deployment costs? 
  - A: Key costs are compute for Whisper and model inference. Whisper-small on CPU is slow; GPU reduces runtime. Storage and bandwidth for video downloads also matter.

**Appendix: Files & Key Modules**
- `backend/app/multimodal/main.py` — pipeline orchestration and CLI
- `backend/app/multimodal/frame_extractor.py` — download and frame sampling
- `backend/app/multimodal/vision_model.py` — EfficientNet-B0 inference
- `backend/app/multimodal/ocr_module.py` — EasyOCR-based OCR
- `backend/app/multimodal/audio_transcriber.py` — ffmpeg audio extraction, Whisper transcription, chunking and keyword classification
- `backend/app/multimodal/metadata_classifier.py` — keyword matching for metadata
- `backend/app/multimodal/pipeline.py` — score normalization and aggregation
- `backend/app/multimodal/keywords.json` — category keyword mappings

**References & Resources**
- Whisper: https://github.com/openai/whisper
- EfficientNet / torchvision: https://pytorch.org/vision/stable/models.html
- CLIP for possible improvement: https://github.com/openai/CLIP
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- yt-dlp: https://github.com/yt-dlp/yt-dlp

**Final notes**
- This design emphasizes explainability and quick iteration. It’s suitable for a final-year project: you can demonstrate end-to-end processing, show per-pipeline outputs, and discuss trade-offs and improvements.
- If you want, I can also:
  - Add an architecture diagram (SVG/PNG) to `DOCS/`.
  - Produce a short README for the `DOCS/` folder with quick-run examples and sample output JSON.
  - Prepare a short slide-deck (6–10 slides) highlighting the architecture, experiments and results for a panel.


---
Generated: December 9, 2025
