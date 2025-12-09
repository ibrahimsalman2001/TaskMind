# Keywords-Based Classification

## Overview

All three classification modalities (CV, Audio, Metadata) use the **keywords.json** file for classification. This ensures:

1. **No training required** - Classification is based on keyword matching
2. **Consistent categories** - All 22 categories from keywords.json are used
3. **Easy updates** - Simply update keywords.json to improve classification

## How It Works

### 1. Computer Vision (CV) Classifier
- Extracts 6 frames from video
- Uses EfficientNet to classify each frame (ImageNet labels)
- Matches ImageNet labels against keywords in keywords.json
- Returns normalized scores for all 22 categories

### 2. Audio Classifier
- Downloads video and extracts audio
- Transcribes audio using Whisper
- Chunks transcription into segments
- Matches transcribed words against keywords in keywords.json
- Returns normalized scores for all 22 categories

### 3. Metadata Classifier
- Takes title, description, and tags
- Extracts all words from the combined text
- Matches words against keywords in keywords.json
- Returns normalized scores for all 22 categories

## Categories in keywords.json

The file contains 22 categories:
1. Academia & Explainer
2. Science & Technology
3. History & Documentaries
4. Finance & Business
5. Comedy & Skits
6. Film & Animation
7. Music & Dance
8. Gaming & Esports
9. Vlogs (General)
10. Travel & Lifestyle
11. Beauty & Fashion
12. Food & Cooking
13. News & Politics
14. Current Events Commentary
15. Health & Wellness
16. Sports (Professional)
17. Islamic/Religious
18. DIY & Craft
19. Autos & Vehicles
20. Pets & Animals
21. Kids
22. Adult/Flagged
23. Other (fallback / low-confidence / ambiguous content)

## Classification Process

For each modality:
1. Extract text/tokens from input
2. Match tokens against keywords in each category
3. Count matches per category (raw scores)
4. Normalize scores to sum to 1.0 (probability distribution)
5. Return scores for all 22 categories

## Final Aggregation

The pipeline:
1. Gets scores from all three modalities
2. Normalizes each set of scores
3. Applies weights (default: equal 1/3 each)
4. Combines scores
5. Returns the category with highest final score

## Updating Keywords

To improve classification:
1. Edit `keywords.json`
2. Add more keywords to relevant categories
3. No retraining needed - changes take effect immediately

## Example

If a video has:
- **Title**: "Python Tutorial for Beginners"
- **Description**: "Learn programming basics"
- **Tags**: "coding, education, python"

The metadata classifier will:
1. Extract tokens: ["python", "tutorial", "beginners", "learn", "programming", "basics", "coding", "education"]
2. Match against keywords.json:
   - "Academia & Explainer": matches "tutorial", "learn", "education" → score: 3
   - "Science & Technology": matches "python", "programming", "coding" → score: 3
   - Other categories: score: 0
3. Normalize: Academia & Explainer: 0.5, Science & Technology: 0.5
4. Return scores for all 22 categories

