# run_pipeline_manual.py
# Simple script to manually input video URL and metadata, then run the classification pipeline

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import classify_video_pipeline
import json


def get_user_input():
    """
    Prompt user for video URL and metadata.
    """
    print("=" * 70)
    print("TaskMind Video Classification - Manual Input")
    print("=" * 70)
    print()
    
    # Get video URL
    video_url = input("Enter YouTube Video URL: ").strip()
    if not video_url:
        print("Error: Video URL is required!")
        return None
    
    # Get title
    print()
    title = input("Enter Video Title (or press Enter to skip): ").strip()
    if not title:
        title = ""
    
    # Get description
    print()
    print("Enter Video Description (press Enter twice when done, or just Enter to skip):")
    description_lines = []
    while True:
        line = input()
        if line == "" and len(description_lines) > 0:
            break
        if line == "" and len(description_lines) == 0:
            break
        description_lines.append(line)
    description = "\n".join(description_lines).strip()
    
    # Get tags
    print()
    tags = input("Enter Video Tags (comma-separated, or press Enter to skip): ").strip()
    if not tags:
        tags = ""
    
    # Get number of frames (optional)
    print()
    num_frames_input = input("Number of frames to extract (default: 6, press Enter for default): ").strip()
    if num_frames_input:
        try:
            num_frames = int(num_frames_input)
        except ValueError:
            print("Invalid input, using default: 6")
            num_frames = 6
    else:
        num_frames = 6
    
    # Get weights (optional)
    print()
    print("Model weights (optional - press Enter for equal weights):")
    print("Format: cv,audio,metadata (e.g., 0.4,0.3,0.3)")
    weights_input = input("Enter weights: ").strip()
    weights = None
    if weights_input:
        try:
            parts = [float(x.strip()) for x in weights_input.split(",")]
            if len(parts) == 3 and sum(parts) > 0:
                total = sum(parts)
                weights = {
                    "cv": parts[0] / total,
                    "audio": parts[1] / total,
                    "metadata": parts[2] / total
                }
                print(f"Using weights: CV={weights['cv']:.2f}, Audio={weights['audio']:.2f}, Metadata={weights['metadata']:.2f}")
            else:
                print("Invalid weights, using equal weights")
        except ValueError:
            print("Invalid weights format, using equal weights")
    
    return {
        "video_url": video_url,
        "title": title,
        "description": description,
        "tags": tags,
        "num_frames": num_frames,
        "weights": weights
    }


def display_results(result):
    """
    Display classification results in a readable format.
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    
    # Final result
    print(f"\n🎯 FINAL CLASSIFICATION:")
    print(f"   Category: {result['top_category']}")
    print(f"   Confidence: {result['top_confidence']:.2%}")
    
    # Top 5 from each modality
    print(f"\n📊 TOP 5 CATEGORIES BY MODALITY:")
    
    print(f"\n   Computer Vision:")
    for i, (cat, score) in enumerate(result['cv_top_5'][:5], 1):
        print(f"      {i}. {cat}: {score:.2%}")
    
    print(f"\n   Audio:")
    for i, (cat, score) in enumerate(result['audio_top_5'][:5], 1):
        print(f"      {i}. {cat}: {score:.2%}")
    
    print(f"\n   Metadata:")
    for i, (cat, score) in enumerate(result['metadata_top_5'][:5], 1):
        print(f"      {i}. {cat}: {score:.2%}")
    
    print(f"\n   Final Aggregated:")
    for i, (cat, score) in enumerate(result['final_top_5'][:5], 1):
        marker = "👑" if i == 1 else "  "
        print(f"      {marker} {i}. {cat}: {score:.2%}")
    
    # Option to save full results
    print("\n" + "=" * 70)
    save = input("Save full results to JSON file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Enter filename (default: classification_result.json): ").strip()
        if not filename:
            filename = "classification_result.json"
        if not filename.endswith('.json'):
            filename += '.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to {filename}")


def main():
    """
    Main function to run the pipeline with manual input.
    """
    try:
        # Get user input
        inputs = get_user_input()
        if not inputs:
            return
        
        # Confirm before running
        print("\n" + "=" * 70)
        print("CONFIRMATION")
        print("=" * 70)
        print(f"Video URL: {inputs['video_url']}")
        print(f"Title: {inputs['title'] or '(empty)'}")
        print(f"Description: {inputs['description'][:50] + '...' if len(inputs['description']) > 50 else inputs['description'] or '(empty)'}")
        print(f"Tags: {inputs['tags'] or '(empty)'}")
        print(f"Frames: {inputs['num_frames']}")
        if inputs['weights']:
            print(f"Weights: CV={inputs['weights']['cv']:.2f}, Audio={inputs['weights']['audio']:.2f}, Metadata={inputs['weights']['metadata']:.2f}")
        else:
            print("Weights: Equal (1/3 each)")
        
        print()
        confirm = input("Proceed with classification? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
        
        # Run pipeline
        print("\n" + "=" * 70)
        print("RUNNING CLASSIFICATION PIPELINE")
        print("=" * 70)
        
        result = classify_video_pipeline(
            video_url=inputs['video_url'],
            title=inputs['title'],
            description=inputs['description'],
            tags=inputs['tags'],
            num_frames=inputs['num_frames'],
            weights=inputs['weights']
        )
        
        # Display results
        display_results(result)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Classification cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

