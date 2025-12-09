# run_pipeline_simple.py
# Even simpler version - just prompts for URL and title, uses defaults for rest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import classify_video_pipeline


def main():
    print("=" * 70)
    print("TaskMind Video Classification - Quick Input")
    print("=" * 70)
    print()
    
    # Get essential inputs only
    video_url = input("Enter YouTube Video URL: ").strip()
    if not video_url:
        print("Error: Video URL is required!")
        return
    
    title = input("Enter Video Title: ").strip()
    if not title:
        print("Warning: No title provided, classification may be less accurate")
        title = ""
    
    description = input("Enter Video Description (optional, press Enter to skip): ").strip()
    tags = input("Enter Video Tags (optional, comma-separated, press Enter to skip): ").strip()
    
    print("\n" + "=" * 70)
    print("Starting classification...")
    print("=" * 70)
    print()
    
    try:
        result = classify_video_pipeline(
            video_url=video_url,
            title=title,
            description=description,
            tags=tags,
            num_frames=6,
            weights=None  # Equal weights
        )
        
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"\n🎯 Final Classification: {result['top_category']}")
        print(f"   Confidence: {result['top_confidence']:.2%}")
        print(f"\nTop 3 Categories:")
        for i, (cat, score) in enumerate(result['final_top_5'][:3], 1):
            print(f"   {i}. {cat}: {score:.2%}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

