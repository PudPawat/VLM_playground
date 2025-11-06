"""
Interactive demo for VideoLM
"""

from pathlib import Path
from videolm import VideoLM

def main():
    print("=" * 60)
    print("VideoLM Interactive Demo")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading model... (this may take a few minutes)")
    model = VideoLM(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        max_frames=8,
        frame_size=(448, 448)
    )
    
    print("\nModel loaded! You can now ask questions about videos.")
    print("Commands:")
    print("  - Type 'new' or 'change' to load a different video")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Otherwise, just ask your question about the current video\n")
    
    current_video = None
    
    while True:
        # Get video path if not already loaded
        if current_video is None:
            video_path = input("Enter video path (or 'quit' to exit): ").strip()
            
            if video_path.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not Path(video_path).exists():
                print(f"Error: Video file not found: {video_path}\n")
                continue
            
            current_video = video_path
            print(f"\n✓ Video loaded: {Path(video_path).name}")
            print("You can now ask multiple questions about this video.\n")
        
        # Question loop for the current video
        while True:
            # Get question
            question = input("Enter your question (or 'new' to change video, 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                return
            
            if question.lower() in ['new', 'change', 'n', 'c']:
                current_video = None
                print("\n" + "-" * 60 + "\n")
                break
            
            if not question:
                print("Error: Please enter a question.\n")
                continue
            
            # Process and answer
            print("\nProcessing video and generating answer...")
            try:
                answer = model.answer_question(
                    video_path=current_video,
                    question=question,
                    max_new_tokens=512,
                    temperature=0.7
                )
                print(f"\nAnswer: {answer}\n")
            except Exception as e:
                print(f"Error: {str(e)}\n")
                # Optionally reset video on error
                # current_video = None
                # break

if __name__ == "__main__":
    main()

