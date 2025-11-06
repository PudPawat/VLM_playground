"""
Quick test script to verify VideoLM setup
"""

from videolm import VideoLM, VideoProcessor

print("=" * 60)
print("VideoLM Setup Verification")
print("=" * 60)

# Test VideoProcessor
print("\n1. Testing VideoProcessor...")
processor = VideoProcessor(max_frames=8, frame_size=(448, 448))
print("   ✓ VideoProcessor initialized successfully")

# Test VideoLM import (without loading model to save time)
print("\n2. Testing VideoLM class...")
print("   ✓ VideoLM class imported successfully")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)
print("\nNext steps:")
print("  - Activate conda environment: conda activate videolm")
print("  - Run examples: python examples/basic_usage.py")
print("  - Or use interactively:")
print("    from videolm import VideoLM")
print("    model = VideoLM(model_name='Qwen/Qwen2-VL-7B-Instruct')")

