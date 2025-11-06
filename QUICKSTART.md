# Quick Start Guide

Get started with VideoLM in 5 minutes!

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Basic Example

```python
from videolm import VideoLM

# Initialize model (first time will download the model)
model = VideoLM(model_name="Qwen/Qwen2-VL-7B-Instruct")

# Ask a question about a video
answer = model.answer_question(
    video_path="your_video.mp4",
    question="What is happening in this video?"
)

print(answer)
```

## Model Size Options

Choose based on your hardware:

```python
# Smallest (2B parameters) - ~4-6 GB VRAM
model = VideoLM(model_name="Qwen/Qwen2-VL-2B-Instruct")

# Medium (7B parameters) - ~14-16 GB VRAM  
model = VideoLM(model_name="Qwen/Qwen2-VL-7B-Instruct")

# Largest (72B parameters) - ~140+ GB VRAM
model = VideoLM(model_name="Qwen/Qwen2-VL-72B-Instruct")
```

## Memory Optimization

If you have limited GPU memory:

```python
# Use 4-bit quantization (reduces memory by ~75%)
model = VideoLM(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    load_in_4bit=True
)

# Or use 8-bit quantization (reduces memory by ~50%)
model = VideoLM(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    load_in_8bit=True
)
```

## Customize Frame Extraction

```python
model = VideoLM(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    max_frames=16,        # Extract more frames
    frame_size=(512, 512)  # Higher resolution
)
```

## Run Examples

```bash
# Basic usage
python examples/basic_usage.py

# Interactive demo
python examples/interactive_demo.py

# Batch processing
python examples/batch_processing.py
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out the [examples/](examples/) directory for more use cases
- Customize the model configuration for your specific needs

## Troubleshooting

**Out of memory?**
- Use a smaller model (2B instead of 7B)
- Enable quantization (`load_in_4bit=True`)
- Reduce `max_frames` or `frame_size`

**Slow processing?**
- Use GPU acceleration (ensure CUDA is installed)
- Use a smaller model variant
- Reduce frame count or resolution

