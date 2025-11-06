# VideoLM: Qwen-based Video Question Answering

A powerful Video Language Model for Question Answering built on top of Qwen2-VL, capable of understanding video content and answering questions about it.

## Features

- 🎥 **Video Understanding**: Process videos and extract meaningful information
- ❓ **Question Answering**: Answer questions about video content
- 🚀 **Multiple Model Sizes**: Support for 2B, 7B, and 72B parameter models
- 🔧 **Flexible Configuration**: Customizable frame extraction and processing
- 📦 **Easy to Use**: Simple API for video QA tasks

## Installation

1. Clone this repository:
```bash
cd Qwen_playground
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU acceleration, ensure you have CUDA-compatible PyTorch installed.

## Quick Start

### Basic Usage

```python
from videolm import VideoLM

# Initialize the model
model = VideoLM(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    max_frames=8,
    frame_size=(448, 448)
)

# Answer a question about a video
answer = model.answer_question(
    video_path="path/to/your/video.mp4",
    question="What is happening in this video?"
)

print(answer)
```

### Interactive Demo

Run the interactive demo:
```bash
python examples/interactive_demo.py
```

### Batch Processing

Process multiple videos at once:
```bash
python examples/batch_processing.py
```

## Model Options

VideoLM supports different Qwen2-VL model sizes:

- **Qwen2-VL-2B-Instruct**: Smallest model, fastest inference, good for resource-constrained environments
- **Qwen2-VL-7B-Instruct**: Balanced model, recommended for most use cases
- **Qwen2-VL-72B-Instruct**: Largest model, best quality, requires significant GPU memory

## Configuration

### VideoLM Parameters

- `model_name`: HuggingFace model identifier (default: "Qwen/Qwen2-VL-7B-Instruct")
- `device`: Device to run on ('cuda', 'cpu', or None for auto-detection)
- `max_frames`: Maximum number of frames to extract from video (default: 8)
- `frame_size`: Target frame resolution as (width, height) tuple (default: (448, 448))
- `load_in_4bit`: Load model with 4-bit quantization (saves memory)
- `load_in_8bit`: Load model with 8-bit quantization (saves memory)

### Answer Generation Parameters

- `max_new_tokens`: Maximum number of tokens in the answer (default: 512)
- `temperature`: Sampling temperature for generation (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.9)

## API Reference

### VideoLM Class

#### `__init__(model_name, device, max_frames, frame_size, load_in_4bit, load_in_8bit)`

Initialize the VideoLM model.

#### `answer_question(video_path, question, max_new_tokens, temperature, top_p)`

Answer a question about a video.

**Parameters:**
- `video_path`: Path to video file
- `question`: Question string
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter

**Returns:** Answer string

#### `answer_question_from_images(images, question, max_new_tokens, temperature, top_p)`

Answer a question about a list of images (video frames).

**Parameters:**
- `images`: List of PIL Images
- `question`: Question string
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter

**Returns:** Answer string

#### `batch_answer(video_paths, questions, max_new_tokens, temperature, top_p)`

Answer multiple questions about multiple videos.

**Parameters:**
- `video_paths`: List of video file paths
- `questions`: List of questions
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter

**Returns:** List of answer strings

## Examples

### Example 1: Basic Video QA

```python
from videolm import VideoLM

model = VideoLM()
answer = model.answer_question(
    video_path="video.mp4",
    question="What activities are shown in this video?"
)
print(answer)
```

### Example 2: Custom Configuration

```python
from videolm import VideoLM

model = VideoLM(
    model_name="Qwen/Qwen2-VL-2B-Instruct",  # Use smaller model
    max_frames=16,  # Extract more frames
    frame_size=(512, 512),  # Higher resolution
    load_in_4bit=True  # Use quantization
)

answer = model.answer_question(
    video_path="video.mp4",
    question="Describe the scene in detail.",
    max_new_tokens=1024,
    temperature=0.5
)
```

### Example 3: Processing Pre-extracted Frames

```python
from videolm import VideoLM
from PIL import Image

model = VideoLM()
images = [Image.open(f"frame_{i}.jpg") for i in range(8)]

answer = model.answer_question_from_images(
    images=images,
    question="What is the main subject in these images?"
)
```

## Supported Video Formats

The model supports common video formats including:
- MP4
- AVI
- MOV
- MKV
- And other formats supported by OpenCV

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.37+
- CUDA (optional, for GPU acceleration)

## Memory Requirements

Approximate memory requirements:

- **Qwen2-VL-2B**: ~4-6 GB VRAM (with quantization: ~2-3 GB)
- **Qwen2-VL-7B**: ~14-16 GB VRAM (with quantization: ~4-6 GB)
- **Qwen2-VL-72B**: ~140+ GB VRAM (requires multiple GPUs or quantization)

For memory-constrained environments, use quantization:
```python
model = VideoLM(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    load_in_4bit=True  # or load_in_8bit=True
)
```

## Performance Tips

1. **Frame Selection**: Adjust `max_frames` based on video length and complexity
2. **Resolution**: Lower `frame_size` for faster processing
3. **Quantization**: Use 4-bit or 8-bit quantization to reduce memory usage
4. **Batch Processing**: Process multiple videos sequentially for better GPU utilization

## Troubleshooting

### Out of Memory Errors

- Use a smaller model (2B instead of 7B)
- Enable quantization (`load_in_4bit=True`)
- Reduce `max_frames` or `frame_size`
- Use CPU mode (slower but uses less memory)

### Slow Processing

- Use GPU acceleration (CUDA)
- Reduce `max_frames` or `frame_size`
- Use a smaller model variant

## License

This project uses Qwen2-VL models. Please refer to the [Qwen2-VL license](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) for model usage terms.

## Citation

If you use this code, please cite the Qwen2-VL paper:

```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Qwen Team},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) by Alibaba Cloud
- Uses [Transformers](https://github.com/huggingface/transformers) library

