# SQA Dataset Zero-Shot Evaluation

This guide explains how to run zero-shot evaluation on the SQA (Scene Question Answering) dataset.

## Dataset Structure

The SQA dataset is located in `dataset/SQA/` and contains:
- **Questions**: `sqa_task/balanced/v1_balanced_questions_{split}_scannetv2.json`
- **Annotations**: `sqa_task/balanced/v1_balanced_sqa_annotations_{split}_scannetv2.json`
- **Videos**: `video/scene{id}_00.mp4`

## Running Evaluation

### Basic Usage

Evaluate on the test split:
```bash
conda activate videolm
python evaluate_sqa.py --split test
```

### Options

- `--split`: Dataset split (`train`, `val`, or `test`) - default: `test`
- `--model-name`: Model to use - default: `Qwen/Qwen2-VL-7B-Instruct`
- `--max-samples`: Limit number of samples (for quick testing) - default: None
- `--output`: Output file path for detailed results - default: `sqa_{split}_results.json`
- `--dataset-dir`: Path to SQA dataset directory - default: `dataset/SQA`
- `--max-frames`: Number of frames to extract - default: 8
- `--load-in-4bit`: Use 4-bit quantization to save memory
- `--no-nlp-eval`: Disable NLP-based evaluation methods (only exact match)

### Examples

**Quick test with 10 samples:**
```bash
python evaluate_sqa.py --split test --max-samples 10
```

**Evaluate with smaller model:**
```bash
python evaluate_sqa.py --split test --model-name Qwen/Qwen2-VL-2B-Instruct
```

**Evaluate with quantization:**
```bash
python evaluate_sqa.py --split test --load-in-4bit
```

**Custom output file:**
```bash
python evaluate_sqa.py --split test --output my_results.json
```

## Output

The script will:
1. Load the dataset
2. Process each video-question pair
3. Compare predictions with ground truth using multiple NLP methods
4. Calculate accuracies for all evaluation metrics
5. Save detailed results to a JSON file

**Evaluation Methods:**
- Exact Match: Normalized string comparison
- Semantic Similarity: Embedding-based cosine similarity
- BLEU Score: N-gram overlap metric
- ROUGE Score: Recall-oriented evaluation
- Fuzzy Matching: Levenshtein distance ratio
- Contains Matching: Checks if GT answer appears in prediction (best for short answers)

### Results Format

The output JSON contains:
```json
{
  "accuracy": 0.75,
  "accuracies": {
    "exact_match": 0.75,
    "semantic": 0.82,
    "bleu": 0.78,
    "rouge": 0.80,
    "fuzzy": 0.77,
    "contains": 0.90
  },
  "correct": 150,
  "total": 200,
  "results": [
    {
      "question_id": 123,
      "scene_id": "scene0000_00",
      "question": "What is in front of me?",
      "gt_answer": "desk",
      "predicted_answer": "There is a desk in front of me",
      "correct": false,
      "correct_semantic": true,
      "correct_bleu": true,
      "correct_rouge": true,
      "correct_fuzzy": true,
      "correct_contains": true,
      "semantic_similarity": 0.95,
      "bleu_score": 0.85,
      "rouge1": 0.90,
      "rougeL": 0.88,
      "fuzzy_similarity": 0.92,
      "contains_score": 1.0,
      ...
    },
    ...
  ]
}
```

**Evaluation Metrics:**
- `correct`: Exact match (normalized string comparison)
- `correct_semantic`: Semantic similarity using embeddings
- `correct_bleu`: BLEU score (n-gram overlap)
- `correct_rouge`: ROUGE score (recall-oriented)
- `correct_fuzzy`: Fuzzy string matching (Levenshtein)
- `correct_contains`: Contains matching (GT answer in prediction) - **Best for short GT answers**

See [NLP_EVALUATION.md](NLP_EVALUATION.md) for detailed information about each metric.

## Performance Tips

1. **Memory**: Use `--load-in-4bit` if you have limited GPU memory
2. **Speed**: Use smaller model (`Qwen2-VL-2B-Instruct`) for faster evaluation
3. **Testing**: Use `--max-samples` to test on a subset first
4. **Frames**: Reduce `--max-frames` for faster processing (may affect accuracy)

## Troubleshooting

**Video not found errors:**
- Check that videos are in `dataset/SQA/video/`
- Verify video filenames match `scene{id}_00.mp4` format

**Out of memory:**
- Use `--load-in-4bit` flag
- Use smaller model variant
- Reduce `--max-frames`
- The script now automatically clears GPU cache between samples
- If OOM occurs, the script will stop and save partial results
- Clear GPU memory manually: `python utils/clear_gpu_memory.py`

**Slow processing:**
- Use GPU if available
- Reduce number of frames
- Use smaller model

