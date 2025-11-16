# NLP-Based Evaluation Methods

The evaluation script now includes multiple NLP-based methods to compare predicted answers with ground truth, going beyond simple exact matching.

## Available Methods

### 1. **Semantic Similarity** (`correct_semantic`)
- **Method**: Uses sentence transformers to compute embeddings and cosine similarity
- **Model**: `all-MiniLM-L6-v2` (default, lightweight and fast)
- **Threshold**: 0.7 (configurable)
- **How it works**: Converts both answers to embeddings and computes cosine similarity
- **Best for**: Capturing semantic equivalence even with different wording
- **Example**: "desk" vs "a desk" vs "the desk" → High similarity

### 2. **BLEU Score** (`correct_bleu`)
- **Method**: Bilingual Evaluation Understudy score
- **Threshold**: 0.3 (configurable)
- **How it works**: Compares n-gram overlap between predicted and ground truth
- **Best for**: Measuring word-level similarity
- **Note**: Uses smoothing to handle short answers

### 3. **ROUGE Score** (`correct_rouge`)
- **Method**: Recall-Oriented Understudy for Gisting Evaluation
- **Metrics**: ROUGE-1 (unigram) and ROUGE-L (longest common subsequence)
- **Threshold**: 0.3 for ROUGE-L F1 (configurable)
- **How it works**: Measures overlap of n-grams and longest common subsequence
- **Best for**: Capturing both word overlap and sequence similarity

### 4. **Fuzzy Matching** (`correct_fuzzy`)
- **Method**: Levenshtein distance ratio
- **Threshold**: 0.7 (configurable)
- **How it works**: Computes edit distance and converts to similarity ratio
- **Best for**: Handling typos and minor variations
- **Example**: "desk" vs "desks" → High similarity

### 5. **Contains Matching** (`correct_contains`)
- **Method**: Substring/word containment check
- **Threshold**: 80% word match for multi-word, any match for single-word
- **How it works**: Checks if ground truth answer (or its words) appears in the predicted answer
- **Best for**: Short ground truth answers that may appear within longer predictions
- **Example**: GT: "desk", Prediction: "There is a desk in front of me" → `true`
- **Note**: Particularly useful for SQA dataset where GT answers are typically short

### 6. **Exact Match** (`correct`)
- **Method**: Normalized string comparison
- **How it works**: Normalizes text (lowercase, remove punctuation) and compares
- **Best for**: Strict matching when exact answer is required

## Output Format

Each result in the JSON output now includes:

```json
{
  "question_id": 123,
  "predicted_answer": "a desk",
  "gt_answer": "desk",
  "correct": false,  // Exact match
  "correct_semantic": true,  // Semantic similarity
  "correct_bleu": true,  // BLEU score
  "correct_rouge": true,  // ROUGE score
  "correct_fuzzy": true,  // Fuzzy matching
  "correct_contains": true,  // Contains matching
  "semantic_similarity": 0.95,
  "bleu_score": 0.85,
  "rouge1": 0.90,
  "rougeL": 0.88,
  "fuzzy_similarity": 0.92,
  "contains_score": 1.0
}
```

## Summary Metrics

The evaluation output includes accuracy for each method:

```json
{
  "accuracy": 0.65,  // Exact match
  "accuracies": {
    "exact_match": 0.65,
    "semantic": 0.82,
    "bleu": 0.78,
    "rouge": 0.80,
    "fuzzy": 0.75,
    "contains": 0.90
  }
}
```

## Usage

### Enable NLP Evaluation (Default)
```bash
python evaluate_sqa.py --split test --max-samples 10
```

### Disable NLP Evaluation
```bash
python evaluate_sqa.py --split test --no-nlp-eval
```

## Customizing Thresholds

You can modify thresholds in `videolm/evaluators.py`:

```python
evaluator = AnswerEvaluator(
    semantic_threshold=0.8,  # Stricter semantic matching
    bleu_threshold=0.4,      # Higher BLEU requirement
    rouge_threshold=0.4,     # Higher ROUGE requirement
    fuzzy_threshold=0.8      # Stricter fuzzy matching
    # Note: contains_match threshold is hardcoded (80% word match)
)
```

## Dependencies

The NLP evaluation methods require:
- `sentence-transformers`: For semantic similarity
- `nltk`: For BLEU score
- `rouge-score`: For ROUGE metrics
- `python-Levenshtein`: For fuzzy matching

Install with:
```bash
pip install sentence-transformers nltk rouge-score python-Levenshtein
```

## Performance Considerations

- **Semantic Similarity**: First run downloads the embedding model (~80MB)
- **BLEU/ROUGE**: Fast, no model loading required
- **Fuzzy Matching**: Very fast, pure string comparison
- **Overall**: NLP evaluation adds ~0.1-0.5 seconds per sample

## When to Use Each Method

- **Exact Match**: When precise answers are required (e.g., "yes"/"no")
- **Semantic Similarity**: Best overall for capturing meaning
- **BLEU**: Good for measuring word overlap
- **ROUGE**: Good for sequence-based answers
- **Fuzzy**: Good for handling typos and minor variations
- **Contains**: Best for short GT answers that may appear in longer predictions

## Recommendations

For VideoQA tasks, we recommend:
1. **Primary metric**: Contains matching (best for short GT answers in SQA dataset)
2. **Secondary metrics**: Semantic similarity and ROUGE-L (capture meaning and sequence)
3. **Exact match**: Keep for comparison with literature
4. **BLEU/Fuzzy**: Useful for debugging and understanding edge cases

