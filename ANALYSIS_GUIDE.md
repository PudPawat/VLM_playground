# Evaluation Results Analysis Guide

## Overview

The `analyze_results.py` script provides comprehensive insights into why predictions are wrong by analyzing evaluation results across multiple dimensions.

## Usage

### Basic Usage

```bash
# Analyze results and print to console
python analyze_results.py --results-file sqa_test_results.json

# Save full report to JSON file
python analyze_results.py --results-file sqa_test_results.json --output analysis_report.json

# Analyze different result files
python analyze_results.py --results-file sqa_test_results_7B.json --output analysis_7B.json
python analyze_results.py --results-file sqa_test_results_2B.json --output analysis_2B.json
```

### Arguments

- `--results-file`: Path to evaluation results JSON file (default: `sqa_test_results.json`)
- `--output`: Optional path to save full JSON report
- `--print`: Print report to console (default: True)

## Analysis Metrics

### 1. **Question Type Analysis**
Analyzes accuracy by question type:
- **What** questions
- **Where** questions  
- **How** questions
- **When** questions
- **Why** questions
- **Who** questions
- **Which** questions
- **Can/Is/Are/Does/Do** (yes/no questions)
- **Count** (how many)
- **Color** questions
- **Direction/Spatial** questions

**Insight**: Identifies which question types the model struggles with most.

### 2. **Answer Type Analysis**
Analyzes accuracy by ground truth answer type (from dataset metadata).

**Insight**: Shows if certain answer categories are harder.

### 3. **Question Vocabulary Analysis**
Identifies keywords in questions that correlate with errors.

**Top error-prone keywords**: Words that appear frequently in questions where predictions fail.

**Insight**: Reveals if specific concepts or terms cause confusion.

### 4. **Situation Vocabulary Analysis**
Identifies keywords in situation descriptions that correlate with errors.

**Insight**: Shows if certain scene descriptions or contexts are problematic.

### 5. **Answer Length Analysis**
Analyzes accuracy by ground truth answer length (in words).

**Insight**: Determines if shorter or longer answers are easier/harder.

### 6. **Prediction Length vs Ground Truth**
Compares prediction length to ground truth:
- **pred_longer**: Predictions much longer than GT
- **pred_shorter**: Predictions much shorter than GT  
- **pred_similar**: Predictions similar length to GT

**Insight**: Reveals if the model tends to over-explain or under-explain.

### 7. **Spatial Questions Analysis**
Special analysis for spatial/directional questions (left, right, front, back, etc.).

**Insight**: Spatial reasoning is often challenging for vision models.

### 8. **Numerical Questions Analysis**
Special analysis for counting/numerical questions (how many, numbers, etc.).

**Insight**: Counting and numerical reasoning performance.

### 9. **Color Questions Analysis**
Special analysis for color-related questions.

**Insight**: Color recognition accuracy.

### 10. **Semantic Similarity Score Analysis**
Compares semantic similarity scores for correct vs incorrect predictions.

**Insight**: Shows if errors have lower semantic similarity (expected) and by how much.

### 11. **Error Examples**
Shows actual examples of failed predictions with:
- Question
- Situation
- Ground truth answer
- Predicted answer

**Insight**: Qualitative understanding of failure modes.

### 12. **Correct Examples**
Shows examples of successful predictions.

**Insight**: Understand what the model does well.

## Output Format

### Console Output
The script prints a formatted report with:
- Summary statistics
- Tables for each analysis dimension
- Top error-prone keywords
- Example errors and successes

### JSON Output
The `--output` flag saves a complete JSON report with all analysis data for:
- Further processing
- Visualization
- Comparison between different model runs

## Example Insights

Based on typical results, you might find:

1. **Spatial questions are harder**: Direction questions have lower accuracy
2. **Short answers are easier**: Single-word answers (like "yes", "no", "right") have higher accuracy
3. **Color questions are easier**: Color recognition is relatively accurate
4. **Over-explanation**: Model often predicts much longer answers than ground truth
5. **Specific keywords cause errors**: Certain terms (e.g., "dryer", "below") appear in many errors

## Comparing Models

To compare different models:

```bash
# Analyze 7B model
python analyze_results.py --results-file sqa_test_results_7B.json --output analysis_7B.json

# Analyze 2B model  
python analyze_results.py --results-file sqa_test_results_2B.json --output analysis_2B.json

# Compare the JSON outputs manually or with a script
```

## Tips

1. **Focus on high error-rate question types**: If "where" questions have 60% error rate, investigate those
2. **Check error-prone keywords**: If "dryer" appears in many errors, the model may struggle with that concept
3. **Review error examples**: Look at actual failures to understand patterns
4. **Compare semantic scores**: Low semantic similarity in errors suggests the model is completely off-track
5. **Answer length patterns**: If longer answers fail more, the model may struggle with complex reasoning

## Integration with Evaluation

The analysis script works with the output from `evaluate_sqa.py`. After running evaluation:

```bash
# 1. Run evaluation
python evaluate_sqa.py --split test --max-samples 100 --output results.json

# 2. Analyze results
python analyze_results.py --results-file results.json --output analysis.json
```

## Next Steps

Based on analysis results, you can:
- Fine-tune on specific question types
- Add training data for error-prone keywords
- Adjust prompts for spatial/numerical questions
- Implement post-processing for answer length
- Focus model improvements on identified weaknesses

