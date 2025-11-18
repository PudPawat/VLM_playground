"""
Zero-shot evaluation script for SQA (Scene Question Answering) dataset
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import torch
import gc

from videolm import VideoLM
from videolm.evaluators import AnswerEvaluator


def load_dataset(
    questions_path: str,
    annotations_path: str,
    video_dir: str
) -> List[Dict]:
    """
    Load SQA dataset
    
    Args:
        questions_path: Path to questions JSON file
        annotations_path: Path to annotations JSON file
        video_dir: Directory containing video files
        
    Returns:
        List of samples with question, answer, and video path
    """
    # Load questions
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
        questions = questions_data.get('questions', questions_data)  # Handle both formats
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)
        annotations = annotations_data.get('annotations', annotations_data)  # Handle both formats
    
    # Create mapping from question_id to annotation
    annotation_map = {ann['question_id']: ann for ann in annotations}
    
    # Combine questions with annotations
    samples = []
    for q in questions:
        question_id = q['question_id']
        scene_id = q['scene_id']
        
        # Find corresponding annotation
        if question_id in annotation_map:
            ann = annotation_map[question_id]
            video_path = os.path.join(video_dir, f"{scene_id}.mp4")
            
            # Get ground truth answer (first answer in the list)
            gt_answer = ann['answers'][0]['answer'].lower().strip()
            
            samples.append({
                'question_id': question_id,
                'scene_id': scene_id,
                'question': q['question'],
                'situation': q.get('situation', ''),
                'video_path': video_path,
                'gt_answer': gt_answer,
                'answer_type': ann.get('answer_type', 'unknown')
            })
    
    return samples


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.lower().strip()
    # Remove common punctuation
    answer = answer.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    return answer


def exact_match(pred: str, gt: str) -> bool:
    """Check if prediction exactly matches ground truth"""
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    return pred_norm == gt_norm


def evaluate(
    model: VideoLM,
    samples: List[Dict],
    output_file: str = None,
    max_samples: int = None,
    use_nlp_eval: bool = True,
    clear_cache: bool = True
) -> Dict:
    """
    Evaluate model on SQA dataset
    
    Args:
        model: VideoLM model instance
        samples: List of samples to evaluate
        output_file: Optional path to save detailed results
        max_samples: Optional limit on number of samples to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    if max_samples:
        samples = samples[:max_samples]
    
    results = []
    correct = 0
    total = 0
    
    # Initialize NLP evaluator if requested
    nlp_evaluator = None
    if use_nlp_eval:
        print("Initializing NLP evaluators...")
        nlp_evaluator = AnswerEvaluator()
    
    # Track metrics for all methods
    metrics = {
        'exact_match': {'correct': 0, 'total': 0},
        'semantic': {'correct': 0, 'total': 0},
        'bleu': {'correct': 0, 'total': 0},
        'rouge': {'correct': 0, 'total': 0},
        'fuzzy': {'correct': 0, 'total': 0},
        'contains': {'correct': 0, 'total': 0}
    }
    
    print(f"\nEvaluating on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Processing"):
        video_path = sample['video_path']
        question = sample['question']
        gt_answer = sample['gt_answer']
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            result = {
                **sample,
                'predicted_answer': 'VIDEO_NOT_FOUND',
                'correct': False
            }
            if nlp_evaluator:
                result.update({
                    'correct_semantic': False,
                    'correct_bleu': False,
                    'correct_rouge': False,
                    'correct_fuzzy': False,
                    'correct_contains': False,
                    'semantic_similarity': 0.0,
                    'bleu_score': 0.0,
                    'rouge1': 0.0,
                    'rougeL': 0.0,
                    'fuzzy_similarity': 0.0,
                    'contains_score': 0.0
                })
            results.append(result)
            total += 1
            continue
        
        try:
            # Get model prediction
            predicted_answer = model.answer_question(
                video_path=video_path,
                question=question,
                max_new_tokens=128,
                temperature=0.1  # Lower temperature for more deterministic answers
            )
            
            # Check exact match
            is_correct_exact = exact_match(predicted_answer, gt_answer)
            
            if is_correct_exact:
                correct += 1
            
            total += 1
            
            # Initialize result dict
            result = {
                **sample,
                'predicted_answer': predicted_answer,
                'correct': bool(is_correct_exact)  # Keep exact match as 'correct' for backward compatibility
            }
            
            # Add NLP-based evaluations
            if nlp_evaluator:
                nlp_results = nlp_evaluator.evaluate(predicted_answer, gt_answer)
                result.update(nlp_results)
                
                # Update metrics counters
                metrics['exact_match']['correct'] += int(is_correct_exact)
                metrics['exact_match']['total'] += 1
                
                if 'correct_semantic' in nlp_results:
                    metrics['semantic']['correct'] += int(bool(nlp_results['correct_semantic']))
                    metrics['semantic']['total'] += 1
                
                if 'correct_bleu' in nlp_results:
                    metrics['bleu']['correct'] += int(bool(nlp_results['correct_bleu']))
                    metrics['bleu']['total'] += 1
                
                if 'correct_rouge' in nlp_results:
                    metrics['rouge']['correct'] += int(bool(nlp_results['correct_rouge']))
                    metrics['rouge']['total'] += 1
                
                if 'correct_fuzzy' in nlp_results:
                    metrics['fuzzy']['correct'] += int(bool(nlp_results['correct_fuzzy']))
                    metrics['fuzzy']['total'] += 1
                
                if 'correct_contains' in nlp_results:
                    metrics['contains']['correct'] += int(bool(nlp_results['correct_contains']))
                    metrics['contains']['total'] += 1
            else:
                metrics['exact_match']['correct'] += int(is_correct_exact)
                metrics['exact_match']['total'] += 1
            
            results.append(result)
            
            # Clear GPU memory after each sample to prevent OOM
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
        except torch.cuda.OutOfMemoryError as e:
            # CUDA OOM - clear cache and raise to stop execution
            print(f"\n❌ CUDA Out of Memory Error processing {video_path}")
            print(f"Error: {str(e)}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print(f"GPU memory cleared. Free memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            print("\n⚠️  Stopping evaluation due to CUDA OOM error.")
            print("Suggestions:")
            print("  - Use --load-in-4bit flag for quantization")
            print("  - Use smaller model (--model-name Qwen/Qwen2-VL-2B-Instruct)")
            print("  - Reduce --max-frames (e.g., --max-frames 4)")
            print("  - Process fewer samples at once (--max-samples)")
            
            # Save partial results before exiting
            if output_file:
                print(f"\n💾 Saving partial results to {output_file}...")
                with open(output_file, 'w') as f:
                    json.dump({
                        'accuracy': correct / total if total > 0 else 0.0,
                        'accuracies': {method: counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0 
                                     for method, counts in metrics.items()},
                        'correct': correct,
                        'total': total,
                        'metrics': metrics,
                        'results': results,
                        'error': 'CUDA_OUT_OF_MEMORY',
                        'error_message': str(e),
                        'stopped_at_sample': len(results)
                    }, f, indent=2)
            
            raise RuntimeError(f"CUDA Out of Memory. Evaluation stopped at sample {len(results) + 1}/{len(samples)}") from e
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error processing {video_path}: {error_msg}")
            
            # Check if it's a memory-related error
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print("\n⚠️  Memory-related error detected. Clearing GPU cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                print("Consider using --load-in-4bit or reducing --max-frames")
            
            result = {
                **sample,
                'predicted_answer': f'ERROR: {error_msg}',
                'correct': False
            }
            if nlp_evaluator:
                result.update({
                    'correct_semantic': False,
                    'correct_bleu': False,
                    'correct_rouge': False,
                    'correct_fuzzy': False,
                    'correct_contains': False,
                    'semantic_similarity': 0.0,
                    'bleu_score': 0.0,
                    'rouge1': 0.0,
                    'rougeL': 0.0,
                    'fuzzy_similarity': 0.0,
                    'contains_score': 0.0
                })
            results.append(result)
            total += 1
            
            # Clear GPU memory after error
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # Calculate accuracies for all methods
    accuracies = {}
    for method, counts in metrics.items():
        if counts['total'] > 0:
            accuracies[method] = float(counts['correct'] / counts['total'])
        else:
            accuracies[method] = 0.0
    
    # Main accuracy (exact match)
    accuracy = float(accuracies.get('exact_match', correct / total if total > 0 else 0.0))
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'accuracies': accuracies,
                'correct': correct,
                'total': total,
                'metrics': metrics,
                'results': results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
    
    return {
        'accuracy': accuracy,
        'accuracies': accuracies,
        'correct': correct,
        'total': total,
        'metrics': metrics,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Zero-shot evaluation on SQA dataset')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2-VL-1B-Instruct', # Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2-VL-72B-Instruct
        help='Model name or path'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save detailed results'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/SQA',
        help='Directory containing SQA dataset'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=8,
        help='Maximum number of frames to extract from video'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load model in 4-bit quantization'
    )
    parser.add_argument(
        '--no-nlp-eval',
        action='store_true',
        help='Disable NLP-based evaluation methods'
    )
    parser.add_argument(
        '--no-clear-cache',
        action='store_true',
        help='Disable automatic GPU cache clearing between samples'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    dataset_dir = Path(args.dataset_dir)
    questions_path = dataset_dir / 'sqa_task' / 'balanced' / f'v1_balanced_questions_{args.split}_scannetv2.json'
    annotations_path = dataset_dir / 'sqa_task' / 'balanced' / f'v1_balanced_sqa_annotations_{args.split}_scannetv2.json'
    video_dir = dataset_dir / 'video'
    
    # Check if files exist
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Load dataset
    print(f"Loading {args.split} split...")
    samples = load_dataset(
        str(questions_path),
        str(annotations_path),
        str(video_dir)
    )
    print(f"Loaded {len(samples)} samples")
    
    # Check GPU memory before starting
    if torch.cuda.is_available():
        print(f"\nGPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9
            free = total - reserved
            
            print(f"  GPU {i} ({props.name}):")
            print(f"    Total: {total:.2f} GB | Allocated: {allocated:.2f} GB | Free: {free:.2f} GB")
        
        # Clear any existing cache
        if not args.no_clear_cache:
            print("\nClearing GPU cache before starting...")
            torch.cuda.empty_cache()
            gc.collect()
    
    # Initialize model
    print(f"\nInitializing model: {args.model_name}")
    model = VideoLM(
        model_name=args.model_name,
        max_frames=args.max_frames,
        frame_size=(448, 448),
        load_in_4bit=args.load_in_4bit
    )
    
    # Evaluate
    results = evaluate(
        model=model,
        samples=samples,
        output_file=args.output or f'sqa_{args.split}_results.json',
        max_samples=args.max_samples,
        use_nlp_eval=not args.no_nlp_eval,
        clear_cache=not args.no_clear_cache
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Exact Match Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    if 'accuracies' in results:
        print("\nNLP-based Accuracies:")
        for method, acc in results['accuracies'].items():
            if method != 'exact_match':
                method_name = method.replace('_', ' ').title()
                counts = results['metrics'][method]
                print(f"  {method_name}: {acc:.4f} ({counts['correct']}/{counts['total']})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

