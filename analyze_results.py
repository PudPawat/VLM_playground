"""
Analyze evaluation results to extract insights on prediction errors
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import argparse
from pathlib import Path


class ResultAnalyzer:
    """Analyze evaluation results for insights"""
    
    def __init__(self, results_file: str):
        """Load results from JSON file"""
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data.get('results', [])
        self.accuracies = self.data.get('accuracies', {})
        
        # Question type patterns
        self.question_patterns = {
            'what': r'\bwhat\b',
            'where': r'\bwhere\b',
            'how': r'\bhow\b',
            'when': r'\bwhen\b',
            'why': r'\bwhy\b',
            'who': r'\bwho\b',
            'which': r'\bwhich\b',
            'can': r'\bcan\b',
            'is': r'\bis\b',
            'are': r'\bare\b',
            'does': r'\bdoes\b',
            'do': r'\bdo\b',
            'count': r'\bhow many\b',
            'color': r'\bcolor\b',
            'direction': r'\b(left|right|front|back|behind|ahead|direction)\b',
            'yes_no': r'\b(is|are|can|does|do|will|would)\s+\w+',
        }
    
    def extract_question_type(self, question: str) -> List[str]:
        """Extract question types from question text"""
        question_lower = question.lower()
        types = []
        
        for qtype, pattern in self.question_patterns.items():
            if re.search(pattern, question_lower, re.IGNORECASE):
                types.append(qtype)
        
        # Special handling
        if 'how many' in question_lower:
            types.append('count')
        if any(word in question_lower for word in ['color', 'colour']):
            types.append('color')
        if any(word in question_lower for word in ['left', 'right', 'front', 'back', 'behind']):
            types.append('spatial')
        
        return types if types else ['other']
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text"""
        # Remove punctuation and split
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # Filter common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        keywords = [w for w in words if w not in stop_words and len(w) >= min_length]
        return keywords
    
    def analyze_by_question_type(self) -> Dict:
        """Analyze accuracy by question type"""
        type_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'samples': []})
        
        for result in self.results:
            question = result.get('question', '')
            question_types = self.extract_question_type(question)
            
            # Use the most specific metric available
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            for qtype in question_types:
                type_stats[qtype]['total'] += 1
                if is_correct:
                    type_stats[qtype]['correct'] += 1
                type_stats[qtype]['samples'].append({
                    'question': question,
                    'correct': is_correct,
                    'predicted': result.get('predicted_answer', ''),
                    'gt': result.get('gt_answer', '')
                })
        
        # Calculate accuracies
        analysis = {}
        for qtype, stats in type_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            analysis[qtype] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'error_rate': 1.0 - accuracy
            }
        
        return analysis
    
    def analyze_by_answer_type(self) -> Dict:
        """Analyze by ground truth answer type"""
        answer_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for result in self.results:
            answer_type = result.get('answer_type', 'unknown')
            
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            answer_type_stats[answer_type]['total'] += 1
            if is_correct:
                answer_type_stats[answer_type]['correct'] += 1
        
        analysis = {}
        for atype, stats in answer_type_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            analysis[atype] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'error_rate': 1.0 - accuracy
            }
        
        return analysis
    
    def analyze_by_question_vocabulary(self) -> Dict:
        """Analyze errors by vocabulary in questions"""
        error_keywords = Counter()
        correct_keywords = Counter()
        
        for result in self.results:
            question = result.get('question', '')
            keywords = self.extract_keywords(question)
            
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            for keyword in keywords:
                if is_correct:
                    correct_keywords[keyword] += 1
                else:
                    error_keywords[keyword] += 1
        
        # Calculate error rates per keyword
        keyword_analysis = {}
        all_keywords = set(error_keywords.keys()) | set(correct_keywords.keys())
        
        for keyword in all_keywords:
            errors = error_keywords.get(keyword, 0)
            corrects = correct_keywords.get(keyword, 0)
            total = errors + corrects
            
            if total > 0:
                error_rate = errors / total
                keyword_analysis[keyword] = {
                    'error_rate': error_rate,
                    'errors': errors,
                    'correct': corrects,
                    'total': total
                }
        
        # Sort by error rate
        sorted_keywords = sorted(
            keyword_analysis.items(), 
            key=lambda x: (x[1]['error_rate'], x[1]['total']), 
            reverse=True
        )
        
        return dict(sorted_keywords[:50])  # Top 50 keywords
    
    def analyze_by_situation_vocabulary(self) -> Dict:
        """Analyze errors by vocabulary in situation descriptions"""
        error_keywords = Counter()
        correct_keywords = Counter()
        
        for result in self.results:
            situation = result.get('situation', '')
            if not situation:
                continue
            
            keywords = self.extract_keywords(situation)
            
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            for keyword in keywords:
                if is_correct:
                    correct_keywords[keyword] += 1
                else:
                    error_keywords[keyword] += 1
        
        # Calculate error rates
        keyword_analysis = {}
        all_keywords = set(error_keywords.keys()) | set(correct_keywords.keys())
        
        for keyword in all_keywords:
            errors = error_keywords.get(keyword, 0)
            corrects = correct_keywords.get(keyword, 0)
            total = errors + corrects
            
            if total >= 3:  # Only keywords that appear at least 3 times
                error_rate = errors / total
                keyword_analysis[keyword] = {
                    'error_rate': error_rate,
                    'errors': errors,
                    'correct': corrects,
                    'total': total
                }
        
        # Sort by error rate
        sorted_keywords = sorted(
            keyword_analysis.items(), 
            key=lambda x: (x[1]['error_rate'], x[1]['total']), 
            reverse=True
        )
        
        return dict(sorted_keywords[:50])
    
    def analyze_answer_length(self) -> Dict:
        """Analyze errors by answer length"""
        length_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for result in self.results:
            gt_answer = result.get('gt_answer', '')
            answer_length = len(gt_answer.split())
            
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            length_stats[answer_length]['total'] += 1
            if is_correct:
                length_stats[answer_length]['correct'] += 1
        
        analysis = {}
        for length, stats in length_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            analysis[length] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return analysis
    
    def analyze_prediction_length_vs_gt(self) -> Dict:
        """Analyze relationship between prediction length and GT length"""
        length_analysis = {
            'pred_longer': {'correct': 0, 'total': 0},
            'pred_shorter': {'correct': 0, 'total': 0},
            'pred_similar': {'correct': 0, 'total': 0}
        }
        
        for result in self.results:
            gt_answer = result.get('gt_answer', '')
            pred_answer = result.get('predicted_answer', '')
            
            if not pred_answer or pred_answer.startswith('ERROR'):
                continue
            
            gt_len = len(gt_answer.split())
            pred_len = len(pred_answer.split())
            
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            if pred_len > gt_len * 1.5:
                category = 'pred_longer'
            elif pred_len < gt_len * 0.5:
                category = 'pred_shorter'
            else:
                category = 'pred_similar'
            
            length_analysis[category]['total'] += 1
            if is_correct:
                length_analysis[category]['correct'] += 1
        
        # Calculate accuracies
        for category in length_analysis:
            stats = length_analysis[category]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        return length_analysis
    
    def analyze_spatial_questions(self) -> Dict:
        """Analyze spatial/directional questions"""
        spatial_keywords = ['left', 'right', 'front', 'back', 'behind', 'ahead', 'direction', 
                          'side', 'clock', 'o\'clock', 'between', 'beside', 'next']
        
        spatial_results = []
        for result in self.results:
            question = result.get('question', '').lower()
            situation = result.get('situation', '').lower()
            
            has_spatial = any(kw in question or kw in situation for kw in spatial_keywords)
            
            if has_spatial:
                is_correct = False
                if 'correct_contains' in result:
                    is_correct = result['correct_contains']
                elif 'correct_semantic' in result:
                    is_correct = result['correct_semantic']
                elif 'correct' in result:
                    is_correct = result['correct']
                
                spatial_results.append({
                    'question': result.get('question', ''),
                    'correct': is_correct,
                    'predicted': result.get('predicted_answer', ''),
                    'gt': result.get('gt_answer', '')
                })
        
        if spatial_results:
            correct_count = sum(1 for r in spatial_results if r['correct'])
            total = len(spatial_results)
            return {
                'accuracy': correct_count / total if total > 0 else 0.0,
                'correct': correct_count,
                'total': total,
                'samples': spatial_results[:20]  # First 20 examples
            }
        return {'accuracy': 0.0, 'correct': 0, 'total': 0, 'samples': []}
    
    def analyze_numerical_questions(self) -> Dict:
        """Analyze numerical/counting questions"""
        numerical_keywords = ['how many', 'number', 'count', 'amount', 'odd', 'even', 
                            'divided by', 'multiply', 'divide']
        
        numerical_results = []
        for result in self.results:
            question = result.get('question', '').lower()
            
            has_numerical = any(kw in question for kw in numerical_keywords)
            
            if has_numerical:
                is_correct = False
                if 'correct_contains' in result:
                    is_correct = result['correct_contains']
                elif 'correct_semantic' in result:
                    is_correct = result['correct_semantic']
                elif 'correct' in result:
                    is_correct = result['correct']
                
                numerical_results.append({
                    'question': result.get('question', ''),
                    'correct': is_correct,
                    'predicted': result.get('predicted_answer', ''),
                    'gt': result.get('gt_answer', '')
                })
        
        if numerical_results:
            correct_count = sum(1 for r in numerical_results if r['correct'])
            total = len(numerical_results)
            return {
                'accuracy': correct_count / total if total > 0 else 0.0,
                'correct': correct_count,
                'total': total,
                'samples': numerical_results[:20]
            }
        return {'accuracy': 0.0, 'correct': 0, 'total': 0, 'samples': []}
    
    def analyze_color_questions(self) -> Dict:
        """Analyze color-related questions"""
        color_results = []
        for result in self.results:
            question = result.get('question', '').lower()
            
            if 'color' in question or 'colour' in question:
                is_correct = False
                if 'correct_contains' in result:
                    is_correct = result['correct_contains']
                elif 'correct_semantic' in result:
                    is_correct = result['correct_semantic']
                elif 'correct' in result:
                    is_correct = result['correct']
                
                color_results.append({
                    'question': result.get('question', ''),
                    'correct': is_correct,
                    'predicted': result.get('predicted_answer', ''),
                    'gt': result.get('gt_answer', '')
                })
        
        if color_results:
            correct_count = sum(1 for r in color_results if r['correct'])
            total = len(color_results)
            return {
                'accuracy': correct_count / total if total > 0 else 0.0,
                'correct': correct_count,
                'total': total,
                'samples': color_results[:20]
            }
        return {'accuracy': 0.0, 'correct': 0, 'total': 0, 'samples': []}
    
    def get_error_examples(self, n: int = 20) -> List[Dict]:
        """Get examples of errors"""
        errors = []
        
        for result in self.results:
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            if not is_correct:
                errors.append({
                    'question': result.get('question', ''),
                    'situation': result.get('situation', ''),
                    'predicted': result.get('predicted_answer', ''),
                    'gt': result.get('gt_answer', ''),
                    'question_id': result.get('question_id', ''),
                    'scene_id': result.get('scene_id', '')
                })
        
        return errors[:n]
    
    def get_correct_examples(self, n: int = 20) -> List[Dict]:
        """Get examples of correct predictions"""
        corrects = []
        
        for result in self.results:
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            if is_correct:
                corrects.append({
                    'question': result.get('question', ''),
                    'situation': result.get('situation', ''),
                    'predicted': result.get('predicted_answer', ''),
                    'gt': result.get('gt_answer', ''),
                    'question_id': result.get('question_id', ''),
                    'scene_id': result.get('scene_id', '')
                })
        
        return corrects[:n]
    
    def analyze_semantic_similarity_scores(self) -> Dict:
        """Analyze semantic similarity scores for errors vs correct"""
        error_scores = []
        correct_scores = []
        
        for result in self.results:
            if 'semantic_similarity' not in result:
                continue
            
            score = result['semantic_similarity']
            is_correct = False
            if 'correct_contains' in result:
                is_correct = result['correct_contains']
            elif 'correct_semantic' in result:
                is_correct = result['correct_semantic']
            elif 'correct' in result:
                is_correct = result['correct']
            
            if is_correct:
                correct_scores.append(score)
            else:
                error_scores.append(score)
        
        def calc_stats(scores):
            if not scores:
                return {}
            return {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }
        
        return {
            'errors': calc_stats(error_scores),
            'correct': calc_stats(correct_scores)
        }
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_samples': len(self.results),
                'overall_accuracies': self.accuracies
            },
            'by_question_type': self.analyze_by_question_type(),
            'by_answer_type': self.analyze_by_answer_type(),
            'by_question_vocabulary': self.analyze_by_question_vocabulary(),
            'by_situation_vocabulary': self.analyze_by_situation_vocabulary(),
            'by_answer_length': self.analyze_answer_length(),
            'by_prediction_length': self.analyze_prediction_length_vs_gt(),
            'spatial_questions': self.analyze_spatial_questions(),
            'numerical_questions': self.analyze_numerical_questions(),
            'color_questions': self.analyze_color_questions(),
            'semantic_similarity_analysis': self.analyze_semantic_similarity_scores(),
            'error_examples': self.get_error_examples(30),
            'correct_examples': self.get_correct_examples(20)
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted report"""
        print("=" * 80)
        print("EVALUATION RESULTS ANALYSIS")
        print("=" * 80)
        
        # Summary
        print("\n📊 SUMMARY")
        print("-" * 80)
        print(f"Total Samples: {report['summary']['total_samples']}")
        print("\nOverall Accuracies:")
        for method, acc in report['summary']['overall_accuracies'].items():
            print(f"  {method:20s}: {acc:.4f}")
        
        # Question Type Analysis
        print("\n" + "=" * 80)
        print("📝 ANALYSIS BY QUESTION TYPE")
        print("=" * 80)
        qtype_analysis = report['by_question_type']
        sorted_types = sorted(qtype_analysis.items(), key=lambda x: x[1]['error_rate'], reverse=True)
        
        print(f"\n{'Question Type':<20} {'Accuracy':<12} {'Error Rate':<12} {'Total':<10} {'Correct':<10}")
        print("-" * 80)
        for qtype, stats in sorted_types:
            print(f"{qtype:<20} {stats['accuracy']:<12.4f} {stats['error_rate']:<12.4f} "
                  f"{stats['total']:<10} {stats['correct']:<10}")
        
        # Answer Type Analysis
        print("\n" + "=" * 80)
        print("📋 ANALYSIS BY ANSWER TYPE")
        print("=" * 80)
        atype_analysis = report['by_answer_type']
        sorted_atypes = sorted(atype_analysis.items(), key=lambda x: x[1]['error_rate'], reverse=True)
        
        print(f"\n{'Answer Type':<20} {'Accuracy':<12} {'Error Rate':<12} {'Total':<10} {'Correct':<10}")
        print("-" * 80)
        for atype, stats in sorted_atypes:
            print(f"{atype:<20} {stats['accuracy']:<12.4f} {stats['error_rate']:<12.4f} "
                  f"{stats['total']:<10} {stats['correct']:<10}")
        
        # Question Vocabulary Analysis
        print("\n" + "=" * 80)
        print("🔤 TOP ERROR-PRONE KEYWORDS IN QUESTIONS")
        print("=" * 80)
        vocab_analysis = report['by_question_vocabulary']
        print(f"\n{'Keyword':<20} {'Error Rate':<12} {'Errors':<10} {'Correct':<10} {'Total':<10}")
        print("-" * 80)
        for keyword, stats in list(vocab_analysis.items())[:20]:
            print(f"{keyword:<20} {stats['error_rate']:<12.4f} {stats['errors']:<10} "
                  f"{stats['correct']:<10} {stats['total']:<10}")
        
        # Situation Vocabulary Analysis
        print("\n" + "=" * 80)
        print("🔤 TOP ERROR-PRONE KEYWORDS IN SITUATIONS")
        print("=" * 80)
        sit_vocab = report['by_situation_vocabulary']
        print(f"\n{'Keyword':<20} {'Error Rate':<12} {'Errors':<10} {'Correct':<10} {'Total':<10}")
        print("-" * 80)
        for keyword, stats in list(sit_vocab.items())[:20]:
            print(f"{keyword:<20} {stats['error_rate']:<12.4f} {stats['errors']:<10} "
                  f"{stats['correct']:<10} {stats['total']:<10}")
        
        # Answer Length Analysis
        print("\n" + "=" * 80)
        print("📏 ANALYSIS BY ANSWER LENGTH")
        print("=" * 80)
        length_analysis = report['by_answer_length']
        sorted_lengths = sorted(length_analysis.items(), key=lambda x: x[0])
        
        print(f"\n{'Length (words)':<15} {'Accuracy':<12} {'Total':<10} {'Correct':<10}")
        print("-" * 80)
        for length, stats in sorted_lengths[:15]:
            print(f"{length:<15} {stats['accuracy']:<12.4f} {stats['total']:<10} {stats['correct']:<10}")
        
        # Prediction Length Analysis
        print("\n" + "=" * 80)
        print("📐 PREDICTION LENGTH VS GROUND TRUTH")
        print("=" * 80)
        pred_length = report['by_prediction_length']
        print(f"\n{'Category':<20} {'Accuracy':<12} {'Total':<10} {'Correct':<10}")
        print("-" * 80)
        for category, stats in pred_length.items():
            print(f"{category:<20} {stats['accuracy']:<12.4f} {stats['total']:<10} {stats['correct']:<10}")
        
        # Special Question Types
        print("\n" + "=" * 80)
        print("🧭 SPATIAL QUESTIONS")
        print("=" * 80)
        spatial = report['spatial_questions']
        print(f"Accuracy: {spatial['accuracy']:.4f} ({spatial['correct']}/{spatial['total']})")
        
        print("\n" + "=" * 80)
        print("🔢 NUMERICAL QUESTIONS")
        print("=" * 80)
        numerical = report['numerical_questions']
        print(f"Accuracy: {numerical['accuracy']:.4f} ({numerical['correct']}/{numerical['total']})")
        
        print("\n" + "=" * 80)
        print("🎨 COLOR QUESTIONS")
        print("=" * 80)
        color = report['color_questions']
        print(f"Accuracy: {color['accuracy']:.4f} ({color['correct']}/{color['total']})")
        
        # Semantic Similarity Analysis
        print("\n" + "=" * 80)
        print("📊 SEMANTIC SIMILARITY SCORES")
        print("=" * 80)
        sem_analysis = report['semantic_similarity_analysis']
        if sem_analysis.get('errors'):
            print("\nErrors:")
            print(f"  Mean: {sem_analysis['errors']['mean']:.4f}")
            print(f"  Min: {sem_analysis['errors']['min']:.4f}")
            print(f"  Max: {sem_analysis['errors']['max']:.4f}")
        if sem_analysis.get('correct'):
            print("\nCorrect:")
            print(f"  Mean: {sem_analysis['correct']['mean']:.4f}")
            print(f"  Min: {sem_analysis['correct']['min']:.4f}")
            print(f"  Max: {sem_analysis['correct']['max']:.4f}")
        
        # Error Examples
        print("\n" + "=" * 80)
        print("❌ ERROR EXAMPLES (First 10)")
        print("=" * 80)
        for i, example in enumerate(report['error_examples'][:10], 1):
            print(f"\n{i}. Question: {example['question']}")
            print(f"   Situation: {example['situation'][:100]}..." if len(example.get('situation', '')) > 100 else f"   Situation: {example.get('situation', 'N/A')}")
            print(f"   GT Answer: {example['gt']}")
            print(f"   Predicted: {example['predicted'][:100]}..." if len(example.get('predicted', '')) > 100 else f"   Predicted: {example.get('predicted', 'N/A')}")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument(
        '--results-file',
        type=str,
        default='sqa_test_results.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for JSON report (optional)'
    )
    parser.add_argument(
        '--print',
        action='store_true',
        default=True,
        help='Print report to console (default: True)'
    )
    
    args = parser.parse_args()
    
    # Load and analyze
    print(f"Loading results from: {args.results_file}")
    analyzer = ResultAnalyzer(args.results_file)
    
    print("Generating analysis report...")
    report = analyzer.generate_full_report()
    
    # Print report
    if args.print:
        analyzer.print_report(report)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Full report saved to: {args.output}")


if __name__ == "__main__":
    main()

