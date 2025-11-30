"""
Visualize evaluation results analysis
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Create visualizations from analysis results"""
    
    def __init__(self, analysis_file: str):
        """Load analysis results"""
        with open(analysis_file, 'r') as f:
            self.data = json.load(f)
        
        self.output_dir = Path(analysis_file).parent
    
    def plot_question_type_accuracy(self, save_path: str = None):
        """Plot accuracy by question type"""
        qtype_data = self.data['by_question_type']
        
        # Sort by accuracy
        sorted_types = sorted(qtype_data.items(), key=lambda x: x[1]['accuracy'])
        types = [t[0] for t in sorted_types]
        accuracies = [t[1]['accuracy'] for t in sorted_types]
        totals = [t[1]['total'] for t in sorted_types]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create color map based on accuracy
        colors = ['#d62728' if acc < 0.3 else '#ff7f0e' if acc < 0.5 else '#2ca02c' 
                 for acc in accuracies]
        
        bars = ax.barh(types, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, acc, total) in enumerate(zip(bars, accuracies, totals)):
            ax.text(acc + 0.01, i, f'{acc:.3f} (n={total})', 
                   va='center', fontsize=9)
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Question Type', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by Question Type', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#d62728', label='Low (<30%)'),
            mpatches.Patch(color='#ff7f0e', label='Medium (30-50%)'),
            mpatches.Patch(color='#2ca02c', label='High (>50%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_answer_length_analysis(self, save_path: str = None):
        """Plot accuracy by answer length"""
        length_data = self.data['by_answer_length']
        
        # Filter to reasonable lengths (1-10 words)
        filtered = {k: v for k, v in length_data.items() if 1 <= int(k) <= 10}
        
        lengths = sorted([int(k) for k in filtered.keys()])
        accuracies = [filtered[str(l)]['accuracy'] for l in lengths]
        totals = [filtered[str(l)]['total'] for l in lengths]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Accuracy by length
        bars = ax1.bar(range(len(lengths)), accuracies, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(lengths)))
        ax1.set_xticklabels(lengths)
        ax1.set_xlabel('Answer Length (words)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy by Answer Length', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Distribution of answer lengths
        ax2.bar(range(len(lengths)), totals, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(lengths)))
        ax2.set_xticklabels(lengths)
        ax2.set_xlabel('Answer Length (words)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Answer Lengths', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, total in enumerate(totals):
            ax2.text(i, total + max(totals)*0.01, f'{total}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_vocabulary_analysis(self, save_path: str = None):
        """Plot top error-prone keywords"""
        q_vocab = self.data['by_question_vocabulary']
        s_vocab = self.data['by_situation_vocabulary']
        
        # Get top 15 from each
        top_q = list(q_vocab.items())[:15]
        top_s = list(s_vocab.items())[:15]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Question vocabulary
        q_words = [w[0] for w in top_q]
        q_rates = [w[1]['error_rate'] for w in top_q]
        q_totals = [w[1]['total'] for w in top_q]
        
        bars1 = ax1.barh(q_words, q_rates, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Error Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Keyword', fontsize=12, fontweight='bold')
        ax1.set_title('Top Error-Prone Keywords in Questions', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1.0)
        
        for i, (bar, rate, total) in enumerate(zip(bars1, q_rates, q_totals)):
            ax1.text(rate + 0.02, i, f'{rate:.2f} (n={total})', 
                    va='center', fontsize=9)
        
        # Situation vocabulary
        s_words = [w[0] for w in top_s]
        s_rates = [w[1]['error_rate'] for w in top_s]
        s_totals = [w[1]['total'] for w in top_s]
        
        bars2 = ax2.barh(s_words, s_rates, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Error Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Keyword', fontsize=12, fontweight='bold')
        ax2.set_title('Top Error-Prone Keywords in Situations', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        
        for i, (bar, rate, total) in enumerate(zip(bars2, s_rates, s_totals)):
            ax2.text(rate + 0.02, i, f'{rate:.2f} (n={total})', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_special_question_types(self, save_path: str = None):
        """Plot comparison of special question types"""
        spatial = self.data['spatial_questions']
        numerical = self.data['numerical_questions']
        color = self.data['color_questions']
        
        categories = ['Spatial', 'Numerical', 'Color']
        accuracies = [
            spatial['accuracy'],
            numerical['accuracy'],
            color['accuracy']
        ]
        totals = [
            spatial['total'],
            numerical['total'],
            color['total']
        ]
        corrects = [
            spatial['correct'],
            numerical['correct'],
            color['correct']
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(categories, accuracies, color=['#9b59b6', '#e67e22', '#1abc9c'], 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by Special Question Types', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        
        # Add value labels
        for bar, acc, corr, tot in zip(bars, accuracies, corrects, totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.3f}\n({corr}/{tot})', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_semantic_similarity_distribution(self, save_path: str = None):
        """Plot distribution of semantic similarity scores"""
        sem_data = self.data['semantic_similarity_analysis']
        
        # We need to extract scores from results, but for now show summary stats
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ['Errors', 'Correct']
        means = [
            sem_data['errors'].get('mean', 0),
            sem_data['correct'].get('mean', 0)
        ]
        mins = [
            sem_data['errors'].get('min', 0),
            sem_data['correct'].get('min', 0)
        ]
        maxs = [
            sem_data['errors'].get('max', 0),
            sem_data['correct'].get('max', 0)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars for mean with error bars
        bars = ax.bar(x, means, width, yerr=[[means[i] - mins[i] for i in range(len(means))],
                                             [maxs[i] - means[i] for i in range(len(means))]],
                     color=['#e74c3c', '#27ae60'], alpha=0.7, edgecolor='black',
                     capsize=5, label='Mean ± Range')
        
        ax.set_ylabel('Semantic Similarity Score', fontsize=12, fontweight='bold')
        ax.set_title('Semantic Similarity: Errors vs Correct Predictions', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.0)
        
        # Add value labels
        for bar, mean, min_val, max_val in zip(bars, means, mins, maxs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'Mean: {mean:.3f}\nRange: [{min_val:.3f}, {max_val:.3f}]',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_prediction_length_analysis(self, save_path: str = None):
        """Plot prediction length vs ground truth analysis"""
        pred_data = self.data['by_prediction_length']
        
        categories = ['Prediction\nLonger', 'Prediction\nShorter', 'Prediction\nSimilar']
        accuracies = [
            pred_data['pred_longer']['accuracy'],
            pred_data['pred_shorter']['accuracy'],
            pred_data['pred_similar']['accuracy']
        ]
        totals = [
            pred_data['pred_longer']['total'],
            pred_data['pred_shorter']['total'],
            pred_data['pred_similar']['total']
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(categories, accuracies, color=['#34495e', '#95a5a6', '#16a085'],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by Prediction Length Category', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, acc, tot in zip(bars, accuracies, totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.3f}\n(n={tot})', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_overall_summary(self, save_path: str = None):
        """Plot overall accuracy summary"""
        accuracies = self.data['summary']['overall_accuracies']
        
        methods = list(accuracies.keys())
        values = list(accuracies.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color code by performance
        colors = ['#d62728' if v < 0.1 else '#ff7f0e' if v < 0.3 else '#2ca02c' 
                 for v in values]
        
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Evaluation Method', fontsize=12, fontweight='bold')
        ax.set_title('Overall Accuracy by Evaluation Method', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#d62728', label='Low (<10%)'),
            mpatches.Patch(color='#ff7f0e', label='Medium (10-30%)'),
            mpatches.Patch(color='#2ca02c', label='High (>30%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_answer_type_analysis(self, save_path: str = None):
        """Plot accuracy by answer type"""
        atype_data = self.data['by_answer_type']
        
        if not atype_data or len(atype_data) == 0:
            print("⚠️  No answer type data available")
            return
        
        # Sort by accuracy
        sorted_types = sorted(atype_data.items(), key=lambda x: x[1]['accuracy'])
        types = [t[0] for t in sorted_types]
        accuracies = [t[1]['accuracy'] for t in sorted_types]
        totals = [t[1]['total'] for t in sorted_types]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(types, accuracies, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Answer Type', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by Answer Type', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels
        for i, (bar, acc, total) in enumerate(zip(bars, accuracies, totals)):
            ax.text(acc + 0.01, i, f'{acc:.3f} (n={total})', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def create_all_visualizations(self, output_dir: str = None):
        """Create all visualizations"""
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Creating visualizations...")
        print("=" * 60)
        
        self.plot_overall_summary(save_path=str(output_path / '01_overall_summary.png'))
        self.plot_question_type_accuracy(save_path=str(output_path / '02_question_type_accuracy.png'))
        self.plot_answer_length_analysis(save_path=str(output_path / '03_answer_length_analysis.png'))
        self.plot_vocabulary_analysis(save_path=str(output_path / '04_vocabulary_analysis.png'))
        self.plot_special_question_types(save_path=str(output_path / '05_special_question_types.png'))
        self.plot_prediction_length_analysis(save_path=str(output_path / '06_prediction_length_analysis.png'))
        self.plot_semantic_similarity_distribution(save_path=str(output_path / '07_semantic_similarity.png'))
        self.plot_answer_type_analysis(save_path=str(output_path / '08_answer_type_analysis.png'))
        
        print("=" * 60)
        print(f"✓ All visualizations saved to: {output_path}")
        print(f"  Total: 8 visualization files")


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results analysis')
    parser.add_argument(
        '--analysis-file',
        type=str,
        required=True,
        help='Path to analysis JSON file (from analyze_results.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as analysis file)'
    )
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        choices=['all', 'summary', 'question_type', 'answer_length', 'vocabulary',
                'special', 'prediction_length', 'semantic', 'answer_type'],
        help='Specific plot to create (default: all)'
    )
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.analysis_file)
    
    if args.plot is None or args.plot == 'all':
        visualizer.create_all_visualizations(args.output_dir)
    else:
        output_path = Path(args.output_dir) if args.output_dir else visualizer.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_map = {
            'summary': ('plot_overall_summary', 'overall_summary.png'),
            'question_type': ('plot_question_type_accuracy', 'question_type_accuracy.png'),
            'answer_length': ('plot_answer_length_analysis', 'answer_length_analysis.png'),
            'vocabulary': ('plot_vocabulary_analysis', 'vocabulary_analysis.png'),
            'special': ('plot_special_question_types', 'special_question_types.png'),
            'prediction_length': ('plot_prediction_length_analysis', 'prediction_length_analysis.png'),
            'semantic': ('plot_semantic_similarity_distribution', 'semantic_similarity.png'),
            'answer_type': ('plot_answer_type_analysis', 'answer_type_analysis.png')
        }
        
        method_name, filename = plot_map[args.plot]
        method = getattr(visualizer, method_name)
        method(save_path=str(output_path / filename))


if __name__ == "__main__":
    main()

