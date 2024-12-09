from typing import Dict, List, Optional
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EvaluationReport:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self):
        self.metrics = None
        self.segment_results = None
        
    def add_metrics(self, metrics: Dict[str, float]):
        """Add overall metrics"""
        self.metrics = metrics
    
    def add_segment_results(self, results: Dict[str, Dict]):
        """Add per-segment results"""
        self.segment_results = results
    
    def generate_report(self) -> str:
        """Generate text report"""
        report = ["=== Sybil Detection Evaluation Report ===\n"]
        
        # Overall metrics
        if self.metrics:
            report.append("Overall Performance Metrics:")
            for metric, value in self.metrics.items():
                report.append(f"{metric}: {value:.3f}")
        
        # Segment results
        if self.segment_results:
            report.append("\nPerformance by Segment:")
            for segment, results in self.segment_results.items():
                report.append(f"\n{segment}:")
                for model, scores in results.items():
                    report.append(f"  {model}:")
                    for metric, value in scores.items():
                        report.append(f"    {metric}: {value:.3f}")
        
        return "\n".join(report)
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'Bot'],
                   yticklabels=['Human', 'Bot'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_probability_distribution(self, probas: np.ndarray):
        """Plot probability distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(probas, bins=50, density=True)
        plt.axvline(x=0.5, color='r', linestyle='--')
        plt.xlabel('Bot Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Bot Probabilities')
        plt.grid(True, alpha=0.3)
        plt.show()