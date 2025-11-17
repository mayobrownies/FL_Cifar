"""
Logging and visualization utilities for FL experiments
"""
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import config


class FLLogger:
    """Logger for federated learning experiments"""

    def __init__(self, output_dir="fl_plots", experiment_name="hetero_proto_fl"):
        # Convert to absolute path if relative
        self.output_dir = os.path.abspath(output_dir)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Metrics storage
        self.metrics = {
            "rounds": [],
            "train_loss": [],
            "test_accuracy": [],
            "test_f1": [],
            "test_precision": [],
            "test_recall": [],
            "per_client_metrics": {},
            "ensemble_accuracy": [],
            "ensemble_f1": [],
            "ensemble_precision": [],
            "ensemble_recall": []
        }

    def log_round(self, round_num, train_loss=None):
        """Log training metrics for a round"""
        self.metrics["rounds"].append(round_num)
        if train_loss is not None:
            self.metrics["train_loss"].append(train_loss)

    def log_evaluation(self, round_num, clients_metrics, ensemble_metrics=None):
        """Log evaluation metrics for all clients and ensemble"""
        # Store per-client metrics
        for client_id, metrics in enumerate(clients_metrics):
            if client_id not in self.metrics["per_client_metrics"]:
                self.metrics["per_client_metrics"][client_id] = {
                    "accuracy": [],
                    "f1": [],
                    "precision": [],
                    "recall": []
                }

            self.metrics["per_client_metrics"][client_id]["accuracy"].append(metrics['accuracy'])
            self.metrics["per_client_metrics"][client_id]["f1"].append(metrics['f1_macro'])
            self.metrics["per_client_metrics"][client_id]["precision"].append(metrics['precision'])
            self.metrics["per_client_metrics"][client_id]["recall"].append(metrics['recall'])

        # Compute weighted average across clients
        avg_acc = np.mean([m['accuracy'] for m in clients_metrics])
        avg_f1 = np.mean([m['f1_macro'] for m in clients_metrics])
        avg_precision = np.mean([m['precision'] for m in clients_metrics])
        avg_recall = np.mean([m['recall'] for m in clients_metrics])

        self.metrics["test_accuracy"].append(avg_acc)
        self.metrics["test_f1"].append(avg_f1)
        self.metrics["test_precision"].append(avg_precision)
        self.metrics["test_recall"].append(avg_recall)

        # Store ensemble metrics if provided
        if ensemble_metrics is not None:
            self.metrics["ensemble_accuracy"].append(ensemble_metrics['accuracy'])
            self.metrics["ensemble_f1"].append(ensemble_metrics['f1_macro'])
            self.metrics["ensemble_precision"].append(ensemble_metrics['precision'])
            self.metrics["ensemble_recall"].append(ensemble_metrics['recall'])

        print(f"\n[Round {round_num}] Avg Client Accuracy: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")
        if ensemble_metrics is not None:
            print(f"[Round {round_num}] Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}, Ensemble F1: {ensemble_metrics['f1_macro']:.4f}")

    def save_results(self):
        """Save results to text file"""
        # Ensure output directory exists and is absolute
        output_dir = os.path.abspath(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Text summary
        summary_file = os.path.join(
            output_dir,
            f"{self.experiment_name}_{self.timestamp}_summary.txt"
        )

        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEDERATED LEARNING EXPERIMENT RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Rounds: {len(self.metrics['rounds'])}\n\n")

            f.write("="*80 + "\n")
            f.write("OVERALL METRICS (Client Average)\n")
            f.write("="*80 + "\n")
            f.write(f"{'Round':<10}{'Accuracy':<15}{'F1 Score':<15}{'Precision':<15}{'Recall':<15}\n")
            f.write("-"*80 + "\n")

            for i, round_num in enumerate(self.metrics['rounds']):
                if i < len(self.metrics['test_accuracy']):
                    f.write(f"{round_num:<10}"
                           f"{self.metrics['test_accuracy'][i]:<15.4f}"
                           f"{self.metrics['test_f1'][i]:<15.4f}"
                           f"{self.metrics['test_precision'][i]:<15.4f}"
                           f"{self.metrics['test_recall'][i]:<15.4f}\n")

            # Add ensemble metrics section
            if self.metrics["ensemble_accuracy"]:
                f.write("\n" + "="*80 + "\n")
                f.write("ENSEMBLE MODEL METRICS (Global Performance)\n")
                f.write("="*80 + "\n")
                f.write(f"{'Round':<10}{'Accuracy':<15}{'F1 Score':<15}{'Precision':<15}{'Recall':<15}\n")
                f.write("-"*80 + "\n")

                for i, round_num in enumerate(self.metrics['rounds']):
                    if i < len(self.metrics['ensemble_accuracy']):
                        f.write(f"{round_num:<10}"
                               f"{self.metrics['ensemble_accuracy'][i]:<15.4f}"
                               f"{self.metrics['ensemble_f1'][i]:<15.4f}"
                               f"{self.metrics['ensemble_precision'][i]:<15.4f}"
                               f"{self.metrics['ensemble_recall'][i]:<15.4f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("PER-CLIENT FINAL METRICS\n")
            f.write("="*80 + "\n")

            for client_id, client_metrics in self.metrics["per_client_metrics"].items():
                f.write(f"\nClient {client_id} ({config.MODEL_TYPES[client_id]}):\n")
                if client_metrics["accuracy"]:
                    final_acc = client_metrics["accuracy"][-1]
                    final_f1 = client_metrics["f1"][-1]
                    final_precision = client_metrics["precision"][-1]
                    final_recall = client_metrics["recall"][-1]

                    f.write(f"  Final Accuracy:  {final_acc:.4f} ({final_acc*100:.2f}%)\n")
                    f.write(f"  Final F1:        {final_f1:.4f}\n")
                    f.write(f"  Final Precision: {final_precision:.4f}\n")
                    f.write(f"  Final Recall:    {final_recall:.4f}\n")

        print(f"\n[RESULTS] Saved to:")
        print(f"  Summary: {summary_file}")

    def plot_metrics(self):
        """Generate plots for training progression"""
        if not self.metrics["rounds"]:
            print("[WARNING] No metrics to plot")
            return

        # Ensure output directory exists and is absolute
        output_dir = os.path.abspath(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Federated Learning Training Progression\n{self.experiment_name}', fontsize=14)

        rounds = self.metrics["rounds"]

        # Plot 1: Accuracy over rounds (Client Average vs Ensemble)
        ax = axes[0, 0]
        if self.metrics["test_accuracy"]:
            ax.plot(rounds[:len(self.metrics["test_accuracy"])],
                   self.metrics["test_accuracy"],
                   'b-o', linewidth=2, markersize=6, label='Avg Clients')
            if self.metrics["ensemble_accuracy"]:
                ax.plot(rounds[:len(self.metrics["ensemble_accuracy"])],
                       self.metrics["ensemble_accuracy"],
                       'r-s', linewidth=2, markersize=6, label='Ensemble')
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('Test Accuracy', fontsize=11)
            ax.set_title('Test Accuracy over Rounds', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        # Plot 2: F1 Score over rounds (Client Average vs Ensemble)
        ax = axes[0, 1]
        if self.metrics["test_f1"]:
            ax.plot(rounds[:len(self.metrics["test_f1"])],
                   self.metrics["test_f1"],
                   'g-o', linewidth=2, markersize=6, label='Avg Clients')
            if self.metrics["ensemble_f1"]:
                ax.plot(rounds[:len(self.metrics["ensemble_f1"])],
                       self.metrics["ensemble_f1"],
                       'r-s', linewidth=2, markersize=6, label='Ensemble')
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('F1 Score', fontsize=11)
            ax.set_title('F1 Score over Rounds', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        # Plot 3: Per-client accuracy comparison
        ax = axes[1, 0]
        colors = ['b', 'orange', 'g']

        for client_id, client_metrics in self.metrics["per_client_metrics"].items():
            if client_metrics["accuracy"]:
                eval_rounds = list(range(1, len(client_metrics["accuracy"]) + 1))
                ax.plot(eval_rounds, client_metrics["accuracy"],
                       marker='o', color=colors[client_id],
                       label=config.MODEL_TYPES[client_id],
                       linewidth=2, markersize=6)

        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Per-Client Accuracy Comparison', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 4: Precision vs Recall
        ax = axes[1, 1]
        if self.metrics["test_precision"] and self.metrics["test_recall"]:
            ax.plot(rounds[:len(self.metrics["test_precision"])],
                   self.metrics["test_precision"],
                   'r-o', label='Precision', linewidth=2, markersize=6)
            ax.plot(rounds[:len(self.metrics["test_recall"])],
                   self.metrics["test_recall"],
                   'm-o', label='Recall', linewidth=2, markersize=6)
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title('Precision & Recall over Rounds', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(
            output_dir,
            f"{self.experiment_name}_{self.timestamp}_plots.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Plot: {plot_file}")
        plt.close()
