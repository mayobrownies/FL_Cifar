import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from . import config

class FLLogger:

    def __init__(self, output_dir="fl_plots", experiment_name="hetero_fl"):
        self.output_dir = os.path.abspath(output_dir)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics = {
            "rounds": [],
            "eval_rounds": [],
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
        self.communication_metrics = None
        self.best_accuracy = None
        self.best_round = None
        self.convergence_round = None

    def log_round(self, round_num, train_loss=None):
        self.metrics["rounds"].append(round_num)
        if train_loss is not None:
            self.metrics["train_loss"].append(train_loss)

    def log_evaluation(self, round_num, clients_metrics, ensemble_metrics=None):
        self.metrics["eval_rounds"].append(round_num)

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

        avg_acc = np.mean([m['accuracy'] for m in clients_metrics])
        avg_f1 = np.mean([m['f1_macro'] for m in clients_metrics])
        avg_precision = np.mean([m['precision'] for m in clients_metrics])
        avg_recall = np.mean([m['recall'] for m in clients_metrics])

        self.metrics["test_accuracy"].append(avg_acc)
        self.metrics["test_f1"].append(avg_f1)
        self.metrics["test_precision"].append(avg_precision)
        self.metrics["test_recall"].append(avg_recall)

        if ensemble_metrics is not None:
            self.metrics["ensemble_accuracy"].append(ensemble_metrics['accuracy'])
            self.metrics["ensemble_f1"].append(ensemble_metrics['f1_macro'])
            self.metrics["ensemble_precision"].append(ensemble_metrics['precision'])
            self.metrics["ensemble_recall"].append(ensemble_metrics['recall'])

        print(f"\n[Round {round_num}] Avg Client Accuracy: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")
        if ensemble_metrics is not None:
            print(f"[Round {round_num}] Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}, Ensemble F1: {ensemble_metrics['f1_macro']:.4f}")

    def log_communication(self, bytes_per_round, total_bytes, total_mb, efficiency):
        self.communication_metrics = {
            "bytes_per_round_per_client": bytes_per_round,
            "total_bytes": total_bytes,
            "total_mb": total_mb,
            "efficiency": efficiency
        }

    def log_best(self, best_acc, best_round):
        self.best_accuracy = best_acc
        self.best_round = best_round

    def check_convergence(self, round_num):
        if self.convergence_round is not None:
            return

        if len(self.metrics["ensemble_accuracy"]) < config.CONVERGENCE_WINDOW:
            return

        recent_accuracies = self.metrics["ensemble_accuracy"][-config.CONVERGENCE_WINDOW:]
        max_acc = max(recent_accuracies)
        min_acc = min(recent_accuracies)

        if max_acc - min_acc <= config.CONVERGENCE_THRESHOLD:
            self.convergence_round = round_num - config.CONVERGENCE_WINDOW + 1
            print(f"\n{'='*80}")
            print(f"CONVERGENCE DETECTED at Round {self.convergence_round}")
            print(f"  Accuracy stable within {config.CONVERGENCE_THRESHOLD*100:.1f}% for {config.CONVERGENCE_WINDOW} rounds")
            print(f"  Range: [{min_acc:.4f}, {max_acc:.4f}]")
            print(f"{'='*80}\n")

    def save_results(self):
        output_dir = os.path.abspath(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        summary_file = os.path.join(
            output_dir,
            f"{self.experiment_name}_{self.timestamp}_summary.txt"
        )

        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Rounds: {len(self.metrics['rounds'])}\n\n")

            f.write("="*80 + "\n")
            f.write("DATA CONFIGURATION\n")
            f.write("="*80 + "\n")
            f.write(f"  Number of Clients:     {config.NUM_CLIENTS}\n")
            f.write(f"  Model Types:           {', '.join(config.MODEL_TYPES)}\n")
            f.write(f"  Batch Size:            {config.BATCH_SIZE}\n")
            f.write(f"  Learning Rate:         {config.LEARNING_RATE}\n")
            f.write(f"\n  Label Heterogeneity:   {config.LABEL_HETEROGENEITY}\n")
            if config.LABEL_HETEROGENEITY == "partial_overlap":
                f.write(f"    Classes per Client:  {config.LABEL_CLASSES_PER_CLIENT}\n")
                f.write(f"    Overlap Size:        {config.LABEL_OVERLAP_SIZE}\n")
            elif config.LABEL_HETEROGENEITY == "non_overlapping":
                f.write(f"    (Classes split evenly with no overlap)\n")
            f.write(f"\n  Data Heterogeneity:    {config.DATA_HETEROGENEITY}\n")
            if config.DATA_HETEROGENEITY in ["transform", "mixed"]:
                f.write(f"    Transform Types:     {', '.join(config.DATA_TRANSFORM_TYPES)}\n")
            f.write("\n")

            if "ULCD" in self.experiment_name and hasattr(config, 'ULCD_USE_PUBLIC_ALIGNMENT'):
                f.write("="*80 + "\n")
                f.write("ABLATION CONFIGURATION\n")
                f.write("="*80 + "\n")
                f.write(f"  Public Alignment:      {config.ULCD_USE_PUBLIC_ALIGNMENT}\n")
                f.write(f"  EMA Smoothing:         {config.ULCD_USE_EMA}\n")
                f.write(f"  Warmup Schedule:       {config.ULCD_USE_WARMUP}\n")
                f.write(f"  Contrastive Consensus: {config.ULCD_USE_CONTRASTIVE}\n\n")

            f.write("="*80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("="*80 + "\n")
            f.write(f"{'Round':<10}{'Accuracy':<15}{'F1 Score':<15}{'Precision':<15}{'Recall':<15}\n")
            f.write("-"*80 + "\n")

            for i, round_num in enumerate(self.metrics['eval_rounds']):
                if i < len(self.metrics['test_accuracy']):
                    f.write(f"{round_num:<10}"
                           f"{self.metrics['test_accuracy'][i]:<15.4f}"
                           f"{self.metrics['test_f1'][i]:<15.4f}"
                           f"{self.metrics['test_precision'][i]:<15.4f}"
                           f"{self.metrics['test_recall'][i]:<15.4f}\n")

            if self.metrics["ensemble_accuracy"]:
                f.write("\n" + "="*80 + "\n")
                f.write("ENSEMBLE MODEL METRICS (Global Performance)\n")
                f.write("="*80 + "\n")
                f.write(f"{'Round':<10}{'Accuracy':<15}{'F1 Score':<15}{'Precision':<15}{'Recall':<15}\n")
                f.write("-"*80 + "\n")

                for i, round_num in enumerate(self.metrics['eval_rounds']):
                    if i < len(self.metrics['ensemble_accuracy']):
                        f.write(f"{round_num:<10}"
                               f"{self.metrics['ensemble_accuracy'][i]:<15.4f}"
                               f"{self.metrics['ensemble_f1'][i]:<15.4f}"
                               f"{self.metrics['ensemble_precision'][i]:<15.4f}"
                               f"{self.metrics['ensemble_recall'][i]:<15.4f}\n")

            # Compute steady-state performance (average over last N evaluations)
            if len(self.metrics["ensemble_accuracy"]) >= config.FINAL_AVERAGE_WINDOW:
                f.write("\n" + "="*80 + "\n")
                f.write(f"STEADY-STATE PERFORMANCE (Last {config.FINAL_AVERAGE_WINDOW} Evaluations)\n")
                f.write("="*80 + "\n")

                last_n_acc = self.metrics["ensemble_accuracy"][-config.FINAL_AVERAGE_WINDOW:]
                last_n_f1 = self.metrics["ensemble_f1"][-config.FINAL_AVERAGE_WINDOW:]
                last_n_rounds = self.metrics["eval_rounds"][-config.FINAL_AVERAGE_WINDOW:]

                mean_acc = np.mean(last_n_acc)
                std_acc = np.std(last_n_acc, ddof=1)  # Sample std dev
                min_acc = np.min(last_n_acc)
                max_acc = np.max(last_n_acc)

                mean_f1 = np.mean(last_n_f1)
                std_f1 = np.std(last_n_f1, ddof=1)

                f.write(f"  Ensemble Accuracy:  {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)\n")
                f.write(f"  Ensemble F1:        {mean_f1:.4f} ± {std_f1:.4f}\n")
                f.write(f"  Accuracy Range:     [{min_acc:.4f}, {max_acc:.4f}]\n")
                f.write(f"  Evaluation Period:  Rounds {last_n_rounds[0]}-{last_n_rounds[-1]}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("PER-CLIENT FINAL METRICS (Round {0})\n".format(self.metrics["eval_rounds"][-1] if self.metrics["eval_rounds"] else 0))
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

            if self.best_accuracy is not None:
                f.write("\n" + "="*80 + "\n")
                f.write("BEST PERFORMANCE\n")
                f.write("="*80 + "\n")
                f.write(f"  Best Ensemble Accuracy: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)\n")
                f.write(f"  Achieved at Round: {self.best_round}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("CONVERGENCE ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"  Convergence Criteria: Accuracy stable within {config.CONVERGENCE_THRESHOLD*100:.1f}% ")
            f.write(f"for {config.CONVERGENCE_WINDOW} consecutive rounds\n")

            if self.convergence_round is not None:
                f.write(f"  Status: CONVERGED\n")
                f.write(f"  Convergence Round: {self.convergence_round}\n")
                if self.convergence_round < len(self.metrics['eval_rounds']):
                    conv_idx = self.metrics['eval_rounds'].index(self.convergence_round) if self.convergence_round in self.metrics['eval_rounds'] else -1
                    if conv_idx >= 0 and conv_idx < len(self.metrics["ensemble_accuracy"]):
                        conv_acc = self.metrics["ensemble_accuracy"][conv_idx]
                        f.write(f"  Accuracy at Convergence: {conv_acc:.4f} ({conv_acc*100:.2f}%)\n")
            else:
                f.write(f"  Status: DID NOT CONVERGE\n")
                if len(self.metrics["ensemble_accuracy"]) >= config.CONVERGENCE_WINDOW:
                    min_variance = float('inf')
                    best_window_start = 0
                    for i in range(len(self.metrics["ensemble_accuracy"]) - config.CONVERGENCE_WINDOW + 1):
                        window = self.metrics["ensemble_accuracy"][i:i + config.CONVERGENCE_WINDOW]
                        variance = max(window) - min(window)
                        if variance < min_variance:
                            min_variance = variance
                            best_window_start = i

                    f.write(f"  Minimum Variance: {min_variance*100:.2f}% (rounds {self.metrics['eval_rounds'][best_window_start]}-{self.metrics['eval_rounds'][best_window_start + config.CONVERGENCE_WINDOW - 1]})\n")
                    f.write(f"  Note: Model remained relatively stable but exceeded {config.CONVERGENCE_THRESHOLD*100:.1f}% threshold\n")

            if self.communication_metrics is not None:
                f.write("\n" + "="*80 + "\n")
                f.write("COMMUNICATION COSTS\n")
                f.write("="*80 + "\n")
                f.write(f"  Bytes per round per client: {self.communication_metrics['bytes_per_round_per_client']:,} bytes ")
                f.write(f"({self.communication_metrics['bytes_per_round_per_client']/1024:.2f} KB)\n")
                f.write(f"  Total communication:        {self.communication_metrics['total_bytes']:,} bytes ")
                f.write(f"({self.communication_metrics['total_mb']:.2f} MB)\n")
                f.write(f"  Communication efficiency:   {self.communication_metrics['efficiency']:.4f} (Accuracy/MB)\n")

        print(f"\nSaved to:")
        print(f"Summary: {summary_file}")

    def plot_metrics(self):
        if not self.metrics["eval_rounds"]:
            print("No metrics to plot")
            return

        output_dir = os.path.abspath(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.experiment_name} Training Progression', fontsize=14)

        eval_rounds = self.metrics["eval_rounds"]

        ax = axes[0, 0]
        if self.metrics["test_accuracy"]:
            ax.plot(eval_rounds[:len(self.metrics["test_accuracy"])],
                   self.metrics["test_accuracy"],
                   'b-o', linewidth=2, markersize=6, label='Avg Clients')
            if self.metrics["ensemble_accuracy"]:
                ax.plot(eval_rounds[:len(self.metrics["ensemble_accuracy"])],
                       self.metrics["ensemble_accuracy"],
                       'r-s', linewidth=2, markersize=6, label='Ensemble')
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('Test Accuracy', fontsize=11)
            ax.set_title('Test Accuracy over Rounds', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        ax = axes[0, 1]
        if self.metrics["test_f1"]:
            ax.plot(eval_rounds[:len(self.metrics["test_f1"])],
                   self.metrics["test_f1"],
                   'g-o', linewidth=2, markersize=6, label='Avg Clients')
            if self.metrics["ensemble_f1"]:
                ax.plot(eval_rounds[:len(self.metrics["ensemble_f1"])],
                       self.metrics["ensemble_f1"],
                       'r-s', linewidth=2, markersize=6, label='Ensemble')
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('F1 Score', fontsize=11)
            ax.set_title('F1 Score over Rounds', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        ax = axes[1, 0]
        colors = ['b', 'orange', 'g']

        for client_id, client_metrics in self.metrics["per_client_metrics"].items():
            if client_metrics["accuracy"]:
                ax.plot(eval_rounds[:len(client_metrics["accuracy"])],
                       client_metrics["accuracy"],
                       marker='o', color=colors[client_id],
                       label=config.MODEL_TYPES[client_id],
                       linewidth=2, markersize=6)

        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Per-Client Accuracy over Rounds', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        ax = axes[1, 1]
        if self.metrics["test_precision"] and self.metrics["test_recall"]:
            ax.plot(eval_rounds[:len(self.metrics["test_precision"])],
                   self.metrics["test_precision"],
                   'b-o', linewidth=2, markersize=6, label='Precision (Avg Clients)')
            ax.plot(eval_rounds[:len(self.metrics["test_recall"])],
                   self.metrics["test_recall"],
                   'b--o', linewidth=2, markersize=6, label='Recall (Avg Clients)')
            if self.metrics["ensemble_precision"] and self.metrics["ensemble_recall"]:
                ax.plot(eval_rounds[:len(self.metrics["ensemble_precision"])],
                       self.metrics["ensemble_precision"],
                       'r-s', linewidth=2, markersize=6, label='Precision (Ensemble)')
                ax.plot(eval_rounds[:len(self.metrics["ensemble_recall"])],
                       self.metrics["ensemble_recall"],
                       'r--s', linewidth=2, markersize=6, label='Recall (Ensemble)')
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title('Precision & Recall over Rounds', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()

        plot_file = os.path.join(
            output_dir,
            f"{self.experiment_name}_{self.timestamp}_plots.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Plot: {plot_file}")
        plt.close()
