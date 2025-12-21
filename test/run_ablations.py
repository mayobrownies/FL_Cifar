import subprocess
import os
import sys
import glob
import shutil

ablation_configs = {
    "ULCD-Full": {
        "ULCD_USE_PUBLIC_ALIGNMENT": True,
        "ULCD_USE_EMA": True,
        "ULCD_USE_WARMUP": True,
        "ULCD_USE_CONTRASTIVE": True,
        "description": "Full ULCD (all features enabled with contrastive consensus)"
    },
    "ULCD-NoPublic": {
        "ULCD_USE_PUBLIC_ALIGNMENT": False,
        "ULCD_USE_EMA": True,
        "ULCD_USE_WARMUP": True,
        "ULCD_USE_CONTRASTIVE": True,
        "description": "Private data alignment with contrastive consensus"
    },
    "ULCD-NoEMA": {
        "ULCD_USE_PUBLIC_ALIGNMENT": True,
        "ULCD_USE_EMA": False,
        "ULCD_USE_WARMUP": True,
        "ULCD_USE_CONTRASTIVE": True,
        "description": "Simple averaging with contrastive consensus"
    },
    "ULCD-NoWarmup": {
        "ULCD_USE_PUBLIC_ALIGNMENT": True,
        "ULCD_USE_EMA": True,
        "ULCD_USE_WARMUP": False,
        "ULCD_USE_CONTRASTIVE": True,
        "description": "Fixed weight with contrastive consensus"
    },
    "ULCD-NoContrastive": {
        "ULCD_USE_PUBLIC_ALIGNMENT": True,
        "ULCD_USE_EMA": True,
        "ULCD_USE_WARMUP": True,
        "ULCD_USE_CONTRASTIVE": False,
        "description": "Without contrastive consensus (old ULCD)"
    },
    "ULCD-Minimal": {
        "ULCD_USE_PUBLIC_ALIGNMENT": False,
        "ULCD_USE_EMA": False,
        "ULCD_USE_WARMUP": False,
        "ULCD_USE_CONTRASTIVE": False,
        "description": "All features disabled (closest to basic FedProto)"
    }
}

def modify_config(flags):
    with open('test/config.py', 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        modified = False
        for flag, value in flags.items():
            if line.strip().startswith(flag + ' ='):
                new_lines.append(f'{flag} = {value}\n')
                modified = True
                break
        if not modified:
            new_lines.append(line)

    with open('test/config.py', 'w') as f:
        f.writelines(new_lines)

def run_ablation(variant_name, config_flags, description):
    print("=" * 80)
    print(f"Running: {variant_name}")
    print(f"Description: {description}")
    print(f"Config: {config_flags}")
    print("=" * 80)

    modify_config(config_flags)

    result = subprocess.run([sys.executable, 'test/ulcd_main.py'],
                          capture_output=False,
                          text=True)

    if result.returncode != 0:
        print(f"ERROR: {variant_name} failed with return code {result.returncode}")
        return False

    output_dir = f"./fl_plots/ablations/{variant_name}"
    os.makedirs(output_dir, exist_ok=True)

    summary_files = glob.glob('./fl_plots/*_summary.txt')
    if summary_files:
        latest_summary = max(summary_files, key=os.path.getmtime)
        shutil.move(latest_summary, f'{output_dir}/results.txt')
        print(f"Moved {latest_summary} -> {output_dir}/results.txt")

    plot_files = glob.glob('./fl_plots/*_plots.png')
    if plot_files:
        latest_plot = max(plot_files, key=os.path.getmtime)
        shutil.move(latest_plot, f'{output_dir}/plots.png')
        print(f"Moved {latest_plot} -> {output_dir}/plots.png")

    checkpoint_dirs = ['./fl_plots/best_ulcd', './fl_plots/checkpoints_ulcd']
    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            dest = f'{output_dir}/{os.path.basename(ckpt_dir)}'
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(ckpt_dir, dest)
            print(f"Moved {ckpt_dir} -> {dest}")

    print(f"Results saved to {output_dir}")
    print()

    return True

def main():
    print("Starting ULCD Ablation Study")
    print("=" * 80)

    original_config = {}
    with open('test/config.py', 'r') as f:
        for line in f:
            for flag in ["ULCD_USE_PUBLIC_ALIGNMENT", "ULCD_USE_EMA", "ULCD_USE_WARMUP", "ULCD_USE_CONTRASTIVE"]:
                if line.strip().startswith(flag + ' ='):
                    original_config[flag] = line.strip()

    results_summary = []

    for variant_name, config in ablation_configs.items():
        flags = {k: v for k, v in config.items() if k != "description"}
        success = run_ablation(variant_name, flags, config["description"])

        if success:
            results_summary.append({
                "variant": variant_name,
                "description": config["description"],
                "config": flags
            })

    os.makedirs('./fl_plots/ablations', exist_ok=True)

    with open('./fl_plots/ablations/summary.txt', 'w') as f:
        f.write("ULCD ABLATION STUDY SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for result in results_summary:
            f.write(f"Variant: {result['variant']}\n")
            f.write(f"Description: {result['description']}\n")
            f.write(f"Configuration:\n")
            for flag, value in result['config'].items():
                f.write(f"  {flag}: {value}\n")
            f.write("\n")

            result_file = f"./fl_plots/ablations/{result['variant']}/results.txt"
            if os.path.exists(result_file):
                with open(result_file, 'r') as rf:
                    lines = rf.readlines()

                    in_best_section = False
                    in_ensemble_section = False
                    ensemble_lines = []

                    for i, line in enumerate(lines):
                        if 'BEST PERFORMANCE' in line:
                            in_best_section = True
                        if 'ENSEMBLE MODEL METRICS' in line:
                            in_ensemble_section = True

                        if in_best_section and ('Best' in line or 'Achieved' in line):
                            f.write(line)

                        if in_ensemble_section and i < len(lines) - 1:
                            ensemble_lines.append(line)
                            if len(ensemble_lines) > 10:
                                break

                    if ensemble_lines:
                        f.write("\n")
                        for line in ensemble_lines[:8]:
                            f.write(line)

            f.write("\n" + "-" * 80 + "\n\n")

    with open('test/config.py', 'r') as orig:
        lines = orig.readlines()

    with open('test/config.py', 'w') as f:
        for line in lines:
            modified = False
            for flag, orig_line in original_config.items():
                if line.strip().startswith(flag + ' ='):
                    f.write(orig_line + '\n')
                    modified = True
                    break
            if not modified:
                f.write(line)

    print("=" * 80)
    print("Ablation study complete!")
    print("Results saved to ./fl_plots/ablations/")
    print("Summary: ./fl_plots/ablations/summary.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()
