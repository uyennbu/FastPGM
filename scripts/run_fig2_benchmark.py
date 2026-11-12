import os
import sys
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FASTPGM_BIN = os.path.join(PROJECT_ROOT, "build", "fastpgm")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ["alarm", "insurance", "hepar2", "munin1"]
DATASET_NAMES = {
    "alarm": "(a) Alarm",
    "insurance": "(b) Insurance",
    "hepar2": "(c) Hepar2",
    "munin1": "(d) Munin1"
}
THREADS_LIST = [1, 2, 4, 8, 12]
GRANULARITIES = [
    (0, "CI-level", "#1f77b4", "s"),      # Blue, square
    (1, "Edge-level", "#ff7f0e", "d"),    # Orange/Yellow, diamond
    (2, "Sample-level", "#2ca02c", "^")   # Green, triangle
]

def run_single(net, sample_size, threads, par_type, group_size=1):
    cmd = [
        FASTPGM_BIN,
        "-j", "0", "-m", "0",
        "-t", str(threads),
        "-p", str(par_type),
        "-g", str(group_size),
        "-a", "0.05",
        "-ss", "0",
        "-f2", f"{net}/{net}_s{sample_size}"
    ]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    time_m = re.search(r"pc-stable:\s*([0-9.eE+-]+)\s*s", res.stdout)
    ci_m   = re.search(r"# of CI-tests =\s*(\d+)", res.stdout)
    t_sec  = float(time_m.group(1)) if time_m else None
    ci_cnt = int(ci_m.group(1)) if ci_m else None
    return t_sec, ci_cnt

def main():
    csv_path = os.path.join(FIGURES_DIR, "fig2_results.csv")
    results = []

    print("=== BẮT ĐẦU BENCHMARK FIG. 2 (CI-level vs Edge-level vs Sample-level) ===")
    for net in DATASETS:
        print(f"\n--- Dataset: {net} (sample size: 5000) ---")
        for par_type, par_name, _, _ in GRANULARITIES:
            for t in THREADS_LIST:
                print(f"Running {net} | {par_name:12} | threads={t:2d}... ", end="", flush=True)
                try:
                    t_sec, ci_cnt = run_single(net, 5000, t, par_type, group_size=1)
                    print(f"Done in {t_sec:.4f}s (CI-tests: {ci_cnt})")
                    results.append({
                        "dataset": net,
                        "parallelism_type": par_type,
                        "granularity": par_name,
                        "threads": t,
                        "execution_time": t_sec,
                        "ci_tests": ci_cnt
                    })
                except Exception as e:
                    print(f"Error: {e}")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nĐã lưu kết quả benchmark vào: {csv_path}")

    # Plotting: Cả 4 hình riêng biệt và 1 hình tổng hợp
    # 1. Vẽ từng đồ thị riêng cho mỗi bộ dữ liệu
    for net in DATASETS:
        plt.figure(figsize=(6, 5))
        sub_df = df[df["dataset"] == net]
        for par_type, par_name, color, marker in GRANULARITIES:
            g_df = sub_df[sub_df["granularity"] == par_name].sort_values("threads")
            plt.plot(g_df["threads"], g_df["execution_time"],
                     label=par_name, color=color, marker=marker, linewidth=2, markersize=7)

        plt.xlabel("Number of threads", fontsize=12, fontweight="bold")
        plt.ylabel("Execution time (sec)", fontsize=12, fontweight="bold")
        plt.title(DATASET_NAMES[net], fontsize=13, fontweight="bold", pad=10)
        plt.xticks(THREADS_LIST)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=11)
        plt.tight_layout()

        out_png = os.path.join(FIGURES_DIR, f"fig2_{net}.png")
        out_pdf = os.path.join(FIGURES_DIR, f"fig2_{net}.pdf")
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)
        plt.close()
        print(f"Đã lưu biểu đồ riêng: {out_png}")

    # 2. Vẽ biểu đồ tổng hợp 4 subplot (2x2) tương tự như bài báo gốc Fig. 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, net in enumerate(DATASETS):
        ax = axes[idx]
        sub_df = df[df["dataset"] == net]
        for par_type, par_name, color, marker in GRANULARITIES:
            g_df = sub_df[sub_df["granularity"] == par_name].sort_values("threads")
            ax.plot(g_df["threads"], g_df["execution_time"],
                    label=par_name, color=color, marker=marker, linewidth=2, markersize=7)

        ax.set_xlabel("Number of threads", fontsize=11, fontweight="bold")
        ax.set_ylabel("Execution time (sec)", fontsize=11, fontweight="bold")
        ax.set_title(DATASET_NAMES[net], fontsize=12, fontweight="bold")
        ax.set_xticks(THREADS_LIST)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=10)

    fig.suptitle("Fig. 2: Execution time of parallel implementations using three different granularities\n(CI-level, Edge-level, and Sample-level)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    comb_png = os.path.join(FIGURES_DIR, "fig2_granularity_comparison.png")
    comb_pdf = os.path.join(FIGURES_DIR, "fig2_granularity_comparison.pdf")
    plt.savefig(comb_png, dpi=300)
    plt.savefig(comb_pdf)
    plt.close()
    print(f"Đã lưu biểu đồ tổng hợp: {comb_png}")

if __name__ == "__main__":
    main()

