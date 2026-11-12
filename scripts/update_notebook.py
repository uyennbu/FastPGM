import json
import os

NOTEBOOK_PATH = "notebooks/reproduce_v1.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 0
nb["cells"][0]["source"] = [
    "# Tái hiện Thực nghiệm Bài báo Fast-BNS (IPDPS 2022) — Fig. 2, Fig. 3 & Fig. 4\n",
    "\n",
    "Notebook này tự động chạy thực nghiệm học cấu trúc mạng Bayes (**PC-Stable / Fast-BNS**) trên 4 mạng chuẩn (`Alarm`, `Insurance`, `Hepar2`, `Munin1`) và vẽ lại các hình thực nghiệm chính trong bài báo [SOURCE_PAPER.md](../docs/papers/SOURCE_PAPER.md):\n",
    "- **Fig. 2**: So sánh 3 mức độ song song hóa (**CI-level**, **Edge-level**, **Sample-level**) theo số lượng luồng (1, 2, 4, 8, 12).\n",
    "- **Fig. 3**: Đánh giá độ tăng tốc (Speedup) của Fast-BNS theo số luồng với các kích thước mẫu 5000, 10000, 15000.\n",
    "- **Fig. 4**: Ảnh hưởng của kích thước nhóm kiểm định (`Group size` - $gs$) đến thời gian thực thi và số phép kiểm định CI phát sinh thêm.\n",
    "\n",
    "> **Môi trường yêu cầu**: Chạy trong môi trường conda `islp` (đã cài đặt `matplotlib`, `pandas`, `jupyter`, `numpy`)."
]

# Update Cell 5: run_fastpgm
nb["cells"][5]["source"] = [
    "def run_fastpgm(net, sample_size, threads, group_size=1, parallelism_type=0):\n",
    "    \"\"\"Chạy ./fastpgm và trả về (thời gian pc-stable (giây), tổng số CI tests).\n",
    "    parallelism_type: 0 = CI-level (mặc định), 1 = Edge-level, 2 = Sample-level\n",
    "    \"\"\"\n",
    "    cmd = [\n",
    "        FASTPGM_BIN,\n",
    "        \"-j\", \"0\", \"-m\", \"0\",\n",
    "        \"-t\", str(threads),\n",
    "        \"-p\", str(parallelism_type),\n",
    "        \"-g\", str(group_size),\n",
    "        \"-a\", \"0.05\",\n",
    "        \"-ss\", \"0\",\n",
    "        \"-f2\", f\"{net}/{net}_s{sample_size}\"\n",
    "    ]\n",
    "    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)\n",
    "    time_m = re.search(r\"pc-stable:\\s*([0-9.eE+-]+)\\s*s\", res.stdout)\n",
    "    ci_m   = re.search(r\"# of CI-tests =\\s*(\\d+)\", res.stdout)\n",
    "    return float(time_m.group(1)), int(ci_m.group(1))\n"
]

# Update Cell 12, 13, 14
nb["cells"][12]["source"] = [
    "## 5. Thực nghiệm & Vẽ **Fig. 2**: So sánh 3 Mức độ Song song hóa (CI-level, Edge-level, Sample-level)\n",
    "\n",
    "Theo Mục IV-A & V-C trong bài báo:\n",
    "1. **CI-level parallelism (Fast-BNS)**: Tận dụng cơ chế *dynamic work pool* phân phối các nhóm CI tests giữa các luồng, cân bằng tải động và loại bỏ hoàn toàn các phép toán `atomic` trên bảng tần số (contingency table).\n",
    "2. **Edge-level parallelism**: Phân phối song song theo từng cạnh ở mỗi độ sâu (*depth*). Hạn chế là các cạnh có số lượng CI test rất chênh lệch, dẫn đến mất cân bằng tải giữa các luồng.\n",
    "3. **Sample-level parallelism**: Song song hóa việc duyệt qua các mẫu bên trong từng CI test. Hạn chế là chi phí khởi tạo luồng liên tục cho mỗi CI test và xung đột ghi trên bảng tần số đòi hỏi nhiều phép toán `#pragma omp atomic`, khiến thời gian thực thi tăng vọt."
]

nb["cells"][14]["source"] = [
    "# Vẽ đồ thị Fig. 2:\n",
    "# 1. Vẽ 4 biểu đồ riêng biệt (mỗi bộ dữ liệu một hình)\n",
    "for net in NETWORKS:\n",
    "    plt.figure(figsize=(6, 5))\n",
    "    sub_df = df_fig2[df_fig2[\"dataset\"] == net]\n",
    "    for par_type, par_name, color, marker in GRANULARITIES:\n",
    "        g_df = sub_df[sub_df[\"granularity\"] == par_name].sort_values(\"threads\")\n",
    "        plt.plot(g_df[\"threads\"], g_df[\"execution_time\"],\n",
    "                 label=par_name, color=color, marker=marker, linewidth=2, markersize=7)\n",
    "    plt.xlabel(\"Number of threads\", fontsize=12, fontweight=\"bold\")\n",
    "    plt.ylabel(\"Execution time (sec)\", fontsize=12, fontweight=\"bold\")\n",
    "    plt.title(NETWORK_NAMES[net], fontsize=13, fontweight=\"bold\", pad=10)\n",
    "    plt.xticks(THREADS_LIST)\n",
    "    plt.grid(True, linestyle=\"--\", alpha=0.5)\n",
    "    plt.legend(frameon=True, fontsize=11)\n",
    "    plt.tight_layout()\n",
    "    \n",
    "    out_png = os.path.join(FIGURES_DIR, f\"fig2_{net}.png\")\n",
    "    plt.savefig(out_png, dpi=300)\n",
    "    plt.show()\n",
    "    print(f\"Đã lưu: {out_png}\")\n",
    "\n",
    "# 2. Vẽ biểu đồ tổng hợp 4 subplot (2x2) theo phong cách bài báo\n",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
    "axes = axes.flatten()\n",
    "for idx, net in enumerate(NETWORKS):\n",
    "    ax = axes[idx]\n",
    "    sub_df = df_fig2[df_fig2[\"dataset\"] == net]\n",
    "    for par_type, par_name, color, marker in GRANULARITIES:\n",
    "        g_df = sub_df[sub_df[\"granularity\"] == par_name].sort_values(\"threads\")\n",
    "        ax.plot(g_df[\"threads\"], g_df[\"execution_time\"],\n",
    "                label=par_name, color=color, marker=marker, linewidth=2, markersize=7)\n",
    "    ax.set_xlabel(\"Number of threads\", fontsize=11, fontweight=\"bold\")\n",
    "    ax.set_ylabel(\"Execution time (sec)\", fontsize=11, fontweight=\"bold\")\n",
    "    ax.set_title(NETWORK_NAMES[net], fontsize=12, fontweight=\"bold\")\n",
    "    ax.set_xticks(THREADS_LIST)\n",
    "    ax.grid(True, linestyle=\"--\", alpha=0.5)\n",
    "    ax.legend(frameon=True, fontsize=10)\n",
    "\n",
    "fig.suptitle(\"Fig. 2: Execution time of parallel implementations using three different granularities\\n(CI-level, Edge-level, and Sample-level)\",\n",
    "             fontsize=14, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "comb_png = os.path.join(FIGURES_DIR, \"fig2_granularity_comparison.png\")\n",
    "plt.savefig(comb_png, dpi=300)\n",
    "plt.show()\n",
    "print(f\"Đã lưu biểu đồ tổng hợp: {comb_png}\")\n"
]

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated notebook cleanly!")
