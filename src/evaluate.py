from utils import *

SAVE_DIR = './results/predictions'

methods_map = [
    ("Early Fusion (LFDNN)",      f"{SAVE_DIR}/early_fusion_predictions.npz"),
    ("Early Fusion + CoT",        f"{SAVE_DIR}/early_fusion_cot_predictions.npz"),
    ("Late Fusion (LF-DNN)",      f"{SAVE_DIR}/late_fusion_predictions.npz"),
    ("Late Fusion + CoT",         f"{SAVE_DIR}/pear_cot_cmff_optimized_predictions.npz"),
    ("Hybrid CENET",              f"{SAVE_DIR}/cenet_predictions.npz"),
    ("Hybrid CENET + CoT (Ours)", f"{SAVE_DIR}/cenet_cot_predictions.npz")
]

summary_rows = []
for name, path in methods_map:
    npz = np.load(path)
    metrics = compute_metrics(npz["y_true"], npz["y_pred"])
    summary_rows.append({"Model": name, **metrics})

df_results = pd.DataFrame(summary_rows)[["Model", "Has0_Acc2", "Has0_F1", "Non0_Acc2", "Non0_F1", "Corr"]]

print(
    df_results.style
    .format({"Has0_Acc2": "{:.2f}%", "Has0_F1": "{:.2f}%", "Non0_Acc2": "{:.2f}%", "Non0_F1": "{:.2f}%", "Corr": "{:.4f}"})
    .hide(axis="index")
    .set_properties(**{"text-align": "center", "padding": "10px", "font-size": "13px", "border": "1px solid #D3D3D3"})
    .set_properties(subset=["Model"], **{"text-align": "left", "font-weight": "bold"})
    .set_table_styles([
        {"selector": "th", "props": [("background-color", "#1F4E78"), ("color", "white"), ("text-align", "center"), ("font-weight", "bold"), ("padding", "10px")]},
        {"selector": "tr:last-child", "props": [("background-color", "#E8F5E9"), ("font-weight", "bold")]}
    ])
)

df_results.to_csv("./results/paper_5_metrics_results.csv", index=False, encoding="utf-8-sig")
df_results.to_excel("./results/paper_5_metrics_results.xlsx", index=False)
print("Saved results table to '/content/paper_5_metrics_results.csv' and '.xlsx'\n")

# Reusable plotting function for Acc-2 and F1-Score
def plot_metric_comparison(base_scores, cot_scores, ylabel, y_lim, out_prefix):
    categories = [
        "Early Fusion\n(Has0)", "Early Fusion\n(Non0)",
        "Late Fusion\n(Has0)",  "Late Fusion\n(Non0)",
        "Hybrid CENET\n(Has0)", "Hybrid CENET\n(Non0)"
    ]
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=300)
    x = np.arange(len(categories))
    width = 0.36

    r1 = ax.bar(x - width/2, base_scores, width, label="Baseline", color="#6BAED6", edgecolor="#2B5C8F", linewidth=0.9, zorder=3)
    r2 = ax.bar(x + width/2, cot_scores,  width, label="+ CoT (Ours)", color="#66C2A5", edgecolor="#1B6B50", linewidth=0.9, zorder=3)

    ax.set_ylabel(ylabel, fontsize=11.5, fontweight="bold", color="#111111")
    ax.set_ylim(*y_lim)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="bold", color="#222222")
    ax.tick_params(axis="y", labelsize=10, colors="#333333")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for rects, color in [(r1, "#2B5C8F"), (r2, "#145A32")]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color)

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(f"./results/{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"./results/{out_prefix}.pdf", bbox_inches="tight")
    plt.show()

acc_base = [df_results.loc[0, "Has0_Acc2"], df_results.loc[0, "Non0_Acc2"], df_results.loc[2, "Has0_Acc2"], df_results.loc[2, "Non0_Acc2"], df_results.loc[4, "Has0_Acc2"], df_results.loc[4, "Non0_Acc2"]]
acc_cot  = [df_results.loc[1, "Has0_Acc2"], df_results.loc[1, "Non0_Acc2"], df_results.loc[3, "Has0_Acc2"], df_results.loc[3, "Non0_Acc2"], df_results.loc[5, "Has0_Acc2"], df_results.loc[5, "Non0_Acc2"]]

f1_base  = [df_results.loc[0, "Has0_F1"],   df_results.loc[0, "Non0_F1"],   df_results.loc[2, "Has0_F1"],   df_results.loc[2, "Non0_F1"],   df_results.loc[4, "Has0_F1"],   df_results.loc[4, "Non0_F1"]]
f1_cot   = [df_results.loc[1, "Has0_F1"],   df_results.loc[1, "Non0_F1"],   df_results.loc[3, "Has0_F1"],   df_results.loc[3, "Non0_F1"],   df_results.loc[5, "Has0_F1"],   df_results.loc[5, "Non0_F1"]]

plot_metric_comparison(acc_base, acc_cot, "Accuracy (Acc-2, %)", (72.0, 91.5), "acc2_comparison_paper")
plot_metric_comparison(f1_base,  f1_cot,  "F1-Score (%)",        (70.0, 90.0), "f1_comparison_paper")
