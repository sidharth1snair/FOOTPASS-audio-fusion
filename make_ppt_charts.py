import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = "ppt_charts"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

# ============================================================
# Chart 1: F1 vs confidence threshold
# ============================================================
thresholds = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
f1_taad =       [0.279, 0.279, 0.280, 0.288, 0.315, 0.368, 0.404, 0.400, 0.282]
f1_taad_audio = [0.387, 0.387, 0.389, 0.396, 0.417, 0.458, 0.472, 0.457, 0.357]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, f1_taad, marker='o', linewidth=2, label='TAAD (video only)', color='#5b8fd4')
ax.plot(thresholds, f1_taad_audio, marker='o', linewidth=2, label='TAAD + audio fusion', color='#e07b39')
ax.axvline(0.70, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.annotate('best threshold (0.70)', xy=(0.70, 0.43), xytext=(0.715, 0.43),
            fontsize=10, color='gray')
ax.set_xlabel('Confidence threshold')
ax.set_ylabel('F1 score')
ax.set_title('F1 vs. confidence threshold: audio fusion improves detection\nacross the whole threshold range')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_f1_vs_threshold.png")
plt.close()

# ============================================================
# Chart 2: Per-class precision comparison (at default threshold 0.15)
# ============================================================
classes = ['drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
prec_taad =       [0.222, 0.370, 0.110, 0.109, 0.087, 0.038, 0.005, 0.026]
prec_taad_audio = [0.312, 0.503, 0.143, 0.108, 0.120, 0.111, 0.007, 0.050]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, prec_taad, width, label='TAAD (video only)', color='#5b8fd4')
ax.bar(x + width/2, prec_taad_audio, width, label='TAAD + audio fusion', color='#e07b39')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel('Precision')
ax.set_title('Per-class precision: audio fusion reduces false alarms\n(header precision nearly triples)')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_per_class_precision.png")
plt.close()

# ============================================================
# Chart 3: Tracking-failure recovery breakdown
# ============================================================
categories = ['No visible bbox\n(tracking failure)', 'Visible bbox\n(normal case)']
missed_pct = [98.3, 48.2]
recovered_pct = [0.4, 20.3]  # % of the MISSED ones that got recovered

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, missed_pct, width, label='Missed by video-only TAAD', color='#c0392b')
ax.bar(x + width/2, recovered_pct, width, label='Recovered by audio fusion', color='#27ae60')

for i, (m, r) in enumerate(zip(missed_pct, recovered_pct)):
    ax.text(i - width/2, m + 1.5, f'{m}%', ha='center', fontsize=11)
    ax.text(i + width/2, r + 1.5, f'{r}%', ha='center', fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Percent')
ax.set_ylim(0, 110)
ax.set_title('Audio fusion does NOT recover tracking-failure misses,\nbut does help ordinary visible-player misses')
ax.legend(loc='upper center')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_tracking_failure_recovery.png")
plt.close()

# ============================================================
# Chart 4: Headline F1 comparison (simple bar, good for title/summary slide)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
models = ['TAAD\n(video only)', 'TAAD +\naudio fusion']
f1_vals = [0.404, 0.472]
colors = ['#5b8fd4', '#e07b39']
bars = ax.bar(models, f1_vals, color=colors, width=0.5)
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 score (at optimal threshold)')
ax.set_title('~17% relative F1 improvement\nfrom audio fusion')
ax.set_ylim(0, 0.55)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_headline_f1_comparison.png")
plt.close()

print("Saved 4 charts to", OUT_DIR)
