import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = "ppt_charts"
os.makedirs(OUT_DIR, exist_ok=True)

classes = ['drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block', 'OVERALL']

# All at confidence threshold 0.15 (matches the paper's own reporting protocol)
def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

def with_f1(pr_dict):
    return {c: (p, r, f1(p, r)) for c, (p, r) in pr_dict.items()}

paper_taad = {  # from FOOTPASS paper, Table 2 (F1 as directly reported)
    'drive':    (23.9, 60.0, 34.2), 'pass':     (43.8, 61.0, 51.0),
    'cross':    (16.6, 66.4, 26.6), 'throw-in': (23.8, 44.4, 31.0),
    'shot':     (12.3, 74.6, 21.1), 'header':   (10.6, 54.1, 17.7),
    'tackle':   (1.5, 16.7, 2.8),   'block':    (3.2, 47.1, 6.0),
    'OVERALL':  (25.6, 59.9, 35.9),
}
our_taad = with_f1({  # model_20, threshold 0.15 (P, R only -- F1 computed below)
    'drive':    (22.2, 61.0), 'pass':     (37.0, 66.0),
    'cross':    (11.0, 61.3), 'throw-in': (10.9, 43.3),
    'shot':     (8.7, 67.2),  'header':   (3.8, 63.6),
    'tackle':   (0.5, 61.5),  'block':    (2.6, 61.5),
    'OVERALL':  (17.9, 63.4),
})
our_taad_audio = with_f1({  # model_40, threshold 0.15 (P, R only -- F1 computed below)
    'drive':    (31.2, 60.8), 'pass':     (50.3, 65.8),
    'cross':    (14.3, 58.6), 'throw-in': (10.8, 46.4),
    'shot':     (12.0, 67.2), 'header':   (11.1, 56.2),
    'tackle':   (0.7, 50.0),  'block':    (5.0, 50.0),
    'OVERALL':  (28.0, 62.8),
})

col_labels = ['Class',
              "Paper's TAAD\nP / R / F1", "Our TAAD\nP / R / F1", "Our TAAD+Audio\nP / R / F1"]

cell_text = []
for c in classes:
    p1 = paper_taad[c]; p2 = our_taad[c]; p3 = our_taad_audio[c]
    cell_text.append([
        c,
        f"{p1[0]:.1f} / {p1[1]:.1f} / {p1[2]:.1f}",
        f"{p2[0]:.1f} / {p2[1]:.1f} / {p2[2]:.1f}",
        f"{p3[0]:.1f} / {p3[1]:.1f} / {p3[2]:.1f}",
    ])

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis('off')

table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# Style header row
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold')

# Style OVERALL row (last row, index = len(classes))
overall_row = len(classes)
for j in range(len(col_labels)):
    cell = table[overall_row, j]
    cell.set_facecolor('#dcdde1')
    cell.set_text_props(fontweight='bold')

# Zebra stripe the rest
for i in range(1, overall_row):
    color = '#f5f6fa' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(color)

for j in range(len(col_labels)):
    table[0, j].set_width(0.28 if j > 0 else 0.16)

ax.set_title('Per-class Precision / Recall / F1 (x100), confidence threshold = 0.15, δ = 12 frames',
              fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/6_full_results_table.png", dpi=160, bbox_inches='tight')
plt.close()
print("Saved results table")
