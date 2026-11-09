import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = "ppt_charts"
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9)
ax.axis('off')

VIDEO_COLOR = '#5b8fd4'
AUDIO_COLOR = '#e07b39'
FUSION_COLOR = '#6b4fa0'
TEXT_COLOR = '#1a1a1a'

def box(x, y, w, h, text, color, fontsize=11, text_color='white'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                           linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)
    return (x + w/2, y)  # bottom-center anchor

def arrow(x1, y1, x2, y2, color='#555555'):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18,
                           linewidth=1.8, color=color)
    ax.add_patch(arr)

# ---- Title ----
ax.text(6.5, 8.6, 'TAAD + Audio Fusion Architecture', ha='center', fontsize=17, fontweight='bold', color=TEXT_COLOR)

# ============================================================
# VIDEO BRANCH (left side)
# ============================================================
ax.text(3.0, 7.9, 'Video Branch (existing FOOTPASS baseline)', ha='center', fontsize=12,
        fontweight='bold', color=VIDEO_COLOR)

v1x, v1y = box(1.3, 7.0, 3.4, 0.65, 'Video clip  (B, 3, T, 352, 640)', VIDEO_COLOR, fontsize=10)
v2x, v2y = box(1.3, 6.0, 3.4, 0.65, 'X3D backbone + feature\npyramid upsampling', VIDEO_COLOR, fontsize=10)
v3x, v3y = box(1.3, 5.0, 3.4, 0.65, 'ROI-Align per player\n(from tracking bboxes)', VIDEO_COLOR, fontsize=10)
v4x, v4y = box(1.3, 4.0, 3.4, 0.65, 'Conv1D + FC classifier head', VIDEO_COLOR, fontsize=10)
v5x, v5y = box(1.3, 3.0, 3.4, 0.65, 'Video logits\n(B, 9, M, T)', VIDEO_COLOR, fontsize=10)

arrow(v1x, v1y, v2x, v2y + 0.65)
arrow(v2x, v2y, v3x, v3y + 0.65)
arrow(v3x, v3y, v4x, v4y + 0.65)
arrow(v4x, v4y, v5x, v5y + 0.65)

# ============================================================
# AUDIO BRANCH (right side)
# ============================================================
ax.text(10.0, 7.9, 'Audio Branch (this work)', ha='center', fontsize=12,
        fontweight='bold', color=AUDIO_COLOR)

a1x, a1y = box(8.3, 7.0, 3.4, 0.65, 'Audio energy  (B, 2, T+2·margin)\ncrowd + commentary', AUDIO_COLOR, fontsize=10)
a2x, a2y = box(8.3, 6.0, 3.4, 0.65, 'AudioEncoder\n(1D temporal CNN)', AUDIO_COLOR, fontsize=10)
a3x, a3y = box(8.3, 5.0, 3.4, 0.65, 'EventConditionedAudioHead\nlearned per-class offset τ', AUDIO_COLOR, fontsize=10)
a4x, a4y = box(8.3, 4.0, 3.4, 0.65, 'sample audio at\n(event_time + τ_class)\nvia differentiable grid_sample', AUDIO_COLOR, fontsize=9.5)
a5x, a5y = box(8.3, 3.0, 3.4, 0.65, 'Audio correction logits\n(B, 9, T)  — class 0 fixed at 0', AUDIO_COLOR, fontsize=10)

arrow(a1x, a1y, a2x, a2y + 0.65)
arrow(a2x, a2y, a3x, a3y + 0.65)
arrow(a3x, a3y, a4x, a4y + 0.65)
arrow(a4x, a4y, a5x, a5y + 0.65)

# ============================================================
# FUSION (bottom center)
# ============================================================
f1x, f1y = box(3.9, 1.7, 5.2, 0.7,
               'pred = video_logits + audio_correction.unsqueeze(2)\n(broadcast audio correction across all M players)',
               FUSION_COLOR, fontsize=10.5)

arrow(v5x, v5y, f1x - 1.0, f1y + 0.7, color=VIDEO_COLOR)
arrow(a5x, a5y, f1x + 1.0, f1y + 0.7, color=AUDIO_COLOR)

f2x, f2y = box(4.65, 0.5, 3.7, 0.65, 'Final prediction  (B, 9, M, T)', '#2c3e50', fontsize=10.5)
arrow(f1x, f1y, f2x, f2y + 0.65)

# ---- Side note on audio_mask ----
ax.annotate('audio_mask = 0 (no audio for this game/sample)\n→ correction zeroed, falls back to pure video',
            xy=(f1x + 1.0, f1y + 0.35), xytext=(10.3, 2.0),
            fontsize=8.5, color='#777777', ha='left',
            arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_architecture_diagram.png", dpi=160, bbox_inches='tight')
plt.close()
print("Saved architecture diagram")
