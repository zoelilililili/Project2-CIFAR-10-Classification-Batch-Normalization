"""
Draw MyCIFARNet architecture diagram using matplotlib.
Saves to reports/figures/architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ========== Output ==========
module_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(module_dir), 'reports', 'figures')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'architecture.png')

# ========== Block definitions ==========
# (label, color, width, height)
blocks = [
    ('Input\n3×32×32',        '#E8E8E8', 1.8),
    ('Conv 3×3, 3→base_ch\nBN, ReLU', '#A8D8EA', 2.8),
    ('Stage 1: 2× ResBlock\nbase_ch→base_ch\n32×32', '#AAEAA8', 3.0),
    ('Stage 2: 2× ResBlock\nbase_ch→2×base_ch\nstride=2, 16×16', '#AAEAA8', 3.0),
    ('Stage 3: 2× ResBlock\n2×base_ch→4×base_ch\nstride=2, 8×8', '#AAEAA8', 3.0),
    ('Stage 4: 2× ResBlock\n4×base_ch→8×base_ch\nstride=2, 4×4', '#AAEAA8', 3.0),
    ('AdaptiveAvgPool2d(1)\n→ Flatten', '#FADA5E', 2.8),
    ('FC 8×base_ch→256\nReLU, Dropout', '#FFB347', 2.8),
    ('FC 256→10', '#FF6B6B', 1.8),
]

# ResBlock detail boxes
res_block_detail = [
    ('Conv 3×3\nBN, ReLU',  '#C8E8C8', 2.0),
    ('Conv 3×3\nBN',         '#C8E8C8', 1.5),
    ('+ skip\nReLU',         '#D8F8D8', 1.5),
]

# ========== Draw ==========
fig = plt.figure(figsize=(16, 10))

# --- Main diagram (left) ---
ax_main = fig.add_axes([0.02, 0.05, 0.68, 0.90])
ax_main.set_xlim(0, 10)
ax_main.set_ylim(0, len(blocks) + 1)
ax_main.axis('off')
ax_main.set_title('MyCIFARNet Architecture (L: base_ch=64, ~11.3M params)',
                  fontsize=14, fontweight='bold', pad=10)

y_positions = []
for i, (label, color, w) in enumerate(blocks[::-1]):
    y = i + 0.5
    y_positions.append(y)
    h = 0.7
    x = (10 - w) / 2
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.15',
                                    facecolor=color, edgecolor='#333', linewidth=1.2)
    ax_main.add_patch(rect)
    ax_main.text(5, y + h/2, label, ha='center', va='center', fontsize=8,
                 fontfamily='monospace')

# Arrows
for i in range(len(blocks) - 1):
    y_from = y_positions[i] + 0.7
    y_to = y_positions[i + 1]
    ax_main.annotate('', xy=(5, y_to), xytext=(5, y_from),
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='#555'))

# S/M/L legend
legend_patches = [
    mpatches.Patch(color='#E8E8E8', label='Input'),
    mpatches.Patch(color='#A8D8EA', label='Conv + BN + ReLU'),
    mpatches.Patch(color='#AAEAA8', label='ResBlock Stage'),
    mpatches.Patch(color='#FADA5E', label='Pooling'),
    mpatches.Patch(color='#FFB347', label='FC + ReLU + Dropout'),
    mpatches.Patch(color='#FF6B6B', label='Output'),
]
ax_main.legend(handles=legend_patches, loc='lower right', fontsize=7)

# --- ResBlock detail (right top) ---
ax_rb = fig.add_axes([0.73, 0.55, 0.25, 0.38])
ax_rb.set_xlim(0, 10)
ax_rb.set_ylim(0, 5)
ax_rb.axis('off')
ax_rb.set_title('ResBlock Detail', fontsize=11, fontweight='bold')

rb_y = []
for i, (label, color, w) in enumerate(res_block_detail[::-1]):
    y = i + 0.6
    rb_y.append(y)
    h = 0.65
    x = (10 - w) / 2
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.1',
                                    facecolor=color, edgecolor='#333', linewidth=1)
    ax_rb.add_patch(rect)
    ax_rb.text(5, y + h/2, label, ha='center', va='center', fontsize=7,
               fontfamily='monospace')

# Arrows in ResBlock
for i in range(len(res_block_detail) - 1):
    y_from = rb_y[i] + 0.65
    y_to = rb_y[i + 1]
    ax_rb.annotate('', xy=(5, y_to), xytext=(5, y_from),
                   arrowprops=dict(arrowstyle='->', lw=1, color='#555'))

# Skip connection arc
ax_rb.annotate('', xy=(6.2, rb_y[0] + 0.65), xytext=(6.2, rb_y[2] + 0.3),
               arrowprops=dict(arrowstyle='->', lw=1, color='#E74C3C',
                               connectionstyle='arc3,rad=0.4'))
ax_rb.text(7.5, rb_y[1] + 0.3, 'skip\n(1×1 conv\nif dims change)',
           fontsize=6, color='#E74C3C', ha='center')

# --- S/M/L variant table (right bottom) ---
ax_var = fig.add_axes([0.73, 0.05, 0.25, 0.42])
ax_var.axis('off')
ax_var.set_title('Variants', fontsize=11, fontweight='bold')

table_data = [['S', '32', '2.9M'],
              ['M', '48', '6.5M'],
              ['L', '64', '11.3M']]
col_labels = ['Name', 'base_ch', 'Params']
table = ax_var.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center',
                     colWidths=[0.08, 0.1, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# ========== Save ==========
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved to {output_path}')
