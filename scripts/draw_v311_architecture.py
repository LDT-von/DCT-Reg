"""Draw DCT v3.11 Fixed architecture diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# Color palette
c_wsi = '#5B8FF9'        # blue for WSI path
c_omic = '#F6BD16'       # gold for omics path
c_ot = '#5AD8A6'         # green for OT
c_event = '#E8684A'      # red for event encoder
c_loss = '#9270CA'       # purple for losses
c_audit = '#6DC8EC'      # cyan for audit

def box(x, y, w, h, label, color, fc_alpha=0.3, fontsize=9, weight='bold'):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2, edgecolor=color, facecolor=color, alpha=fc_alpha
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#222')

def arrow(x1, y1, x2, y2, color='#555', style='-', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle=style))

def arrow_label(x, y, text, fontsize=7, color='#666'):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha='center',
            style='italic', bbox=dict(facecolor='white', alpha=0.8,
                                       edgecolor='none', pad=1))

# ===== Title =====
ax.text(8, 10.6, 'DCT v3.11 Fixed Architecture',
        ha='center', fontsize=18, fontweight='bold')
ax.text(8, 10.2, 'OT-Aligned Slot Survival with Diversity Regularization',
        ha='center', fontsize=11, style='italic', color='#555')

# ===== Inputs =====
box(0.2, 9.0, 2.6, 0.7, 'WSI Patches\n[B,512,1024]', c_wsi, 0.4)
box(0.2, 6.0, 2.6, 0.7, 'Omics Tokens\n[B,64,256]', c_omic, 0.4)

# ===== Stage labels (left) =====
ax.text(-0.1, 9.35, 'INPUT', ha='left', va='center', fontsize=10,
        fontweight='bold', color=c_wsi, rotation=90)
ax.text(-0.1, 6.35, 'INPUT', ha='left', va='center', fontsize=10,
        fontweight='bold', color=c_omic, rotation=90)

# ===== M1: Projectors =====
box(3.5, 9.0, 2.4, 0.7, 'M1a WSI\nProjector', c_wsi, 0.25)
box(3.5, 6.0, 2.4, 0.7, 'M1b Omics\nProjector', c_omic, 0.25)
arrow(2.8, 9.35, 3.5, 9.35, c_wsi)
arrow(2.8, 6.35, 3.5, 6.35, c_omic)

# ===== M2: Slot Attention =====
box(6.5, 9.0, 2.4, 0.7, 'M2a WSI Slot\nAttention (8 slots)', c_wsi, 0.25)
box(6.5, 6.0, 2.4, 0.7, 'M2b Omics Slot\nAttention (8 slots)', c_omic, 0.25)
arrow(5.9, 9.35, 6.5, 9.35, c_wsi)
arrow(5.9, 6.35, 6.5, 6.35, c_omic)
arrow_label(6.2, 9.05, '[B,512,256]', fontsize=7)
arrow_label(6.2, 6.05, '[B,64,256]', fontsize=7)
arrow_label(7.7, 9.85, '[B,8,256]', fontsize=7)
arrow_label(7.7, 6.85, '[B,8,256]', fontsize=7)

# ===== M5: Per-Slot Hazard (CRITICAL NEW) =====
box(9.5, 9.0, 2.6, 0.9, 'M5a Per-Slot\nHazard Head\n(WSI)', c_wsi, 0.3)
box(9.5, 6.0, 2.6, 0.9, 'M5b Per-Slot\nHazard Head\n(Omics)', c_omic, 0.3)
arrow(8.9, 9.35, 9.5, 9.35, c_wsi, lw=2)
arrow(8.9, 6.35, 9.5, 6.35, c_omic, lw=2)
arrow_label(8.9, 9.85, 'short-gradient', fontsize=6, color='#c0392b')

# ===== Per-Slot NLL Loss (purple) =====
box(12.7, 7.5, 2.8, 0.9, 'L3 Per-Slot NLL\n(w=0.05)', c_loss, 0.4)
arrow(12.1, 9.4, 12.7, 8.3, c_loss, lw=1.2, style='--')
arrow(12.1, 6.5, 12.7, 7.6, c_loss, lw=1.2, style='--')
ax.text(12.5, 8.85, '← diversity (L4, w=0.02)',
        fontsize=6.5, color=c_loss, style='italic')

# ===== M3: OT Transport (center) =====
box(6.5, 3.0, 2.4, 2.0, '', c_ot, 0.15)
ax.text(7.7, 4.85, 'M3 OT Transport', ha='center', fontsize=11,
        fontweight='bold', color=c_ot)

box(6.6, 4.0, 2.2, 0.4, 'M3a Cost + Marginals', c_ot, 0.25, fontsize=7.5)
box(6.6, 3.55, 2.2, 0.4, 'M3b Sinkhorn (ε=0.05)', c_ot, 0.25, fontsize=7.5)
box(6.6, 3.1, 2.2, 0.4, 'M3c Plan × Value', c_ot, 0.25, fontsize=7.5)

# Cross-modal arrows into OT
arrow(7.7, 5.9, 7.7, 5.05, c_wsi, lw=2)  # from WSI slots down
arrow(7.7, 5.9, 8.2, 5.05, c_omic, lw=2)  # from omics slots down
ax.text(8.3, 5.6, 'cross-modal', fontsize=7, color='#666', style='italic')

# Plan output
arrow(7.7, 3.05, 7.7, 2.5, c_ot, lw=2)
arrow_label(7.7, 2.7, 'plan [B,4,8,8]', fontsize=7, color=c_ot)

# Stage embeddings annotation
ax.text(9.4, 4.0, '+ Stage\nEmbeddings\n[1,4,256]', ha='left', va='center',
        fontsize=7, color=c_ot, style='italic',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor=c_ot, boxstyle='round,pad=0.3'))

# ===== M4: Event Encoder =====
box(10.0, 1.7, 2.6, 0.9, 'M4 Event Encoder\nTransformer (2L, 4H)', c_event, 0.3)
arrow(8.9, 2.3, 10.0, 2.15, c_event, lw=2)

# ===== Hazard + Gate =====
box(13.0, 1.7, 2.5, 0.9, 'Hazard Head\n+ Gate → logits', c_event, 0.3)
arrow(12.6, 2.15, 13.0, 2.15, c_event, lw=2)

# ===== Risk output =====
box(13.0, 0.3, 2.5, 0.7, 'Risk = -Σ cumprob(1-h)\n[B]', '#333', 0.2)
arrow(14.25, 1.7, 14.25, 1.0, '#333', lw=1.5)

# ===== Main NLL + IPCW losses =====
box(10.0, 0.3, 2.6, 0.7, 'L1 NLL (w=1.0)\nL2 IPCW Rank (w=0.10)', c_loss, 0.3)
arrow(13.0, 0.65, 12.6, 0.65, c_loss, lw=1.2, style='--')

# ===== Risk anchor / Audit box =====
box(0.2, 1.5, 2.6, 1.7, '', c_audit, 0.1)
ax.text(1.5, 3.0, 'Risk Anchors\n+ Intervention\nAudit', ha='center', va='center',
        fontsize=10, fontweight='bold', color=c_audit)
ax.text(1.5, 2.2, '(EMA buffer)\nLow/High cost\nshape [2,4,8,8]',
        ha='center', va='center', fontsize=7.5, color=c_audit, style='italic')

# Arrow from anchors to OT
arrow(2.8, 2.5, 6.5, 3.5, c_audit, lw=1.5, style=':')

# Audit arrow going down to losses
box(0.2, 0.3, 2.6, 0.7, 'Audit Score\n(v3.10:0.23 → v3.11:0.61)', c_audit, 0.3)
arrow(1.5, 1.45, 1.5, 1.05, c_audit, lw=1.2, style='--')

# ===== Side legend =====
legend_x = 15.7
ax.text(legend_x, 9.3, 'Module Colors:', fontsize=8, fontweight='bold')
ax.scatter([legend_x], [9.0], color=c_wsi, s=60, alpha=0.5, edgecolors=c_wsi)
ax.text(legend_x + 0.15, 9.0, 'WSI path', fontsize=7, va='center')
ax.scatter([legend_x], [8.7], color=c_omic, s=60, alpha=0.5, edgecolors=c_omic)
ax.text(legend_x + 0.15, 8.7, 'Omics path', fontsize=7, va='center')
ax.scatter([legend_x], [8.4], color=c_ot, s=60, alpha=0.5, edgecolors=c_ot)
ax.text(legend_x + 0.15, 8.4, 'OT module', fontsize=7, va='center')
ax.scatter([legend_x], [8.1], color=c_event, s=60, alpha=0.5, edgecolors=c_event)
ax.text(legend_x + 0.15, 8.1, 'Event head', fontsize=7, va='center')
ax.scatter([legend_x], [7.8], color=c_loss, s=60, alpha=0.5, edgecolors=c_loss)
ax.text(legend_x + 0.15, 7.8, 'Losses', fontsize=7, va='center')
ax.scatter([legend_x], [7.5], color=c_audit, s=60, alpha=0.5, edgecolors=c_audit)
ax.text(legend_x + 0.15, 7.5, 'Audit', fontsize=7, va='center')

# Annotation key
ax.text(legend_x, 6.7, 'Arrows:', fontsize=8, fontweight='bold')
arrow(legend_x - 0.15, 6.4, legend_x + 0.15, 6.4, '#555', lw=1.5)
ax.text(legend_x + 0.3, 6.4, 'forward', fontsize=7, va='center')
arrow(legend_x - 0.15, 6.1, legend_x + 0.15, 6.1, c_loss, lw=1.2, style='--')
ax.text(legend_x + 0.3, 6.1, 'loss', fontsize=7, va='center')
arrow(legend_x - 0.15, 5.8, legend_x + 0.15, 5.8, c_audit, lw=1.2, style=':')
ax.text(legend_x + 0.3, 5.8, 'audit', fontsize=7, va='center')

# Footer notes
ax.text(0.2, 0.05, 'Key innovation: Per-Slot Hazard Head (M5) provides short-gradient supervision,',
        fontsize=7, style='italic', color='#555')
ax.text(0.2, -0.2, 'enabling v3.11 intervention audit to pass (0.61 > 0.5), where v3.10 failed (0.23).',
        fontsize=7, style='italic', color='#555')

plt.tight_layout()
plt.savefig('paper/jpg/dct_v311_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print('saved paper/jpg/dct_v311_architecture.png')
