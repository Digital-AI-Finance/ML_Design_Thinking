"""
Create 4 new mathematical charts for Week 7 rewrite
- fairness_roc_space.pdf
- impossibility_proof.pdf
- optimization_tradeoff.pdf
- information_theory_bias.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom colors matching Week 7 Nature Professional theme
ForestGreen = '#145335'
Teal = '#0D9488'
Amber = '#F59E0B'
mlred = '#D62728'
mlgreen = '#2CA02C'
mlblue = '#0066CC'

# ============================================================================
# Chart 1: fairness_roc_space.pdf
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Group A ROC curve (better performance)
fpr_a = np.array([0.00, 0.02, 0.05, 0.08, 0.12, 0.18, 0.30, 0.50, 1.00])
tpr_a = np.array([0.50, 0.70, 0.82, 0.90, 0.94, 0.97, 0.99, 1.00, 1.00])

# Group B ROC curve (worse performance - fairness gap)
fpr_b = np.array([0.00, 0.04, 0.10, 0.14, 0.20, 0.28, 0.42, 0.60, 1.00])
tpr_b = np.array([0.45, 0.65, 0.78, 0.86, 0.91, 0.95, 0.98, 1.00, 1.00])

# Plot ROC curves
ax.plot(fpr_a, tpr_a, 'o-', linewidth=3, markersize=8,
        color=Teal, label='Group A (Advantaged)', alpha=0.8)
ax.plot(fpr_b, tpr_b, 's-', linewidth=3, markersize=8,
        color=Amber, label='Group B (Disadvantaged)', alpha=0.8)

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2,
        alpha=0.5, label='Random Classifier')

# Highlight specific operating points
point_a_idx = 3  # 90% TPR, 8% FPR
point_b_idx = 3  # 86% TPR, 14% FPR

ax.plot(fpr_a[point_a_idx], tpr_a[point_a_idx], 'o',
        markersize=15, color=Teal, markeredgewidth=3,
        markeredgecolor='black', zorder=5)
ax.plot(fpr_b[point_b_idx], tpr_b[point_b_idx], 's',
        markersize=15, color=Amber, markeredgewidth=3,
        markeredgecolor='black', zorder=5)

# Draw distance arrow
ax.annotate('', xy=(fpr_b[point_b_idx], tpr_b[point_b_idx]),
            xytext=(fpr_a[point_a_idx], tpr_a[point_a_idx]),
            arrowprops=dict(arrowstyle='<->', lw=2.5, color=mlred))

# Distance calculation text
mid_x = (fpr_a[point_a_idx] + fpr_b[point_b_idx]) / 2
mid_y = (tpr_a[point_a_idx] + tpr_b[point_b_idx]) / 2
ax.text(mid_x + 0.03, mid_y - 0.03,
        r'd = 7.2%' + '\n' + r'Fairness Gap',
        fontsize=14, fontweight='bold', color=mlred,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor=mlred, linewidth=2))

# Annotate points
ax.text(fpr_a[point_a_idx] - 0.06, tpr_a[point_a_idx] + 0.03,
        'Group A\n(90%, 8%)',
        fontsize=11, ha='right', color=Teal, fontweight='bold')
ax.text(fpr_b[point_b_idx] + 0.05, tpr_b[point_b_idx] - 0.03,
        'Group B\n(86%, 14%)',
        fontsize=11, ha='left', color=Amber, fontweight='bold')

# Perfect classifier point
ax.plot(0, 1, '*', markersize=20, color=mlgreen,
        markeredgewidth=2, markeredgecolor='black', zorder=5)
ax.text(0.02, 0.98, 'Perfect\n(100%, 0%)',
        fontsize=11, color=mlgreen, fontweight='bold')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
ax.set_title('ROC Space: Visualizing Fairness as Geometric Distance',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

# Add interpretation box
textstr = 'Fairness Interpretation:\n• Curves close together = Fair\n• Large distance = Bias\n• Distance quantifies unfairness'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.15, linewidth=2)
ax.text(0.55, 0.25, textstr, fontsize=11, verticalalignment='top',
        bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('../charts/fairness_roc_space.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/fairness_roc_space.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created fairness_roc_space.pdf")

# ============================================================================
# Chart 2: impossibility_proof.pdf
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# ROC space
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, alpha=0.5)

# Draw constraint regions
# Constraint 1: Calibration (diagonal-ish line)
calibration_x = np.linspace(0, 1, 100)
calibration_y = 0.85 + 0.15 * calibration_x
ax.plot(calibration_x, calibration_y, linewidth=4, color=mlblue,
        label='Constraint 1: Calibration', alpha=0.7)
ax.fill_between(calibration_x, calibration_y - 0.03, calibration_y + 0.03,
                 alpha=0.2, color=mlblue)

# Constraint 2: Demographic Parity (vertical - same FPR)
dp_x = 0.11
ax.axvline(dp_x, linewidth=4, color=mlgreen,
           label='Constraint 2: Demographic Parity (same FPR)', alpha=0.7)
ax.axvspan(dp_x - 0.015, dp_x + 0.015, alpha=0.2, color=mlgreen)

# Constraint 3: Equal Opportunity (horizontal - same TPR)
eo_y = 0.88
ax.axhline(eo_y, linewidth=4, color=Amber,
           label='Constraint 3: Equal Opportunity (same TPR)', alpha=0.7)
ax.axhspan(eo_y - 0.015, eo_y + 0.015, alpha=0.2, color=Amber)

# Show the impossible intersection point
impossible_x = dp_x
impossible_y = eo_y
ax.plot(impossible_x, impossible_y, 'X', markersize=25,
        color=mlred, markeredgewidth=3, markeredgecolor='black',
        zorder=5, label='Impossible Point')

# But calibration requires different point
calibrated_y = 0.85 + 0.15 * impossible_x
ax.plot(impossible_x, calibrated_y, 'o', markersize=20,
        color=mlblue, markeredgewidth=3, markeredgecolor='black',
        zorder=5)

# Draw conflict arrows
ax.annotate('', xy=(impossible_x, calibrated_y),
            xytext=(impossible_x, impossible_y),
            arrowprops=dict(arrowstyle='<->', lw=3, color=mlred,
                          linestyle='dashed'))
ax.text(impossible_x + 0.08, (calibrated_y + impossible_y) / 2,
        'CONFLICT!\n' + r'$\Delta$ = ' + f'{abs(calibrated_y - impossible_y):.2f}',
        fontsize=13, fontweight='bold', color=mlred,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                  edgecolor=mlred, linewidth=3))

# Mathematical proof text
proof_text = (
    'Chouldechova Impossibility Theorem:\n\n'
    '3 Constraints (Calibration, DP, EO)\n'
    '2 Degrees of Freedom (x, y in ROC space)\n'
    '━━━━━━━━━━━━━━━━━━━━━━\n'
    '3 equations, 2 unknowns\n'
    '= Overdetermined system\n'
    '= NO SOLUTION when base rates differ'
)
props = dict(boxstyle='round,pad=1', facecolor='lightyellow',
             alpha=0.9, linewidth=3, edgecolor=mlred)
ax.text(0.58, 0.45, proof_text, fontsize=12, verticalalignment='top',
        bbox=props, fontfamily='monospace', fontweight='bold')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
ax.set_title('Impossibility Theorem: Why You Cannot Satisfy All Fairness Metrics',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.5, 1.02)

plt.tight_layout()
plt.savefig('../charts/impossibility_proof.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/impossibility_proof.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created impossibility_proof.pdf")

# ============================================================================
# Chart 3: optimization_tradeoff.pdf
# ============================================================================

fig, ax = plt.subplots(figsize=(11, 8))

# Generate Pareto frontier (accuracy vs fairness violation)
fairness_violation = np.linspace(0, 0.35, 50)
# Accuracy decreases as we enforce stricter fairness
accuracy = 0.73 + 0.12 * np.exp(-5 * fairness_violation**0.8)

# Plot Pareto frontier
ax.plot(fairness_violation * 100, accuracy * 100, linewidth=4,
        color=ForestGreen, label='Pareto Frontier\n(Optimal Trade-offs)',
        alpha=0.8, zorder=3)
ax.fill_between(fairness_violation * 100, 70, accuracy * 100,
                 alpha=0.15, color=ForestGreen)

# Mark specific points with different lambda values
lambda_points = [
    (0.30, 0.85, 0.0, 'Unconstrained\n(λ=0)'),
    (0.048, 0.823, 0.3, 'Optimal\n(λ=0.3)'),
    (0.01, 0.78, 1.0, 'Max Fairness\n(λ=1.0)'),
]

colors_lambda = [mlred, mlgreen, mlblue]
for i, (fv, acc, lam, label) in enumerate(lambda_points):
    ax.plot(fv * 100, acc * 100, 'o', markersize=18,
            color=colors_lambda[i], markeredgewidth=3,
            markeredgecolor='black', zorder=5)

    # Annotate
    if i == 0:  # Unconstrained
        offset_x, offset_y = 3, -2
    elif i == 1:  # Optimal
        offset_x, offset_y = -2, 3
    else:  # Max fairness
        offset_x, offset_y = -2, -3

    ax.annotate(label + f'\n({fv*100:.1f}%, {acc*100:.1f}%)',
                xy=(fv * 100, acc * 100),
                xytext=(fv * 100 + offset_x, acc * 100 + offset_y),
                fontsize=11, fontweight='bold', color=colors_lambda[i],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor=colors_lambda[i], linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2,
                              color=colors_lambda[i]))

# Add infeasible region
infeasible_x = fairness_violation * 100
infeasible_y = accuracy * 100 + 5
ax.fill_between(infeasible_x, infeasible_y, 100,
                 alpha=0.15, color='gray', label='Infeasible Region')
ax.text(12, 91, 'INFEASIBLE\n(Cannot achieve)',
        fontsize=12, fontweight='bold', color='gray',
        ha='center', style='italic')

# Add suboptimal region
ax.fill_between(fairness_violation * 100, 65, accuracy * 100,
                 alpha=0.1, color='red')
ax.text(25, 72, 'Suboptimal\n(Dominated solutions)',
        fontsize=11, fontweight='bold', color=mlred,
        ha='center', alpha=0.7, style='italic')

# Trade-off annotation
ax.annotate('', xy=(0.048 * 100, 0.823 * 100),
            xytext=(0.30 * 100, 0.85 * 100),
            arrowprops=dict(arrowstyle='<->', lw=3, color='purple'))
ax.text(10, 83.5,
        'Trade-off:\n-2.7% accuracy\nfor -84% bias',
        fontsize=12, fontweight='bold', color='purple',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lavender',
                  edgecolor='purple', linewidth=2))

ax.set_xlabel('Fairness Violation (Demographic Parity %)',
              fontsize=14, fontweight='bold')
ax.set_ylabel('Model Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Fairness-Accuracy Pareto Frontier: Lagrangian Optimization',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 36)
ax.set_ylim(70, 92)

# Add lambda interpretation box
lambda_text = (
    'Lagrangian: L(θ, λ) = Acc(θ) - λ·Violation(θ)\n\n'
    'λ = 0: Ignore fairness (max accuracy)\n'
    'λ = 0.3: Balance trade-off (OPTIMAL)\n'
    'λ = 1.0: Prioritize fairness (min violation)'
)
props = dict(boxstyle='round,pad=0.8', facecolor='lightcyan',
             alpha=0.9, linewidth=2, edgecolor=Teal)
ax.text(0.97, 0.52, lambda_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', ha='right',
        bbox=props, fontfamily='monospace')

plt.tight_layout()
plt.savefig('../charts/optimization_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/optimization_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created optimization_tradeoff.pdf")

# ============================================================================
# Chart 4: information_theory_bias.pdf
# ============================================================================

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# --- Subplot 1: Venn diagram for I(D; A) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Mutual Information: I(D; A)', fontsize=14, fontweight='bold', pad=15)

# Draw Venn circles
circle_D = Circle((3.5, 5), 2.5, color=Teal, alpha=0.3, linewidth=3,
                   edgecolor=Teal, label='H(D)')
circle_A = Circle((6.5, 5), 2.5, color=Amber, alpha=0.3, linewidth=3,
                   edgecolor=Amber, label='H(A)')
ax1.add_patch(circle_D)
ax1.add_patch(circle_A)

# Labels
ax1.text(2.5, 5, 'H(D|A)', fontsize=13, ha='center', va='center', fontweight='bold')
ax1.text(7.5, 5, 'H(A|D)', fontsize=13, ha='center', va='center', fontweight='bold')
ax1.text(5, 5, 'I(D;A)', fontsize=14, ha='center', va='center',
         fontweight='bold', color='darkred',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))

# Decision and Protected attribute labels
ax1.text(5, 8.5, 'Decision D\n(Approve/Deny)', fontsize=12, ha='center',
         fontweight='bold', color=Teal)
ax1.text(5, 1.5, 'Protected Attribute A\n(Race, Gender, etc.)', fontsize=12,
         ha='center', fontweight='bold', color=Amber)

# Formula
ax1.text(5, 9.5, r'I(D; A) = H(D) - H(D|A) = H(A) - H(A|D)',
         fontsize=11, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                   edgecolor='black', linewidth=2))

# --- Subplot 2: Bias interpretation ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Bias Interpretation', fontsize=14, fontweight='bold', pad=15)

# Two scenarios
y_start = 8.5
scenarios = [
    ('No Bias', 'I(D; A) = 0', 'Decisions independent of group', mlgreen),
    ('Bias Present', 'I(D; A) > 0', 'Decisions leak group information', mlred),
]

for i, (title, formula, description, color) in enumerate(scenarios):
    y = y_start - i * 3.5

    # Box
    rect = FancyBboxPatch((0.5, y - 1.2), 9, 2.8,
                          boxstyle="round,pad=0.15",
                          linewidth=3, edgecolor=color,
                          facecolor=color, alpha=0.15)
    ax2.add_patch(rect)

    # Text
    ax2.text(5, y + 0.8, title, fontsize=13, ha='center',
             fontweight='bold', color=color)
    ax2.text(5, y + 0.2, formula, fontsize=12, ha='center',
             style='italic', fontfamily='monospace')
    ax2.text(5, y - 0.5, description, fontsize=11, ha='center')

# Example calculation
ax2.text(5, 0.8, 'Example: Loan approval with I(D; A) = 0.21 bits\n' +
         '→ Knowing group reduces decision uncertainty by 21%',
         fontsize=10, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                   edgecolor='black', linewidth=2))

# --- Subplot 3: Shannon entropy quantification ---
ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Shannon Entropy: Quantifying Unmeasurable Discrimination',
              fontsize=14, fontweight='bold', pad=15)

# Draw information flow diagram
boxes = [
    (1, 7, 'Protected\nAttributes', '6 categories\n(Race, Gender,\nAge, etc.)'),
    (1, 3.5, 'Subgroups', '490,140\ncombinations'),
    (4.5, 7, 'Discrimination\nSpace', 'H = log₂(490,140)\n= 18.9 bits'),
    (4.5, 3.5, 'Measurement\nCapacity', 'B = log₂(18)\n= 4.2 bits'),
    (8, 5.2, 'Information\nLoss', 'Loss = 18.9 - 4.2\n= 14.7 bits\n= 97% HIDDEN'),
]

colors_boxes = [Teal, Amber, mlblue, mlgreen, mlred]
for i, (x, y, title, content) in enumerate(boxes):
    rect = FancyBboxPatch((x - 0.7, y - 0.8), 1.8, 1.6,
                          boxstyle="round,pad=0.1",
                          linewidth=2.5, edgecolor=colors_boxes[i],
                          facecolor=colors_boxes[i], alpha=0.2)
    ax3.add_patch(rect)

    ax3.text(x + 0.2, y + 0.4, title, fontsize=11, ha='center',
             fontweight='bold', color=colors_boxes[i])
    ax3.text(x + 0.2, y - 0.3, content, fontsize=9, ha='center')

# Arrows showing flow
arrows = [
    ((1.9, 7), (3.8, 7), ''),
    ((1.9, 3.5), (3.8, 3.5), ''),
    ((2, 6.3), (2, 4.3), 'Intersectionality\nexplosion'),
    ((5.5, 6.3), (5.5, 4.3), 'Typical\naudit'),
    ((6.3, 7), (7.3, 5.8), ''),
    ((6.3, 3.5), (7.3, 4.6), ''),
]

for (x1, y1), (x2, y2), label in arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='black', alpha=0.6)
    ax3.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax3.text(mid_x + 0.5, mid_y, label, fontsize=9,
                ha='left', style='italic')

# Summary box
summary = (
    'Result: 97% of discrimination patterns are INVISIBLE\n'
    'Only 4.2 bits measured out of 18.9 bits total discrimination space\n'
    '→ Measurement bottleneck makes bias undetectable at scale'
)
props = dict(boxstyle='round,pad=0.8', facecolor='mistyrose',
             alpha=0.9, linewidth=3, edgecolor=mlred)
ax3.text(5, 1, summary, fontsize=11, ha='center', va='center',
        bbox=props, fontweight='bold')

plt.suptitle('Information Theory of Bias: I(D; A) and Shannon Entropy',
             fontsize=17, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('../charts/information_theory_bias.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/information_theory_bias.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created information_theory_bias.pdf")

print("\n" + "="*60)
print("ALL 4 NEW MATHEMATICAL CHARTS CREATED SUCCESSFULLY!")
print("="*60)
print("\nFiles created:")
print("  - fairness_roc_space.pdf (ROC geometric visualization)")
print("  - impossibility_proof.pdf (Chouldechova theorem)")
print("  - optimization_tradeoff.pdf (Pareto frontier)")
print("  - information_theory_bias.pdf (I(D;A) and entropy)")
print("\nNext: Run 'python compile.py 20251001_1700_main.tex'")
