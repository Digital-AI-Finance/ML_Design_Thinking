"""Structured Output Reliability - JSON schema validation quadrant"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

fig, ax = plt.subplots(figsize=(10, 6))

# Quadrant labels
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

# Fill quadrants
ax.fill([0, 0.5, 0.5, 0], [0.5, 0.5, 1, 1], color='#f1c40f', alpha=0.3)  # Top-left: Flexible but unreliable
ax.fill([0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1], color='#2ecc71', alpha=0.3)  # Top-right: JSON Schema (best)
ax.fill([0, 0.5, 0.5, 0], [0, 0, 0.5, 0.5], color='#e74c3c', alpha=0.3)  # Bottom-left: Worst
ax.fill([0.5, 1, 1, 0.5], [0, 0, 0.5, 0.5], color='#3498db', alpha=0.3)  # Bottom-right: Structured but rigid

# Quadrant text
ax.text(0.25, 0.75, 'Flexible\nbut Unreliable', ha='center', va='center', fontsize=10, fontweight='bold', color='#f39c12')
ax.text(0.75, 0.75, 'JSON Schema\n(Optimal)', ha='center', va='center', fontsize=11, fontweight='bold', color='#27ae60')
ax.text(0.25, 0.25, 'Unstructured\nUnreliable', ha='center', va='center', fontsize=10, fontweight='bold', color='#c0392b')
ax.text(0.75, 0.25, 'Structured\nbut Rigid', ha='center', va='center', fontsize=10, fontweight='bold', color='#2980b9')

# Place markers for different approaches
approaches = [
    {'name': 'Free text', 'x': 0.15, 'y': 0.6, 'color': '#e74c3c'},
    {'name': 'Regex', 'x': 0.7, 'y': 0.35, 'color': '#3498db'},
    {'name': 'JSON Schema', 'x': 0.85, 'y': 0.85, 'color': '#2ecc71'},
    {'name': 'Pydantic', 'x': 0.75, 'y': 0.8, 'color': '#2ecc71'},
]

for approach in approaches:
    ax.scatter(approach['x'], approach['y'], s=150, c=approach['color'],
              edgecolors='white', linewidth=2, zorder=5)
    ax.annotate(approach['name'], (approach['x'], approach['y']),
               xytext=(5, 5), textcoords='offset points', fontsize=8)

# Axis labels
ax.set_xlabel('Structure Level', fontsize=11, fontweight='bold')
ax.set_ylabel('Reliability', fontsize=11, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels(['Unstructured', 'Structured'])
ax.set_yticks([0.25, 0.75])
ax.set_yticklabels(['Low', 'High'])

ax.set_title('Output Reliability Framework', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("17_structured_output_reliability created")
