import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Timeline data
years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
adoption_rates = [2, 5, 15, 35, 60, 85, 95]  # % of companies using GenAI

# Key milestones
milestones = [
    (2020.5, 10, "GPT-3\nLaunch"),
    (2021.8, 20, "GitHub\nCopilot"),
    (2022.5, 30, "DALL-E 2\nMidjourney"),
    (2022.9, 40, "Stable\nDiffusion"),
    (2023.2, 50, "ChatGPT\nBoom"),
    (2024.0, 65, "Enterprise\nAdoption"),
    (2025.5, 88, "Industry\nStandard")
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot adoption curve
ax.plot(years, adoption_rates, 'o-', linewidth=3, markersize=10,
        color='#3498db', label='GenAI Adoption Rate')

# Add smooth curve
from scipy.interpolate import make_interp_spline
years_smooth = np.linspace(2020, 2026, 300)
spl = make_interp_spline(years, adoption_rates, k=3)
adoption_smooth = spl(years_smooth)
ax.plot(years_smooth, adoption_smooth, '--', alpha=0.5, linewidth=2, color='#3498db')

# Fill area under curve
ax.fill_between(years_smooth, 0, adoption_smooth, alpha=0.2, color='#3498db')

# Add milestones
for year, height, label in milestones:
    ax.annotate(label, xy=(year, height), xytext=(year, height + 8),
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#f39c12', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                               color='#e67e22', lw=2))

# Add adoption phases
phase_y = -12
ax.text(2021, phase_y, 'Early\nAdopters', ha='center', fontsize=10,
        color='#7f8c8d', fontweight='bold')
ax.text(2023, phase_y, 'Early\nMajority', ha='center', fontsize=10,
        color='#7f8c8d', fontweight='bold')
ax.text(2025, phase_y, 'Late\nMajority', ha='center', fontsize=10,
        color='#7f8c8d', fontweight='bold')

# Styling
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Enterprise Adoption (%)', fontsize=12, fontweight='bold')
ax.set_title('Generative AI Adoption Timeline in Design & Prototyping',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(2019.5, 2026.5)
ax.set_ylim(-20, 105)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=11)

# Add growth indicators
ax.text(2022, 25, '300% YoY\nGrowth', fontsize=10, fontweight='bold',
        color='#27ae60', ha='center')
ax.text(2024.5, 75, 'Mainstream\nAdoption', fontsize=10, fontweight='bold',
        color='#2c3e50', ha='center')

# Save the figure
plt.tight_layout()
plt.savefig('../charts/adoption_timeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../charts/adoption_timeline.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created adoption_timeline chart")