"""
Week 8: Structured Output & Reliable AI Systems
Chart Generation Script

Generates all 15 visualizations for the lecture slides.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import seaborn as sns
from matplotlib.patches import Wedge
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create charts directory if it doesn't exist
os.makedirs('../charts', exist_ok=True)

# Color palette
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'gray': '#7f7f7f'
}

def save_chart(filename):
    """Save chart as both PDF and PNG"""
    plt.savefig(f'../charts/{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'../charts/{filename}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created {filename}")


# Chart 1: Reliability Cost Impact
def create_reliability_cost_impact():
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Manual\nFixes', 'Lost\nCustomers', 'Support\nTickets', 'Delayed\nLaunches', 'Refunds']
    costs = [45, 120, 35, 85, 25]  # in thousands
    colors = [COLORS['red'], COLORS['orange'], COLORS['purple'], COLORS['blue'], COLORS['green']]

    bars = ax.bar(categories, costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost}K',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Annual Cost (thousands $)', fontsize=12, fontweight='bold')
    ax.set_title('Cost of Unreliable AI Outputs\nAnnual Impact Per 1000 Users',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 140)

    # Add total annotation
    total = sum(costs)
    ax.text(0.98, 0.95, f'Total: ${total}K/year',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            ha='right', va='top')

    plt.tight_layout()
    save_chart('reliability_cost_impact')


# Chart 2: Structured vs Unstructured
def create_structured_vs_unstructured():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Unstructured output
    ax1.text(0.5, 0.9, 'Unstructured Output', ha='center', fontsize=14,
             fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.7, 'The restaurant was amazing! I\'d give\n'
             'it 5 stars. Great food quality and\n'
             'service was excellent. Price was\n'
             'moderate around $30 per person.',
             fontsize=10, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor=COLORS['red'], alpha=0.2))

    ax1.text(0.1, 0.35, 'Problems:', fontsize=11, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.25, '- No standard format', fontsize=9, transform=ax1.transAxes)
    ax1.text(0.1, 0.18, '- Requires parsing', fontsize=9, transform=ax1.transAxes)
    ax1.text(0.1, 0.11, '- Error-prone extraction', fontsize=9, transform=ax1.transAxes)
    ax1.text(0.1, 0.04, '- No validation', fontsize=9, transform=ax1.transAxes)

    ax1.axis('off')

    # Structured output
    ax2.text(0.5, 0.9, 'Structured Output (JSON)', ha='center', fontsize=14,
             fontweight='bold', transform=ax2.transAxes)

    json_text = '''{
  "rating": 5,
  "food_quality": 5,
  "service": 5,
  "price_level": "moderate",
  "avg_price_per_person": 30,
  "recommended_for": ["date", "friends"]
}'''

    ax2.text(0.1, 0.7, json_text, fontsize=9, transform=ax2.transAxes,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.2))

    ax2.text(0.1, 0.28, 'Benefits:', fontsize=11, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.1, 0.21, '- Standard JSON format', fontsize=9, transform=ax2.transAxes)
    ax2.text(0.1, 0.14, '- Direct integration', fontsize=9, transform=ax2.transAxes)
    ax2.text(0.1, 0.07, '- Type validation', fontsize=9, transform=ax2.transAxes)
    ax2.text(0.1, 0.00, '- Reliable parsing', fontsize=9, transform=ax2.transAxes)

    ax2.axis('off')

    plt.tight_layout()
    save_chart('structured_vs_unstructured')


# Chart 3: JSON Schema Example
def create_json_schema():
    fig, ax = plt.subplots(figsize=(10, 8))

    schema_text = '''{
  "type": "object",
  "properties": {
    "rating": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "food_quality": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "price_level": {
      "type": "string",
      "enum": ["cheap", "moderate", "expensive"]
    },
    "recommended_for": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["date", "family", "business", "friends"]
      }
    }
  },
  "required": ["rating", "food_quality", "price_level"]
}'''

    ax.text(0.05, 0.95, 'JSON Schema Example', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.05, 0.90, 'Restaurant Review Validation', fontsize=12,
            transform=ax.transAxes, style='italic')

    ax.text(0.05, 0.12, schema_text, fontsize=9, transform=ax.transAxes,
            family='monospace', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Add annotations
    ax.annotate('Type constraints', xy=(0.75, 0.70), xytext=(0.85, 0.75),
                transform=ax.transAxes, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2))

    ax.annotate('Value validation', xy=(0.75, 0.55), xytext=(0.85, 0.60),
                transform=ax.transAxes, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))

    ax.annotate('Required fields', xy=(0.75, 0.20), xytext=(0.85, 0.15),
                transform=ax.transAxes, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))

    ax.axis('off')
    plt.tight_layout()
    save_chart('json_schema_example')


# Chart 4: Prompt Patterns Comparison
def create_prompt_patterns():
    fig, ax = plt.subplots(figsize=(12, 7))

    patterns = ['Basic\nPrompt', 'Role-Based', 'Step-by-Step', 'Few-Shot', 'Chain-of-\nThought']
    success_rates = [72, 81, 88, 92, 95]
    consistency = [68, 79, 91, 93, 96]

    x = np.arange(len(patterns))
    width = 0.35

    bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate (%)',
                   color=COLORS['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, consistency, width, label='Consistency (%)',
                   color=COLORS['green'], alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prompt Engineering Patterns: Success Rate & Consistency',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_chart('prompt_patterns_comparison')


# Chart 5: Temperature vs Reliability
def create_temperature_reliability():
    fig, ax = plt.subplots(figsize=(10, 6))

    temperature = np.linspace(0, 2, 100)
    reliability = 100 / (1 + 2 * temperature)  # Inverse relationship
    creativity = 100 * (1 - np.exp(-1.5 * temperature))  # Saturation curve

    ax.plot(temperature, reliability, linewidth=3, label='Reliability', color=COLORS['blue'])
    ax.plot(temperature, creativity, linewidth=3, label='Creativity', color=COLORS['orange'])

    # Mark optimal zones
    ax.axvspan(0, 0.3, alpha=0.2, color=COLORS['green'], label='Structured Output Zone')
    ax.axvspan(0.7, 1.3, alpha=0.2, color=COLORS['purple'], label='Creative Zone')

    # Add annotations
    ax.annotate('Use for\nstructured data', xy=(0.15, 85), xytext=(0.4, 70),
                fontsize=10, arrowprops=dict(arrowstyle='->', lw=2))

    ax.annotate('Use for\ncreative content', xy=(1.0, 80), xytext=(1.4, 65),
                fontsize=10, arrowprops=dict(arrowstyle='->', lw=2))

    ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Temperature Impact on Reliability vs Creativity',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='center left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    save_chart('temperature_reliability')


# Chart 6: Function Calling Flow
def create_function_calling_flow():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define components
    components = [
        ('User\nQuery', 0.1, 0.8),
        ('LLM', 0.3, 0.8),
        ('Function\nDefinition', 0.5, 0.8),
        ('Function\nCall', 0.7, 0.8),
        ('Function\nResult', 0.7, 0.5),
        ('LLM\nProcesses', 0.5, 0.5),
        ('Structured\nResponse', 0.3, 0.5),
        ('User', 0.1, 0.5)
    ]

    # Draw boxes
    for label, x, y in components:
        color = COLORS['blue'] if 'LLM' in label else COLORS['green'] if 'Function' in label else COLORS['orange']
        box = FancyBboxPatch((x-0.05, y-0.05), 0.1, 0.1,
                             boxstyle="round,pad=0.01",
                             facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows
    arrows = [
        (0.15, 0.8, 0.25, 0.8),
        (0.35, 0.8, 0.45, 0.8),
        (0.55, 0.8, 0.65, 0.8),
        (0.7, 0.75, 0.7, 0.6),
        (0.65, 0.5, 0.55, 0.5),
        (0.45, 0.5, 0.35, 0.5),
        (0.25, 0.5, 0.15, 0.5),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Add step numbers
    steps = ['1', '2', '3', '4', '5', '6', '7']
    step_positions = [(0.2, 0.85), (0.4, 0.85), (0.6, 0.85), (0.75, 0.65),
                      (0.6, 0.45), (0.4, 0.45), (0.2, 0.45)]

    for step, (x, y) in zip(steps, step_positions):
        ax.text(x, y, step, ha='center', va='center', fontsize=12,
                fontweight='bold', color='red',
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='red', linewidth=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)
    ax.axis('off')
    ax.set_title('Function Calling Flow Architecture', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('function_calling_flow')


# Chart 7: Validation Pipeline
def create_validation_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6))

    stages = [
        ('Raw\nAI Output', 0.1, COLORS['gray']),
        ('Schema\nValidation', 0.3, COLORS['blue']),
        ('Type\nChecking', 0.5, COLORS['green']),
        ('Business\nRules', 0.7, COLORS['orange']),
        ('Final\nOutput', 0.9, COLORS['purple'])
    ]

    for label, x, color in stages:
        circle = plt.Circle((x, 0.5), 0.08, color=color, alpha=0.6, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 0.5, label, ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw connections
    for i in range(len(stages)-1):
        x1 = stages[i][1] + 0.08
        x2 = stages[i+1][1] - 0.08
        ax.arrow(x1, 0.5, x2-x1, 0, head_width=0.03, head_length=0.02, fc='black', ec='black', lw=2)

    # Add validation checks below
    checks = [
        ('Valid JSON?', 0.2),
        ('Correct types?', 0.4),
        ('Rules pass?', 0.6),
        ('Ready!', 0.8)
    ]

    for check, x in checks:
        ax.text(x, 0.25, check, ha='center', va='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.9)
    ax.axis('off')
    ax.set_title('Multi-Stage Validation Pipeline', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('validation_pipeline')


# Chart 8: Error Handling Strategies
def create_error_handling():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Decision tree structure
    ax.text(0.5, 0.95, 'AI Request', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['blue'], alpha=0.5))

    ax.text(0.5, 0.85, 'Success?', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Yes branch
    ax.text(0.7, 0.75, 'Yes', ha='center', fontsize=10, color='green', fontweight='bold')
    ax.text(0.7, 0.65, 'Valid schema?', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(0.85, 0.55, 'Return result', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.5))

    # No branch
    ax.text(0.3, 0.75, 'No', ha='center', fontsize=10, color='red', fontweight='bold')
    ax.text(0.3, 0.65, 'Retry count < 3?', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.text(0.15, 0.55, 'Yes: Retry\nwith backoff', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.5))

    ax.text(0.45, 0.55, 'No: Use\nfallback', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=COLORS['red'], alpha=0.5))

    # Schema validation failure
    ax.text(0.55, 0.55, 'No: Fix\nprompt', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.5))

    # Fallback options
    ax.text(0.5, 0.35, 'Fallback Options:', ha='center', fontsize=11, fontweight='bold')
    fallbacks = ['1. Simpler model', '2. Rule-based system', '3. Human review', '4. Default values']
    for i, fb in enumerate(fallbacks):
        ax.text(0.5, 0.28 - i*0.05, fb, ha='center', fontsize=9)

    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    arrows = [
        ((0.5, 0.92), (0.5, 0.88)),
        ((0.55, 0.83), (0.65, 0.77)),
        ((0.45, 0.83), (0.35, 0.77)),
        ((0.7, 0.72), (0.7, 0.68)),
        ((0.3, 0.72), (0.3, 0.68)),
        ((0.25, 0.63), (0.18, 0.58)),
        ((0.35, 0.63), (0.42, 0.58)),
        ((0.75, 0.63), (0.78, 0.58)),
        ((0.65, 0.63), (0.58, 0.58)),
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 1)
    ax.axis('off')
    ax.set_title('Error Handling Decision Tree', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('error_handling_strategies')


# Chart 9: Production Architecture
def create_production_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Layers
    layers = [
        ('User Interface\nAPI Gateway', 0.5, 0.9, 0.8),
        ('Request Queue\nLoad Balancer', 0.5, 0.75, 0.7),
        ('AI Service\nFunction Calling', 0.25, 0.55, 0.3),
        ('Validation\nService', 0.55, 0.55, 0.3),
        ('Cache Layer\nRedis', 0.75, 0.55, 0.25),
        ('Monitoring\nLogging', 0.5, 0.35, 0.6),
        ('Database\nPostgreSQL', 0.5, 0.15, 0.4)
    ]

    for label, x, y, width in layers:
        color = COLORS['blue'] if 'AI' in label else COLORS['green'] if 'Validation' in label else COLORS['orange'] if 'Cache' in label else COLORS['purple']
        rect = FancyBboxPatch((x-width/2, y-0.05), width, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=color, alpha=0.3, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw connections
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    connections = [
        ((0.5, 0.88), (0.5, 0.80)),
        ((0.5, 0.73), (0.3, 0.60)),
        ((0.5, 0.73), (0.55, 0.60)),
        ((0.6, 0.73), (0.75, 0.60)),
        ((0.25, 0.53), (0.5, 0.40)),
        ((0.55, 0.53), (0.5, 0.40)),
        ((0.5, 0.33), (0.5, 0.20)),
    ]

    for start, end in connections:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1)
    ax.axis('off')
    ax.set_title('Production Architecture for Structured AI', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('production_architecture')


# Chart 10: UX Reliability Patterns
def create_ux_patterns():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Pattern 1: Loading States
    ax1.text(0.5, 0.9, 'Progressive Loading', ha='center', fontsize=12, fontweight='bold', transform=ax1.transAxes)
    stages = ['Request sent', 'Processing...', 'Validating...', 'Ready!']
    colors_stage = [COLORS['blue'], COLORS['orange'], COLORS['purple'], COLORS['green']]
    for i, (stage, color) in enumerate(zip(stages, colors_stage)):
        y = 0.7 - i*0.15
        ax1.add_patch(Rectangle((0.2, y-0.03), 0.6, 0.08, facecolor=color, alpha=0.3, transform=ax1.transAxes))
        ax1.text(0.5, y, stage, ha='center', transform=ax1.transAxes, fontsize=10)
    ax1.axis('off')

    # Pattern 2: Confidence Display
    ax2.text(0.5, 0.9, 'Show Confidence', ha='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    confidences = [('High: 95%', COLORS['green']), ('Medium: 75%', COLORS['orange']), ('Low: 45%', COLORS['red'])]
    for i, (conf, color) in enumerate(confidences):
        y = 0.65 - i*0.2
        ax2.add_patch(Rectangle((0.1, y-0.05), 0.8, 0.12, facecolor=color, alpha=0.3, transform=ax2.transAxes))
        ax2.text(0.5, y, conf, ha='center', transform=ax2.transAxes, fontsize=11, fontweight='bold')
        if 'Low' in conf:
            ax2.text(0.5, y-0.06, 'Please review', ha='center', transform=ax2.transAxes, fontsize=9, style='italic')
    ax2.axis('off')

    # Pattern 3: Error Recovery
    ax3.text(0.5, 0.9, 'Graceful Error Recovery', ha='center', fontsize=12, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.7, 'Error Occurred', ha='center', transform=ax3.transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor=COLORS['red'], alpha=0.3))
    ax3.text(0.5, 0.5, 'Retry', ha='center', transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.3))
    ax3.text(0.3, 0.3, 'Fallback\nOption', ha='center', transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor=COLORS['blue'], alpha=0.3))
    ax3.text(0.7, 0.3, 'Manual\nInput', ha='center', transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor=COLORS['purple'], alpha=0.3))
    ax3.axis('off')

    # Pattern 4: Review & Edit
    ax4.text(0.5, 0.9, 'Human-in-the-Loop', ha='center', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.7, 'AI Suggestion', ha='center', transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor=COLORS['blue'], alpha=0.3))
    ax4.text(0.3, 0.45, 'Accept', ha='center', transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.5))
    ax4.text(0.5, 0.45, 'Edit', ha='center', transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.5))
    ax4.text(0.7, 0.45, 'Reject', ha='center', transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor=COLORS['red'], alpha=0.5))
    ax4.text(0.5, 0.2, 'User maintains control', ha='center', transform=ax4.transAxes, fontsize=9, style='italic')
    ax4.axis('off')

    plt.suptitle('UX Patterns for Reliable AI', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_chart('ux_reliability_patterns')


# Chart 11: Testing Pyramid
def create_testing_pyramid():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw pyramid
    pyramid_layers = [
        ('Unit Tests\n70%', 0, 0.6, COLORS['green']),
        ('Integration Tests\n20%', 0.2, 0.4, COLORS['orange']),
        ('End-to-End Tests\n10%', 0.4, 0.2, COLORS['blue'])
    ]

    for label, bottom, height, color in pyramid_layers:
        width_bottom = 0.8 - bottom
        width_top = width_bottom - 0.2

        # Draw trapezoid
        left_x = 0.1 + bottom
        right_x = 0.9 - bottom
        y_bottom = 0.15 + bottom
        y_top = y_bottom + height

        trap = plt.Polygon([[left_x, y_bottom], [right_x, y_bottom],
                           [right_x - 0.1, y_top], [left_x + 0.1, y_top]],
                          facecolor=color, alpha=0.6, edgecolor='black', linewidth=2)
        ax.add_patch(trap)

        ax.text(0.5, y_bottom + height/2, label, ha='center', va='center',
                fontsize=12, fontweight='bold')

    # Add descriptions
    descriptions = [
        ('Schema validation\nType checking\nBusiness rules', 0.35),
        ('API integration\nValidation pipeline\nError handling', 0.65),
        ('Full workflow\nUser scenarios\nEdge cases', 0.85)
    ]

    for desc, y in descriptions:
        ax.text(1.05, y, desc, ha='left', va='center', fontsize=9, transform=ax.transData)

    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Testing Pyramid for Structured AI', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('testing_pyramid')


# Chart 12: Monitoring Dashboard
def create_monitoring_dashboard():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Metric 1: Success Rate
    hours = np.arange(24)
    success_rate = 95 + np.random.normal(0, 2, 24)
    success_rate = np.clip(success_rate, 85, 100)

    ax1.plot(hours, success_rate, linewidth=2, color=COLORS['green'], marker='o')
    ax1.axhline(y=90, color='red', linestyle='--', label='Threshold')
    ax1.fill_between(hours, 90, success_rate, where=(success_rate >= 90), alpha=0.3, color='green')
    ax1.set_title('Success Rate (24h)', fontweight='bold')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Success %')
    ax1.set_ylim(80, 100)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Metric 2: Response Time
    response_times = np.array([0.3, 0.5, 0.8, 1.2, 2.1])
    percentiles = ['P50', 'P75', 'P90', 'P95', 'P99']
    colors_rt = [COLORS['green'], COLORS['green'], COLORS['orange'], COLORS['orange'], COLORS['red']]

    bars = ax2.barh(percentiles, response_times, color=colors_rt, alpha=0.7)
    ax2.set_title('Response Time Percentiles', fontweight='bold')
    ax2.set_xlabel('Seconds')
    ax2.axvline(x=1.0, color='red', linestyle='--', label='SLA: 1s')
    ax2.legend()

    # Add values
    for bar, val in zip(bars, response_times):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}s', va='center', fontsize=9)

    # Metric 3: Error Types
    errors = ['Schema\nInvalid', 'Timeout', 'API\nError', 'Validation\nFailed', 'Other']
    error_counts = [45, 23, 18, 12, 8]

    wedges, texts, autotexts = ax3.pie(error_counts, labels=errors, autopct='%1.1f%%',
                                        colors=[COLORS['red'], COLORS['orange'], COLORS['purple'],
                                               COLORS['blue'], COLORS['gray']],
                                        startangle=90)
    ax3.set_title('Error Distribution', fontweight='bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Metric 4: Cost Trend
    days = np.arange(30)
    daily_cost = 120 + 5 * days + np.random.normal(0, 10, 30)

    ax4.plot(days, daily_cost, linewidth=2, color=COLORS['purple'])
    ax4.fill_between(days, daily_cost, alpha=0.3, color=COLORS['purple'])
    ax4.set_title('Daily API Cost Trend', fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Cost ($)')
    ax4.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(days, daily_cost, 1)
    p = np.poly1d(z)
    ax4.plot(days, p(days), "r--", linewidth=2, label='Trend')
    ax4.legend()

    plt.suptitle('Production Monitoring Dashboard', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_chart('monitoring_dashboard')


# Chart 13: Innovation Pipeline Week 8
def create_innovation_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6))

    stages = [
        ('Empathize\nWeeks 1-3', 0.1, COLORS['blue']),
        ('Define\nWeek 4', 0.25, COLORS['orange']),
        ('Ideate\nWeek 5', 0.4, COLORS['green']),
        ('Prototype\nWeek 6', 0.55, COLORS['purple']),
        ('Refine\nWeek 8', 0.7, COLORS['red']),
        ('Test\nWeeks 9-10', 0.85, COLORS['gray'])
    ]

    for label, x, color in stages:
        circle = plt.Circle((x, 0.5), 0.06, color=color, alpha=0.6 if 'Week 8' not in label else 1.0,
                           ec='black', linewidth=3 if 'Week 8' in label else 2)
        ax.add_patch(circle)
        ax.text(x, 0.5, label.split('\n')[0], ha='center', va='center',
               fontsize=10 if 'Week 8' not in label else 12,
               fontweight='bold' if 'Week 8' in label else 'normal')
        ax.text(x, 0.35, label.split('\n')[1], ha='center', va='top',
               fontsize=8, style='italic')

    # Draw connections
    for i in range(len(stages)-1):
        x1 = stages[i][1] + 0.06
        x2 = stages[i+1][1] - 0.06
        ax.arrow(x1, 0.5, x2-x1, 0, head_width=0.025, head_length=0.02,
                fc='black', ec='black', lw=2)

    # Highlight Week 8
    ax.text(0.7, 0.7, 'You are here!', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.annotate('', xy=(0.7, 0.58), xytext=(0.7, 0.68),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.axis('off')
    ax.set_title('Design Thinking Journey: Week 8 Position', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('innovation_pipeline_week8')


# Chart 14: ROI Calculator
def create_roi_calculator():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Before and After comparison
    categories = ['Time per\nRequest', 'Error Rate', 'Manual\nReview', 'Customer\nSatisfaction', 'Monthly\nCost']
    before = [120, 15, 80, 65, 8500]
    after = [8, 2, 10, 92, 6200]

    # Normalize for visualization (different units)
    before_norm = [val / max(before[i], after[i]) * 100 for i, val in enumerate(before)]
    after_norm = [val / max(before[i], after[i]) * 100 for i, val in enumerate(after)]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.barh([i - width/2 for i in x], before_norm, width,
                    label='Before (Unstructured)', color=COLORS['red'], alpha=0.7)
    bars2 = ax.barh([i + width/2 for i in x], after_norm, width,
                    label='After (Structured)', color=COLORS['green'], alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Relative Performance (lower is better for cost/time/errors)', fontsize=10)
    ax.set_title('ROI: Unstructured vs Structured Outputs', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)

    # Add actual values as text
    labels_before = ['120 min', '15%', '80%', '65/100', '$8.5K']
    labels_after = ['8 min', '2%', '10%', '92/100', '$6.2K']

    for i, (val_before, val_after) in enumerate(zip(labels_before, labels_after)):
        ax.text(105, i - width/2, val_before, va='center', fontsize=9, color=COLORS['red'], fontweight='bold')
        ax.text(105, i + width/2, val_after, va='center', fontsize=9, color=COLORS['green'], fontweight='bold')

    # Add improvement percentages
    improvements = ['-93%', '-87%', '-88%', '+42%', '-27%']
    for i, improvement in enumerate(improvements):
        color = COLORS['green'] if '-' in improvement or '+' in improvement else COLORS['red']
        ax.text(-5, i, improvement, ha='right', va='center', fontsize=10,
                fontweight='bold', color=color)

    ax.set_xlim(-10, 120)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_chart('roi_calculator')


# Chart 15: Best Practices Checklist
def create_best_practices():
    fig, ax = plt.subplots(figsize=(10, 10))

    practices = [
        ('Design', [
            'Define clear JSON schema',
            'Document required vs optional fields',
            'Use enums for constrained values',
            'Include examples in schema'
        ]),
        ('Implementation', [
            'Set temperature to 0-0.3',
            'Use function calling when available',
            'Implement multi-stage validation',
            'Add retry logic with backoff'
        ]),
        ('Testing', [
            'Unit test schema validation',
            'Integration test full pipeline',
            'Test edge cases and errors',
            'Validate with real data'
        ]),
        ('Production', [
            'Monitor success rates',
            'Log failed validations',
            'Set up alerting',
            'Track costs and optimize'
        ])
    ]

    y_pos = 0.95
    colors_section = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]

    for (section, items), color in zip(practices, colors_section):
        # Section header
        ax.text(0.05, y_pos, section, fontsize=14, fontweight='bold',
                transform=ax.transAxes, color=color)
        y_pos -= 0.05

        # Items
        for item in items:
            ax.text(0.1, y_pos, f'• {item}', fontsize=10, transform=ax.transAxes)
            y_pos -= 0.04

        y_pos -= 0.03  # Extra space between sections

    ax.axis('off')
    ax.set_title('Best Practices for Structured AI', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_chart('best_practices_checklist')


# Main execution
if __name__ == '__main__':
    print("Generating Week 8 charts...\n")

    create_reliability_cost_impact()
    create_structured_vs_unstructured()
    create_json_schema()
    create_prompt_patterns()
    create_temperature_reliability()
    create_function_calling_flow()
    create_validation_pipeline()
    create_error_handling()
    create_production_architecture()
    create_ux_patterns()
    create_testing_pyramid()
    create_monitoring_dashboard()
    create_innovation_pipeline()
    create_roi_calculator()
    create_best_practices()

    print("\n✓ All 15 charts generated successfully!")
    print("Charts saved to ../charts/ as both PDF (300 dpi) and PNG (150 dpi)")