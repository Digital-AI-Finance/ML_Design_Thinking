"""
Week 8 V2 Chart Generation Script
Creates all 11 conceptual charts for the compact rewrite
NO FAKE DATA - all charts are conceptual visualizations
Uses mllavender color palette
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette (mllavender)
colors = {
    'mlblue': '#0066CC',
    'mlpurple': '#3333B2',
    'mllavender': '#ADADE0',
    'mllavender2': '#C1C1E8',
    'mllavender3': '#CCCCEB',
    'mllavender4': '#D6D6EF',
    'mlorange': '#FF7F0E',
    'mlgreen': '#2CA02C',
    'mlred': '#D62728',
    'mlgray': '#7F7F7F',
    'lightgray': '#F0F0F0',
    'white': '#FFFFFF'
}

def save_chart(fig, name):
    """Save chart as both PDF and PNG"""
    fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    print(f"Created: {name}.pdf and {name}.png")
    plt.close(fig)


def chart1_unpredictability_problem():
    """Same input, different outputs - the unpredictability problem"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Input box
    input_box = FancyBboxPatch((0.5, 2.5), 2, 1, boxstyle="round,pad=0.1",
                                edgecolor=colors['mlpurple'], facecolor=colors['mllavender4'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 3, 'Input:\n"iPhone 15 Pro\n128GB $999"',
            ha='center', va='center', fontsize=10, weight='bold')

    # Arrow
    arrow1 = FancyArrowPatch((2.7, 3), (4.3, 4.5), arrowstyle='->', lw=2, color=colors['mlgray'])
    arrow2 = FancyArrowPatch((2.7, 3), (4.3, 3), arrowstyle='->', lw=2, color=colors['mlgray'])
    arrow3 = FancyArrowPatch((2.7, 3), (4.3, 1.5), arrowstyle='->', lw=2, color=colors['mlgray'])
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)

    # Output boxes (3 different formats)
    outputs = [
        (5, 4.2, "Output 1:\niPhone 15 Pro\ncosts $999\nwith 128GB"),
        (5, 2.7, "Output 2:\nProduct: iPhone 15 Pro\nPrice: 999 USD\nStorage: 128 GB"),
        (5, 1.2, "Output 3:\n{price: $999,\nname: iPhone 15 Pro,\nstorage: 128GB}")
    ]

    for x, y, text in outputs:
        box = FancyBboxPatch((x, y), 3.5, 0.8, boxstyle="round,pad=0.05",
                             edgecolor=colors['mlred'], facecolor=colors['lightgray'], linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1.75, y + 0.4, text, ha='center', va='center', fontsize=8, family='monospace')

    # Title
    ax.text(5, 5.5, 'The Unpredictability Problem', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])
    ax.text(5, 0.3, 'Same input → Three different output formats (unparseable, inconsistent)',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'unpredictability_problem')


def chart2_integration_challenge():
    """Database/API rejecting invalid data"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # AI Output
    ai_box = FancyBboxPatch((0.5, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                            edgecolor=colors['mlorange'], facecolor=colors['mllavender4'], linewidth=2)
    ax.add_patch(ai_box)
    ax.text(1.75, 4.25, 'AI Output:\n"Price: 999 USD\nStorage: 128 GB"',
            ha='center', va='center', fontsize=9, family='monospace')

    # Arrow to database
    arrow = FancyArrowPatch((3.2, 4.25), (5.3, 4.25), arrowstyle='->', lw=3, color=colors['mlgray'])
    ax.add_patch(arrow)

    # Database expecting structure
    db_box = FancyBboxPatch((5.5, 3.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                            edgecolor=colors['mlblue'], facecolor=colors['lightgray'], linewidth=2)
    ax.add_patch(db_box)
    ax.text(7.25, 4.5, 'Database Expects:', ha='center', fontsize=10, weight='bold', color=colors['mlblue'])
    ax.text(7.25, 4, 'price: FLOAT\nstorage_gb: INT', ha='center', va='center',
            fontsize=9, family='monospace', color=colors['mlpurple'])

    # X mark (rejection)
    ax.plot([4, 4.8], [4.25, 4.25], 'r-', lw=4)
    ax.plot([4.2, 4.6], [4.45, 4.05], 'r-', lw=4)
    ax.plot([4.2, 4.6], [4.05, 4.45], 'r-', lw=4)

    # Error message
    error_box = Rectangle((3, 2), 4, 1, facecolor='#FFE6E6', edgecolor=colors['mlred'], linewidth=2)
    ax.add_patch(error_box)
    ax.text(5, 2.5, 'ERROR: Invalid data type\n"999 USD" cannot convert to FLOAT',
            ha='center', va='center', fontsize=9, color=colors['mlred'], weight='bold', family='monospace')

    # Title
    ax.text(5, 5.5, 'The Integration Challenge', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])
    ax.text(5, 0.5, 'AI generates text, but systems need typed, structured data',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'integration_challenge')


def chart3_prompt_engineering_patterns():
    """Five prompt engineering techniques"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, 'Five Prompt Engineering Patterns', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])

    patterns = [
        ("1. Detailed Instructions", "Specify exactly what to extract,\nlist all required fields", 1.5, 7.5),
        ("2. Few-Shot Examples", "Show 3-5 example outputs\nto demonstrate format", 5.5, 7.5),
        ("3. Role-Playing", '"You are an expert..."\nsets context and expectations', 1.5, 5),
        ("4. Step-by-Step", "Break into steps:\n1. Identify... 2. Extract...", 5.5, 5),
        ("5. Format Specification", '"Return as JSON with..."\ndescribe desired structure', 3.5, 2.5)
    ]

    for title, desc, x, y in patterns:
        # Box
        box = FancyBboxPatch((x - 1.2, y - 0.6), 2.4, 1.2, boxstyle="round,pad=0.1",
                             edgecolor=colors['mlblue'], facecolor=colors['mllavender3'], linewidth=2)
        ax.add_patch(box)

        # Text
        ax.text(x, y + 0.3, title, ha='center', fontsize=11, weight='bold', color=colors['mlpurple'])
        ax.text(x, y - 0.2, desc, ha='center', va='center', fontsize=8, color=colors['mlgray'])

    ax.text(5, 0.8, 'All techniques improve quality through clearer communication',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'prompt_engineering_patterns')


def chart4_success_examples():
    """Visual showing working cases"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Title
    ax.text(5, 5.5, 'Success: When Prompt Engineering Works', ha='center', fontsize=14, weight='bold', color=colors['mlgreen'])

    examples = [
        ("Input:\niPhone 15 Pro\n128GB $999", "Output:\nProduct: iPhone 15 Pro\nPrice: $999\nStorage: 128GB", 2, 3),
        ("Input:\nMacBook Air M2\n256GB $1199", "Output:\nProduct: MacBook Air M2\nPrice: $1199\nStorage: 256GB", 5.5, 3),
    ]

    for inp, out, x, y in examples:
        # Input box
        input_box = FancyBboxPatch((x - 0.8, y + 0.5), 1.6, 1.2, boxstyle="round,pad=0.05",
                                   edgecolor=colors['mlblue'], facecolor=colors['lightgray'], linewidth=1.5)
        ax.add_patch(input_box)
        ax.text(x, y + 1.1, inp, ha='center', va='center', fontsize=8, family='monospace')

        # Arrow
        arrow = FancyArrowPatch((x, y + 0.4), (x, y - 0.2), arrowstyle='->', lw=2, color=colors['mlgreen'])
        ax.add_patch(arrow)

        # Output box
        output_box = FancyBboxPatch((x - 0.9, y - 1.5), 1.8, 1.2, boxstyle="round,pad=0.05",
                                    edgecolor=colors['mlgreen'], facecolor='#E6FFE6', linewidth=2)
        ax.add_patch(output_box)
        ax.text(x, y - 0.9, out, ha='center', va='center', fontsize=8, family='monospace')

        # Checkmark
        ax.text(x + 1.1, y - 0.9, '✓', ha='center', fontsize=20, color=colors['mlgreen'], weight='bold')

    ax.text(5, 0.5, 'On clean, simple inputs: Consistent, parseable outputs',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'success_examples')


def chart5_failure_pattern():
    """Visual showing breaking cases"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Title
    ax.text(5, 5.5, 'Failure: When Complexity Breaks Prompt Engineering', ha='center', fontsize=14, weight='bold', color=colors['mlred'])

    # Complex input
    input_box = FancyBboxPatch((0.5, 3), 4, 1.5, boxstyle="round,pad=0.05",
                               edgecolor=colors['mlorange'], facecolor=colors['lightgray'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 3.75, 'Complex Input:\n"iPhone 15 Pro Max 256 or 512GB\noptions starting at $1099\n(with trade-in discount)"',
            ha='center', va='center', fontsize=8, family='monospace')

    # Arrow
    arrow = FancyArrowPatch((4.7, 3.75), (5.3, 3.75), arrowstyle='->', lw=3, color=colors['mlgray'])
    ax.add_patch(arrow)

    # Bad outputs
    outputs = [
        ("Price: $1099\nStorage: 256 or 512GB", 7, 4.5, "X Type error:\ncan't parse range"),
        ("price: 1099 USD\nstorage_capacity: 256GB", 7, 3, "X Field mismatch:\nwrong keys"),
        ("product=$1099,\nstorage=256/512", 7, 1.5, "X Parse error:\ninvalid format")
    ]

    for text, x, y, error in outputs:
        # Output box
        box = FancyBboxPatch((x - 1.3, y - 0.3), 2.6, 0.6, boxstyle="round,pad=0.03",
                             edgecolor=colors['mlred'], facecolor='#FFE6E6', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, family='monospace')

        # Error label
        ax.text(x + 1.5, y, error, ha='left', va='center', fontsize=7, color=colors['mlred'], weight='bold')

    ax.text(5, 0.5, 'On complex, messy inputs: Inconsistent, unparseable outputs',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'failure_pattern')


def chart6_human_consistency_methods():
    """How humans ensure data consistency"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Title
    ax.text(5, 5.5, 'How Humans Ensure Data Consistency', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])

    steps = [
        ("1. Define\nSchema", colors['mlblue'], 1.5),
        ("2. Enter\nData", colors['mlgreen'], 3.5),
        ("3. Validate\nAgainst Schema", colors['mlorange'], 5.5),
        ("4. Reject\nInvalid", colors['mlred'], 7.5)
    ]

    for i, (text, color, x) in enumerate(steps):
        # Circle
        circle = Circle((x, 3), 0.6, facecolor=color, edgecolor='white', linewidth=3, alpha=0.3)
        ax.add_patch(circle)
        ax.text(x, 3, str(i+1), ha='center', va='center', fontsize=16, weight='bold', color=color)

        # Label
        ax.text(x, 2, text, ha='center', va='center', fontsize=9, weight='bold', color=color)

        # Arrow (except last)
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((x + 0.7, 3), (x + 1.3, 3), arrowstyle='->', lw=3, color=colors['mlgray'])
            ax.add_patch(arrow)

    ax.text(5, 0.8, 'The pattern: Structure-first, validate-always, reject-invalid',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'human_consistency_methods')


def chart7_prompts_vs_schemas():
    """Suggestions vs Contracts comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Title
    ax.text(5, 5.5, 'Suggestions vs Contracts', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])

    # Left side - Prompts (Weak)
    prompt_box = FancyBboxPatch((0.5, 2), 4, 2.5, boxstyle="round,pad=0.1",
                                edgecolor=colors['mlorange'], facecolor=colors['lightgray'], linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(2.5, 4.2, 'Prompts (Suggestions)', ha='center', fontsize=12, weight='bold', color=colors['mlorange'])
    ax.text(2.5, 3.6, '"Please return as JSON..."', ha='center', fontsize=9, style='italic')
    ax.text(2.5, 3, 'Characteristics:', ha='center', fontsize=9, weight='bold')
    ax.text(2.5, 2.5, '• AI can ignore\n• No enforcement\n• Variable compliance',
            ha='center', va='center', fontsize=8)

    # VS
    ax.text(5, 3.25, 'VS', ha='center', va='center', fontsize=16, weight='bold', color=colors['mlgray'])

    # Right side - Schemas (Strong)
    schema_box = FancyBboxPatch((5.5, 2), 4, 2.5, boxstyle="round,pad=0.1",
                                edgecolor=colors['mlgreen'], facecolor='#E6FFE6', linewidth=3)
    ax.add_patch(schema_box)
    ax.text(7.5, 4.2, 'Schemas (Contracts)', ha='center', fontsize=12, weight='bold', color=colors['mlgreen'])
    ax.text(7.5, 3.6, 'API-level enforcement', ha='center', fontsize=9, weight='bold')
    ax.text(7.5, 3, 'Characteristics:', ha='center', fontsize=9, weight='bold')
    ax.text(7.5, 2.5, '• Guaranteed structure\n• API enforces\n• Predictable output',
            ha='center', va='center', fontsize=8)

    ax.text(5, 0.8, 'Enforcement beats suggestion - contracts ensure reliability',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'prompts_vs_schemas')


def chart8_three_layer_architecture():
    """The complete 3-layer system"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    # Title
    ax.text(5, 6.5, 'The 3-Layer Architecture', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])

    # Layer 1: Schema Definition
    layer1 = FancyBboxPatch((0.5, 4.5), 2.5, 1.3, boxstyle="round,pad=0.1",
                            edgecolor=colors['mlblue'], facecolor=colors['mllavender3'], linewidth=3)
    ax.add_patch(layer1)
    ax.text(1.75, 5.6, 'Layer 1', ha='center', fontsize=10, weight='bold', color=colors['mlblue'])
    ax.text(1.75, 5.3, 'Schema Definition', ha='center', fontsize=9, weight='bold')
    ax.text(1.75, 4.9, 'Define contract\nwith Pydantic', ha='center', va='center', fontsize=8)

    # Arrow 1
    arrow1 = FancyArrowPatch((3.2, 5.15), (3.8, 5.15), arrowstyle='->', lw=3, color=colors['mlgray'])
    ax.add_patch(arrow1)

    # Layer 2: Function Calling
    layer2 = FancyBboxPatch((4, 4.5), 2.5, 1.3, boxstyle="round,pad=0.1",
                            edgecolor=colors['mlgreen'], facecolor='#E6FFE6', linewidth=3)
    ax.add_patch(layer2)
    ax.text(5.25, 5.6, 'Layer 2', ha='center', fontsize=10, weight='bold', color=colors['mlgreen'])
    ax.text(5.25, 5.3, 'Function Calling', ha='center', fontsize=9, weight='bold')
    ax.text(5.25, 4.9, 'Enforce schema\nat API level', ha='center', va='center', fontsize=8)

    # Arrow 2
    arrow2 = FancyArrowPatch((6.7, 5.15), (7.3, 5.15), arrowstyle='->', lw=3, color=colors['mlgray'])
    ax.add_patch(arrow2)

    # Layer 3: Validation
    layer3 = FancyBboxPatch((7.5, 4.5), 2, 1.3, boxstyle="round,pad=0.1",
                            edgecolor=colors['mlorange'], facecolor=colors['lightgray'], linewidth=3)
    ax.add_patch(layer3)
    ax.text(8.5, 5.6, 'Layer 3', ha='center', fontsize=10, weight='bold', color=colors['mlorange'])
    ax.text(8.5, 5.3, 'Validation', ha='center', fontsize=9, weight='bold')
    ax.text(8.5, 4.9, 'Check + Retry\non errors', ha='center', va='center', fontsize=8)

    # Flow diagram below
    flow_items = [
        ("Input\nText", 1, 3),
        ("Schema\n(Contract)", 3, 3),
        ("LLM with\nFunction Call", 5, 3),
        ("Structured\nJSON", 7, 3),
        ("Validated\nOutput", 9, 3)
    ]

    for i, (text, x, y) in enumerate(flow_items):
        box = FancyBboxPatch((x - 0.4, y - 0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                             edgecolor=colors['mlpurple'], facecolor=colors['mllavender4'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=7)

        if i < len(flow_items) - 1:
            ax.arrow(x + 0.5, y, 0.7, 0, head_width=0.15, head_length=0.15, fc=colors['mlgray'], ec=colors['mlgray'])

    # Error path (retry loop)
    retry_arrow = FancyArrowPatch((8.5, 2.5), (5, 2.5), arrowstyle='->', lw=2, color=colors['mlred'],
                                  linestyle='dashed')
    ax.add_patch(retry_arrow)
    ax.text(6.75, 2.2, 'Retry on validation error', ha='center', fontsize=7, color=colors['mlred'], style='italic')

    ax.text(5, 0.5, 'Three layers ensure reliability: Define → Enforce → Validate',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'three_layer_architecture')


def chart9_before_after_qualitative():
    """Qualitative before/after comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Title
    ax.text(5, 5.5, 'Before and After: The Transformation', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])

    # Left side - Before
    before_box = FancyBboxPatch((0.5, 1), 4, 3.5, boxstyle="round,pad=0.1",
                                edgecolor=colors['mlred'], facecolor='#FFE6E6', linewidth=3)
    ax.add_patch(before_box)
    ax.text(2.5, 4.2, 'BEFORE: Prompt Engineering', ha='center', fontsize=11, weight='bold', color=colors['mlred'])

    before_items = [
        ("Works on simple cases", "✓", colors['mlgreen']),
        ("Breaks on complexity", "X", colors['mlred']),
        ("Variable formats", "X", colors['mlred']),
        ("Unpredictable errors", "X", colors['mlred']),
        ("Manual intervention", "X", colors['mlred']),
        ("Not production-ready", "X", colors['mlred'])
    ]

    y = 3.7
    for text, mark, color in before_items:
        ax.text(0.8, y, mark, ha='center', fontsize=12, weight='bold', color=color)
        ax.text(1.3, y, text, ha='left', fontsize=8)
        y -= 0.4

    # Right side - After
    after_box = FancyBboxPatch((5.5, 1), 4, 3.5, boxstyle="round,pad=0.1",
                               edgecolor=colors['mlgreen'], facecolor='#E6FFE6', linewidth=3)
    ax.add_patch(after_box)
    ax.text(7.5, 4.2, 'AFTER: Structured Outputs', ha='center', fontsize=11, weight='bold', color=colors['mlgreen'])

    after_items = [
        ("Works across complexity", "✓", colors['mlgreen']),
        ("Handles messy data", "✓", colors['mlgreen']),
        ("Consistent format", "✓", colors['mlgreen']),
        ("Predictable errors", "✓", colors['mlgreen']),
        ("Automatic retry", "✓", colors['mlgreen']),
        ("Production-grade", "✓", colors['mlgreen'])
    ]

    y = 3.7
    for text, mark, color in after_items:
        ax.text(5.8, y, mark, ha='center', fontsize=12, weight='bold', color=color)
        ax.text(6.3, y, text, ha='left', fontsize=8)
        y -= 0.4

    ax.text(5, 0.4, 'Qualitative improvement through structure and validation',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'before_after_qualitative')


def chart10_production_architecture_unified():
    """Complete production system with all components"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    # Title
    ax.text(7, 7.5, 'Production Architecture: All Layers Working Together', ha='center',
            fontsize=14, weight='bold', color=colors['mlpurple'])

    # Components
    components = [
        ("Input\nData", 1, 5, colors['mlblue'], 1.2, 1),
        ("Schema\nDefinition\n(Pydantic)", 3.5, 5, colors['mlblue'], 1.5, 1.2),
        ("LLM API\nFunction\nCalling", 6.5, 5, colors['mlgreen'], 1.5, 1.2),
        ("Structured\nJSON", 9.5, 5, colors['mlgreen'], 1.2, 1),
        ("Validation\nLayer", 12, 5, colors['mlorange'], 1.2, 1),
        ("Output", 12, 3, colors['mlpurple'], 1, 0.8)
    ]

    for name, x, y, color, w, h in components:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1",
                             edgecolor=color, facecolor=colors['lightgray'], linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, weight='bold')

    # Arrows (main flow)
    arrows = [
        (1.6, 5, 2.7, 5),
        (5.2, 5, 5.7, 5),
        (8.2, 5, 8.8, 5),
        (10.1, 5, 10.8, 5),
        (12, 4.5, 12, 3.8)
    ]

    for x1, y1, x2, y2 in arrows:
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.2, fc=colors['mlgray'], ec=colors['mlgray'], lw=2)

    # Retry loop (dashed)
    retry_path = FancyArrowPatch((11.5, 4.5), (7, 4.2), arrowstyle='->', lw=2,
                                 color=colors['mlred'], linestyle='dashed')
    ax.add_patch(retry_path)
    ax.text(9, 4, 'Retry on error', ha='center', fontsize=8, color=colors['mlred'], style='italic')

    # Annotations
    annotations = [
        (3.5, 3.5, "Contract\ndefines structure", colors['mlblue']),
        (6.5, 3.5, "API enforces\nschema", colors['mlgreen']),
        (12, 6, "Validation\ncatches errors", colors['mlorange'])
    ]

    for x, y, text, color in annotations:
        ax.text(x, y, text, ha='center', va='center', fontsize=7, color=color, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['lightgray'], edgecolor=color, alpha=0.7))

    ax.text(7, 0.8, 'Complete system: Schema → Enforcement → Validation → Recovery',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'production_architecture_unified')


def chart11_modern_applications_map():
    """Real companies using structured outputs (NO METRICS)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    # Title
    ax.text(5, 6.5, 'Modern Applications in Production (2024)', ha='center',
            fontsize=14, weight='bold', color=colors['mlpurple'])

    applications = [
        ("GitHub Copilot\nWorkspace", "Code\nGeneration", colors['mlblue'], 1.5, 4),
        ("Stripe\nInvoices", "Payment\nProcessing", colors['mlgreen'], 4.5, 4),
        ("Healthcare\nSystems", "Clinical\nNotes", colors['mlorange'], 7.5, 4),
        ("E-commerce\nPlatforms", "Product\nCatalogs", colors['mlred'], 4.5, 1.5)
    ]

    for company, use_case, color, x, y in applications:
        # Company box
        company_box = FancyBboxPatch((x - 0.9, y + 0.3), 1.8, 0.6, boxstyle="round,pad=0.05",
                                     edgecolor=color, facecolor=colors['lightgray'], linewidth=2)
        ax.add_patch(company_box)
        ax.text(x, y + 0.6, company, ha='center', va='center', fontsize=9, weight='bold', color=color)

        # Use case box
        use_box = FancyBboxPatch((x - 0.7, y - 0.6), 1.4, 0.5, boxstyle="round,pad=0.03",
                                 edgecolor=color, facecolor=colors['mllavender4'], linewidth=1.5)
        ax.add_patch(use_box)
        ax.text(x, y - 0.35, use_case, ha='center', va='center', fontsize=8)

    # Central concept
    center_box = FancyBboxPatch((4, 2.5), 2, 0.8, boxstyle="round,pad=0.1",
                                edgecolor=colors['mlpurple'], facecolor=colors['mllavender3'], linewidth=3)
    ax.add_patch(center_box)
    ax.text(5, 2.9, 'Structured Outputs\n+ Validation', ha='center', va='center',
            fontsize=10, weight='bold', color=colors['mlpurple'])

    # Arrows from center to applications
    arrow_coords = [(5, 3.3, 2, 4.3), (5, 3.3, 4.5, 3.7), (5, 3.3, 7, 4.3), (5, 2.7, 4.5, 2)]
    for x1, y1, x2, y2 in arrow_coords:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', lw=2, color=colors['mlgray'], alpha=0.5)
        ax.add_patch(arrow)

    ax.text(5, 0.3, 'Real production systems across diverse industries',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'modern_applications_map')


def chart12_common_pitfalls():
    """Common pitfalls with structured outputs - 3 categories"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Title
    ax.text(6, 7.3, 'Common Pitfalls with Structured Outputs', ha='center', fontsize=14, weight='bold', color=colors['mlpurple'])
    ax.text(6, 6.8, 'Predictable failure modes and how to prevent them', ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    # Three categories
    categories = [
        {
            'x': 1.5,
            'y': 4.5,
            'width': 3,
            'title': 'Design Pitfalls',
            'color': colors['mlred'],
            'items': [
                'Over-engineering\nsimple tasks',
                'Schema too rigid\nfor evolution',
                'Validation\ntoo strict',
                'No versioning\nstrategy'
            ]
        },
        {
            'x': 4.8,
            'y': 4.5,
            'width': 3,
            'title': 'Operational Pitfalls',
            'color': colors['mlorange'],
            'items': [
                'Cost\nblindness',
                'No fallback when\nvalidation fails',
                'Single point\nof failure',
                'Inadequate error\nlogging'
            ]
        },
        {
            'x': 8.1,
            'y': 4.5,
            'width': 3,
            'title': 'Data Pitfalls',
            'color': colors['mlblue'],
            'items': [
                'Ignoring\nedge cases',
                'Insufficient testing\non real data',
                'No monitoring\nfor drift',
                'Missing confidence\nscores'
            ]
        }
    ]

    for cat in categories:
        # Category box
        box = FancyBboxPatch((cat['x'], cat['y']), cat['width'], 3.5,
                            boxstyle="round,pad=0.1",
                            edgecolor=cat['color'], facecolor=colors['mllavender4'], linewidth=2.5)
        ax.add_patch(box)

        # Category title
        ax.text(cat['x'] + cat['width']/2, cat['y'] + 3.2, cat['title'],
                ha='center', va='center', fontsize=11, weight='bold', color=cat['color'])

        # Items
        y_start = cat['y'] + 2.6
        for i, item in enumerate(cat['items']):
            # Item box
            item_box = Rectangle((cat['x'] + 0.2, y_start - i*0.7), cat['width'] - 0.4, 0.6,
                                edgecolor=colors['mlgray'], facecolor='white', linewidth=1, alpha=0.7)
            ax.add_patch(item_box)

            # Item text
            ax.text(cat['x'] + cat['width']/2, y_start - i*0.7 + 0.3, item,
                   ha='center', va='center', fontsize=8, color=colors['mlpurple'])

    # Prevention strategies at bottom
    prevention_box = FancyBboxPatch((1.5, 0.8), 9.6, 3.2,
                                   boxstyle="round,pad=0.1",
                                   edgecolor=colors['mlgreen'], facecolor=colors['lightgray'], linewidth=2.5)
    ax.add_patch(prevention_box)

    ax.text(6, 3.5, 'Prevention Strategies', ha='center', fontsize=11, weight='bold', color=colors['mlgreen'])

    strategies = [
        ('Test Extensively', 'Use real messy data, not clean examples'),
        ('Design for Failure', 'Graceful degradation, fallback mechanisms'),
        ('Monitor Continuously', 'Track validation failures, schema drift'),
        ('Version Schemas', 'Enable evolution without breaking changes'),
        ('Start Simple', 'Add complexity only when needed'),
        ('Document Edge Cases', 'Known limitations and workarounds')
    ]

    y_pos = 2.8
    for i, (title, desc) in enumerate(strategies):
        col = i % 2
        row = i // 2
        x_pos = 2 + col * 5
        y = y_pos - row * 0.7

        ax.text(x_pos, y, f'{title}:', ha='left', fontsize=9, weight='bold', color=colors['mlgreen'])
        ax.text(x_pos + 0.1, y - 0.25, desc, ha='left', fontsize=7, color=colors['mlgray'])

    # Bottom caption
    ax.text(6, 0.3, 'Robust systems anticipate and prevent these pitfalls from day one',
            ha='center', fontsize=10, color=colors['mlgray'], style='italic')

    save_chart(fig, 'common_pitfalls_structured_outputs')


if __name__ == "__main__":
    print("Generating Week 8 V2 charts (conceptual, no fake data)...")
    print()

    chart1_unpredictability_problem()
    chart2_integration_challenge()
    chart3_prompt_engineering_patterns()
    chart4_success_examples()
    chart5_failure_pattern()
    chart6_human_consistency_methods()
    chart7_prompts_vs_schemas()
    chart8_three_layer_architecture()
    chart9_before_after_qualitative()
    chart10_production_architecture_unified()
    chart11_modern_applications_map()
    chart12_common_pitfalls()

    print()
    print("All 12 charts generated successfully!")
    print("Charts are conceptual visualizations - no unverified data used")
    print("NEW: Added common_pitfalls chart for pedagogical framework compliance")
