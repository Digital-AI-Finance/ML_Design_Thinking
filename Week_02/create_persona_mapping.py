"""
Week 2: Persona Mapping from Clusters
Transform cluster analysis into human-centered design personas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def load_and_cluster_data(n_clusters=5):
    """Load data and apply optimal clustering"""
    try:
        X = np.load('fintech_X.npy')
        y_true = np.load('fintech_y_true.npy')
        segments = np.load('fintech_segments.npy', allow_pickle=True)

        with open('fintech_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]

        # Preprocess
        X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        return X, X_scaled, cluster_labels, feature_names, segments
    except FileNotFoundError:
        print("Dataset not found. Generating...")
        import generate_fintech_dataset
        generate_fintech_dataset.main()
        return load_and_cluster_data(n_clusters)

def analyze_cluster_personas(X, cluster_labels, feature_names):
    """Convert clusters to detailed personas"""

    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = cluster_labels

    personas = []

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_means = cluster_data[feature_names].mean()
        cluster_size = len(cluster_data)

        # Define persona based on characteristics
        persona = {}
        persona['cluster_id'] = cluster_id
        persona['size'] = cluster_size
        persona['percentage'] = cluster_size / len(df) * 100

        # Analyze key behaviors
        persona['transaction_frequency'] = cluster_means['transaction_frequency']
        persona['transaction_volume'] = cluster_means['transaction_volume']
        persona['savings_behavior'] = cluster_means['savings_behavior']
        persona['credit_utilization'] = cluster_means['credit_utilization']
        persona['international_activity'] = cluster_means['international_activity']
        persona['support_contacts'] = cluster_means['support_contacts']
        persona['account_age'] = cluster_means['account_age']
        persona['session_duration'] = cluster_means['session_duration']

        # Create persona narrative
        if cluster_means['transaction_frequency'] > 8 and cluster_means['transaction_volume'] > 8000:
            persona['name'] = "Power Professional Patricia"
            persona['age_range'] = "28-45"
            persona['occupation'] = "Business Owner / Senior Manager"
            persona['tech_savvy'] = "High"
            persona['primary_need'] = "Efficiency & Advanced Features"
            persona['pain_points'] = ["Transaction limits", "Batch processing delays", "Limited API access"]
            persona['opportunities'] = ["Premium features", "Business accounts", "Priority support"]
            persona['quote'] = "I need banking that moves at the speed of my business"
            persona['emoji'] = "💼"
            persona['color'] = '#2E86AB'

        elif cluster_means['savings_behavior'] > 50 and cluster_means['credit_utilization'] < 30:
            persona['name'] = "Saver Samuel"
            persona['age_range'] = "35-60"
            persona['occupation'] = "Professional / Established Career"
            persona['tech_savvy'] = "Moderate"
            persona['primary_need'] = "Security & Growth"
            persona['pain_points'] = ["Low interest rates", "Complex investment options", "Unclear fees"]
            persona['opportunities'] = ["Savings goals", "Investment products", "Financial planning"]
            persona['quote'] = "I want my money to work as hard as I do"
            persona['emoji'] = "💰"
            persona['color'] = '#A23B72'

        elif cluster_means['international_activity'] > 0.4:
            persona['name'] = "Global Gina"
            persona['age_range'] = "25-40"
            persona['occupation'] = "Consultant / Digital Nomad"
            persona['tech_savvy'] = "Very High"
            persona['primary_need'] = "International Flexibility"
            persona['pain_points'] = ["Currency conversion fees", "ATM access abroad", "Time zone support"]
            persona['opportunities'] = ["Multi-currency accounts", "Travel benefits", "Global ATM network"]
            persona['quote'] = "My office is wherever I open my laptop"
            persona['emoji'] = "✈️"
            persona['color'] = '#F18F01'

        elif cluster_means['support_contacts'] > 2 and cluster_means['account_age'] < 90:
            persona['name'] = "Newcomer Nancy"
            persona['age_range'] = "18-30"
            persona['occupation'] = "Student / Entry-level"
            persona['tech_savvy'] = "Learning"
            persona['primary_need'] = "Guidance & Simplicity"
            persona['pain_points'] = ["Confusing features", "Hidden fees", "No credit history"]
            persona['opportunities'] = ["Onboarding tutorials", "Starter products", "Credit building"]
            persona['quote'] = "I just want banking to be simple and transparent"
            persona['emoji'] = "🎓"
            persona['color'] = '#C73E1D'

        else:
            persona['name'] = "Casual Chris"
            persona['age_range'] = "25-50"
            persona['occupation'] = "Various"
            persona['tech_savvy'] = "Average"
            persona['primary_need'] = "Convenience & Reliability"
            persona['pain_points'] = ["App complexity", "Unexpected downtime", "Generic offers"]
            persona['opportunities'] = ["Personalized recommendations", "Loyalty rewards", "Simplified UI"]
            persona['quote'] = "I want banking that just works"
            persona['emoji'] = "👤"
            persona['color'] = '#6C7B95'

        personas.append(persona)

    return personas

def create_persona_cards(personas):
    """Create visual persona cards"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, persona in enumerate(personas[:5]):
        ax = axes[idx]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        # Card background
        card = FancyBboxPatch((0.5, 0.5), 9, 11,
                              boxstyle="round,pad=0.1",
                              facecolor='white',
                              edgecolor=persona['color'],
                              linewidth=3)
        ax.add_patch(card)

        # Header with emoji and name
        header = FancyBboxPatch((0.5, 9), 9, 2,
                               boxstyle="round,pad=0.05",
                               facecolor=persona['color'],
                               alpha=0.3)
        ax.add_patch(header)

        # Persona details
        ax.text(5, 10.5, persona['emoji'], fontsize=40, ha='center')
        ax.text(5, 9.5, persona['name'], fontsize=16, fontweight='bold', ha='center')
        ax.text(5, 9, f"{persona['percentage']:.1f}% of users", fontsize=11, ha='center', style='italic')

        # Demographics
        y_pos = 8
        ax.text(1, y_pos, 'Demographics:', fontsize=11, fontweight='bold')
        y_pos -= 0.5
        ax.text(1.5, y_pos, f"• Age: {persona['age_range']}", fontsize=9)
        y_pos -= 0.4
        ax.text(1.5, y_pos, f"• Job: {persona['occupation']}", fontsize=9)
        y_pos -= 0.4
        ax.text(1.5, y_pos, f"• Tech: {persona['tech_savvy']}", fontsize=9)

        # Behaviors (with mini bar charts)
        y_pos = 6
        ax.text(1, y_pos, 'Key Behaviors:', fontsize=11, fontweight='bold')
        y_pos -= 0.5

        behaviors = [
            ('Transactions/day', persona['transaction_frequency'], 10),
            ('Volume ($1000s)', persona['transaction_volume']/1000, 20),
            ('Savings score', persona['savings_behavior'], 100),
            ('Support needs', persona['support_contacts'], 5)
        ]

        for behavior, value, max_val in behaviors:
            ax.text(1.5, y_pos, f"{behavior}:", fontsize=8)
            # Mini bar
            bar_width = 3 * (value / max_val)
            bar = FancyBboxPatch((5, y_pos - 0.1), bar_width, 0.2,
                                facecolor=persona['color'], alpha=0.5)
            ax.add_patch(bar)
            ax.text(8.5, y_pos, f"{value:.1f}", fontsize=8, ha='right')
            y_pos -= 0.4

        # Pain points
        y_pos = 3.5
        ax.text(1, y_pos, 'Pain Points:', fontsize=11, fontweight='bold', color='#d62728')
        y_pos -= 0.5
        for pain in persona['pain_points'][:2]:
            ax.text(1.5, y_pos, f"• {pain}", fontsize=8, wrap=True)
            y_pos -= 0.4

        # Opportunities
        y_pos = 2.2
        ax.text(1, y_pos, 'Opportunities:', fontsize=11, fontweight='bold', color='#2ca02c')
        y_pos -= 0.5
        for opp in persona['opportunities'][:2]:
            ax.text(1.5, y_pos, f"• {opp}", fontsize=8, wrap=True)
            y_pos -= 0.4

        # Quote
        ax.text(5, 0.8, f'"{persona["quote"]}"', fontsize=9,
               ha='center', style='italic', wrap=True, color='#666666')

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('FinTech User Personas from Cluster Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()

    return fig

def create_empathy_maps(personas):
    """Create empathy maps for each persona"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, persona in enumerate(personas[:5]):
        ax = axes[idx]
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.axis('off')

        # Draw empathy map quadrants
        ax.axhline(y=0, color='gray', linewidth=2)
        ax.axvline(x=0, color='gray', linewidth=2)

        # Add diagonal lines
        ax.plot([-5, 5], [5, -5], color='gray', linewidth=1, alpha=0.5)
        ax.plot([-5, 5], [-5, 5], color='gray', linewidth=1, alpha=0.5)

        # Center circle for persona
        center_circle = Circle((0, 0), 1.5, facecolor=persona['color'], alpha=0.3)
        ax.add_patch(center_circle)
        ax.text(0, 0.3, persona['emoji'], fontsize=30, ha='center', va='center')
        ax.text(0, -0.5, persona['name'].split()[0], fontsize=10, ha='center', fontweight='bold')

        # Think (top)
        ax.text(0, 4.5, 'THINKS', fontsize=11, fontweight='bold', ha='center')
        thinks = [
            f"Is this worth ${persona['transaction_volume']:.0f}/month?",
            "How secure is my money?",
            "Can I trust this platform?"
        ]
        for i, thought in enumerate(thinks[:2]):
            ax.text(0, 3.8 - i*0.5, thought, fontsize=8, ha='center', style='italic')

        # Feel (top right)
        ax.text(3.5, 3.5, 'FEELS', fontsize=11, fontweight='bold', ha='center')
        if persona['support_contacts'] > 2:
            feels = ["Confused", "Overwhelmed", "Curious"]
        elif persona['transaction_frequency'] > 8:
            feels = ["Empowered", "Busy", "Efficient"]
        else:
            feels = ["Comfortable", "Secure", "Satisfied"]

        for i, feel in enumerate(feels[:2]):
            ax.text(3.5, 2.8 - i*0.4, f"• {feel}", fontsize=8, ha='center')

        # See (left)
        ax.text(-3.5, 0, 'SEES', fontsize=11, fontweight='bold', ha='center', rotation=90)
        sees = ["Competitors' offers", "Social media ads", "Friend recommendations"]
        for i, see in enumerate(sees[:2]):
            ax.text(-3, -1 + i*0.5, f"• {see[:15]}...", fontsize=7)

        # Say (right)
        ax.text(3.5, 0, 'SAYS', fontsize=11, fontweight='bold', ha='center', rotation=270)
        ax.text(2.5, 0, f'"{persona["quote"][:30]}..."', fontsize=7, style='italic', rotation=270, ha='center')

        # Do (bottom left)
        ax.text(-2.5, -3.5, 'DOES', fontsize=11, fontweight='bold', ha='center')
        does = [
            f"{persona['transaction_frequency']:.0f} transactions/day",
            f"Uses app {persona['session_duration']:.0f} min/session"
        ]
        for i, do in enumerate(does):
            ax.text(-2.5, -4 - i*0.3, do, fontsize=7, ha='center')

        # Hear (bottom right)
        ax.text(2.5, -3.5, 'HEARS', fontsize=11, fontweight='bold', ha='center')
        hears = ["Reviews online", "Expert opinions", "News about fintech"]
        for i, hear in enumerate(hears[:2]):
            ax.text(2.5, -4 - i*0.3, f"• {hear}", fontsize=7, ha='center')

        # Pain points (bottom)
        ax.text(0, -4.5, 'PAINS', fontsize=10, fontweight='bold', ha='center', color='#d62728')
        ax.text(0, -4.8, persona['pain_points'][0][:40], fontsize=7, ha='center')

        # Gains (between sections)
        gains_text = persona['opportunities'][0] if persona['opportunities'] else "Better experience"
        ax.text(0, 2, f"GAIN: {gains_text[:30]}", fontsize=8, ha='center',
               bbox=dict(boxstyle="round", facecolor='#90EE90', alpha=0.3))

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Empathy Maps: Understanding User Psychology', fontsize=18, fontweight='bold')
    plt.tight_layout()

    return fig

def create_journey_maps(personas):
    """Create customer journey maps for different personas"""

    # Define journey stages
    stages = ['Awareness', 'Consideration', 'Onboarding', 'Active Use', 'Loyalty']

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, persona in enumerate(personas[:5]):
        ax = axes[idx]

        # Generate journey data based on persona characteristics
        if persona['tech_savvy'] == "High" or persona['tech_savvy'] == "Very High":
            satisfaction = [3, 4, 4, 5, 4]  # Generally positive
            effort = [2, 2, 1, 1, 2]  # Low effort
        elif persona['support_contacts'] > 2:
            satisfaction = [3, 3, 2, 3, 4]  # Struggles early
            effort = [3, 4, 5, 3, 2]  # High initial effort
        else:
            satisfaction = [3, 3, 3, 4, 4]  # Moderate throughout
            effort = [2, 3, 3, 2, 2]  # Moderate effort

        x = np.arange(len(stages))

        # Create dual axis
        ax2 = ax.twinx()

        # Plot satisfaction (line)
        line1 = ax.plot(x, satisfaction, 'o-', color=persona['color'],
                       linewidth=3, markersize=10, label='Satisfaction')

        # Plot effort (bars)
        bars = ax2.bar(x, effort, alpha=0.3, color='red', width=0.5, label='Effort')

        # Customize axes
        ax.set_xlabel('Journey Stage', fontsize=10)
        ax.set_ylabel('Satisfaction', fontsize=10, color=persona['color'])
        ax2.set_ylabel('Effort Required', fontsize=10, color='red')

        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 5.5)
        ax2.set_ylim(0, 5.5)

        # Add title
        ax.set_title(f"{persona['emoji']} {persona['name']}'s Journey",
                    fontsize=11, fontweight='bold')

        # Add touchpoint annotations
        touchpoints = {
            0: "Social media ad",
            1: "Website comparison",
            2: "App download",
            3: "Daily transactions",
            4: "Premium upgrade"
        }

        for stage_idx, touchpoint in touchpoints.items():
            if stage_idx < len(stages):
                ax.annotate(touchpoint, xy=(stage_idx, satisfaction[stage_idx]),
                          xytext=(stage_idx, satisfaction[stage_idx] + 0.5),
                          fontsize=7, ha='center',
                          arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f9f9f9')

        # Add legend
        if idx == 0:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=8)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Customer Journey Maps by Persona', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def create_design_opportunities_matrix(personas):
    """Create a matrix of design opportunities by persona"""

    # Define opportunity categories
    opportunities = [
        'Mobile Features',
        'Security Options',
        'Savings Tools',
        'International Services',
        'Customer Support',
        'Analytics Dashboard',
        'Social Features',
        'Rewards Program'
    ]

    # Create importance matrix (personas x opportunities)
    importance_matrix = np.zeros((5, len(opportunities)))

    for i, persona in enumerate(personas[:5]):
        if 'Power' in persona['name']:
            importance_matrix[i] = [5, 4, 3, 3, 2, 5, 2, 3]  # Power user priorities
        elif 'Saver' in persona['name']:
            importance_matrix[i] = [3, 5, 5, 2, 3, 4, 1, 4]  # Saver priorities
        elif 'Global' in persona['name']:
            importance_matrix[i] = [5, 3, 2, 5, 3, 3, 3, 4]  # Global user priorities
        elif 'Newcomer' in persona['name']:
            importance_matrix[i] = [4, 3, 3, 1, 5, 2, 3, 2]  # Newcomer priorities
        else:
            importance_matrix[i] = [4, 4, 3, 2, 3, 3, 2, 3]  # Casual user priorities

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(importance_matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

    # Set ticks
    ax.set_xticks(np.arange(len(opportunities)))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(opportunities, rotation=45, ha='right')
    ax.set_yticklabels([p['name'].split()[0] + ' ' + p['emoji'] for p in personas[:5]])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Priority Level', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(5):
        for j in range(len(opportunities)):
            value = importance_matrix[i, j]
            color = 'white' if value > 3 else 'black'
            text = ax.text(j, i, f'{value:.0f}', ha='center', va='center', color=color, fontweight='bold')

    ax.set_title('Design Opportunity Priority Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature/Service Categories', fontsize=11)
    ax.set_ylabel('User Personas', fontsize=11)

    # Add grid
    ax.set_xticks(np.arange(len(opportunities) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(6) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    return fig

def create_persona_comparison_radar(personas):
    """Create radar chart comparing persona characteristics"""

    categories = ['Tech Savvy', 'Transaction\nVolume', 'Savings\nFocus',
                 'Support\nNeeds', 'International\nActivity', 'Account\nMaturity']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for idx, persona in enumerate(personas[:5]):
        ax = axes[idx]

        # Normalize values for radar chart (0-5 scale)
        values = [
            {'High': 5, 'Very High': 5, 'Moderate': 3, 'Learning': 2, 'Average': 3}.get(persona['tech_savvy'], 3),
            min(5, persona['transaction_volume'] / 4000),  # Scale to 0-5
            min(5, persona['savings_behavior'] / 20),  # Scale to 0-5
            min(5, persona['support_contacts']),  # Already 0-5 range
            min(5, persona['international_activity'] * 10),  # Scale to 0-5
            min(5, persona['account_age'] / 150)  # Scale to 0-5
        ]

        # Number of categories
        N = len(categories)

        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=persona['color'])
        ax.fill(angles, values, alpha=0.25, color=persona['color'])

        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

        # Set y-axis limits and labels
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=7)

        # Add title
        ax.set_title(f"{persona['emoji']} {persona['name']}", fontsize=10, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Persona Characteristics Radar Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def main():
    """Generate all persona mapping visualizations"""

    print("Loading data and creating personas...")
    X, X_scaled, cluster_labels, feature_names, true_segments = load_and_cluster_data(n_clusters=5)

    # Analyze clusters to create personas
    print("Analyzing clusters and building personas...")
    personas = analyze_cluster_personas(X, cluster_labels, feature_names)

    # Print persona summaries
    print("\nPersona Summaries:")
    print("=" * 60)
    for persona in personas:
        print(f"\n{persona['emoji']} {persona['name']}")
        print(f"   Size: {persona['size']} users ({persona['percentage']:.1f}%)")
        print(f"   Primary Need: {persona['primary_need']}")
        print(f"   Key Pain Point: {persona['pain_points'][0]}")

    # Create visualizations
    print("\nGenerating persona visualizations...")

    # 1. Persona cards
    print("Creating persona cards...")
    fig1 = create_persona_cards(personas)
    fig1.savefig('personas_cards.png', dpi=300, bbox_inches='tight')
    fig1.savefig('personas_cards.pdf', bbox_inches='tight')

    # 2. Empathy maps
    print("Creating empathy maps...")
    fig2 = create_empathy_maps(personas)
    fig2.savefig('personas_empathy_maps.png', dpi=300, bbox_inches='tight')
    fig2.savefig('personas_empathy_maps.pdf', bbox_inches='tight')

    # 3. Journey maps
    print("Creating journey maps...")
    fig3 = create_journey_maps(personas)
    fig3.savefig('personas_journey_maps.png', dpi=300, bbox_inches='tight')
    fig3.savefig('personas_journey_maps.pdf', bbox_inches='tight')

    # 4. Design opportunities matrix
    print("Creating design opportunities matrix...")
    fig4 = create_design_opportunities_matrix(personas)
    fig4.savefig('personas_opportunities_matrix.png', dpi=300, bbox_inches='tight')
    fig4.savefig('personas_opportunities_matrix.pdf', bbox_inches='tight')

    # 5. Persona comparison radar
    print("Creating persona comparison radar charts...")
    fig5 = create_persona_comparison_radar(personas)
    fig5.savefig('personas_radar_comparison.png', dpi=300, bbox_inches='tight')
    fig5.savefig('personas_radar_comparison.pdf', bbox_inches='tight')

    # Save persona data to CSV
    personas_df = pd.DataFrame(personas)
    personas_df.to_csv('personas_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("PERSONA MAPPING COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - personas_cards.png/pdf - Visual persona cards")
    print("  - personas_empathy_maps.png/pdf - Empathy maps")
    print("  - personas_journey_maps.png/pdf - Customer journeys")
    print("  - personas_opportunities_matrix.png/pdf - Design priorities")
    print("  - personas_radar_comparison.png/pdf - Characteristic comparison")
    print("  - personas_summary.csv - Persona data export")
    print("\nThese personas bridge data science with human-centered design!")

if __name__ == "__main__":
    main()