import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import beta, binom
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def save_chart(fig, name):
    fig.savefig(f'../charts/{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'../charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Generated {name}')

def chart1_ab_testing_fundamentals():
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    control = np.random.binomial(1, 0.05, 1000)
    treatment = np.random.binomial(1, 0.06, 1000)

    ax1.bar(['Control\n(5.0% CTR)', 'Treatment\n(6.0% CTR)'],
            [control.mean()*100, treatment.mean()*100],
            color=['#3176b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Click-Through Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A/B Test Results', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 8])
    ax1.grid(axis='y', alpha=0.3)

    for i, v in enumerate([control.mean()*100, treatment.mean()*100]):
        ax1.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    days = np.arange(1, 15)
    control_conversion = 0.05 + np.random.normal(0, 0.003, len(days))
    treatment_conversion = 0.06 + np.random.normal(0, 0.003, len(days))

    ax2.plot(days, control_conversion*100, marker='o', label='Control', linewidth=2, color='#3176b4')
    ax2.plot(days, treatment_conversion*100, marker='s', label='Treatment', linewidth=2, color='#ff7f0e')
    ax2.axhline(y=5.0, color='#3176b4', linestyle='--', alpha=0.5)
    ax2.axhline(y=6.0, color='#ff7f0e', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Temporal Stability Check', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(fig, 'ab_testing_fundamentals')

def chart2_statistical_power_curves():
    fig, ax = plt.subplots(figsize=(10, 6))

    sample_sizes = np.arange(100, 10000, 100)
    effect_sizes = [0.01, 0.02, 0.05, 0.10]

    for effect in effect_sizes:
        p1 = 0.05
        p2 = p1 + effect
        pooled_p = (p1 + p2) / 2

        powers = []
        for n in sample_sizes:
            se = np.sqrt(2 * pooled_p * (1 - pooled_p) / n)
            z_crit = 1.96
            z_beta = (effect - z_crit * se) / se
            power = stats.norm.cdf(z_beta)
            powers.append(power)

        ax.plot(sample_sizes, powers, linewidth=2.5,
                label=f'Effect: {effect*100:.0f}% → {(p1+effect)*100:.0f}% CTR')

    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% Power Target')
    ax.set_xlabel('Sample Size per Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Statistical Power', fontsize=12, fontweight='bold')
    ax.set_title('Power Curves: Sample Size vs Detection Ability', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    save_chart(fig, 'statistical_power_curves')

def chart3_type_i_ii_errors():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-4, 4, 1000)
    null_dist = stats.norm.pdf(x, 0, 1)
    alt_dist = stats.norm.pdf(x, 2, 1)

    crit_value = 1.96

    ax1.fill_between(x, 0, null_dist, where=(x > crit_value),
                     color='red', alpha=0.5, label='Type I Error (α = 5%)')
    ax1.plot(x, null_dist, 'b-', linewidth=2, label='Null Distribution (H0)')
    ax1.axvline(crit_value, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Test Statistic', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('Type I Error: False Positive', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, 0, alt_dist, where=(x < crit_value),
                     color='orange', alpha=0.5, label='Type II Error (β = 20%)')
    ax2.fill_between(x, 0, alt_dist, where=(x >= crit_value),
                     color='green', alpha=0.5, label='Power (1-β = 80%)')
    ax2.plot(x, alt_dist, 'g-', linewidth=2, label='Alternative Distribution (H1)')
    ax2.axvline(crit_value, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Test Statistic', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title('Type II Error vs Power', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(fig, 'type_i_ii_errors')

def chart4_confidence_intervals():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(10, 8))

    true_mean = 0.06
    n_experiments = 30
    sample_size = 1000

    means = []
    cis_lower = []
    cis_upper = []

    for i in range(n_experiments):
        data = np.random.binomial(1, true_mean, sample_size)
        mean = data.mean()
        se = np.sqrt(mean * (1 - mean) / sample_size)
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se

        means.append(mean)
        cis_lower.append(ci_lower)
        cis_upper.append(ci_upper)

    colors = ['green' if (l <= true_mean <= u) else 'red'
              for l, u in zip(cis_lower, cis_upper)]

    for i in range(n_experiments):
        ax.plot([cis_lower[i]*100, cis_upper[i]*100], [i, i],
                color=colors[i], linewidth=2, alpha=0.7)
        ax.scatter(means[i]*100, i, color=colors[i], s=50, zorder=3)

    ax.axvline(true_mean*100, color='blue', linestyle='--', linewidth=3,
               label='True Value (6.0%)')
    ax.set_xlabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Experiment Number', fontsize=12, fontweight='bold')
    ax.set_title('95% Confidence Intervals: 30 Repeated Experiments',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='x')

    n_capture = sum([1 for c in colors if c == 'green'])
    ax.text(0.98, 0.98, f'{n_capture}/30 capture true value ({n_capture/30*100:.0f}%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_chart(fig, 'confidence_intervals')

def chart5_bayesian_ab_posterior():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    conversions_a = 50
    visitors_a = 1000
    conversions_b = 65
    visitors_b = 1000

    x = np.linspace(0, 0.15, 1000)

    prior_alpha, prior_beta = 1, 1
    posterior_a = beta(prior_alpha + conversions_a, prior_beta + visitors_a - conversions_a)
    posterior_b = beta(prior_alpha + conversions_b, prior_beta + visitors_b - conversions_b)

    ax1.plot(x*100, posterior_a.pdf(x), linewidth=2.5, label='Control (50/1000)', color='#3176b4')
    ax1.plot(x*100, posterior_b.pdf(x), linewidth=2.5, label='Treatment (65/1000)', color='#ff7f0e')
    ax1.axvline(posterior_a.mean()*100, color='#3176b4', linestyle='--', alpha=0.5)
    ax1.axvline(posterior_b.mean()*100, color='#ff7f0e', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('Posterior Distributions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    n_samples = 10000
    samples_a = posterior_a.rvs(n_samples)
    samples_b = posterior_b.rvs(n_samples)
    prob_b_better = (samples_b > samples_a).mean()

    lift = (samples_b - samples_a) / samples_a * 100
    ax2.hist(lift, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax2.set_xlabel('Relative Lift (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'P(Treatment > Control) = {prob_b_better:.1%}',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    save_chart(fig, 'bayesian_ab_posterior')

def chart6_sequential_testing_boundaries():
    fig, ax = plt.subplots(figsize=(10, 6))

    looks = np.arange(1, 11)
    alpha_total = 0.05

    obrien_fleming = 1.96 * np.sqrt(10 / looks)
    pocock = np.full_like(looks, 2.36, dtype=float)
    naive = np.full_like(looks, 1.96, dtype=float)

    ax.plot(looks, obrien_fleming, marker='o', linewidth=2.5,
            label="O'Brien-Fleming (Conservative early)", color='blue')
    ax.plot(looks, pocock, marker='s', linewidth=2.5,
            label='Pocock (Constant)', color='green')
    ax.plot(looks, naive, marker='^', linewidth=2.5,
            label='Naive (5% each look)', color='red', linestyle='--')

    ax.set_xlabel('Analysis Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z-score Threshold', fontsize=12, fontweight='bold')
    ax.set_title('Sequential Testing Boundaries (α = 5% total)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(looks)

    save_chart(fig, 'sequential_testing_boundaries')

def chart7_multi_armed_bandit_exploration():
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    true_rates = [0.05, 0.06, 0.055, 0.07]
    n_arms = len(true_rates)
    n_rounds = 1000

    epsilon = 0.1
    pulls = np.zeros(n_arms)
    rewards = np.zeros(n_arms)

    pull_history = [[] for _ in range(n_arms)]
    cumulative_reward = []
    total_reward = 0

    for t in range(n_rounds):
        if np.random.random() < epsilon or t < n_arms:
            arm = np.random.randint(n_arms)
        else:
            avg_rewards = rewards / (pulls + 1e-5)
            arm = np.argmax(avg_rewards)

        pulls[arm] += 1
        reward = np.random.binomial(1, true_rates[arm])
        rewards[arm] += reward
        total_reward += reward
        cumulative_reward.append(total_reward)

        for i in range(n_arms):
            pull_history[i].append(pulls[i])

    rounds = np.arange(1, n_rounds + 1)
    for i in range(n_arms):
        ax1.plot(rounds, pull_history[i], linewidth=2,
                label=f'Arm {i+1} ({true_rates[i]*100:.1f}% CTR)')

    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Pulls', fontsize=12, fontweight='bold')
    ax1.set_title('Epsilon-Greedy Exploration (ε = 10%)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    best_arm = np.argmax(true_rates)
    optimal_reward = np.arange(1, n_rounds + 1) * true_rates[best_arm]
    regret = optimal_reward - np.array(cumulative_reward)

    ax2.plot(rounds, regret, linewidth=2.5, color='red')
    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Regret', fontsize=12, fontweight='bold')
    ax2.set_title('Exploration Cost vs Optimal Strategy', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(fig, 'multi_armed_bandit_exploration')

def chart8_thompson_sampling_demo():
    np.random.seed(42)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    true_rates = [0.05, 0.07]
    n_rounds = 100

    alpha = [1, 1]
    beta_param = [1, 1]

    snapshots = [10, 30, 50, 100]
    axes = [ax1, ax2, ax3, ax4]

    for t in range(1, n_rounds + 1):
        theta_samples = [np.random.beta(alpha[i], beta_param[i]) for i in range(2)]
        arm = np.argmax(theta_samples)

        reward = np.random.binomial(1, true_rates[arm])
        alpha[arm] += reward
        beta_param[arm] += 1 - reward

        if t in snapshots:
            ax = axes[snapshots.index(t)]
            x = np.linspace(0, 0.15, 500)

            for i in range(2):
                dist = beta(alpha[i], beta_param[i])
                ax.plot(x*100, dist.pdf(x), linewidth=2.5,
                       label=f'Arm {i+1} (α={alpha[i]}, β={beta_param[i]})')
                ax.axvline(true_rates[i]*100, color=f'C{i}',
                          linestyle='--', alpha=0.5)

            ax.set_xlabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
            ax.set_title(f'Round {t}: Belief Distributions', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    save_chart(fig, 'thompson_sampling_demo')

def chart9_causal_inference_dag():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    nodes = {
        'Treatment': (2, 7),
        'Outcome': (8, 7),
        'Confounder': (5, 9),
        'Mediator': (5, 5),
        'Collider': (5, 3)
    }

    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.5, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold')

    arrows = [
        ('Treatment', 'Outcome', 'Direct Causal', 'green'),
        ('Confounder', 'Treatment', 'Backdoor', 'red'),
        ('Confounder', 'Outcome', 'Backdoor', 'red'),
        ('Treatment', 'Mediator', 'Front-door', 'blue'),
        ('Mediator', 'Outcome', 'Front-door', 'blue'),
        ('Treatment', 'Collider', 'Spurious', 'orange'),
        ('Outcome', 'Collider', 'Spurious', 'orange')
    ]

    for start, end, label, color in arrows:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length
        dy_norm = dy / length

        start_x = x1 + 0.5 * dx_norm
        start_y = y1 + 0.5 * dy_norm
        end_x = x2 - 0.5 * dx_norm
        end_y = y2 - 0.5 * dy_norm

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=color))

    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=3, label='Causal path'),
        plt.Line2D([0], [0], color='red', lw=3, label='Confounding (bias)'),
        plt.Line2D([0], [0], color='blue', lw=3, label='Mediation'),
        plt.Line2D([0], [0], color='orange', lw=3, label='Spurious association')
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=11, ncol=2)

    ax.set_title('Directed Acyclic Graph: Randomization Breaks Confounding',
                fontsize=14, fontweight='bold')

    save_chart(fig, 'causal_inference_dag')

def chart10_experiment_velocity_dashboard():
    np.random.seed(42)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    experiments = [12, 15, 18, 22, 25, 28]
    ax1.bar(months, experiments, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.plot(months, experiments, marker='o', color='red', linewidth=2, markersize=8)
    ax1.set_ylabel('Experiments Launched', fontsize=11, fontweight='bold')
    ax1.set_title('Monthly Experiment Velocity', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(experiments):
        ax1.text(i, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')

    ax2 = fig.add_subplot(gs[1, 0])
    win_rates = [0.35, 0.42, 0.38, 0.45, 0.40, 0.48]
    ax2.plot(months, win_rates, marker='s', linewidth=2.5, color='green', markersize=8)
    ax2.axhline(y=0.4, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target: 40%')
    ax2.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Experiment Success Rate', fontsize=13, fontweight='bold')
    ax2.set_ylim([0.3, 0.55])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    avg_duration = [14, 12, 10, 9, 8, 7]
    ax3.plot(months, avg_duration, marker='^', linewidth=2.5, color='purple', markersize=8)
    ax3.set_ylabel('Days', fontsize=11, fontweight='bold')
    ax3.set_title('Average Experiment Duration', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :])
    categories = ['Product\nFeatures', 'Pricing', 'UX/UI', 'Marketing', 'Infrastructure']
    experiments_by_cat = [45, 28, 35, 22, 15]
    wins_by_cat = [18, 10, 16, 8, 7]

    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, experiments_by_cat, width, label='Total Tests',
           color='lightblue', edgecolor='black')
    ax4.bar(x + width/2, wins_by_cat, width, label='Wins',
           color='lightgreen', edgecolor='black')
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Experiments by Category (6 months)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    save_chart(fig, 'experiment_velocity_dashboard')

def chart11_metric_tree_example():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    nodes = {
        'Revenue': (5, 11, '#ff6b6b', 'North Star'),
        'ARPU': (2.5, 9, '#4ecdc4', 'Primary'),
        'Conversion': (7.5, 9, '#4ecdc4', 'Primary'),
        'Sessions': (1, 7, '#95e1d3', 'Secondary'),
        'Purchases': (4, 7, '#95e1d3', 'Secondary'),
        'CTR': (6, 7, '#95e1d3', 'Secondary'),
        'Cart Adds': (9, 7, '#95e1d3', 'Secondary'),
        'Page Load': (0.5, 5, '#f9ca24', 'Guardrail'),
        'Engagement': (2, 5, '#f9ca24', 'Guardrail'),
        'Error Rate': (4.5, 5, '#f9ca24', 'Guardrail'),
        'Bounce Rate': (7, 5, '#f9ca24', 'Guardrail'),
        'Support': (9.5, 5, '#f9ca24', 'Guardrail')
    }

    for name, (x, y, color, level) in nodes.items():
        size = 0.6 if level == 'North Star' else 0.5 if level == 'Primary' else 0.4
        circle = plt.Circle((x, y), size, color=color, ec='black', linewidth=2, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

    connections = [
        ('Revenue', 'ARPU'),
        ('Revenue', 'Conversion'),
        ('ARPU', 'Sessions'),
        ('ARPU', 'Purchases'),
        ('Conversion', 'CTR'),
        ('Conversion', 'Cart Adds'),
        ('Sessions', 'Page Load'),
        ('Sessions', 'Engagement'),
        ('CTR', 'Error Rate'),
        ('Cart Adds', 'Bounce Rate'),
        ('Purchases', 'Support')
    ]

    for start, end in connections:
        x1, y1 = nodes[start][:2]
        x2, y2 = nodes[end][:2]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.5)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b',
                  markersize=12, label='North Star Metric'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4',
                  markersize=12, label='Primary Metrics'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95e1d3',
                  markersize=12, label='Secondary Metrics'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f9ca24',
                  markersize=12, label='Guardrail Metrics')
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=11, ncol=4)

    ax.set_title('Metric Tree: Hierarchical Decomposition',
                fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'metric_tree_example')

def chart12_canary_deployment_timeline():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    stages = ['Canary\n1%', 'Monitor\n5%', 'Expand\n25%', 'Scale\n50%', 'Full\n100%']
    hours = [0, 2, 6, 12, 24]
    traffic = [1, 5, 25, 50, 100]

    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#00d2d3']
    ax1.bar(stages, traffic, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Traffic Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Progressive Canary Rollout', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, (v, h) in enumerate(zip(traffic, hours)):
        ax1.text(i, v + 3, f'{v}%\n({h}h)', ha='center', fontsize=10, fontweight='bold')

    timeline = np.arange(0, 25)
    error_rate = np.concatenate([
        np.random.normal(0.1, 0.02, 2),
        np.random.normal(0.15, 0.03, 4),
        np.random.normal(0.12, 0.02, 6),
        np.random.normal(0.11, 0.02, 6),
        np.random.normal(0.1, 0.02, 7)
    ])

    latency_p95 = np.concatenate([
        np.random.normal(150, 10, 2),
        np.random.normal(180, 15, 4),
        np.random.normal(165, 10, 6),
        np.random.normal(155, 8, 6),
        np.random.normal(150, 10, 7)
    ])

    ax2_twin = ax2.twinx()

    l1 = ax2.plot(timeline, error_rate, marker='o', linewidth=2,
                 label='Error Rate', color='red', markersize=5)
    l2 = ax2_twin.plot(timeline, latency_p95, marker='s', linewidth=2,
                      label='P95 Latency', color='blue', markersize=5)

    for h in [2, 6, 12, 24]:
        ax2.axvline(h, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    ax2.axhline(0.2, color='red', linestyle='--', alpha=0.3, label='Error threshold')
    ax2_twin.axhline(200, color='blue', linestyle='--', alpha=0.3, label='Latency threshold')

    ax2.set_xlabel('Hours Since Deployment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold', color='red')
    ax2_twin.set_ylabel('P95 Latency (ms)', fontsize=12, fontweight='bold', color='blue')
    ax2.set_title('Real-Time Guardrail Monitoring', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=10)

    plt.tight_layout()
    save_chart(fig, 'canary_deployment_timeline')

def chart13_simpsons_paradox():
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    n_new = 200
    n_power = 800

    new_control = np.random.binomial(1, 0.10, n_new)
    new_treatment = np.random.binomial(1, 0.12, n_new)
    power_control = np.random.binomial(1, 0.08, n_power)
    power_treatment = np.random.binomial(1, 0.09, n_power)

    segments = ['New Users', 'Power Users']
    control_rates = [new_control.mean()*100, power_control.mean()*100]
    treatment_rates = [new_treatment.mean()*100, power_treatment.mean()*100]

    x = np.arange(len(segments))
    width = 0.35

    ax1.bar(x - width/2, control_rates, width, label='Control',
           color='#3176b4', edgecolor='black', alpha=0.7)
    ax1.bar(x + width/2, treatment_rates, width, label='Treatment',
           color='#ff7f0e', edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('By Segment: Treatment Wins Both', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(segments)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    for i in range(len(segments)):
        for j, rate in enumerate([control_rates[i], treatment_rates[i]]):
            ax1.text(i + (j - 0.5) * width, rate + 0.3, f'{rate:.1f}%',
                    ha='center', fontsize=10, fontweight='bold')

    overall_control = np.concatenate([new_control, power_control])
    overall_treatment = np.concatenate([new_treatment, power_treatment])

    overall_rates = ['Overall']
    overall_control_rate = overall_control.mean() * 100
    overall_treatment_rate = overall_treatment.mean() * 100

    ax2.bar([0 - width/2], [overall_control_rate], width, label='Control',
           color='#3176b4', edgecolor='black', alpha=0.7)
    ax2.bar([0 + width/2], [overall_treatment_rate], width, label='Treatment',
           color='#ff7f0e', edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title("Simpson's Paradox: Control Wins Overall!",
                 fontsize=14, fontweight='bold')
    ax2.set_xticks([0])
    ax2.set_xticklabels(overall_rates)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 12])

    ax2.text(0 - width/2, overall_control_rate + 0.3, f'{overall_control_rate:.1f}%',
            ha='center', fontsize=10, fontweight='bold')
    ax2.text(0 + width/2, overall_treatment_rate + 0.3, f'{overall_treatment_rate:.1f}%',
            ha='center', fontsize=10, fontweight='bold')

    ax2.text(0.5, 0.95, 'Why? Imbalanced sample sizes\nPower users: 80% vs 80%\nNew users: 20% vs 20%',
            transform=ax2.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    save_chart(fig, 'simpsons_paradox')

def chart14_experiment_decision_matrix():
    fig, ax = plt.subplots(figsize=(10, 8))

    categories = ['Strong Win\n(p<0.01, +15%)', 'Marginal Win\n(p=0.04, +3%)',
                  'Null\n(p=0.3, +1%)', 'Guardrail Fail\n(Latency +50%)',
                  'Strong Loss\n(p<0.01, -10%)']
    decisions = ['SHIP', 'ITERATE', 'STOP', 'ROLLBACK', 'STOP']
    colors = ['#2ecc71', '#f39c12', '#95a5a6', '#e74c3c', '#c0392b']

    y_positions = np.arange(len(categories))

    for i, (cat, dec, col) in enumerate(zip(categories, decisions, colors)):
        ax.barh(i, 1, color=col, alpha=0.7, edgecolor='black', linewidth=2)
        ax.text(0.5, i, f'{dec}', ha='center', va='center',
               fontsize=16, fontweight='bold', color='white')
        ax.text(-0.05, i, cat, ha='right', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim([-0.5, 1.2])
    ax.set_ylim([-0.5, len(categories)-0.5])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_title('Experiment Decision Framework', fontsize=16, fontweight='bold', pad=20)

    ax.text(0.5, -1.2, 'Always consider: Statistical significance + Practical significance + Guardrails',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_chart(fig, 'experiment_decision_matrix')

def chart15_continuous_improvement_loop():
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    stages = [
        ('OBSERVE\nMetric Degradation', 0, 1.2, '#e74c3c'),
        ('HYPOTHESIZE\nRoot Cause', 1.04, 0.6, '#f39c12'),
        ('DESIGN\nExperiment', 1.04, -0.6, '#f1c40f'),
        ('IMPLEMENT\nA/B Test', 0, -1.2, '#2ecc71'),
        ('ANALYZE\nResults', -1.04, -0.6, '#3498db'),
        ('DECIDE\nShip/Iterate/Stop', -1.04, 0.6, '#9b59b6')
    ]

    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 7)

    for i, (stage, x, y, color) in enumerate(stages):
        circle = plt.Circle((x, y), 0.35, color=color, ec='black', linewidth=3, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, stage, ha='center', va='center', fontsize=10,
               fontweight='bold', color='white')

        next_i = (i + 1) % len(stages)
        next_x, next_y = stages[next_i][1], stages[next_i][2]

        dx = next_x - x
        dy = next_y - y
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length
        dy_norm = dy / length

        start_x = x + 0.35 * dx_norm
        start_y = y + 0.35 * dy_norm
        end_x = next_x - 0.35 * dx_norm
        end_y = next_y - 0.35 * dy_norm

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=4, color='black'))

    center_text = ax.text(0, 0, 'CONTINUOUS\nIMPROVEMENT\nLOOP',
                         ha='center', va='center', fontsize=14, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='lightgray',
                                 edgecolor='black', linewidth=2))

    ax.set_title('The Iteration Cycle: Never Stop Learning',
                fontsize=16, fontweight='bold', pad=30)

    ax.text(0, -1.8, 'Average cycle time: 7-14 days | Target: 100+ experiments/year',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_chart(fig, 'continuous_improvement_loop')

def chart16_validation_method_decision():
    """Decision tree for when to use A/B testing vs alternative validation methods"""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'When to Use A/B Testing: Decision Framework',
            ha='center', fontsize=16, fontweight='bold')

    # Root question
    root_box = FancyBboxPatch((5.5, 8.2), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(root_box)
    ax.text(7, 8.6, 'How should we validate\nthis change?', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Three main branches based on traffic
    # Branch 1: High Traffic → A/B Testing
    ax.arrow(6.5, 8.2, -2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    high_box = FancyBboxPatch((2.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor='#2ecc71', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(high_box)
    ax.text(4, 6.2, 'HIGH TRAFFIC\n10K+ users/day', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax.text(4, 5.4, 'Need causal rigor\nCan afford 1-2 week wait\nReversible decision\nMeasurable outcome',
            ha='center', va='top', fontsize=8)

    ax.arrow(4, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_high = FancyBboxPatch((2.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                 facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(result_high)
    ax.text(4, 4.5, 'A/B TESTING', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(4, 4.1, 'Classical or Bayesian\n1-2 week experiment\nStatistical rigor\nProduction environment',
            ha='center', va='center', fontsize=8)

    # Branch 2: Medium Traffic → Adaptive Methods
    ax.arrow(7, 8.2, 0, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    med_box = FancyBboxPatch((5.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#f39c12', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(med_box)
    ax.text(7, 6.2, 'MEDIUM TRAFFIC\n1K-10K users/day', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax.text(7, 5.4, 'Need faster decisions\nCan adapt in real-time\nExploration-exploitation\nOptimization goal',
            ha='center', va='top', fontsize=8)

    ax.arrow(7, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_med = FancyBboxPatch((5.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(result_med)
    ax.text(7, 4.5, 'ADAPTIVE METHODS', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(7, 4.1, 'Multi-armed bandits\nThompson sampling\nSequential testing\nFaster convergence',
            ha='center', va='center', fontsize=8)

    # Branch 3: Low Traffic → Alternative Methods
    ax.arrow(7.5, 8.2, 2, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    low_box = FancyBboxPatch((8.5, 5.8), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#95a5a6', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(low_box)
    ax.text(10, 6.2, 'LOW TRAFFIC\n<1K users/day', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax.text(10, 5.4, 'Qualitative insights needed\nRapid iteration critical\nEarly stage / MVP\nObvious improvement',
            ha='center', va='top', fontsize=8)

    ax.arrow(10, 5.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    result_low = FancyBboxPatch((8.5, 3.8), 3, 0.9, boxstyle="round,pad=0.1",
                                facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(result_low)
    ax.text(10, 4.5, 'ALTERNATIVES', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(10, 4.1, 'User research\nAnalytics dashboards\nPrototypes/MVPs\nExpert judgment',
            ha='center', va='center', fontsize=8)

    # Additional decision factors box
    factors_box = FancyBboxPatch((0.5, 0.5), 13, 2.5, boxstyle="round,pad=0.1",
                                  facecolor='#ecf0f1', edgecolor='black', linewidth=2)
    ax.add_patch(factors_box)
    ax.text(7, 2.7, 'Additional Decision Factors', ha='center', va='center',
            fontsize=12, fontweight='bold')

    factors_text = """
SKIP A/B Testing When:
• Obvious bug fix or broken feature → Ship directly
• One-way door decision (irreversible) → More validation needed (user research + prototype)
• Qualitative question ("Why do users...?") → User interviews, usability testing
• High risk to users → Canary deployment + monitoring instead
• Need immediate action → Ship with monitoring, iterate fast
• Regulatory/legal constraints → Compliance review first

USE A/B Testing When:
• Measurable outcome exists (CTR, conversion, revenue)
• Reversible decision (can rollback)
• Unclear which option is better (need data)
• Stakeholders need objective evidence
• Production environment required for realistic test
• Can afford 1-2 week experiment duration
    """
    ax.text(7, 1.5, factors_text, ha='center', va='center', fontsize=7,
            family='monospace')

    # Bottom principle
    ax.text(7, 0.15, 'Principle: Match validation method to traffic, stakes, and reversibility - A/B testing for safe, measurable, reversible decisions',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    save_chart(fig, 'validation_method_decision')

if __name__ == '__main__':
    print('Generating Week 10 A/B Testing charts...\n')

    chart1_ab_testing_fundamentals()
    chart2_statistical_power_curves()
    chart3_type_i_ii_errors()
    chart4_confidence_intervals()
    chart5_bayesian_ab_posterior()
    chart6_sequential_testing_boundaries()
    chart7_multi_armed_bandit_exploration()
    chart8_thompson_sampling_demo()
    chart9_causal_inference_dag()
    chart10_experiment_velocity_dashboard()
    chart11_metric_tree_example()
    chart12_canary_deployment_timeline()
    chart13_simpsons_paradox()
    chart14_experiment_decision_matrix()
    chart15_continuous_improvement_loop()
    chart16_validation_method_decision()

    print('\nAll 16 charts generated successfully!')
    print('Location: Week_10/charts/')