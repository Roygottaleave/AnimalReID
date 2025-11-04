import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from your comparison
datasets = ['Dataset 1', 'Dataset 2']
metrics = {
    'Total Images': [121, 1700],
    'Number of Individuals': [8, 34],
    'Training Samples': [78, 1088],
    'Validation Samples': [19, 272],
    'Test Samples': [24, 340],
    'Avg Images per Individual': [15.1, 50.0],
    'Min Images per Individual': [8, 50],
    'Max Images per Individual': [32, 50]
}

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Dataset Comparison: Animal ReID Analysis', fontsize=16, fontweight='bold')

# Plot 1: Basic Statistics
basic_metrics = ['Total Images', 'Number of Individuals']
basic_data = [metrics[metric] for metric in basic_metrics]
x = np.arange(len(basic_metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, [d[0] for d in basic_data], width, label='Dataset 1', color='#FF6B6B')
bars2 = ax1.bar(x + width/2, [d[1] for d in basic_data], width, label='Dataset 2', color='#4ECDC4')

ax1.set_title('A. Dataset Scale Comparison', fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xticks(x)
ax1.set_xticklabels(basic_metrics)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom')

# Plot 2: Data Distribution
distribution_metrics = ['Min Images per Individual', 'Avg Images per Individual', 'Max Images per Individual']
distribution_data = [metrics[metric] for metric in distribution_metrics]
x = np.arange(len(distribution_metrics))

bars1 = ax2.bar(x - width/2, [d[0] for d in distribution_data], width, label='Dataset 1', color='#FF6B6B')
bars2 = ax2.bar(x + width/2, [d[1] for d in distribution_data], width, label='Dataset 2', color='#4ECDC4')

ax2.set_title('B. Data Distribution Analysis', fontweight='bold')
ax2.set_ylabel('Number of Images')
ax2.set_xticks(x)
ax2.set_xticklabels(['Min', 'Average', 'Max'])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}', ha='center', va='bottom')

# Plot 3: Data Split Comparison
split_metrics = ['Training Samples', 'Validation Samples', 'Test Samples']
split_data = [metrics[metric] for metric in split_metrics]
x = np.arange(len(split_metrics))

bars1 = ax3.bar(x - width/2, [d[0] for d in split_data], width, label='Dataset 1', color='#FF6B6B')
bars2 = ax3.bar(x + width/2, [d[1] for d in split_data], width, label='Dataset 2', color='#4ECDC4')

ax3.set_title('C. Training-Validation-Test Split', fontweight='bold')
ax3.set_ylabel('Number of Samples')
ax3.set_xticks(x)
ax3.set_xticklabels(split_metrics)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom')

# Plot 4: Quality Assessment Radar Chart
categories = ['Data Quantity', 'Class Balance', 'Sample Adequacy', 'Split Ratio', 'Training Stability']
dataset1_scores = [2, 3, 3, 4, 3]  # Scale 1-5
dataset2_scores = [5, 5, 5, 5, 5]  # Scale 1-5

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
dataset1_scores += dataset1_scores[:1]
dataset2_scores += dataset2_scores[:1]
angles += angles[:1]

ax4 = plt.subplot(224, polar=True)
ax4.plot(angles, dataset1_scores, 'o-', linewidth=2, label='Dataset 1', color='#FF6B6B')
ax4.fill(angles, dataset1_scores, alpha=0.25, color='#FF6B6B')
ax4.plot(angles, dataset2_scores, 'o-', linewidth=2, label='Dataset 2', color='#4ECDC4')
ax4.fill(angles, dataset2_scores, alpha=0.25, color='#4ECDC4')

ax4.set_yticklabels([])
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)
ax4.set_title('D. Dataset Quality Assessment', fontweight='bold', pad=20)
ax4.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Additional: Individual Distribution Chart
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))

# Dataset 1 Individual Distribution
dataset1_individuals = ['BAS', 'AVV', 'BW', 'MB', 'EE', 'BO', 'AI', 'AKA']
dataset1_counts = [32, 18, 17, 13, 11, 11, 11, 8]

bars = ax5.bar(dataset1_individuals, dataset1_counts, color='#FF6B6B')
ax5.set_title('Dataset 1: Individual Distribution (Uneven)', fontweight='bold')
ax5.set_ylabel('Number of Images')
ax5.set_xlabel('Individuals')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

# Dataset 2 Individual Distribution (sample of 10)
dataset2_individuals = ['Rupee', 'Star', 'Meg', 'Lala', 'Tes', 'Teal', 'Tea', 'Maj', 'Mindy', 'Verity']
dataset2_counts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

bars = ax6.bar(dataset2_individuals, dataset2_counts, color='#4ECDC4')
ax6.set_title('Dataset 2: Individual Distribution (Perfectly Balanced)', fontweight='bold')
ax6.set_ylabel('Number of Images')
ax6.set_xlabel('Individuals (First 10 of 34)')
ax6.tick_params(axis='x', rotation=45)
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print summary statistics
print("="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"{'Metric':<30} {'Dataset 1':<12} {'Dataset 2':<12} {'Improvement'}")
print("-"*60)
for metric, values in metrics.items():
    if metric in ['Total Images', 'Number of Individuals', 'Training Samples', 
                  'Validation Samples', 'Test Samples']:
        improvement = f"+{values[1]-values[0]:,} ({values[1]/values[0]:.1f}x)"
        print(f"{metric:<30} {values[0]:<12} {values[1]:<12} {improvement}")
    else:
        improvement = f"+{values[1]-values[0]:.1f}"
        print(f"{metric:<30} {values[0]:<12} {values[1]:<12} {improvement}")