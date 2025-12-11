import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Load training data
rewards = np.load("rewards.npy")
losses = np.load("losses.npy")

# Configuration
num_episodes = len(rewards)
window_size = 100  # For moving average

print("="*60)
print("DQN Lunar Lander - Training Analysis")
print("="*60)
print(f"\nTotal Episodes Trained: {num_episodes}")
print(f"Batch Size: 32")
print(f"Gamma (γ): 0.85")
print(f"Epsilon (ε): Start=1.0, End=0.2")
print(f"Replay Buffer Size: 100,000")
print("\n" + "="*60)

# Calculate statistics
print("\nReward Statistics:")
print(f"  Mean Reward: {np.mean(rewards):.2f}")
print(f"  Std Deviation: {np.std(rewards):.2f}")
print(f"  Min Reward: {np.min(rewards):.2f}")
print(f"  Max Reward: {np.max(rewards):.2f}")
print(f"  Final 100 Episodes Avg: {np.mean(rewards[-100:]):.2f}")

# Find best 100 consecutive episodes
best_avg = -float('inf')
best_idx = 0
for i in range(len(rewards) - 100):
    avg = np.mean(rewards[i:i+100])
    if avg > best_avg:
        best_avg = avg
        best_idx = i

print(f"  Best 100 Consecutive Avg: {best_avg:.2f} (Episodes {best_idx}-{best_idx+100})")

print("\nLoss Statistics:")
print(f"  Mean Loss: {np.mean(losses):.4f}")
print(f"  Final 100 Episodes Loss: {np.mean(losses[-100:]):.4f}")
print(f"  Loss Reduction: {((losses[0] - np.mean(losses[-100:])) / losses[0] * 100):.2f}%")

print("\n" + "="*60)

# Create comprehensive plots
fig = plt.figure(figsize=(16, 12))

# 1. Episode Returns (every 100 episodes as specified)
ax1 = plt.subplot(3, 2, 1)
episodes_100 = np.arange(100, num_episodes+1, 100)
rewards_100 = [rewards[i-1] for i in episodes_100]
ax1.plot(episodes_100, rewards_100, 'o-', color='steelblue', markersize=4, linewidth=1.5)
ax1.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Target (200)')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax1.set_ylabel('Return', fontsize=11, fontweight='bold')
ax1.set_title('Returns Every 100 Episodes', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. All Episode Returns with Moving Average
ax2 = plt.subplot(3, 2, 2)
ax2.plot(rewards, alpha=0.3, color='lightblue', label='Raw Returns')
moving_avg = uniform_filter1d(rewards, size=window_size, mode='nearest')
ax2.plot(moving_avg, color='darkblue', linewidth=2, label=f'{window_size}-Episode Moving Avg')
ax2.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Target (200)')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax2.set_ylabel('Return', fontsize=11, fontweight='bold')
ax2.set_title('All Episode Returns with Moving Average', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. MSE (Network Loss)
ax3 = plt.subplot(3, 2, 3)
ax3.plot(losses, alpha=0.4, color='lightcoral', label='Raw Loss')
loss_moving_avg = uniform_filter1d(losses, size=window_size, mode='nearest')
ax3.plot(loss_moving_avg, color='darkred', linewidth=2, label=f'{window_size}-Episode Moving Avg')
ax3.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax3.set_ylabel('MSE Loss', fontsize=11, fontweight='bold')
ax3.set_title('Mean Squared Error (Network Loss)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_yscale('log')  # Log scale for better visualization

# 4. Mean Scores (binned by 100 episodes)
ax4 = plt.subplot(3, 2, 4)
num_bins = num_episodes // 100
mean_scores = [np.mean(rewards[i*100:(i+1)*100]) for i in range(num_bins)]
bin_centers = [(i*100 + (i+1)*100) / 2 for i in range(num_bins)]
ax4.bar(bin_centers, mean_scores, width=80, alpha=0.7, color='teal', edgecolor='black')
ax4.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Target (200)')
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax4.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
ax4.set_title('Mean Scores per 100 Episodes', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()

# 5. Learning Progress: Cumulative Average
ax5 = plt.subplot(3, 2, 5)
cumulative_avg = [np.mean(rewards[:i+1]) for i in range(len(rewards))]
ax5.plot(cumulative_avg, color='purple', linewidth=2)
ax5.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Target (200)')
ax5.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax5.set_ylabel('Cumulative Average Return', fontsize=11, fontweight='bold')
ax5.set_title('Cumulative Average Return (Learning Curve)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. Return Distribution (Histogram)
ax6 = plt.subplot(3, 2, 6)
ax6.hist(rewards, bins=50, alpha=0.7, color='orange', edgecolor='black')
ax6.axvline(x=np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
ax6.axvline(x=200, color='green', linestyle='--', linewidth=2, label='Target: 200')
ax6.set_xlabel('Return', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Return Distribution', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.legend()

plt.tight_layout()
plt.savefig('dqn_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive analysis plot saved as 'dqn_comprehensive_analysis.png'")
plt.show()

# Additional detailed analysis plot for loss
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss over episodes
axes[0, 0].plot(losses, alpha=0.5, color='salmon')
axes[0, 0].plot(loss_moving_avg, color='darkred', linewidth=2, label='Moving Average')
axes[0, 0].set_xlabel('Episode', fontweight='bold')
axes[0, 0].set_ylabel('MSE Loss', fontweight='bold')
axes[0, 0].set_title('Training Loss Over Episodes', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Loss vs Reward correlation
axes[0, 1].scatter(losses[::10], rewards[::10], alpha=0.5, c=range(0, len(losses), 10), cmap='viridis')
axes[0, 1].set_xlabel('MSE Loss', fontweight='bold')
axes[0, 1].set_ylabel('Return', fontweight='bold')
axes[0, 1].set_title('Loss vs Return (Every 10th Episode)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
cbar.set_label('Episode', fontweight='bold')

# Loss distribution
axes[1, 0].hist(losses, bins=50, alpha=0.7, color='crimson', edgecolor='black')
axes[1, 0].axvline(x=np.mean(losses), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
axes[1, 0].set_xlabel('MSE Loss', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Loss Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].legend()

# Loss reduction over time (binned)
loss_bins = [np.mean(losses[i*100:(i+1)*100]) for i in range(num_bins)]
axes[1, 1].plot(bin_centers, loss_bins, 'o-', color='maroon', markersize=6, linewidth=2)
axes[1, 1].set_xlabel('Episode', fontweight='bold')
axes[1, 1].set_ylabel('Mean Loss per 100 Episodes', fontweight='bold')
axes[1, 1].set_title('Loss Reduction Progress', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dqn_loss_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Detailed loss analysis plot saved as 'dqn_loss_analysis.png'")
plt.show()

# Print analysis summary
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

# Convergence analysis
first_500_avg = np.mean(rewards[:500])
last_500_avg = np.mean(rewards[-500:])
improvement = ((last_500_avg - first_500_avg) / abs(first_500_avg) * 100) if first_500_avg != 0 else 0

print(f"\nLearning Progress:")
print(f"  First 500 Episodes Avg: {first_500_avg:.2f}")
print(f"  Last 500 Episodes Avg: {last_500_avg:.2f}")
print(f"  Improvement: {improvement:.2f}%")

# Success rate (episodes with reward > 200)
success_rate = (np.array(rewards) > 200).sum() / len(rewards) * 100
print(f"\nPerformance Metrics:")
print(f"  Episodes with Return > 200: {(np.array(rewards) > 200).sum()}/{len(rewards)} ({success_rate:.2f}%)")
print(f"  Episodes with Return > 0: {(np.array(rewards) > 0).sum()}/{len(rewards)} ({(np.array(rewards) > 0).sum()/len(rewards)*100:.2f}%)")

# Training stability
std_first_half = np.std(rewards[:num_episodes//2])
std_second_half = np.std(rewards[num_episodes//2:])
print(f"\nTraining Stability:")
print(f"  First Half Std Dev: {std_first_half:.2f}")
print(f"  Second Half Std Dev: {std_second_half:.2f}")
print(f"  Stability Improvement: {((std_first_half - std_second_half) / std_first_half * 100):.2f}%")
