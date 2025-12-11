"""
Simple script to visualize trained agent with live rendering using pygame.
No video/gif recording - just watch it play!
"""
import gymnasium as gym
import torch
import numpy as np
from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Live rendering with pygame
env = gym.make("LunarLander-v3", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load("dqn_lander.pth", map_location=device))
policy_net.eval()

print("="*60)
print("Watching trained DQN agent play Lunar Lander...")
print("Close the window or press Ctrl+C to stop")
print("="*60 + "\n")

NUM_EPISODES = 10
rewards = []

try:
    for ep in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {step_count}")

except KeyboardInterrupt:
    print("\nStopped by user")

env.close()

if rewards:
    print(f"\nAverage Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success Rate (>200): {(np.array(rewards) > 200).sum()}/{len(rewards)}")
