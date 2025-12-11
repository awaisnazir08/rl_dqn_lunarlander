import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from model import DQN
from utils import ReplayBuffer, seed_everything

BATCH_SIZE = 64
GAMMA = 0.85
LR = 5e-4
BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 200_000
MAX_EPISODES = 2500
MAX_STEPS = 1500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("LunarLander-v3", render_mode=None) 
seed_everything()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# ε-greedy action selection function
def select_action(state, steps_done):
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()

# Perform training
steps_done = 0
episode_rewards = []
episode_losses = []  # Track losses for analysis

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    episode_loss = []  # Track loss per episode

    for step in range(MAX_STEPS):
        action = select_action(state, steps_done)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps_done += 1

        # Only start training after collecting enough experiences
        if len(replay_buffer) >= max(1000, BATCH_SIZE * 10):  # Warmup period
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            # Q(s, a)
            q_values = policy_net(states).gather(1, actions)

            # target Q(s', a')
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)
            episode_loss.append(loss.item())  # Track loss value

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
            optimizer.step()

        # Target network sync
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    episode_rewards.append(total_reward)
    # Store average loss for this episode
    avg_loss = np.mean(episode_loss) if episode_loss else 0.0
    episode_losses.append(avg_loss)
    
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        recent_loss = np.mean(episode_losses[-10:]) if len(episode_losses) >= 10 else avg_loss
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Loss: {recent_loss:.4f}")
    
    # Check if target achieved (solved criteria: avg reward > 200 over last 100 episodes)
    if episode >= 100:
        avg_last_100 = np.mean(episode_rewards[-100:])
        if avg_last_100 >= 200:
            print(f"\n🎉 Target Achieved at Episode {episode}!")
            print(f"Average Reward (last 100 episodes): {avg_last_100:.2f}")
            break

# Save model
torch.save(policy_net.state_dict(), "dqn_lander.pth")

# Save rewards and losses for analysis
np.save("rewards.npy", episode_rewards)
np.save("losses.npy", episode_losses)

print("\n=== Training Complete ===")
print(f"Total Episodes: {len(episode_rewards)}")
print(f"Final Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
if len(episode_rewards) >= 100:
    print(f"Best Average Reward (any 100 consecutive): {max([np.mean(episode_rewards[i:i+100]) for i in range(len(episode_rewards)-100)]):.2f}")
print(f"Model saved to: dqn_lander.pth")
