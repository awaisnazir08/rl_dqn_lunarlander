import gymnasium as gym
import torch
import numpy as np
from model import DQN
import imageio
from PIL import Image, ImageDraw, ImageFont
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CONFIGURATION - Change these settings
# =============================================================================
RENDER_MODE = "rgb_array"  # Options: "human" (live), "rgb_array" (for recording), None (no render)
SAVE_VIDEO = True          # Save episodes as MP4 video
SAVE_GIF = True            # Save episodes as GIF
NUM_EVAL_EPISODES = 5      # Number of episodes to evaluate
NUM_RECORD_EPISODES = 3    # Number of episodes to save (video/gif)

# =============================================================================

env = gym.make("LunarLander-v3", render_mode=RENDER_MODE)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load("dqn_lander.pth", map_location=device))
policy_net.eval()

# Create output directory for recordings
if SAVE_VIDEO or SAVE_GIF:
    os.makedirs("recordings", exist_ok=True)

rewards = []

def add_text_to_frame(frame, text, position=(10, 10)):
    """Add text overlay to frame"""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Add text with background for readability
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill='black')
    draw.text(position, text, fill='white', font=font)
    return np.array(img)

print("="*60)
print("DQN LUNAR LANDER - AGENT EVALUATION")
print("="*60)
print(f"Render Mode: {RENDER_MODE}")
print(f"Episodes to Evaluate: {NUM_EVAL_EPISODES}")
print(f"Episodes to Record: {NUM_RECORD_EPISODES if (SAVE_VIDEO or SAVE_GIF) else 0}")
print(f"Save Video: {SAVE_VIDEO}")
print(f"Save GIF: {SAVE_GIF}")
print("="*60 + "\n")

for ep in range(NUM_EVAL_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    frames = []
    step_count = 0

    # Decide if we should record this episode
    record_episode = (ep < NUM_RECORD_EPISODES) and (SAVE_VIDEO or SAVE_GIF) and (RENDER_MODE == "rgb_array")

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        # Capture frame if recording
        if record_episode:
            frame = env.render()
            if frame is not None:
                # Add episode info to frame
                text = f"Episode {ep+1} | Step {step_count} | Reward: {total_reward:.1f}"
                frame_with_text = add_text_to_frame(frame, text)
                frames.append(frame_with_text)

    rewards.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {step_count}")

    # Save video and/or GIF
    if record_episode and len(frames) > 0:
        if SAVE_VIDEO:
            video_path = f"recordings/episode_{ep+1}_reward_{total_reward:.0f}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  → Video saved: {video_path}")
        
        if SAVE_GIF:
            gif_path = f"recordings/episode_{ep+1}_reward_{total_reward:.0f}.gif"
            # Downsample frames for smaller GIF size (every 2nd frame)
            gif_frames = frames[::2]
            imageio.mimsave(gif_path, gif_frames, fps=15, loop=0)
            print(f"  → GIF saved: {gif_path}")

env.close()

avg_reward = np.mean(rewards)
std_reward = np.std(rewards)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"Min Reward: {np.min(rewards):.2f}")
print(f"Max Reward: {np.max(rewards):.2f}")
print(f"Success Rate (>200): {(np.array(rewards) > 200).sum()}/{NUM_EVAL_EPISODES} ({(np.array(rewards) > 200).sum()/NUM_EVAL_EPISODES*100:.1f}%)")

if SAVE_VIDEO or SAVE_GIF:
    print(f"\nRecordings saved in: ./recordings/")
print("="*60)
