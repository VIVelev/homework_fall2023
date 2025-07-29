#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "gym[mujoco]",
#     "torch",
#     "numpy",
#     "opencv-python",
#     "imageio",
#     "imageio-ffmpeg"
# ]
# ///

"""
Script to record video evaluations of expert policies in their respective environments.
This script loads expert policies and records their performance as videos.
"""

import os
import sys
import gym
import torch
import numpy as np
import imageio
from pathlib import Path

# Add the cs285 module to the path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.utils import sample_n_trajectories

# Configuration
EXPERT_POLICIES = {
    "Ant-v4": "cs285/policies/experts/Ant.pkl",
    "Walker2d-v4": "cs285/policies/experts/Walker2d.pkl", 
    "HalfCheetah-v4": "cs285/policies/experts/HalfCheetah.pkl",
    "Hopper-v4": "cs285/policies/experts/Hopper.pkl",
    "Humanoid-v4": "cs285/policies/experts/Humanoid.pkl"
}

def setup_pytorch():
    """Initialize PyTorch settings."""
    ptu.init_gpu(use_gpu=False, gpu_id=0)
    torch.manual_seed(42)
    np.random.seed(42)

def load_expert_policy(policy_path):
    """Load an expert policy from file."""
    if not os.path.exists(policy_path):
        print(f"Warning: Policy file {policy_path} not found")
        return None
    
    try:
        policy = LoadedGaussianPolicy(policy_path)
        policy.to(ptu.device)
        return policy
    except Exception as e:
        print(f"Error loading policy from {policy_path}: {e}")
        return None

def sample_trajectory_with_video(env, policy, max_path_length):
    """Sample a rollout with video recording, compatible with newer Gym API."""
    ob, _ = env.reset()
    
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    
    while True:
        # Render frame for video
        try:
            if hasattr(env, 'sim'):
                # MuJoCo environments
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                # Other environments - use the new Gym API
                img = env.render()
            
            if img is not None:
                import cv2
                image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
        except Exception as e:
            print(f"    Warning: Could not render frame: {e}")
            # Create a black frame as fallback
            image_obs.append(np.zeros((250, 250, 3), dtype=np.uint8))
        
        # Get action from policy
        ac = policy.get_action(ob)
        if len(ac.shape) > 1:
            ac = ac[0]  # Remove batch dimension if present
        
        # Take action
        next_ob, rew, terminated, truncated, _ = env.step(ac)
        done = terminated or truncated
        
        # Check if rollout should end
        steps += 1
        rollout_done = done or steps >= max_path_length
        
        # Record data
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)
        
        ob = next_ob
        
        if rollout_done:
            break
    
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32)
    }

def record_policy_video(env_name, policy, num_episodes=3, max_episode_length=1000, output_dir="expert_videos"):
    """Record video of policy performance in environment."""
    print(f"Recording {num_episodes} episodes for {env_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    try:
        env = gym.make(env_name, render_mode='rgb_array')
        env.reset(seed=42)
    except Exception as e:
        print(f"Error creating environment {env_name}: {e}")
        return
    
    # Get episode length from environment spec if available
    if hasattr(env.spec, 'max_episode_steps') and env.spec.max_episode_steps:
        max_episode_length = min(max_episode_length, env.spec.max_episode_steps)
    
    # Record episodes
    for episode in range(num_episodes):
        print(f"  Recording episode {episode + 1}/{num_episodes}...")
        
        try:
            # Sample trajectory with video recording using our custom function
            path = sample_trajectory_with_video(env, policy, max_episode_length)
            
            # Extract video frames
            if 'image_obs' in path and len(path['image_obs']) > 0:
                frames = path['image_obs']
                
                # Save video
                video_filename = f"{output_dir}/{env_name}_expert_episode_{episode + 1}.mp4"
                try:
                    with imageio.get_writer(video_filename, fps=30) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    print(f"    Saved video: {video_filename}")
                    
                    # Print episode statistics
                    total_reward = np.sum(path['reward'])
                    episode_length = len(path['reward'])
                    print(f"    Episode reward: {total_reward:.2f}, Length: {episode_length} steps")
                    
                except Exception as e:
                    print(f"    Error saving video {video_filename}: {e}")
            else:
                print(f"    No video frames recorded for episode {episode + 1}")
                
        except Exception as e:
            print(f"    Error recording episode {episode + 1}: {e}")
    
    env.close()

def main():
    """Main function to record videos for all expert policies."""
    print("Recording expert policy videos...")
    print("=" * 50)
    
    setup_pytorch()
    
    # Create output directory
    output_dir = "expert_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_recordings = 0
    total_policies = 0
    
    # Record videos for each expert policy
    for env_name, policy_path in EXPERT_POLICIES.items():
        total_policies += 1
        print(f"\nProcessing {env_name}...")
        print("-" * 30)
        
        # Load expert policy
        policy = load_expert_policy(policy_path)
        if policy is None:
            print(f"Skipping {env_name} due to policy loading error")
            continue
        
        print(f"Successfully loaded expert policy for {env_name}")
        
        # Record video
        try:
            record_policy_video(env_name, policy, num_episodes=2, output_dir=output_dir)
            successful_recordings += 1
        except Exception as e:
            print(f"Error recording videos for {env_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Recording complete!")
    print(f"Successfully recorded videos for {successful_recordings}/{total_policies} environments")
    print(f"Videos saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()