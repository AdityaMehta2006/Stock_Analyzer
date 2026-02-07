import torch
import gymnasium as gym

# 1. Verify the Engine is ready
if torch.cuda.is_available():
    print(f"Vibe Check Passed: Running on NVIDIA GPU ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    print("Vibe Check Passed: Running on Apple Silicon (MPS)")
else:
    print("Warning: Running on CPU (Slower for Deep Learning)")

# 2. Vibe Code: "Load the game and let me watch it learn"
# render_mode="human" is what makes the window pop up (impossible on Colab!)
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

print("Launching Simulation...")
for _ in range(1000):
    env.render()
    
    # Random Action (Vibe coding: we'll replace this with a Neural Net later)
    action = env.action_space.sample() 
    
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("Simulation Finished.")