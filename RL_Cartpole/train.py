import gymnasium as gym
from agent import DQNAgent
import torch

episode_count = 300

env = gym.make("CartPole-v1")
env.metadata['render_fps'] = 120
state, info = env.reset()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

for i in range(episode_count):
    done = False
    total_reward = 0
    state, info = env.reset()
    while done == False:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
    print(f"Episode {i+1}  score: {total_reward}")

torch.save(agent.policy_net.state_dict(), "cartpole_dqn.pth")
print("Model saved")