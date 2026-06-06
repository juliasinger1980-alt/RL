import gymnasium as gym
from agent import DQNAgent
import torch
import pygame

episode_count = 10

env = gym.make("CartPole-v1", render_mode="human")
env.metadata['render_fps'] = 30
state, info = env.reset()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.policy_net.load_state_dict(torch.load("cartpole_dqn.pth"))
agent.epsilon = 0

try:
    for i in range(episode_count):
        done = False
        total_reward = 0
        state, info = env.reset()
        while done == False:
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        exit()
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            keys = pygame.key.get_pressed()
            pygame.event.pump()
            if keys[pygame.K_LEFT]:
                env.unwrapped.state[2] -= 0.01
                print("pushing left")
            if keys[pygame.K_RIGHT]:
                env.unwrapped.state[2] += 0.01
                print("pushing right")
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {i+1}  score: {total_reward}")
except KeyboardInterrupt:
     pass