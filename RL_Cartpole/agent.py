import torch
import torch.nn as nn
from collections import deque
import random as rndm

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,action_size))

    def forward(self, x):
        x = self.net(x)
        return x
    

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.steps = 0

        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        rndmnumber = rndm.random()
        if rndmnumber < self.epsilon:
            return rndm.randrange(self.action_size)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            output = self.policy_net(state_t)
            return output.argmax().item()
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = rndm.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        max_next_q = self.target_net(next_states_t).max(1)[0]
        target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())