import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, action_space)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value