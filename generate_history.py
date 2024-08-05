import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from time import sleep

from dark_room import DarkRoom
from utils import print_grid


seed = 3434
torch.manual_seed(seed)
np.random.seed(seed)

 
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_pi = nn.Linear(128, n_actions)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        return pi, v

# Worker class
class Worker(mp.Process):
    def __init__(self, global_model, optimizer, global_episode , res_queue, name):
        super(Worker, self).__init__()
        self.name = f"w{name:02d}"
        self.global_episode = global_episode 
        self.global_model, self.optimizer = global_model, optimizer
        self.local_model = ActorCritic(2, 5)
        self.env = DarkRoom()
        self.res_queue = res_queue

    def run(self):
        while self.global_episode.value < 1000:
            current_state = self.env.reset()
            current_state = torch.FloatTensor(current_state)
            done = False
            episode_reward = 0
            self.local_model.load_state_dict(self.global_model.state_dict())
            # history contains the observation, action, reward
            episode_history = []
 
            total_step = 0
            while not done:
                logits, value = self.local_model(current_state)
                policy = F.softmax(logits, dim=-1)
                action = torch.multinomial(policy, 1).item()

                episode_history.append([current_state, action, episode_reward])

                new_state, reward, done, _ = self.env.step(action)
                new_state = torch.FloatTensor(new_state)
                episode_reward += reward

                self.optimize(current_state, new_state, episode_reward, action, done, value)

                current_state = new_state
                total_step += 1

                if total_step > 1000:
                    print(f"Reaced 1000 steps {self.name}, episode: {self.global_episode.value}, reward: {episode_reward}")
                    current_state = self.env.reset()
                    current_state = torch.FloatTensor(current_state)
                    episode_reward = 0
                    total_step = 0
                    break

                if done:
                    with self.global_episode.get_lock():
                        self.global_episode.value += 1 
                    self.res_queue.put([self.global_episode.value, episode_reward, total_step, episode_history])

                    print(f"{self.name}, episode: {self.global_episode.value}, reward: {episode_reward}, total steps: {total_step}")
                    break

    def optimize(self, state, new_state, reward, action, done, value):
        self.optimizer.zero_grad()
        
        _, new_value = self.local_model(new_state)
        td_target = reward + 0.99 * new_value * (1 - int(done))
        td_error = td_target - value
        
        actor_loss = -self.local_model(state)[0][action] * td_error.detach()
        critic_loss = td_error.pow(2)
        
        loss = actor_loss + 0.5 * critic_loss
        loss.backward()

        for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()

if __name__ == "__main__":
    env = DarkRoom()
    global_model = ActorCritic(2, 5)
    global_model.share_memory()

    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)
    
    global_episode = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0)
    res_queue = mp.Queue()

    num_processes = 4
    processes = []
    for i in range(num_processes):
        p = Worker(global_model, optimizer, global_episode , res_queue, i)
        p.start()
        processes.append(p)
    global_history = []
    [p.join() for p in processes]

    while not res_queue.empty():
        episode, reward, total_step = res_queue.get()
        global_history.append(episode_history)
        print(f"Episode {episode}, Reward: {reward}, Total_step: {total_step}")

    global_model.eval()
    env = DarkRoom(size=16)
    state = env.reset() 
    action_history = []
    for i in range(1000):
        policy, value = global_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action = torch.softmax(policy, dim=-1).argmax().item()
        action_history.append(action)
        # action = Categorical(policy).sample().item()
        state, reward, terminated, info = env.step(action)

        if terminated:
            print(f"terminated")
            sleep(3)
            state = env.reset()
        if(reward == 1):
            print(f"Goal reached in {i} steps")
            print(f"Action history: {action_history}")
            sleep(3)
            state = env.reset()
            action_history = []
        
        print_grid(env.render())
        print(f"Action history: {action_history}")
        sleep(0.1)