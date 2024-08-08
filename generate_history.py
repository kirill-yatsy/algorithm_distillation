import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
from time import sleep
import torch.nn.functional as F
from dark_room import DarkRoom
from utils import print_grid
import csv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

env = DarkRoom()
input_dim = env.observation_space.shape[0]
action_space = env.action_space.n

action_to_text = {0: "left", 1: "right", 2: "up", 3: "down", 4: "stay"}

class Worker(mp.Process):
    def __init__(self, global_model, optimizer, input_dim, action_space, global_ep, res_queue, global_lock, worker_id):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.env = DarkRoom()
        self.global_ep = global_ep
        self.res_queue = res_queue
        self.global_lock = global_lock
        self.worker_id = worker_id
        self.gamma = 0.99

        self.local_model = ActorCritic(input_dim, action_space).to(device)
        self.global_model = global_model
        self.optimizer = optimizer

        self.data_file = f"worker_{self.worker_id}_data.csv"
        self.initialize_data_file()
        # generate a unique guid  for this worker 

    def initialize_data_file(self):
        # if file exists, do nothing
        if os.path.exists(self.data_file):
            return
        with open(self.data_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'Action', 'Reward', 'X', 'Y'])

    def save_step_data(self, step, action, reward, x, y):
        with open(self.data_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.uniq_guid, step, action, reward, x, y])

    def run(self):
        num_episodes = 1000
        while self.global_ep.value < num_episodes:
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            done = False
            total_reward = 0
            seq_len = 0
            self.uniq_guid = str(uuid.uuid4())
            while not done:
                policy, value = self.local_model(state)
                action = self.select_action(policy)
                next_state, reward, done = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                seq_len += 1
                total_reward += reward
                
                self.save_step_data(seq_len, action, reward, next_state[0, 0].item(), next_state[0, 1].item())
                
                self.optimize(state, next_state, reward, action, done, value)
                state = next_state

                if seq_len > 2000:
                    print(f"Worker {self.worker_id} reached max steps")
                    state = self.env.reset()
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    seq_len = 0
                    total_reward = 0
            
            with self.global_ep.get_lock():
                current_ep = self.global_ep.value
                self.global_ep.value += 1
            self.res_queue.put(current_ep + 1)
            print(f"Worker {self.worker_id} finished episode {current_ep + 1} with total reward: {total_reward}, seq_len: {seq_len}")
    
    def select_action(self, policy):
        policy = torch.softmax(policy, dim=-1)
        action = Categorical(policy).sample().item()
        return action
    
    def optimize(self, state, new_state, reward, action, done, value):
        self.optimizer.zero_grad()

        _, new_value = self.local_model(new_state)
        td_target = reward + self.gamma * new_value * (1 - int(done))
        td_error = td_target - value
        
        actor_loss = -self.local_model(state)[0][0, action] * td_error.detach()
        critic_loss = F.smooth_l1_loss(value, td_target.detach())
        
        loss = actor_loss + 0.5 * critic_loss
        loss.backward()
 

        with self.global_lock:
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad
                else:
                    global_param._grad += local_param.grad
            self.optimizer.step()

        for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
            # without next magic steps reach 2000 very often
            local_param.data.copy_(0.999 * local_param.data + 0.001 * global_param.data)

def evaluate_model(global_model, env, num_episodes=10):
    global_model.eval()
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            policy, _ = global_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = torch.softmax(policy, dim=-1).argmax().item()
            state, reward, done = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def generate_history(seed, num_workers=4): 
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    global_ep = mp.Value('i', 0)
    res_queue = mp.Queue()
    global_model = ActorCritic(input_dim, action_space).to(device)
    global_model.share_memory()
    global_lock = mp.Lock()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)
    
    
    workers = [Worker(global_model, optimizer, input_dim, action_space, global_ep, res_queue, global_lock, i) for i in range(num_workers)]
    [w.start() for w in workers]
    [w.join() for w in workers]

    print("Training finished")

    return global_model

if __name__ == "__main__":
    mp.set_start_method('spawn')
    num_workers = 8
    generate_history(45*1, num_workers)
    generate_history(45*2, num_workers)
    generate_history(45*3, num_workers)
    generate_history(45*4, num_workers)
    generate_history(45*5, num_workers)
    global_model = generate_history(45*6)

    # Evaluate the trained model
    env = DarkRoom(size=9)
    global_model = global_model.cpu().eval()
    average_reward = evaluate_model(global_model, env)
    print(f"Average reward over 10 episodes: {average_reward}")

    

    with torch.no_grad():
        state = env.reset()
        done = False
        while not done:
            policy, _ = global_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = torch.softmax(policy, dim=-1).argmax().item()
            state, reward, done = env.step(action)
            
            print_grid(env.render())
            print(f"Action: {action_to_text[action]}, Reward: {reward}")
            sleep(0.5)

        print("Visualization complete")

    
    # merge all csv files into one and delete the individual files
    data_files = [f"worker_{i}_data.csv" for i in range(num_workers)]
    with open("all_data.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Step', 'Action', 'Reward', 'X', 'Y'])
        for data_file in data_files:
            with open(data_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if "Episode" in row:
                        continue
                    writer.writerow(row)
            os.remove(data_file)