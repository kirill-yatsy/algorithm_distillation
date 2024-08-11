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
from a3c.ActorCritic import ActorCritic
from a3c.Worker import Worker
from dark_room import DarkRoom
from utils import print_grid
import csv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
env = DarkRoom()
input_dim = env.observation_space.shape[0]
action_space = env.action_space.n

action_to_text = {0: "left", 1: "right", 2: "up", 3: "down", 4: "stay"}
 

def generate_history(seed, num_workers=4): 
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    global_ep = mp.Value('i', 0)
    res_queue = mp.Queue()
    global_model = ActorCritic(input_dim, action_space).to(device)
    global_model.share_memory()
    global_lock = mp.Lock()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)
    
    
    workers = [Worker(global_model, optimizer, input_dim, action_space, global_ep, res_queue, global_lock, i, device) for i in range(num_workers)]
    [w.start() for w in workers]
    [w.join() for w in workers]

    print("Training finished")

    return global_model

if __name__ == "__main__":
    mp.set_start_method('spawn')
    num_workers = 8
    generate_history(45*1, num_workers)
    # generate_history(45*2, num_workers)
    # generate_history(45*3, num_workers)
    generate_history(45*4, num_workers)
    generate_history(45*5, num_workers)
    global_model = generate_history(45*6)

    # save the model
    model_path = "models/a3c_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(global_model.state_dict(), "models/a3c_model.pth")

    print("Data collection complete")
    # Evaluate the trained model
    env = DarkRoom(size=9)
    global_model = global_model.cpu().eval()
    
    # merge all csv files into one and delete the individual files
    data_files = [f"data/worker_{i}_data.csv" for i in range(num_workers)] 
    with open("data/all_data.csv", 'w', newline='') as file:
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