import uuid
import torch
import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch.nn.functional as F
from a3c.ActorCritic import ActorCritic
from dark_room import DarkRoom
import csv
import os


class Worker(mp.Process):
    def __init__(
        self,
        global_model,
        optimizer,
        input_dim,
        action_space,
        global_ep,
        res_queue,
        global_lock,
        worker_id,
        device,
    ):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.env = DarkRoom()
        self.global_ep = global_ep
        self.res_queue = res_queue
        self.global_lock = global_lock
        self.worker_id = worker_id
        self.gamma = 0.99
        self.device = device

        self.local_model = ActorCritic(input_dim, action_space).to(device)
        self.global_model = global_model
        self.optimizer = optimizer

        self.data_file = f"data/worker_{self.worker_id}_data.csv"
        self.history = []
        self.initialize_data_file()

    def initialize_data_file(self):
        # if file exists, do nothing
        if os.path.exists(self.data_file):
            return
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Step", "Action", "Reward", "X", "Y"])

    def save_step_data(self, step, action, reward, x, y):
        self.history.append([self.uniq_guid, step, action, reward, x, y])

    def save_history(self):
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, "a", newline="") as file:
            writer = csv.writer(file) 
            writer.writerows(self.history)
    
    def run(self):
        num_episodes = 500
        while self.global_ep.value < num_episodes:
            state = self.env.reset(use_random_agent_pos=True)
            state = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            done = False
            total_reward = 0
            seq_len = 0
            self.uniq_guid = str(uuid.uuid4())
            while not done:
                policy, value = self.local_model(state)
                action = self.select_action(policy)
                next_state, reward, done = self.env.step(action)
                next_state = (
                    torch.tensor(next_state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                seq_len += 1
                total_reward += reward 

                self.history.append([self.uniq_guid, seq_len, action, reward, next_state[0, 0].item(), next_state[0, 1].item()])


                self.optimize(state, next_state, reward, action, done, value)
                state = next_state

                

                if seq_len > 1000:
                    print(f"Worker {self.worker_id} reached max steps")
                    break
                    state = self.env.reset(use_random_agent_pos=True)
                    state = (
                        torch.tensor(state, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    seq_len = 0
                    total_reward = 0
                    self.history = []

            with self.global_ep.get_lock():
                current_ep = self.global_ep.value
                self.global_ep.value += 1
            self.res_queue.put(current_ep + 1)
            print(
                f"Worker {self.worker_id} finished episode {current_ep + 1} with total reward: {total_reward}, seq_len: {seq_len}"
            )
        self.save_history()

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

        # entropy = Categorical(torch.softmax(self.local_model(state)[0], dim=-1)).entropy()

        loss = actor_loss + 0.5 * critic_loss  
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 0.5)


        with self.global_lock:
            for local_param, global_param in zip(
                self.local_model.parameters(), self.global_model.parameters()
            ):
                if global_param.grad is None:
                    global_param._grad = local_param.grad
                else:
                    global_param._grad += local_param.grad
            self.optimizer.step()

        for local_param, global_param in zip(
            self.local_model.parameters(), self.global_model.parameters()
        ): 
            local_param.data.copy_(global_param.data)
