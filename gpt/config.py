import torch


class Config:
    # tokens
    start_token = 0 
    padding_token = 1
    end_token = 2

    # model parameters
    block_size = 512
    batch_size = 64
    
    n_embd = 128
    n_head = 2
    n_layer = 3
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 3e-4
    epoch = 10
    
    # environment parameters
    action_dim = 5 # left, right, up, down, stay
    grid_dim = 50 # size of the grid
    system_dim = 3 # system tokens (start, padding, end)
    reward_dim = 2 # 1, -1

    vocab_size = 0
    
    def __init__(self):
        self.vocab_size = self.system_dim + self.grid_dim * 2 + self.action_dim + self.reward_dim



CFG = Config()