import torch


class Config:
    # tokens
    start_token = 0 
    padding_token = 1
    end_token = 2

    # model parameters
    context_size = 128
    batch_size = 256
    
    n_embd = 256
    n_head = 6
    n_layer = 12
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 1e-5
    epoch = 30
    
    # environment parameters
    action_dim = 5 # left, right, up, down, stay
    grid_dim = 50 # size of the grid
    system_dim = 3 # system tokens (start, padding, end)
    reward_dim = 4 # 1, -1, -2, -3

    vocab_size = 0
    
    def __init__(self):
        self.vocab_size = self.system_dim + self.grid_dim * 2 + self.action_dim + self.reward_dim



CFG = Config()