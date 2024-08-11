import numpy as np

from gpt.config import CFG


class Tokenizer:
    vocab_size = CFG.system_dim + CFG.grid_dim * 2 + CFG.action_dim + CFG.reward_dim

    def encode(self, x: np.array):
        x = np.array(x)
        tokens = []
        for i in range(0, len(x)):
            step = np.array(x[i])
            # I checked DecisionTransformer too late, that is why I used "work around" with offsets
            # Don't have time to fix it now
            tokens.extend(
                [
                    step[0] + CFG.system_dim,
                    step[1] + CFG.system_dim + CFG.grid_dim,
                    step[2] + CFG.system_dim + CFG.grid_dim * 2,
                    step[3] + CFG.system_dim + CFG.grid_dim * 2 + CFG.action_dim,
                ]
            )
        return [CFG.start_token] + tokens + [CFG.end_token]

    def pad(self, x):
        return x + [CFG.padding_token] * (CFG.block_size - len(x))

    def __call__(self, x):
        encoded = self.encode(x)
        padded = self.pad(encoded)
        if len(padded) != CFG.block_size:
            print("padding error")
        return padded
