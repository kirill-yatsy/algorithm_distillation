import numpy as np

from gpt.config import CFG


class Tokenizer:
    vocab_size = CFG.system_dim + CFG.grid_dim + CFG.action_dim + CFG.reward_dim

    def map_reward(self, reward):
        if reward == 1:
            return 3
        elif reward == -1:
            return 2
        elif reward == -2:
            return 1
        elif reward == -3:
            return 0
        raise ValueError(f"Reward {reward} is not in the list")

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
                    step[1] + CFG.system_dim,
                    step[2] + CFG.system_dim + CFG.grid_dim,
                    self.map_reward(step[3])
                    + CFG.system_dim
                    + CFG.grid_dim
                    + CFG.action_dim,
                ]
            )
        return [CFG.start_token] + tokens + [CFG.end_token]

    def pad(self, x):
        return x + [CFG.padding_token] * (CFG.context_size - len(x))

    def __call__(self, x):
        encoded = self.encode(x)
        padded = self.pad(encoded)
        if len(padded) != CFG.context_size:
            print("padding error")
        return padded


class Tokenizer_not_working:
    mapper = dict(
        {
            CFG.start_token: CFG.start_token,
            CFG.padding_token: CFG.padding_token,
            CFG.end_token: CFG.end_token,
        }
    )
    token_counter = 3

    # gets one unicode number. It should check if the unicode number is already in the mapper. If not, it should add it. Returns the number.
    def get_tokenized_code(self, x):
        if x not in self.mapper:
            self.mapper[x] = self.token_counter
            self.token_counter += 1
        return self.mapper[x]

    def encode(self, x: np.array):
        x = np.array(x)
        tokens = []
        for i in range(0, len(x)):
            step = np.array(x[i])
            step = ",".join(step.astype(str))
            for i in range(len(step)):
                unicode_numb = ord(step[i])
                tokens.append(self.get_tokenized_code(unicode_numb))
        return [self.mapper[CFG.start_token]] + tokens + [self.mapper[CFG.end_token]]

    def cut_to_max_len(self, x):
        tokenized_steps = []
        length = 0
        for i in range(0, len(x)):
            step = x[i]
            tokenized_step = self.encode([step])
            if length + len(tokenized_step) > CFG.block_size - 2:
                return x[:i]
            tokenized_steps.append(tokenized_step)
            length += len(tokenized_step)

        return x

    def pad(self, x):
        return x + [CFG.padding_token] * (CFG.block_size - len(x))

    def call(self, x):
        return self.pad(self.encode(x))
