import torch
import torch.nn as nn

from gpt.Block import Block
from gpt.config import CFG


class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(CFG.vocab_size, CFG.n_embd)
        self.position_embedding_table = nn.Embedding(CFG.block_size, CFG.n_embd)
        self.blocks = nn.Sequential(
            *[Block(CFG.n_embd, n_head=CFG.n_head) for _ in range(CFG.n_layer)]
        )
        self.ln_f = nn.LayerNorm(CFG.n_embd)  # final layer norm
        self.lm_head = nn.Linear(CFG.n_embd, CFG.action_dim)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)
        # tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=CFG.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x[:, -1, :])  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # B, T, C = logits.shape
            # logits = logits.view(B, C * T)
            # targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss