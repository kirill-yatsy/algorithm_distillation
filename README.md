# Algorithm Distillation

This repository contains ~~an implementation~~ try to implement of Algorithm Distillation, a method for distilling reinforcement learning algorithms into neural networks by modeling their training histories with a causal sequence model. For more details, please refer to the original paper: [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/pdf/2210.14215).

## Quick Start

1. Clone the repository:
```bash
git clone git@github.com:kirill-yatsy/algorithm_distillation.git
cd algorithm_distillation
```
 
2. Create a conda environment based on the env.yml file
```bash
conda env create -f env.yml
```

3. Activate the environment
```bash
conda activate airi
```

4. Generate the dataset
```bash
python a3c/train.py
```

5. Train the transformer model
```bash
python gpt/train.py
```

6. Open the notebook notebooks/show_in_context_learning.ipynb run the cells to see the comparison between a3c and transformer model.


