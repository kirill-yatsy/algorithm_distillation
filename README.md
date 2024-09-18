# Algorithm Distillation

Training transformer to learn offline A3C learning history

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
conda activate al_dis
```

4. Generate the dataset
```bash
PYTHONPATH=$(pwd) python a3c/train.py
```

5. Train the transformer model
```bash
PYTHONPATH=$(pwd) python gpt/train.py
```

6. Open the notebook notebooks/show_in_context_learning.ipynb run the cells to see the comparison between a3c and transformer model.


