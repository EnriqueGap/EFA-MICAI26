# EFA-MICAI26

This repo contains a small sandbox for experimenting with ML classification models optimized with Optuna.

It is inspired from [Zhining Liu et al](https://arxiv.org/abs/2505.17451v2) and [Ruizhe Liu, Jiaqi Luo](https://arxiv.org/abs/2605.14915v1) taking ideas from their benchmarking protocol and using it to test different ML algorithms on our EFA dataset.

In simple terms, if we take our high imbalanced EFA dataset, prepare it in a consistent way, and try several models, which one performs better and under what settings?

For privacy reasons, this repo run over a _wine classification dataset_, and no other dataset is provided.

This repo is useful when you want to compare different classification models under the same workflow, allowing custom search-spaces for each ML algorithm to fit.

## How To Start

1. Install the dependencies from `requirements.txt`.
2. Open `initial_process.ipynb`.
3. Run the notebook with the example dataset.
4. Replace the example data with your own dataset.
