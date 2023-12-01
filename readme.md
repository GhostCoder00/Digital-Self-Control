# Digital Self-Control
This is the repository for Digital Self-Control. This repository is anonymized for review.

## Teaser
![teaser](https://github.com/wmd0701/Digital-Self-Control/assets/34072813/6baafcb5-138e-4006-8b88-984842a052b5)

## Usage
run `experiments_FL.py` for federated learning.

run `experiments_nFL` for centralized learning.

For detailed argument settings please check `utils.py`. 

## Environment
Important installed libraries and their versions by **2023 September 7th**:

| Library | Version |
| --- | ----------- |
| Python | 3.10.12 by Anaconda|
| PyTorch | 2.0.1 for CUDA 11.7 |
| TorchMetrics | 0.11.4 |
| Scikit-Learn | 1.2.2 |
| NumPy | 1.25.0 |

Others:
- We used the library **Weights & Bias** (https://wandb.ai/site) instead of the commonly used tensorboard for data visualization. Please install and set up it properly beforehand.

- We used the function `match` in our implementation. This function only exists for Python version >= 3.10. Please replace it with `if-elif-else` statement if needed.

## Datasets
We conducted experiments using four datasets: Colorado (https://ieeexplore.ieee.org/abstract/document/8680698), Korea (https://nmsl.kaist.ac.kr/projects/attention/), Engagenet (https://github.com/engagenet/engagenet_baselines), and DAISEE (https://people.iith.ac.in/vineethnb/resources/daisee/index.html). Please contact the corresponding authors for data access if necessary.

Please dive into the `raw_data_preprocessing` directory for further instructions on data preprocessing.
