# Digital Self-Control

> **[UMAP 2024] Digital Self-Control: A Solution for Enhancing Self-Regulation in Online Learning**.

> Digital Self-Control a video-based cognitive distraction detection system based on facial expressions and gaze features, which could aid learners in keeping their attention. It utilizes federated learning to safeguards users' data privacy. Through its privacy-by-design approach, Digitial Self-Control introduces an innovative and novel solution that has the potential for real-time support. It also addresses the challenges associated with eyeglass wearers and improves the model performance with additional features containing information about the presence of glasses. 

This repository is anonymized for review.

## 🖼️ Teaser
<img src="https://github.com![teaser_mw](https://github.com/GhostCoder00/Digital-Self-Control/assets/144137539/6f627f0e-e5d7-4554-bd53-1f29314208e9)" width="700">

## 💁 Usage
1. Download data and carry out data preprocessing following the instructions below.

2. Create conda environment with `conda env create -f environment.yml`.

3. Run `experiments_nFL.py` for centralized learning and `experiments_FL.py` for federated learning.

For detailed argument settings please check `utils.py`. 

## 🔧 Environment
Important libraries and their versions by **January 25th, 2024**:

| Library | Version |
| --- | ----------- |
| Python | 3.11.7 by Anaconda|
| PyTorch | 2.1.2 for CUDA 12.1 |
| TorchMetrics | 1.2.1 |
| Scikit-Learn | 1.4.0 |
| WandB | 0.16.2 |

Others:
- We used **Weights & Bias** (https://wandb.ai/site) for figures instead of tensorboard. Please install and set up it properly beforehand.

- We used the Python function `match` in our implementation. This function only exists for Python version >= 3.10. Please replace it with `if-elif-else` statement if needed.

## 🗺 Instructions on data preprocessing
We conducted experiments using four datasets: Colorado (https://ieeexplore.ieee.org/abstract/document/8680698), Korea (https://nmsl.kaist.ac.kr/projects/attention/), Engagenet (https://github.com/engagenet/engagenet_baselines), and DAISEE (https://people.iith.ac.in/vineethnb/resources/daisee/index.html). Please contact the corresponding authors for data access if necessary.

Please dive into the `raw_data_preprocessing` directory for further instructions on data preprocessing.
