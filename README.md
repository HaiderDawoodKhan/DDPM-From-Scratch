# DDPM on FashionMNIST (PA1)

This project implements a baseline DDPM with:

- Dataset: FashionMNIST, scaled to `[-1, 1]`
- Timesteps: `L=1000`
- Schedule: linear betas from `1e-4` to `0.02` (plus cosine option for ablation)
- Model: small U-Net (`32,64,128` channels), GroupNorm + SiLU, no attention
- Training objective: `L_simple = MSE(eps_hat, eps)`
- Sampling: ancestral DDPM with all `L` reverse steps

## Structure

- `diffusion/schedule.py`: beta/alpha/alpha_bar and precomputed scalars
- `diffusion/forward.py`: `q_sample`, timestep sampling, `(t, eps, x_t)` utility
- `diffusion/posterior.py`: posterior formulas and reverse-step mechanics
- `diffusion/ddpm.py`: ancestral sampler
- `models/unet.py`: epsilon network `eps_theta(x_t, t)`
- `train.py`: training loop + CSV logging + periodic sample grids
- `eval.py`: diagnostics, ablations, nearest-neighbor checks, and trajectory generation
- `visualizations.ipynb`: required visualizations
- `outputs/training`: contains the generated samples across training steps for schedulers (cosine + linear)
- `notebooks/`: contains the notebooks used for training the models