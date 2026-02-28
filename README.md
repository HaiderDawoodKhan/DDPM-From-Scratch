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

## Train

Default run (100k steps):

```bash
python train.py --steps 100000 --batch-size 256
```

Key outputs:

- `outputs/train_log.csv`
- `outputs/samples/step_*.png`
- `outputs/checkpoints/ddpm_step_*.pt`

## Evaluate / Sample

```bash
python eval.py --checkpoint outputs/checkpoints/ddpm_step_100000.pt --samples 64 --capture-every 100
```

Run Task 6/7 diagnostics (step ablation + fixed trajectory + nearest-neighbor check):

```bash
python eval.py \
	--checkpoint outputs/checkpoints/ddpm_step_100000.pt \
	--samples 64 \
	--sampling-steps-ablation 1000,250,100,50 \
	--run-nn-check
```

Optional schedule-ablation comparison in eval:

```bash
python eval.py \
	--checkpoint outputs_linear/checkpoints/ddpm_step_100000.pt \
	--compare-checkpoint outputs_cosine/checkpoints/ddpm_step_100000.pt \
	--train-log outputs_linear/train_log.csv \
	--train-log-alt outputs_cosine/train_log.csv \
	--train-log-alt-label cosine
```

Outputs:

- `outputs/eval/sample_grid_final.png`
- `outputs/eval/intermediates/snapshot_*.png`
- `outputs/eval/diagnostics.txt`
- `outputs/eval/schedule_ablation_alpha_snr.png`
- `outputs/eval/schedule_ablation_training_curves.png` (if both logs supplied)
- `outputs/eval/ablation_steps/sample_grid_steps_*.png`
- `outputs/eval/ablation_steps/failure_mode_narrative.txt`
- `outputs/eval/fixed_seed_denoising_trajectory.png`
- `outputs/eval/nearest_neighbor_pairs_pixel_l2.png` (if `--run-nn-check`)
