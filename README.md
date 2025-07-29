<h1 align="center">G.O.D Subnet</h1>


🚀 Welcome to the [Gradients on Demand](https://gradients.io) Subnet

> Distributed intelligence for LLM and diffusion model training. Where the world's best AutoML minds compete.

## 🎯 Two Training Systems

### 1. **Real-Time Serving** 
Miners compete to train models for [Gradients.io](https://gradients.io) customers who use our 4-click interface to fine-tune AI models.

### 2. **Tournaments** 🏆
Competitive events where validators execute miners' open-source training scripts on dedicated infrastructure.

- **Duration**: 4-7 days per tournament
- **Frequency**: New tournaments start 24 hours after the previous one ends
- **Rewards**: Significantly higher weight potential for top performers
- **Open Source**: Winning AutoML scripts are released when tournaments complete
- [Tournament Overview](docs/tournament_overview.md)
- [Tournament Miner Guide](docs/tourn_miner.md)

## Setup Guides

- [Real-Time Miner Setup](docs/miner_setup.md)
- [Tournament Miner Guide](docs/tourn_miner.md)
- [Validator Setup Guide](docs/validator_setup.md)

## Recommended Compute Requirements

[Compute Requirements](docs/compute.md)

## Miner Advice

[Miner Advice](docs/miner_advice.md)



## Running evaluations on your own
You can re-evaluate existing tasks on your own machine. Or you can run non-submitted models to check if they are good. 
This works for tasks not older than 7 days.

Make sure to build the latest docker images before running the evaluation.
```bash
docker build -f dockerfiles/validator.dockerfile -t weightswandering/tuning_vali:latest .
docker build -f dockerfiles/validator-diffusion.dockerfile -t diagonalge/tuning_validator_diffusion:latest .
```

To see the available options, run:
```bash
python -m utils.run_evaluation --help
```

To re-evaluate a task, run:
```bash
python -m utils.run_evaluation --task_id <task_id>
```

To run a non-submitted model, run:
```bash
python -m utils.run_evaluation --task_id <task_id> --models <model_name>
```