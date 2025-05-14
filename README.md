<h1 align="center">G.O.D Subnet</h1>


🚀 Welcome to the [Gradients on Demand](https://gradients.io) Subnet

> Providing access to Bittensor network for on-demand training at scale.


## Setup Guides

- [Miner Setup Guide](docs/miner_setup.md)
- [Validator Setup Guide](docs/validator_setup.md)

## Recommended Compute Requirements

[Compute Requirements](docs/compute.md)

## Miner Advice

[Miner Advice](docs/miner_advice.md)



## Running evaluations on your own
You can re-evaluate existing tasks on your own machine. Or you can run non-submitted models to check if they are good. 
This works for tasks not older than 7 days.

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