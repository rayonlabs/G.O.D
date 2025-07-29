# Real-Time Miner Setup Guide

This guide covers setting up a miner for real-time serving of [Gradients.io](https://gradients.io) customer requests.

For tournament participation, see the [Tournament Miner Guide](tourn_miner.md).

## Prerequisites

- Docker
- Python 3.8+
- Hugging Face account and API token
- WANDB token for training



## Setup Steps

0. Clone the repo
    ```bash
    git clone https://github.com/rayonlabs/G.O.D.git
    cd G.O.D
    ```

1. Install system dependencies:
    ```bash
    sudo -E ./bootstrap.sh
    source $HOME/.bashrc
    source $HOME/.venv/bin/activate
    ```

2. Install the python packages:

```bash
task install
```

**FOR DEV**
```bash
pip install -e '.[dev]'
pre-commit install
```

2. Set up your wallet:

You prob know how to do this, but just in case:

Make sure you have bittensor installed to the latest version then:

```bash
btcli wallet new-coldkey
btcli wallet new-hotkey
```

To check see your wallet address'

```bash
btcli wallet list
```

3. Register to the subnet.

**FOR PROD**
```bash
btcli s register
```

**FOR DEV**
```bash
btcli s register --network test
```

4. Register on metagraph

Then register your IP on the metagraph using fiber. Get your external ip with `curl ifconfig.me`.

**FOR PROD**
```bash
fiber-post-ip --netuid 56 --subtensor.network finney --external_port 7999 --wallet.name default --wallet.hotkey default --external_ip [YOUR-IP]
```

**FOR DEV**
```bash
fiber-post-ip --netuid 241 --subtensor.network test --external_port 7999 --wallet.name default --wallet.hotkey default --external_ip [YOUR-IP]
```


5. Configure environment variables:
    ```bash
    python3 -m core.create_config --miner
    ```
NOTE: You will need a wandb token & a huggingface token.

6. Update the wandb & huggingface configuration:
    - Update your `wandb_entity` in the wandb section of the config to be your wandb username+org_name [here](../core/config/base.yml) - if you are part of a team.

7. Start the miner service:


```bash
task miner
```
