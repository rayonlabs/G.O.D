# Validator Setup Guide

Steps to set-up the validator.

## Prerequisites

- Docker
- Hugging Face account
- s3 bucket [Backblaze B2 example](s3_setup.md)

## Setup Steps

0. Clone the repo
```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
```

1. Install system dependencies (Ubuntu 24.04 LTS):

```bash
sudo -E ./bootstrap.sh
source $HOME/.bashrc
source $HOME/.venv/bin/activate
```

2. Get your key onto the VM:

You know how to do this, don't you ;)

3. Create and set up the `.vali.env` file:

```bash
python -m core.create_config
```

Make sure the VALIDATOR_PORT is not being used by anything else, and is exposed in your firewall settings.

4. Install the dependencies:

```bash
task install
```

**FOR DEV**
```bash
pip install -e '.[dev]'
pre-commit install
```




5. Run the Validator

```bash
task autoupdates
```

**FOR DEV**

```bash
task validator
```


6. Make sure you have outgoing and incoming connections exposed

Gradients allows anyone in the world to train a model on Bittensor. We communicate to 'api.gradients.io' to facilitate these user requests.
Technically it will work without this connection, but then the subnet is just synthetics. Make sure to allow it!


7. IF YOU ARE DEVVING, ADD THIS (key needs to be upgraded)
```bash
echo "NINETEEN_API_KEY=<your-nineteen-api-key>" >> .vali.env
```
