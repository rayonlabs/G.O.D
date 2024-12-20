# Validator Setup Guide

Steps to set-up the validator.

If you dont intend to utilise the bandwidth independently of the gradients platform,
child hotkeying would not be a bad idea at all.

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
task bootstrap
```

2. Get your key onto the VM:

You know how to do this, don't you ;)

3. Create and set up the `.vali.env` file:

Within this you'll be asked for the gpu(s) you'd like to use for validation. If you select the default, then gpu at index [0] will be used to evaluate all models. However, you can select whichever you'd like "3,4,5" would mean that evaluation jobs are spread between your third, fourth and fifth gpu, for example.

```bash
task config
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




IF YOU ARE DEVVING AND ONLY IF YOU ARE A DEVELOPER, ADD THIS (key needs to be upgraded)
```bash
echo "NINETEEN_API_KEY=<your-nineteen-api-key>" >> .vali.env
```
