# Compute Requirements

The subnet operates on a **100% tournament-based system**. Miners compete by submitting open-source training code that validators execute on their infrastructure.

## Miner Requirements

**Miners don't need compute for tournaments** - you submit open-source training repositories that validators execute on their infrastructure.

### What You Need:

- Development machine for coding/testing
- GitHub repository with Docker training scripts
- Optional: Local GPU for testing your implementations

## Validator Infrastructure (For Reference)

Validators run your code on dedicated **trainer nodes** with:

#### GPU Allocation by Model Size

**Text Tasks (Instruct, DPO, GRPO):**

- ≤4B parameters: 1x H100
- 4-12B parameters: 2x H100
- 12-40B parameters: 4x H100
- > 40B parameters: 8x H100

_DPO tasks get 3x GPU multiplier, GRPO tasks get 2x multiplier_

**Image Tasks (SDXL, Flux):**

- All models: 1x A100

#### Container Limits

- **Memory**: 135GB per GPU (dynamically allocated)
- **CPU**: 24 cores per GPU (dynamically allocated)
- **Network**: Isolated (no internet during training)
