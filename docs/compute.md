# Miner Compute Requirements

## Real-Time Serving

For competitive real-time serving of [Gradients.io](https://gradients.io) customer requests:

### Recommended Setup
- **Proxy Node**: Route jobs to specialized GPUs based on task type
- **Text Training**: Multiple H100s/H200s for LLM fine-tuning
- **Image Training**: Multiple A100s for diffusion model training

| Component    | Specification                           |
|-------------|----------------------------------------|
| VRAM        | 80+GB per GPU                          |
| Storage     | 1TB+ (recommended for model caching)   |
| RAM         | 64GB+ per node                         |
| Example GPUs| **Text**: H100/H200 cluster<br>**Image**: A100 cluster |
| vCPUs       | 12+ per node                           |

*Scale based on job acceptance strategy and competitive positioning.*

## Tournaments

**Miners don't need compute for tournaments** - you submit open-source training repositories that validators execute on their infrastructure.

### What You Need:
- Development machine for coding/testing
- GitHub repository with Docker training scripts
- Optional: Local GPU for testing your implementations

### What Validators Use (For Reference)

Validators run your code on dedicated **trainer nodes** with:

#### GPU Allocation by Model Size
**Text Tasks (Instruct, DPO, GRPO):**
- ≤4B parameters: 1x H100
- 4-12B parameters: 2x H100  
- 12-40B parameters: 4x H100
- >40B parameters: 8x H100

*DPO tasks get 3x GPU multiplier, GRPO tasks get 2x multiplier*

**Image Tasks (SDXL, Flux):**
- All models: 1x A100

#### Container Limits
- **Memory**: 24GB per training job
- **CPU**: 8 cores per training job
- **Network**: Isolated (no internet during training)
- **Storage**: 2TB+ for model/dataset caching
