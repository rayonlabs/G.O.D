# Advanced Gradients Mining Guide 🚀 💎

Welcome to the world of Gradients mining! This guide compiles wisdom from a few experienced miners and adds practical tips to help you successfully train LLMs and outperform the competition. 🏆

## 🧠 Understanding What Makes Gradients Special ✨

Gradients is focused on making AI model training accessible to everyone. While users can fine-tune models with just four clicks (selecting a base model, uploading data, picking columns, and setting training time), miners like you are the AI peeps who make the magic happen behind the scenes. 🪄

When users submit jobs, they're not looking for basic fine-tuning - they're counting on miners to produce results that outperform models trained on major platforms like AWS SageMaker, Google Vertex AI, and others. As a miner, your technical expertise becomes part of a competitive marketplace that consistently delivers models with significant performance improvements over baseline. 📈

## 🛠️ Essential Setup Components 🔩

### Learning Rate Optimisation 📊

Finding the optimal learning rate is perhaps the single most important factor for successful training outcomes:

- Consider implementing an automated LR finder that actively tests the loss landscape (like a customised version of Smith's "LR range test") 🔍
- For most LLM fine-tuning, starting points between 1e-5 and 3e-4 often yield good results, but the optimal rate varies significantly by model size and data domain
- Prioritise model-specific LR strategies: smaller models (1-3B) generally benefit from higher learning rates, while larger models (>7B) typically need lower rates
- Don't just set and forget - adaptive LR schedules that respond to validation metrics often outperform static approaches 🧪

### Job Selection & Acceptance 🎯

Being strategic about which jobs to accept might improve your performance:

- One approach could be starting with jobs you're confident you can complete successfully
- The base miner is quite restrictive in what it accepts by default
- You might want to modify the `job_accept_endpoint` to gradually accept more model families
- Keep track of different task types - standard instruction tuning tasks and DPO (Direct Preference Optimisation) tasks have different requirements ⚖️
- Keeping track of which model families work with your specific setup could save you from failed jobs

As shown in the example code, a basic acceptance strategy might filter by:
```python
# Simple example of job acceptance logic
if request.task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK]:
    return MinerTaskResponse(message="Only accepting text instruction and DPO tasks", accepted=False)

if "llama" not in request.model.lower():
    return MinerTaskResponse(message="Only accepting llama-type jobs currently", accepted=False)

if request.hours_to_complete < 13:
    return MinerTaskResponse(message="Accepting your job", accepted=True)
```

You'll want to optimise this based on your infrastructure capabilities and specialisation. 🛠️

### Smart Proxy Management 🔄

A proxy in this context means having a node that accepts all/more jobs and then distributes them to different GPUs based on job characteristics:

- Consider developing a routing system that matches models to the GPUs best suited to handle them 🧩
- Build a knowledge base of which model families work with specialised optimisations (Flash Attention, etc.)
- Track success rates by model type and size to make intelligent decisions about job acceptance
- Advanced miners will use a triage system that analyses job requirements before binding GPU resources 🔬

## 🔄 Advanced Training Dynamics 🧬

Former top miners suggest that adaptive training approaches significantly improved their results:

### Advanced Training Techniques 🔥

Here are some high-level approaches that have proven effective for LLM fine-tuning:

1. **Exponential Moving Average (EMA)** 📉
   - Maintaining an EMA of model weights often produces more stable models
   - EMA models typically generalise better to unseen data compared to last-checkpoint models
   - Consider using EMA decay rates between 0.998-0.9999 depending on training duration

2. **Effective LoRA Configurations** ⚙️
   - Higher ranks (128-256) often outperform the commonly used default of 8 or 16
   - Consider task-specific adapters for different parts of the model (attention vs. feed-forward)
   - Experiment with newer variants like QLoRA, GaLore, or DoRA for improved performance

3. **Dynamic Training Schedules** ⏱️
   - Implement training that automatically extends when improvement continues
   - Use early stopping when validation metrics plateau to prevent overfitting
   - Consider multi-stage approaches: broad learning followed by focused refinement

4. **Validation-Driven Adjustments** 📋
   - Set up systems that can backtrack to earlier checkpoints when performance degrades
   - Monitor the generalisation gap between training and validation metrics
   - Implement automatic hyperparameter adjustments based on validation signals

### Adaptive Rank Selection for LoRA 🧮

Instead of using fixed ranks across all adapter matrices, you might consider:
- Dynamically assigning different ranks to different layers based on attention pattern analysis
- Higher ranks for later transformer layers for example where most learning takes place 
- Minimal adaptation for embedding layers to preserve fundamental knowledge

### Model Size-Specific Strategies 📏

Different approaches might work better for different model sizes:

**For small models (<3B parameters)**: 🐣
- Full fine-tuning might yield better results when GPU memory allows
- Consider tuning all layers if your hardware can handle it

**For larger models**: 🐘
- It could also be the case that full-finetuning is better but it really depends on the amount of data you have 
- High-rank LoRA (rank 256+) has been reported to perform well
- Full fine-tune the final layer(s) could be a good compromise
- Being strategic about which parameters to freeze might balance performance and resource usage

**For DPO tasks**: 👍👎
- These require special handling as you're optimising based on preference pairs
- Consider different learning rates compared to standard instruction tuning
- Pay special attention to preventing overfitting, as DPO can be particularly sensitive

## 🔧 Technical Implementation Tips 💻

### Framework Selection 🏗️

While Axolotl is provided as a starting point, your choice of framework could make a difference:

- Different frameworks offer varying memory optimisations
- Experimentation could help you find what works best for your hardware setup
- Developing your own training pipeline might give you maximum control

### Memory Management 💾

When working with limited GPU resources, some approaches to consider:
- Monitoring memory usage throughout training
- Trying gradient checkpointing to reduce memory footprint
- Experimenting with precision adjustments (bf16 where supported)
- Testing efficient attention mechanisms when compatible with your models

### Scheduler Optimisation ⏰

Rather than static training schedules, some winning approaches implement:
- Dynamic decay patterns that respond to validation metrics
- Advanced schedulers that prevent overfitting while ensuring convergence
- Domain-specific adjustments for different types of data

## ⚙️ Job Scheduling Improvements 📆

The base code uses a single element blocking queue, but there might be better approaches:

- You could implement priority queuing based on expected completion time
- Job batching for similar model types might improve throughput
- Parallel preparation while training occurs could reduce downtime
- A cooldown system for resource-intensive jobs might prevent overloading

## 🚫 Common Pitfalls to Avoid ⚠️

Based on experiences shared by successful miners:

1. **Accepting jobs you can't complete** - There's a significant punishment factor! 😱
2. **Using a one-size-fits-all approach** - Different data domains need different strategies
3. **Ignoring early warning signs** - Monitoring training dynamics could save you time ⏱️
4. **Overlooking memory management** - This becomes especially critical for larger models
5. **Not adapting to training signals** - Static hyperparameters rarely work best
6. **Failing to distinguish between task types** - DPO tasks and standard instruction tuning need different handling

## 🌟 Getting Started Steps 🚶‍♂️

A possible pathway to success:

1. Set up your basic miner and validate its functionality ✅
2. Consider implementing an automated LR finder 🔎
3. You might want to start with a restricted set of model families that you know work
4. Create a dynamic config that changes parameters (like batch-size) depending on the job details
5. Gradually expanding to more model types as you gain confidence 🌱
6. Collecting data on what works and what doesn't could inform future decisions 📊
7. Optimising your job scheduling logic might improve throughput
8. Adding adaptive callbacks could optimise your training 🔄

## 🤝 Final Thoughts 💭

Mining on Gradients is challenging but rewarding. The advanced approaches required will make you a better ML engineer, and your contributions directly help users achieve the incredible results that outperform what they'd get from major tech platforms. 🏆

Good luck and happy mining! 🚀 💰 💎

---

*Special thanks to contributing miners who have shared their expertise and code, particularly NeuralWonderland. These contributions have helped advance the entire mining community.* 🙏
