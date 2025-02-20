# Tips on Training Diffusion LoRA Models

The tricky part in the diffusion training is that there can be 2-3 different types of training objectives, its upto the miner to figure that out and optimize the parameters based on that information. Lets divide the most common training objectives into 2 categories:

## Training a style

The default config is optimized for style LoRA training on 10-15 images. In my research, styles work better with lower learning rates and higher number of repeats. For the most part, you will need to worry about these parameters:

**Diffusion Repeats:**  Set to 10 as a default, can be decreased for person/concept/object training.
**Epochs:** Set to 10 as a default. In my research this range is fine for styles, though for person/concept/object, this can very easily overfit the model so reducing it got better results.
**Learning Rate**: "learning_rate", "unet_lr" and "text_encoder_lr" are all set to 1e-5 as a default. Should be increased for person/concept/object.
**Train Batch Size**: Set to 4 as a default, got me better results on styles with the rest of the parameters as mentioned above.

## Training a person/concept/object

These trainings tend to overfit very quickly. Sharing some parameters that got me good results on that:

**Diffusion Repeats:** Lower repeats, ~8 worked better in my working.
**Epochs:** The model converges much faster, reducing the total epochs is a good idea. 8 worked well for me for training a person.
**Learning Rate:** Should be increased in this case, 4e-5 got me good results for a person on 20-25 images. Can be decreased if the training data size is lower.
**Train Batch Size:** With the other mentioned parameters, a batch size of 2 worked fine.

We use Kohya's engine for training. Remember that the above mentioned tips are shared with the optimizer set to "AdamW8Bit" as a default. The dynamics change a lot with different optimizer types and you are free to experiment as long as it produces good results. For example "Prodigy" is a very commonly used optimizer that adapts the learning rate automatically.
