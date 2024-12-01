# A few golden nuggets for you miners

Although all models work out-the-box for  validation, it’s not the case for training. For example, some models work with flash attention, others don’t, the fun here is to find out what models work with what settings.

Since there’s a punishment factor on accepting and subsequently failing jobs, it’s super important to get this bit right. If you accept a job that you can’t do, you’ll be punished hard.

The base miner is very restricted on the jobs it accepts (see [job_accept_endpoint](https://github.com/rayonlabs/G.O.D/blob/2edec909aa0dca4a7231ead5cf4ad00bbb7d2782/miner/endpoints/tuning.py#L106)) so you'll want to increase the model families it accepts when you feel confident of completing the job. 

A decent approach would be to start with a setting (say flash attention on - Lora only for k and values, as a crude example), find the key model families that work well in this setting and then only accept jobs for these model families.

Then expand to other settings.

Part two of the fun is then looking at how best to schedule jobs, the base code is a single element blocking queue - perhaps not optimal.

Good luck
