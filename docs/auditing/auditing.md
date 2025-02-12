# Auditing

### What is Auditing?

Auditing is the process of verifying that the subnet developer validator is functioning fairly. It ensures that miners are assigned weights for their actual work done, and that these weights are calculated in a transparent and unbiased manner.

### Why Do We Want Auditing?

By providing a transparent and verifiable way to evaluate miner performance, auditing helps to:

* Ensure that miners are rewarded fairly for their work
* Prevent manipulation or bias in the weight calculation process
* Provide a clear and understandable picture of how miner scores are determined

The biggest benefit, is that it allows just one validator to run the validator code, giving immense power to subnet owners and the rapildy improving our ecosystem.

### How Does Auditing Work?

1. **Record every detail**: Record every task that contributes to a miner's score, including the reasons for their relative score.
2. **Provide score verification**: Allow auditors to verify miner scores by running evaluations ad-hoc.
3. **Run synthetic jobs**: Periodically run synthetic jobs and check that they appear in the scoring API.

### Implementation

**Auditing endpoints**:

1. Have an endpoint to retrieve the most recent X tasks and their headline details for a given hotkey, with filtering by hotkey, [here](../validator/endpoints/auditing.py#L10)
2. Have an endpoint to retrieve full details for a specific task, [here](../validator/endpoints/auditing.py#L16)
3. Have a script to send a synthetic job and ping the endpoint 24 hours later to verify its presence in the scoring API.
4. Have a script get the validator on chain weights set and pull the explanations and check they match

### Phase 2: Future Development

In the future, we plan to:

* **Improve score evaluation**:

Develop an easier way to audit the evaluation of quality scores. This is all currently possible, but would need some manual work from the auditor which we want to avoid.

* **Implement miner reporting**: Allow miners to report their own scores and provide additional transparency.
Pesky miners be pesky, so we probably need some 'trusted' miners on rotation, perhaps run by the auditors. This is a marginal improvement though and can be in phase 2.

### FAQs

* **How do I run auditing?**: Auditing can be ran by following the instructions in [auditing_setup.md](auditing_setup.md)
* **How do I know it's working?**: Auditing is working if the scores are being calculated correctly and the synthetic jobs are appearing in the scoring API - and your weights are set if you are running a separate validator.
* **Isn't this weight copying?**: No, it has none of the downsides of weight copying. This is basically CHK with additional measures to ensure fairness.
