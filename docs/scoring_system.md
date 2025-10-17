# Scoring System and Weight Distribution

## Constants Reference

The scoring system uses these key constants (see [`validator/core/constants.py`](../validator/core/constants.py)):

- `TOURNAMENT_TEXT_WEIGHT` - Proportion for text tournaments (0.55 = 55%)
- `TOURNAMENT_IMAGE_WEIGHT` - Proportion for image tournaments (0.45 = 45%)
- `TOURNAMENT_PARTICIPATION_WEIGHT` - Weight for active tournament participants
- `EMISSION_MULTIPLIER_THRESHOLD` - Performance threshold for emission increases
- `EMISSION_BURN_HOTKEY` - Address that receives burned emissions

## Overview

The Gradient subnet operates as a **tournament-only system**. All emissions are distributed between:

1. **Tournament winners** (text and image tournaments, calculated separately)
2. **Active participants** (small reward for tournament participation)
3. **Burn address** (all remaining weight goes to `EMISSION_BURN_HOTKEY`)

There is no legacy miner system - only tournament participants receive rewards.

## Weight Distribution

### Base Allocation

Total weight allocation is calculated as:

```python
text_base_weight = BASE_TOURNAMENT_WEIGHT * TOURNAMENT_TEXT_WEIGHT
image_base_weight = BASE_TOURNAMENT_WEIGHT * TOURNAMENT_IMAGE_WEIGHT
burn_weight = 1.0 - text_base_weight - image_base_weight
```

### Performance-Based Adjustments

When tournaments perform well against synthetic benchmarks, they receive emission multipliers (see [`calculate_emission_multiplier`](../validator/core/weight_setting.py)):

```python
# Only applies if performance exceeds threshold
if performance_diff > EMISSION_MULTIPLIER_THRESHOLD:
    emission_increase = (performance_diff - EMISSION_MULTIPLIER_THRESHOLD) * 2.0
    tournament_weight = base_weight + emission_increase
```

When tournaments perform poorly, no increase is applied, and more weight remains with the burn address.

## Tournament Weights

### Text and Image Separation

Text and image tournaments are completely independent:

- Text tournament winners receive `text_tournament_weight` distributed by performance
- Image tournament winners receive `image_tournament_weight` distributed by performance
- A miner can win both tournaments and receive rewards from each

### Weight Calculation

Within each tournament, weights are distributed based on tournament performance (see [`get_tournament_weights_from_data`](../validator/evaluation/tournament_scoring.py)):

1. Final round determines the winner
2. Earlier rounds receive progressively smaller allocations
3. Higher-ranking participants in each round receive more weight

## Participation Rewards

Active tournament participants receive a small fixed reward:

```python
participation_weight = TOURNAMENT_PARTICIPATION_WEIGHT  # per participant
```

This incentivizes participation independent of performance.

### Scaling for Participation

When many participants are active, tournament and burn weights are scaled down proportionally:

```python
participation_total = len(participants) * TOURNAMENT_PARTICIPATION_WEIGHT
scale_factor = 1.0 - participation_total

scaled_text_weight = text_tournament_weight * scale_factor
scaled_image_weight = image_tournament_weight * scale_factor
scaled_burn_weight = burn_weight * scale_factor
```

## Example Scenario

Starting conditions:

- `BASE_TOURNAMENT_WEIGHT = 0.9`
- `TOURNAMENT_TEXT_WEIGHT = 0.55`
- `TOURNAMENT_IMAGE_WEIGHT = 0.45`
- Text tournament performance: Good (exceeds threshold by 0.1)
- Image tournament performance: Average (no threshold exceeded)
- 10 active participants

**Step 1: Calculate base weights**

```
text_base_weight = 0.9 * 0.55 = 0.495
image_base_weight = 0.9 * 0.45 = 0.405
burn_weight = 1.0 - 0.495 - 0.405 = 0.1
```

**Step 2: Apply performance multipliers**

```
text_emission_increase = 0.1 * 2.0 = 0.2
text_tournament_weight = 0.495 + 0.2 = 0.695

image_emission_increase = 0.0
image_tournament_weight = 0.405 + 0.0 = 0.405

burn_weight = 1.0 - 0.695 - 0.405 = -0.1 (capped at minimum)
```

Note: When emission increases exceed available weight, the system caps allocations appropriately.

**Step 3: Scale for participation**

```
participation_total = 10 * 0.01 = 0.1  # (assuming TOURNAMENT_PARTICIPATION_WEIGHT = 0.01)
scale_factor = 1.0 - 0.1 = 0.9

scaled_text_weight = 0.695 * 0.9 = 0.626
scaled_image_weight = 0.405 * 0.9 = 0.365
scaled_burn_weight = -0.1 * 0.9 = -0.09 (capped at 0)
participation_weights = 0.1
```

**Final Distribution:**

- Text tournament: 62.6%
- Image tournament: 36.5%
- Participation: 10%
- Burn: ~0% (text tournament performed well, consuming most available weight)

## Key Characteristics

**Tournament-Only**: No legacy/regular mining - only tournament participants receive rewards.

**Performance Incentives**: Tournaments that outperform benchmarks receive emission multipliers, increasing their share.

**Burn as Default**: All weight not allocated to tournaments or participation goes to the burn address.

**Type-Specific Rewards**: Text and image tournaments are independent - good performance in one doesn't affect the other.

**Participation Encouraged**: Small guaranteed reward for active participation, independent of performance.

## Implementation Details

The main entry point is [`get_node_weights_from_tournament_audit_data`](../validator/core/weight_setting.py) which orchestrates the entire process. Key functions include:

- [`get_tournament_burn_details`](../validator/core/weight_setting.py) - Calculates weight allocations based on performance
- [`get_tournament_weights_from_data`](../validator/evaluation/tournament_scoring.py) - Distributes weights within tournaments
- [`apply_tournament_weights`](../validator/core/weight_setting.py) - Applies final weight assignments
- [`calculate_performance_difference`](../validator/tournament/performance_calculator.py) - Compares tournament vs benchmark performance
