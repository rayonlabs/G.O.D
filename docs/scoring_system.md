# Scoring System and Weight Distribution

## Constants Reference

The scoring system uses these key constants (see [`validator/core/constants.py`](../validator/core/constants.py)):

- `BURN_REDUCTION_RATE` - Burn multiplier (determines burn rate per % performance difference)
- `LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT` - Legacy boost multiplier
- `TOURNAMENT_TEXT_WEIGHT` - Base text tournament weight
- `TOURNAMENT_IMAGE_WEIGHT` - Base image tournament weight
- `BASE_TOURNAMENT_WEIGHT` - Base tournament weight allocation
- `BASE_REGULAR_WEIGHT` - Base regular weight allocation
- `EMISSION_BURN_HOTKEY` - Address that receives burned emissions
- `MAX_BURN_PROPORTION` - Maximum proportion of weight that can be burned

## Overview

The Gradient subnet balances regular task performance with tournament results using separated burn dynamics. Text and image miners are treated differently based on how their respective tournaments perform compared to regular tasks.

## How Miners Are Classified

### Tournament Miners
Miners get classified based on which tournaments they participate in (see [`get_tournament_participation_data`](../validator/db/sql/tournaments.py)):

- **Text-only**: Only participated in text tournaments
- **Image-only**: Only participated in image tournaments
- **Both**: Participated in both tournament types (weighted by `TOURNAMENT_TEXT_WEIGHT` and `TOURNAMENT_IMAGE_WEIGHT`)

### Legacy Miners

Miners who do real time tasks separate from the tournaments. They're classified by their task completion history over the past 7 days - what proportion of their tasks were text vs image (see [`get_weekly_task_participation_data`](../validator/db/sql/tournaments.py)).


## Tournament Performance Tracking

The system compares tournament winners against the best legacy miner scores for the same tasks (see [`calculate_performance_difference`](../validator/core/weight_setting.py)). When tournaments underperform, this triggers burn dynamics.

Performance differences are calculated as:
```
performance_difference = (tournament_score - best_legacy_score_for_same_task) / best_legacy_score_for_same_task
```

## Separated Burn System

### The Core Mechanism
When tournaments underperform, weight gets redistributed (see [`get_tournament_burn_details_separated`](../validator/core/weight_setting.py)):

1. **Tournament miners lose weight** - proportional to their tournament type's underperformance
2. **Legacy miners gain weight** - they get boosted because their approach outperformed tournaments
3. **Excess weight goes to burn** - sent to `EMISSION_BURN_HOTKEY`

### Text vs Image Separation
The key insight is that text and image tournaments are evaluated separately:

- If text tournaments perform poorly but image tournaments perform well, only text miners face penalties
- Legacy miners get different boosts for their text vs image task work
- Weight redistribution happens proportionally based on `TOURNAMENT_TEXT_WEIGHT` and `TOURNAMENT_IMAGE_WEIGHT`

### The Burn Rate
The system uses `BURN_REDUCTION_RATE` as a multiplier:
```
burn_proportion = min(MAX_BURN_PROPORTION, abs(performance_difference) * BURN_REDUCTION_RATE)
```

This creates strong incentives for tournament performance while capping maximum burn at `MAX_BURN_PROPORTION`.

## Weight Application

### Tournament Miners
Tournament miners receive weights based on their performance and participation (see [`apply_tournament_weights_separated`](../validator/core/weight_setting.py)):

- Their tournament ranking (better performers get more weight)
- Which tournaments they participated in
- How well those tournaments performed overall
- The scaled weight allocation after burn adjustments

### Legacy Miners
Legacy miners receive their base performance weight plus boosts (see [`apply_regular_weights_separated`](../validator/core/weight_setting.py)):
```
boost = performance_difference * LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT
```
- Boosts are calculated from tournament underperformance
- Applied proportionally to their text vs image task participation over the past 7 days

### Burn Allocation
All burned weight goes to `EMISSION_BURN_HOTKEY` rather than being lost, maintaining the total emission while redistributing it based on performance.

## Example Scenario

Consider a scenario where:
- Text tournaments underperform by 14% compared to best legacy miner scores on the same tasks (`text_performance_diff = -0.14`)
- Image tournaments underperform by 4% compared to best legacy miner scores on the same tasks (`image_performance_diff = -0.04`)

**Burn Calculations (using BURN_REDUCTION_RATE = 5.0, MAX_BURN_PROPORTION = 0.75):**
```
text_burn_proportion = min(0.75, 0.14 * 5.0) = 0.70
image_burn_proportion = min(0.75, 0.04 * 5.0) = 0.20
```

**Weight Redistribution (using BASE_TOURNAMENT_WEIGHT = 0.525, TOURNAMENT_TEXT_WEIGHT = 0.55, TOURNAMENT_IMAGE_WEIGHT = 0.45, BASE_REGULAR_WEIGHT = 0.15, LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT = 0.25):**

```
# Calculate tournament burn amounts
text_tournament_burn = 0.525 * 0.55 * 0.70 = 0.202
image_tournament_burn = 0.525 * 0.45 * 0.20 = 0.047

# Tournament weights after burn
text_tournament_weight = (0.525 * 0.55) - 0.202 = 0.087
image_tournament_weight = (0.525 * 0.45) - 0.047 = 0.189

# Regular weights get base + portion of tournament burn
text_regular_weight = 0.15 + (0.202 * 0.25) = 0.201
image_regular_weight = 0.15 + (0.047 * 0.25) = 0.162

# Burn weight gets remainder
total_tournament_burn = 0.202 + 0.047 = 0.249
burn_weight = (1 - 0.15 - 0.525) + (0.249 * (1 - 0.25)) = 0.512
```

**Legacy Boosts (using LEGACY_PERFORM_DIFF_EMISSION_GAIN_PERCENT = 0.25):**

```
text_legacy_boost = -0.14 * 0.25 = -0.035 (3.5% boost applied to individual legacy miners' scores)
image_legacy_boost = -0.04 * 0.25 = -0.01 (1% boost applied to individual legacy miners' scores)
```

**Results:**

- Text tournament weight reduced from 0.289 to 0.087 (70% reduction)
- Image tournament weight reduced from 0.236 to 0.189 (20% reduction)
- Text regular weight boosted from 0.15 to 0.201 (34% increase)
- Image regular weight boosted from 0.15 to 0.162 (8% increase)
- Legacy miners doing text tasks get 3.5% boost to their individual scores
- Legacy miners doing image tasks get 1% boost to their individual scores
- Miners doing both get proportional boosts based on their 7-day task mix
- 51.2% of total weight goes to `EMISSION_BURN_HOTKEY`


## Key Benefits

**Performance Incentives**: Tournament underperformance directly reduces tournament miner rewards while boosting legacy miners who outperformed.

**Type-Specific Fairness**: Text and image miners aren't penalized for the other type's poor performance.

**Proportional Participation**: Miners participating in both tournaments get appropriately weighted contributions from each.

**Legacy Protection**: Regular task miners get rewarded when their approach outperforms tournaments.

**System Stability**: All weight redistribution is conservative and bounded (max burn at `MAX_BURN_PROPORTION`) to prevent extreme swings.

This system creates a competitive balance where tournament participation is rewarded when it produces superior results, but legacy approaches are protected and boosted when tournaments fail to deliver improvements.

## Implementation Details

The main entry point is [`get_node_weights_from_period_scores_separated`](../validator/core/weight_setting.py) which orchestrates the entire process. Key functions include:

- [`get_tournament_weights_from_data_separated`](../validator/evaluation/tournament_scoring.py) - Calculates separate text/image tournament weights
- [`tournament_scores_to_weights`](../validator/evaluation/tournament_scoring.py) - Converts tournament scores to weight distributions
- [`calculate_final_round_winner`](../validator/evaluation/tournament_scoring.py) - Determines tournament winners

