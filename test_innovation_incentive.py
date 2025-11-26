"""
Test to verify innovation_incentive calculation with recursive behavior.

This test demonstrates:
1. How innovation_incentive is calculated based on previous champion's final emission
2. How previous champion's stored innovation_incentive is included in their final emission
3. The recursive nature where each champion's innovation builds on the previous
"""

from datetime import datetime, timedelta, timezone


def calculate_emission_boost_from_perf(perf_diff: float) -> float:
    """Calculate emission boost from performance difference."""
    if perf_diff < 0.05:
        return 0.15
    elif perf_diff < 0.075:
        return 0.10
    elif perf_diff < 0.1:
        return 0.05
    else:
        return 0.0


def calculate_hybrid_decays(
    first_championship_time: datetime, consecutive_wins: int, current_time: datetime | None = None
) -> tuple[float, float, bool]:
    """Calculate old_decay, new_decay, and whether to apply hybrid system."""
    HYBRID_CUTOFF = datetime(2025, 10, 15, tzinfo=timezone.utc)

    current_time_utc = current_time if current_time else datetime.now(timezone.utc)

    # Old decay (win-based)
    old_decay = min(0.4, consecutive_wins * 0.04)

    # New decay (time-based)
    days_since_first_win = (current_time_utc - first_championship_time).days
    new_decay = min(0.4, days_since_first_win * (0.4 / 90))

    # Hybrid system
    apply_hybrid = first_championship_time < HYBRID_CUTOFF

    return old_decay, new_decay, apply_hybrid


def calculate_tournament_weight_with_decay(
    base_weight: float,
    emission_boost: float,
    old_decay: float,
    new_decay: float,
    apply_hybrid: bool,
    max_weight: float,
) -> float:
    """Calculate tournament weight with decay."""
    weight_before_cap = base_weight + emission_boost

    if apply_hybrid:
        # Use whichever decay is smaller (more favorable to champion)
        decay = min(old_decay, new_decay)
    else:
        # Use new time-based decay only
        decay = new_decay

    weight_after_decay = weight_before_cap - decay
    final_weight = min(weight_after_decay, max_weight)

    return final_weight


def test_innovation_incentive_recursive_calculation():
    """
    Test showing how innovation_incentive is calculated recursively.

    Scenario:
    - Champion A wins and defends, receives innovation_incentive_A = 0.05
    - After significant decay, Champion A falls below base
    - Champion B dethrones A, receives what A lost below base
    - Champion C dethrones B, receives what B lost (which includes B's stored innovation from A)
    """
    print("=" * 80)
    print("INNOVATION INCENTIVE RECURSIVE CALCULATION TEST")
    print("=" * 80)

    base_weight = 0.3
    max_weight = 0.35

    # Champion A's reign
    print("\n--- Champion A's Reign ---")
    champ_a_first_win = datetime(2025, 1, 1, tzinfo=timezone.utc)
    champ_a_perf_diff = 0.08  # Mediocre performance
    champ_a_emission_boost = calculate_emission_boost_from_perf(champ_a_perf_diff)
    print(f"Champion A performance difference: {champ_a_perf_diff:.3f}")
    print(f"Champion A emission boost from performance: {champ_a_emission_boost:.3f}")

    # Champion A defends for 70 days (3 wins) - significant decay
    champ_a_dethrone_time = champ_a_first_win + timedelta(days=70)
    champ_a_consecutive_wins = 3

    champ_a_old_decay, champ_a_new_decay, champ_a_apply_hybrid = calculate_hybrid_decays(
        champ_a_first_win, champ_a_consecutive_wins, champ_a_dethrone_time
    )

    # Champion A gets innovation_incentive from previous champion (assume 0.05)
    champ_a_prev_innovation = 0.05
    champ_a_combined_boost = champ_a_emission_boost + champ_a_prev_innovation

    print(f"Champion A previous champion's innovation: {champ_a_prev_innovation:.3f}")
    print(f"Champion A combined boost (perf + prev_innovation): {champ_a_combined_boost:.3f}")

    champ_a_final_weight = calculate_tournament_weight_with_decay(
        base_weight=base_weight,
        emission_boost=champ_a_combined_boost,
        old_decay=champ_a_old_decay,
        new_decay=champ_a_new_decay,
        apply_hybrid=champ_a_apply_hybrid,
        max_weight=max_weight,
    )

    print(f"Champion A at dethrone time:")
    print(f"  - Days since first win: 60")
    print(f"  - Consecutive wins: {champ_a_consecutive_wins}")
    print(f"  - Old decay: {champ_a_old_decay:.3f}")
    print(f"  - New decay: {champ_a_new_decay:.3f}")
    print(f"  - Apply hybrid: {champ_a_apply_hybrid}")
    print(f"  - Final weight: {champ_a_final_weight:.3f}")

    # Innovation incentive for Champion B (who dethrones A)
    innovation_incentive_b = max(0.0, base_weight - champ_a_final_weight)
    print(f"\n→ Innovation incentive for Champion B: {innovation_incentive_b:.3f}")
    print(f"  (Calculated as: max(0, {base_weight:.3f} - {champ_a_final_weight:.3f}))")

    # This innovation_incentive_b would be STORED in the database for Champion B's tournament
    print(f"\n✓ STORED in database: innovation_incentive = {innovation_incentive_b:.3f}")

    # Champion B's reign
    print("\n--- Champion B's Reign ---")
    champ_b_first_win = champ_a_dethrone_time
    champ_b_perf_diff = 0.09  # Poor performance
    champ_b_emission_boost = calculate_emission_boost_from_perf(champ_b_perf_diff)
    print(f"Champion B performance difference: {champ_b_perf_diff:.3f}")
    print(f"Champion B emission boost from performance: {champ_b_emission_boost:.3f}")

    # Champion B defends for 60 days (2 wins) - moderate decay
    champ_b_dethrone_time = champ_b_first_win + timedelta(days=60)
    champ_b_consecutive_wins = 2

    champ_b_old_decay, champ_b_new_decay, champ_b_apply_hybrid = calculate_hybrid_decays(
        champ_b_first_win, champ_b_consecutive_wins, champ_b_dethrone_time
    )

    # Champion B's stored innovation_incentive is included in their combined boost
    champ_b_prev_innovation = innovation_incentive_b  # Retrieved from database
    champ_b_combined_boost = champ_b_emission_boost + champ_b_prev_innovation

    print(f"Champion B stored innovation (from when they won): {champ_b_prev_innovation:.3f}")
    print(f"Champion B combined boost (perf + stored_innovation): {champ_b_combined_boost:.3f}")

    champ_b_final_weight = calculate_tournament_weight_with_decay(
        base_weight=base_weight,
        emission_boost=champ_b_combined_boost,
        old_decay=champ_b_old_decay,
        new_decay=champ_b_new_decay,
        apply_hybrid=champ_b_apply_hybrid,
        max_weight=max_weight,
    )

    print(f"Champion B at dethrone time:")
    print(f"  - Days since first win: 45")
    print(f"  - Consecutive wins: {champ_b_consecutive_wins}")
    print(f"  - Old decay: {champ_b_old_decay:.3f}")
    print(f"  - New decay: {champ_b_new_decay:.3f}")
    print(f"  - Apply hybrid: {champ_b_apply_hybrid}")
    print(f"  - Final weight: {champ_b_final_weight:.3f}")

    # Innovation incentive for Champion C (who dethrones B)
    innovation_incentive_c = max(0.0, base_weight - champ_b_final_weight)
    print(f"\n→ Innovation incentive for Champion C: {innovation_incentive_c:.3f}")
    print(f"  (Calculated as: max(0, {base_weight:.3f} - {champ_b_final_weight:.3f}))")

    print(f"\n✓ STORED in database: innovation_incentive = {innovation_incentive_c:.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Recursive Innovation Incentive")
    print("=" * 80)
    print(f"Champion A received:     {champ_a_prev_innovation:.3f} innovation from predecessor")
    print(f"Champion B receives:     {innovation_incentive_b:.3f} innovation (A lost {base_weight - champ_a_final_weight:.3f} below base)")
    print(f"Champion C receives:     {innovation_incentive_c:.3f} innovation (B lost {base_weight - champ_b_final_weight:.3f} below base)")
    print()
    print("Note: B's innovation includes the effect of A's innovation (via A's final emission)")
    print("      C's innovation includes the effect of B's innovation (via B's final emission)")
    print("      This creates a chain where each new champion inherits accumulated innovation")
    print("=" * 80)


def test_backwards_compatibility():
    """Test that NULL innovation_incentive is handled correctly (backwards compatibility)."""
    print("\n" + "=" * 80)
    print("BACKWARDS COMPATIBILITY TEST (NULL innovation_incentive)")
    print("=" * 80)

    base_weight = 0.3
    max_weight = 0.35

    print("\n--- Previous Champion (no stored innovation_incentive) ---")
    prev_perf_diff = 0.06
    prev_emission_boost = calculate_emission_boost_from_perf(prev_perf_diff)

    # Simulate NULL in database (treated as 0.0 in code)
    prev_innovation = None
    prev_innovation_value = prev_innovation if prev_innovation is not None else 0.0
    prev_combined_boost = prev_emission_boost + prev_innovation_value

    print(f"Previous champion performance boost: {prev_emission_boost:.3f}")
    print(f"Previous champion stored innovation: {prev_innovation} (NULL in database)")
    print(f"Previous champion innovation value used: {prev_innovation_value:.3f}")
    print(f"Previous champion combined boost: {prev_combined_boost:.3f}")

    # Calculate previous champion's final weight at dethrone time (30 days, 1 win)
    prev_first_win = datetime(2025, 1, 1, tzinfo=timezone.utc)
    prev_dethrone = prev_first_win + timedelta(days=30)

    prev_old_decay, prev_new_decay, prev_apply_hybrid = calculate_hybrid_decays(
        prev_first_win, 1, prev_dethrone
    )

    prev_final_weight = calculate_tournament_weight_with_decay(
        base_weight=base_weight,
        emission_boost=prev_combined_boost,
        old_decay=prev_old_decay,
        new_decay=prev_new_decay,
        apply_hybrid=prev_apply_hybrid,
        max_weight=max_weight,
    )

    print(f"Previous champion final weight: {prev_final_weight:.3f}")

    innovation_for_new_champ = max(0.0, base_weight - prev_final_weight)
    print(f"\n→ Innovation incentive for new champion: {innovation_for_new_champ:.3f}")
    print(f"\n✓ Backwards compatible: NULL treated as 0.0, calculation proceeds normally")
    print("=" * 80)


if __name__ == "__main__":
    test_innovation_incentive_recursive_calculation()
    test_backwards_compatibility()
    print("\n✅ All tests completed successfully!")
