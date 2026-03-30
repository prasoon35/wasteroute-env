"""
Graders for WasteRoute-Env.
Each grader takes an episode result and returns a score 0.0-1.0.
"""

def grade_easy(obs_history: list, final_obs) -> float:
    """
    Easy grader: weighted by fill levels of collected bins.
    Score = avg fill of collected bins + fuel bonus
    """
    total_bins = 5
    collected = final_obs.collected_bins

    if not collected:
        return 0.0

    # Get fill levels at collection time from obs history
    fill_rewards = []
    for obs in obs_history:
        if obs.message and "Collected bin" in obs.message:
            # Extract fill from message e.g "fill=0.65"
            try:
                fill = float(obs.message.split("fill=")[1].split(")")[0])
                fill_rewards.append(fill)
            except:
                fill_rewards.append(0.5)

    # Coverage score — did agent collect all bins?
    coverage = len(collected) / total_bins

    # Quality score — how full were collected bins?
    quality = sum(fill_rewards) / len(fill_rewards) if fill_rewards else 0.5

    score = (coverage * 0.6) + (quality * 0.4)

    # Fuel bonus
    if final_obs.fuel_remaining > 0.5:
        score = min(1.0, score + 0.1)

    return round(min(1.0, score), 3)


def grade_medium(obs_history: list, final_obs) -> float:
    """
    Medium grader: coverage + quality + fuel management.
    """
    total_bins = 15
    collected = final_obs.collected_bins

    if not collected:
        return 0.0

    # Coverage
    coverage = len(collected) / total_bins

    # Quality from messages
    fill_rewards = []
    for obs in obs_history:
        if obs.message and "Collected bin" in obs.message:
            try:
                fill = float(obs.message.split("fill=")[1].split(")")[0])
                fill_rewards.append(fill)
            except:
                fill_rewards.append(0.5)

    quality = sum(fill_rewards) / len(fill_rewards) if fill_rewards else 0.5

    # Fuel efficiency
    fuel_score = (final_obs.fuel_remaining / 0.7) * 0.2
    fuel_score = max(0.0, fuel_score)

    score = (coverage * 0.5) + (quality * 0.3) + fuel_score
    return round(min(1.0, score), 3)


def grade_hard(obs_history: list, final_obs) -> float:
    """
    Hard grader: coverage + quality + fuel + overflow penalties.
    """
    total_bins = 20
    collected = final_obs.collected_bins

    if not collected:
        return 0.0

    # Coverage
    coverage = len(collected) / total_bins

    # Quality from messages
    fill_rewards = []
    for obs in obs_history:
        if obs.message and "Collected bin" in obs.message:
            try:
                fill = float(obs.message.split("fill=")[1].split(")")[0])
                fill_rewards.append(fill)
            except:
                fill_rewards.append(0.5)

    quality = sum(fill_rewards) / len(fill_rewards) if fill_rewards else 0.5

    # Fuel score
    fuel_score = (final_obs.fuel_remaining / 0.5) * 0.2
    fuel_score = max(0.0, fuel_score)

    # Overflow penalty capped at 0.2
    overflow_count = sum(
        1 for obs in obs_history
        if "OVERFLOWED" in obs.message
    )
    overflow_penalty = min(0.2, overflow_count * 0.02)

    score = (coverage * 0.5) + (quality * 0.3) + fuel_score - overflow_penalty
    return round(max(0.0, min(1.0, score)), 3)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

def grade(task: str, obs_history: list, final_obs) -> float:
    """Main grader entry point."""
    if task not in GRADERS:
        raise ValueError(f"Unknown task: {task}")
    return GRADERS[task](obs_history, final_obs)
