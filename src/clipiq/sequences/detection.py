def find_anticipation_payoff_pairs(blocks, max_gap=10):
    """
    Find adjacent speech blocks where anticipation is followed by payoff.
    """

    pairs = []

    for i in range(len(blocks) - 1):
        current = blocks[i]
        following = blocks[i + 1]

        gap = following["start"] - current["end"]

        has_anticipation = current.get("anticipation", 0) > 0

        has_payoff = (
            following.get("excitement", 0) > 0
            or following.get("payoff", 0) > 0
        )

        if has_anticipation and has_payoff and 0 <= gap <= max_gap:
            pairs.append({
                "setup": current,
                "payoff": following,
                "gap": gap,
                "start": current["start"],
                "end": following["end"]
            })

    return pairs


def find_extended_sequences(blocks, max_gap=60):
    """
    Find anticipation or tactical-decision blocks followed by
    action or payoff blocks within a larger time window.
    """

    sequences = []

    setup_categories = {"anticipation", "tactical_decision"}
    payoff_categories = {"action", "payoff"}

    for i, setup in enumerate(blocks):
        if setup.get("category") not in setup_categories:
            continue

        for payoff in blocks[i + 1:]:
            gap = payoff["start"] - setup["end"]

            if gap > max_gap:
                break

            if gap < 0:
                continue

            if payoff.get("category") not in payoff_categories:
                continue

            sequences.append({
                "start": setup["start"],
                "end": payoff["end"],
                "setup": setup,
                "payoff": payoff,
                "gap": gap,
                "human_verified": False
            })

    return sequences
