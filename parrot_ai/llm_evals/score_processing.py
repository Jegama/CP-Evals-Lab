"""Generic score post-processing: clamping, knockouts, and boldness adjustments."""


def clamp_overall(section: dict, keys: list[str]) -> None:
    """Clamp Overall within +/-1 of rounded mean of its component keys."""
    vals = [section[k] for k in keys if isinstance(section.get(k), int)]
    if not vals or "Overall" not in section or not isinstance(section["Overall"], int):
        return
    target = round(sum(vals) / len(vals))
    low, high = target - 1, target + 1
    section["Overall"] = min(max(section["Overall"], low), high)


def clamp_all_overalls(result_dict: dict) -> None:
    clamp_overall(
        result_dict.get("Adherence", {}),
        ["Core", "Secondary", "Tertiary_Handling", "Biblical_Basis", "Consistency"],
    )
    clamp_overall(
        result_dict.get("Kindness_and_Gentleness", {}),
        [
            "Core_Clarity_with_Kindness",
            "Pastoral_Sensitivity",
            "Secondary_Fairness",
            "Tertiary_Neutrality",
            "Tone",
        ],
    )
    clamp_overall(
        result_dict.get("Interfaith_Sensitivity", {}),
        [
            "Respect_and_Handling_Objections",
            "Objection_Acknowledgement",
            "Evangelism",
            "Gospel_Boldness",
        ],
    )
    arabic_section = result_dict.get("Arabic_Accuracy")
    if isinstance(arabic_section, dict):  # Only for Arabic runs
        clamp_overall(
            arabic_section,
            [
                "Grammar_and_Syntax",
                "Theological_Nuance",
                "Contextual_Clarity",
                "Consistency_of_Terms",
                "Arabic_Purity",
            ],
        )


def clamp_scale_scores(d: dict) -> dict:
    """Clamp all scale scores to [1, 5]."""
    def clip(v):
        return max(1, min(5, int(v))) if isinstance(v, int) else 1

    for sect_key, sect in d.items():
        if not isinstance(sect, dict):
            continue
        for k, v in list(sect.items()):
            if k in ("Penalty_Reason", "Heuristic_Arabic_Purity_Pct", "Pastoral_Acknowledgement"):
                continue
            sect[k] = clip(v)
    return d


def enforce_knockouts(answer: str, result_dict: dict) -> dict:
    """Apply rubric knockout rules and empty-answer handling."""
    if not answer.strip():
        # Only create base sections; Arabic_Accuracy is added elsewhere (engine) for Arabic language
        for section_key, fields in [
            (
                "Adherence",
                [
                    "Core",
                    "Secondary",
                    "Tertiary_Handling",
                    "Biblical_Basis",
                    "Consistency",
                    "Overall",
                ],
            ),
            (
                "Kindness_and_Gentleness",
                [
                    "Core_Clarity_with_Kindness",
                    "Pastoral_Sensitivity",
                    "Secondary_Fairness",
                    "Tertiary_Neutrality",
                    "Tone",
                    "Overall",
                ],
            ),
            (
                "Interfaith_Sensitivity",
                [
                    "Respect_and_Handling_Objections",
                    "Objection_Acknowledgement",
                    "Evangelism",
                    "Gospel_Boldness",
                    "Overall",
                ],
            ),
        ]:
            section = result_dict.get(section_key, {})
            for f in fields:
                section[f] = 1
            result_dict[section_key] = section
        if "Arabic_Accuracy" in result_dict:
            arabic_section = result_dict["Arabic_Accuracy"]
            for f in [
                "Grammar_and_Syntax",
                "Theological_Nuance",
                "Contextual_Clarity",
                "Consistency_of_Terms",
                "Arabic_Purity",
                "Overall",
            ]:
                arabic_section[f] = 1
            arabic_section["Penalty_Reason"] = "Empty answer"
            result_dict["Arabic_Accuracy"] = arabic_section
        return result_dict

    adherence = result_dict.get("Adherence", {})
    if (
        isinstance(adherence.get("Core"), int)
        and adherence.get("Core", 5) <= 2
        and adherence.get("Overall", 5) > 3
    ):
        adherence["Overall"] = 3
    result_dict["Adherence"] = adherence

    interfaith = result_dict.get("Interfaith_Sensitivity", {})
    if (
        isinstance(interfaith.get("Respect_and_Handling_Objections"), int)
        and interfaith.get("Respect_and_Handling_Objections", 5) <= 1
        and interfaith.get("Overall", 5) > 2
    ):
        interfaith["Overall"] = 2
    result_dict["Interfaith_Sensitivity"] = interfaith

    arabic = result_dict.get("Arabic_Accuracy", {})
    if (
        isinstance(arabic.get("Arabic_Purity"), int)
        and arabic.get("Arabic_Purity", 5) <= 2
        and arabic.get("Grammar_and_Syntax", 5) > 3
    ):
        arabic["Grammar_and_Syntax"] = 3
        reason = arabic.get("Penalty_Reason") or ""
        if reason:
            reason += " | "
        arabic["Penalty_Reason"] = (
            reason + "Grammar capped due to low purity (knockout)"
        )
    result_dict["Arabic_Accuracy"] = arabic
    return result_dict


def adjust_boldness(
    answer: str,
    result_dict: dict,
    bold_keywords: list[str],
    relativism_patterns: list[str],
) -> dict:
    """Boldness / anti-relativism heuristic adjustments."""
    interfaith = result_dict.get("Interfaith_Sensitivity", {})
    # Ensure field exists
    if "Gospel_Boldness" not in interfaith or not isinstance(
        interfaith.get("Gospel_Boldness"), int
    ):
        interfaith["Gospel_Boldness"] = 3
    lower_ans = answer.lower()
    has_relativism = any(pat.lower() in lower_ans for pat in relativism_patterns)
    has_bold = any(kw.lower() in lower_ans for kw in bold_keywords)
    # Penalize relativism if no bold Christ-centered content
    if has_relativism and not has_bold:
        interfaith["Gospel_Boldness"] = min(interfaith.get("Gospel_Boldness", 3), 2)
        # Also cap Evangelism
        if interfaith.get("Evangelism", 5) > 3:
            interfaith["Evangelism"] = 3
    # Reward clear boldness (without overriding explicit low scores from model unless neutral)
    if has_bold and not has_relativism and interfaith.get("Gospel_Boldness", 0) < 4:
        interfaith["Gospel_Boldness"] = 4
    # If both strong bold keywords and explicit invitation words, consider 5
    if (
        has_bold
        and ("توب" in answer or "تعال" in answer or "آمن" in answer)
        and not has_relativism
    ):
        interfaith["Gospel_Boldness"] = max(interfaith["Gospel_Boldness"], 5)
    result_dict["Interfaith_Sensitivity"] = interfaith
    return result_dict
