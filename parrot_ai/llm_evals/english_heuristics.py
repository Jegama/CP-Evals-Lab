"""English-specific heuristic functions for ceiling-compression calibration."""

import re

# Regex for Scripture citations: "Book Chapter:Verse" patterns
# Matches patterns like "John 3:16", "1 Corinthians 15:3-4", "Romans 5:8"
_SCRIPTURE_RE = re.compile(
    r"\b(?:[123]\s?)?"                           # optional book number prefix
    r"(?:Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|Samuel|Kings"
    r"|Chronicles|Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs|Ecclesiastes"
    r"|Song\s+of\s+Solomon|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel"
    r"|Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah"
    r"|Haggai|Zechariah|Malachi"
    r"|Matthew|Mark|Luke|John|Acts|Romans|Corinthians|Galatians|Ephesians"
    r"|Philippians|Colossians|Thessalonians|Timothy|Titus|Philemon|Hebrews"
    r"|James|Peter|Jude|Revelation)"
    r"\s+\d{1,3}:\d{1,3}"                       # chapter:verse
    r"(?:\s*[-–]\s*\d{1,3})?",                  # optional verse range
    re.IGNORECASE,
)

_THEOLOGICAL_TERMS = [
    "substitutionary atonement", "penal substitution", "justification by faith",
    "justification by grace", "sola fide", "sola gratia", "sola scriptura",
    "solus christus", "soli deo gloria", "hypostatic union", "incarnation",
    "original sin", "total depravity", "unconditional election",
    "limited atonement", "particular redemption", "irresistible grace",
    "effectual calling", "perseverance of the saints", "imputation",
    "propitiation", "expiation", "sanctification", "glorification",
    "regeneration", "covenant theology", "federal headship",
    "means of grace", "general revelation", "special revelation",
    "inerrancy", "infallibility", "sovereignty of god",
    "trinity", "triune", "deity of christ", "virgin birth",
    "bodily resurrection", "second coming", "final judgment",
    "cessationism", "continuationism", "complementarian", "egalitarian",
]


def has_scripture_citation(answer: str) -> bool:
    """Return True if the answer contains at least one Scripture citation."""
    return bool(_SCRIPTURE_RE.search(answer))


def has_theological_terminology(answer: str) -> bool:
    """Return True if the answer uses at least one recognized theological term."""
    lower = answer.lower()
    return any(term in lower for term in _THEOLOGICAL_TERMS)


def calibrate_english_scores(question: str, answer: str, result_dict: dict) -> dict:
    """Cap inflated Adherence/Kindness scores when heuristic evidence is absent.

    Modeled on the sermon calibration pattern: if the judge gave a high score
    but the answer text lacks observable evidence, cap the score.
    """
    adherence = result_dict.get("Adherence", {})
    kindness = result_dict.get("Kindness_and_Gentleness", {})

    # Biblical_Basis > 3 but no Scripture citation -> cap at 3
    if (
        isinstance(adherence.get("Biblical_Basis"), int)
        and adherence["Biblical_Basis"] > 3
        and not has_scripture_citation(answer)
    ):
        adherence["Biblical_Basis"] = 3

    # Core > 4 but no theological terminology -> cap at 4
    if (
        isinstance(adherence.get("Core"), int)
        and adherence["Core"] > 4
        and not has_theological_terminology(answer)
    ):
        adherence["Core"] = 4

    # Pastoral_Sensitivity cap based on judge-provided Pastoral_Acknowledgement:
    #   "no"      -> cap at 3 (jumped straight to doctrine, no acknowledgment)
    #   "partial" -> cap at 4 (acknowledgment present but brief or generic)
    #   "yes"     -> no cap  (full acknowledgment; let judge score stand)
    #   None      -> no cap  (legacy result without this field; no action taken)
    pastoral_ack = kindness.get("Pastoral_Acknowledgement")
    if isinstance(kindness.get("Pastoral_Sensitivity"), int):
        if pastoral_ack == "no" and kindness["Pastoral_Sensitivity"] > 3:
            kindness["Pastoral_Sensitivity"] = 3
        elif pastoral_ack == "partial" and kindness["Pastoral_Sensitivity"] > 4:
            kindness["Pastoral_Sensitivity"] = 4

    result_dict["Adherence"] = adherence
    result_dict["Kindness_and_Gentleness"] = kindness
    return result_dict
