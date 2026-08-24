"""AI reference patterns from Appendix B.3.

The patterns are applied with re.IGNORECASE to the stored continuation text as archived, 
with newlines already collapsed to spaces and no separate lowercasing.
Matching uses re.search over the text field of block 2 rows, and a continuation counts once per pattern it matches.
rho is the number of continuations matching FIRST_PERSON divided by the number matching THIRD_PERSON.
"""

import re

NARROW = re.compile(
    r"(?:\bas an ai\b|\blanguage model\b)",
    re.IGNORECASE,
)

FIRST_PERSON = re.compile(
    r"(?:\bas an ai\b[^.?!]{0,60}?\bi\b"
    r"|\bi\s*(?:'?m|am)\s+(?:just\s+|only\s+|merely\s+|simply\s+)?"
    r"(?:an?\s+)?(?:ai\b|a\.i\.|artificial intelligence"
    r"|language model|large language model|llm\b|chatbot|bot\b"
    r"|machine\b|computer program|computer\b|program\b|algorithm"
    r"|robot|software|neural network|model\b|assistant\b)"
    r"|\bi\s+(?:do not|don'?t|cannot|can'?t)\s+have\s+(?:a\s+)?"
    r"(?:feelings|emotions|consciousness|body|physical"
    r"|subjective|personal experien))",
    re.IGNORECASE,
)

THIRD_PERSON = re.compile(
    r"(?:\bthe (?:chatbot|bot|ai|model|system|program|machine"
    r"|assistant|computer)\b"
    r"|\bit (?:replied|answered|responded|said)\b)",
    re.IGNORECASE,
)