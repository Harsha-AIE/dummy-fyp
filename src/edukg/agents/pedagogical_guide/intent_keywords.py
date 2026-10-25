from __future__ import annotations

from edukg.core.models.enums import IntentCategory

# Assumption (flagged, same shape as modules/subject_detection/subject_keywords.py):
# no intent-classification training data exists, so this offline,
# dependency-free keyword bank is the tier-1 strategy — swappable for a real
# trained classifier later without touching PedagogicalGuideAgent, which only
# depends on the IntentClassificationStrategy interface.
INTENT_KEYWORDS: dict[IntentCategory, list[str]] = {
    IntentCategory.MISCONCEPTION: [
        "i thought", "wait, why", "but i thought", "doesn't make sense", "dont make sense",
        "i'm confused", "im confused", "confused about", "don't understand", "dont understand",
        "why isn't", "why doesn't", "that's wrong", "thats wrong", "i don't get", "i dont get",
        "shouldn't it", "shouldnt it", "isn't that", "isnt that",
    ],
    IntentCategory.EXAMPLE_REQUEST: [
        "example", "for instance", "can you show", "show me", "give me an example",
        "like what", "such as", "demonstrate", "walk me through",
    ],
    IntentCategory.QUESTION: [
        "what is", "what are", "how does", "how do", "why does", "why do", "when does",
        "can you explain", "explain", "what does", "how is", "what happens", "how can",
    ],
    IntentCategory.OFF_TOPIC: [
        "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "lol", "ok", "okay",
        "cool", "nice", "great",
    ],
}
