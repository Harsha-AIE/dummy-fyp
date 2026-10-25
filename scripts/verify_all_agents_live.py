"""Run: python scripts/verify_all_agents_live.py (with PYTHONPATH=src)

Live, end-to-end verification of all three tutor agents against the real
running backend (http://127.0.0.1:8000) — Pedagogical Guide intent
classification across every intent class, Knowledge Manager retrieval
across every ingested document, and Knowledge Tracing's mastery evolution
over a multi-turn conversation (including a concept outside the KG at DKT
training time, to confirm the BKT-only fallback works). Uses a fresh random
student id each run, so it never interferes with anyone else's mastery
state. Requires the backend to be running — start it first.
"""

from __future__ import annotations

import json
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"
STUDENT = f"deep-test-{uuid.uuid4()}"


def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------------------------------------------------------------------------
# 1. Pedagogical Guide — intent classification across varied phrasings
# ---------------------------------------------------------------------------
section("Pedagogical Guide — intent classification")
PG_CASES = [
    ("What is the role of genome sequencing in discovering antibiotics?", "question"),
    ("How does DNA sequencing actually work?", "question"),
    ("I think antibiotics are made by viruses, not bacteria or fungi.", "misconception"),
    ("I don't think fMRI can measure blood flow, I believe it directly reads neuron activity.", "misconception"),
    ("Can you give me an example of a biosynthetic gene cluster?", "example_request"),
    ("Show me an example of how tangent space embedding is used.", "example_request"),
    ("hi", "off_topic"),
    ("what's for lunch today", "off_topic"),
]
pg_correct = 0
for message, expected in PG_CASES:
    resp = post("/api/agents/classify", {"message": message, "student_id": STUDENT})
    got = resp["intent"]
    ok = got == expected
    pg_correct += ok
    concepts = [c["canonical_name"] for c in resp["detected_concepts"]]
    print(f"{'OK ' if ok else 'FAIL'} expected={expected:16s} got={got:16s} escalated={resp['escalated_to_llm']!s:5s} concepts={concepts}  <- {message[:60]}")
print(f"\nPedagogical Guide: {pg_correct}/{len(PG_CASES)} correct")

# ---------------------------------------------------------------------------
# 2. Knowledge Manager — retrieval across every ingested document's topic
# ---------------------------------------------------------------------------
section("Knowledge Manager — retrieval across all 4 documents")
KM_CASES = [
    ("what is DNA sequencing used for", "DNA Sequencing"),
    ("how does genome mining help discover antibiotics", "antibiotic"),
    ("what dataset was used for the autism fMRI study", "ABIDE"),
    ("what is tangent space embedding", "Tangent"),
    ("how do plants convert sunlight into energy", "Photosynthesis"),
    ("what do stomata do", "Stomata"),
]
km_correct = 0
for query, expected_keyword in KM_CASES:
    resp = post("/api/agents/retrieve", {"query": query, "top_k": 3, "student_id": STUDENT})
    results = resp["results"]
    top_text = results[0]["text"] if results else ""
    ok = expected_keyword.lower() in top_text.lower()
    km_correct += ok
    top_score = results[0]["final_score"] if results else None
    print(f"{'OK ' if ok else 'FAIL'} expects '{expected_keyword}' in top result (score={top_score})  <- {query}")
    if not ok and results:
        print(f"      got instead: {top_text[:100]!r}")
print(f"\nKnowledge Manager: {km_correct}/{len(KM_CASES)} correct")

# ---------------------------------------------------------------------------
# 3. Knowledge Tracing — mastery must evolve sensibly over a conversation
# ---------------------------------------------------------------------------
section("Knowledge Tracing — mastery evolution over a conversation (DKT-covered concept: antibiotics)")
turns = [
    ("I think antibiotics are made by viruses, not bacteria or fungi.", "misconception #1"),
    ("Actually I also think antibiotics work by starving the immune system.", "misconception #2"),
    ("What is the role of genome sequencing in discovering new antibiotics?", "question"),
]
mastery_trace = []
for message, label in turns:
    resp = post("/api/agents/classify", {"message": message, "student_id": STUDENT})
    updated = resp["updated_mastery"]
    antibiotics_estimate = next((m for m in updated if "antibiotic" in m["concept_name"].lower()), None)
    if antibiotics_estimate:
        mastery_trace.append((label, antibiotics_estimate["ensemble_mastery"], antibiotics_estimate["explanation"]))
        print(f"[{label}] ensemble_mastery={antibiotics_estimate['ensemble_mastery']:.3f}  bkt={antibiotics_estimate['bkt_mastery']:.3f}  dkt={antibiotics_estimate['dkt_mastery']}")
        print(f"    explanation: {antibiotics_estimate['explanation']}")
    else:
        print(f"[{label}] !! no mastery estimate returned for antibiotics concept")

if len(mastery_trace) >= 2:
    first_mastery = mastery_trace[0][1]
    after_two_misconceptions = mastery_trace[1][1] if len(mastery_trace) > 1 else None
    print(f"\nMastery after 2 misconceptions ({after_two_misconceptions}) should be lower than after just 1 ({first_mastery}): "
          f"{'OK' if after_two_misconceptions is not None and after_two_misconceptions <= first_mastery else 'CHECK'}")

section("Knowledge Tracing — concept NOT in DKT's trained vocab (photosynthesis, ingested after training)")
resp = post("/api/agents/classify", {"message": "I think photosynthesis happens at night, not during the day.", "student_id": STUDENT})
photo_estimate = next((m for m in resp["updated_mastery"] if "photosynthesis" in m["concept_name"].lower()), None)
if photo_estimate:
    print(f"bkt_mastery={photo_estimate['bkt_mastery']:.3f}  dkt_mastery={photo_estimate['dkt_mastery']}  (expect dkt_mastery=None, since this concept postdates DKT training)")
    print(f"explanation: {photo_estimate['explanation']}")
else:
    print("!! no mastery estimate returned for photosynthesis concept")

section("Knowledge Tracing — mastery vector endpoint reflects the full session")
vector = get(f"/api/kt/mastery/{STUDENT}")
print(f"student {STUDENT} has {len(vector['estimates'])} tracked concepts:")
for e in vector["estimates"]:
    print(f"  {e['concept_name']:20s} ensemble={e['ensemble_mastery']:.3f}  bkt={e['bkt_mastery']:.3f}  dkt={e['dkt_mastery']}")

events = get(f"/api/kt/events/{STUDENT}")
print(f"\n{len(events)} raw kt_events recorded for this student session")

print("\n\nDONE.")
