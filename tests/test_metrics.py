from cia.eval.metrics import (
    groundedness,
    hallucinated,
    keyword_accuracy,
    recall_at_k,
    token_f1,
)
from cia.guardrails import NO_ANSWER

CTX = ["Patient p-001. Active conditions: Diabetes mellitus type 2; Essential hypertension."]


def test_keyword_accuracy():
    assert keyword_accuracy("Has diabetes mellitus type 2 [1].", ["Diabetes mellitus type 2"]) == 1
    assert keyword_accuracy("Has asthma [1].", ["Diabetes mellitus type 2"]) == 0
    # with no expected facts, only a refusal counts as correct
    assert keyword_accuracy(NO_ANSWER, []) == 1
    assert keyword_accuracy("Paris.", []) == 0


def test_token_f1():
    assert token_f1("diabetes and hypertension", "diabetes and hypertension") == 1.0
    assert 0 < token_f1("diabetes only", "diabetes and hypertension") < 1
    assert token_f1("asthma", "diabetes") == 0.0


def test_groundedness_and_hallucination():
    grounded = "Patient p-001 has diabetes mellitus type 2 [1]."
    invented = "Patient p-001 has diabetes [1]. They also had heart surgery in Paris last year."
    assert groundedness(grounded, CTX) == 1.0 and hallucinated(grounded, CTX) == 0.0
    assert groundedness(invented, CTX) == 0.5 and hallucinated(invented, CTX) == 1.0
    assert groundedness(NO_ANSWER, CTX) == 1.0


def test_recall_at_k():
    assert recall_at_k(["patient:p-001", "policy:a"], ["patient:p-001"]) == 1.0
    assert recall_at_k(["policy:a"], ["patient:p-001"]) == 0.0
