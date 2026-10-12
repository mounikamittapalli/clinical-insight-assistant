from cia.attribution import add_citations

PASSAGES = [
    "Patient p-001. Active conditions: Diabetes mellitus type 2; Essential hypertension.",
    "Medicare covers continuous glucose monitors for insulin-treated diabetes.",
]


def test_adds_best_matching_citation_per_sentence():
    out = add_citations(
        "Patient p-001 has essential hypertension. Medicare covers glucose monitors for "
        "insulin-treated diabetes.",
        PASSAGES,
    )
    assert out == (
        "Patient p-001 has essential hypertension [1]. Medicare covers glucose monitors "
        "for insulin-treated diabetes [2]."
    )


def test_drops_unsupported_sentences():
    out = add_citations("Patient p-001 has essential hypertension. She flew to Paris.", PASSAGES)
    assert out == "Patient p-001 has essential hypertension [1]."


def test_returns_none_when_nothing_is_supported():
    assert add_citations("The capital of France is Paris.", PASSAGES) is None


def test_verify_drops_sentence_citing_unrelated_passage():
    from cia.attribution import verify_citations

    assert verify_citations("Paris [1].", PASSAGES) is None
    out = verify_citations("Patient p-001 has essential hypertension [2].", PASSAGES)
    assert out == "Patient p-001 has essential hypertension [1]."
    kept = verify_citations("Patient p-001 has essential hypertension [1].", PASSAGES)
    assert kept == "Patient p-001 has essential hypertension [1]."
