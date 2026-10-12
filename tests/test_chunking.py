import pytest

from cia.chunking import chunk_documents, chunk_text
from cia.schema import Document


def test_short_text_is_one_chunk():
    assert chunk_text("A short note.", chunk_size=100, overlap=10) == ["A short note."]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} about coverage rules." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share some text (the overlap)
    assert any(word in chunks[1] for word in chunks[0].split()[-3:])


def test_prefers_paragraph_boundaries():
    text = "First paragraph about diabetes.\n\nSecond paragraph about hypertension."
    chunks = chunk_text(text, chunk_size=45, overlap=0)
    assert chunks == ["First paragraph about diabetes.", "Second paragraph about hypertension."]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_documents_keeps_source_and_ids():
    docs = [Document("d1", "Alpha. " * 100, "policy:a.pdf", {"type": "policy"})]
    chunks = chunk_documents(docs, chunk_size=120, overlap=20)
    assert chunks[0].chunk_id == "d1::0"
    assert {c.source for c in chunks} == {"policy:a.pdf"}
    assert all(c.metadata["type"] == "policy" for c in chunks)
