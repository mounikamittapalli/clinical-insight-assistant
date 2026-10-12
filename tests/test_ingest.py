import sqlite3

from cia.ingest.documents import load_policy_documents
from cia.ingest.synthea import load_tables, patient_documents, write_sqlite


def test_patient_summaries(data_dir):
    frames = load_tables(data_dir / "raw" / "synthea" / "csv")
    docs = {d.metadata["patient_id"]: d for d in patient_documents(frames)}
    assert set(docs) == {"p-001", "p-002", "p-003"}
    p1 = docs["p-001"].text
    assert "Diabetes mellitus type 2" in p1 and "Essential hypertension" in p1
    assert "Resolved conditions: Acute bronchitis" in p1
    assert "Metformin 500 MG Oral Tablet" in p1
    assert "ambulatory 2" in p1
    # names are dropped; patients are referenced by ID only
    assert "Jane" not in p1 and "Doe" not in p1
    assert "Allergy to peanuts" in docs["p-002"].text


def test_max_patients_limits_all_tables(data_dir):
    frames = load_tables(data_dir / "raw" / "synthea" / "csv", max_patients=1)
    assert len(frames["patients"]) == 1
    assert set(frames["conditions"]["PATIENT"]) == {"p-001"}


def test_write_sqlite(data_dir, tmp_path):
    frames = load_tables(data_dir / "raw" / "synthea" / "csv")
    db = write_sqlite(frames, tmp_path / "p.db")
    with sqlite3.connect(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM conditions WHERE PATIENT='p-001'").fetchone()[0]
    assert n == 3


def test_load_policy_documents(data_dir):
    docs = load_policy_documents(data_dir / "raw" / "policies")
    assert len(docs) == 1
    assert docs[0].source == "policy:NCD-280.14-cgm.txt"
    assert "continuous glucose monitors" in docs[0].text.lower()
