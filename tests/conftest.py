"""Shared fixtures: a tiny Synthea-style dataset, a fake LLM, and an offline embedder."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from cia.config import Config
from cia.embeddings import HashingEmbedder

PATIENTS = """Id,BIRTHDATE,DEATHDATE,FIRST,LAST,GENDER,RACE,ETHNICITY,STATE
p-001,1970-03-02,,Jane,Doe,F,white,nonhispanic,Texas
p-002,1985-07-19,,John,Roe,M,black,nonhispanic,Ohio
p-003,1992-11-30,,Ana,Diaz,F,white,hispanic,Florida
"""
CONDITIONS = """START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION
2015-01-01,,p-001,e1,44054006,Diabetes mellitus type 2
2016-05-01,,p-001,e2,59621000,Essential hypertension
2010-01-01,2010-02-01,p-001,e3,10509002,Acute bronchitis
2019-03-03,,p-002,e4,195967001,Asthma
2021-08-08,,p-003,e5,35489007,Depressive disorder
"""
MEDICATIONS = """START,STOP,PATIENT,PAYER,ENCOUNTER,CODE,DESCRIPTION,BASE_COST,TOTALCOST
2015-01-02,,p-001,x,e1,860975,Metformin 500 MG Oral Tablet,1,25.5
2016-05-02,,p-001,x,e2,314076,Lisinopril 10 MG Oral Tablet,1,12.0
2019-03-04,,p-002,x,e4,895994,Albuterol Inhaler,1,40.0
"""
ENCOUNTERS = """Id,START,PATIENT,ENCOUNTERCLASS,DESCRIPTION,TOTAL_CLAIM_COST
e1,2015-01-01,p-001,ambulatory,Checkup,120.0
e2,2016-05-01,p-001,ambulatory,Follow-up,90.0
e3,2010-01-01,p-001,emergency,ER visit,800.0
e4,2019-03-03,p-002,outpatient,Consult,150.0
e5,2021-08-08,p-003,wellness,Annual visit,100.0
"""
ALLERGIES = """START,PATIENT,DESCRIPTION
2000-01-01,p-002,Allergy to peanuts
"""
POLICY = """National Coverage Determination for Continuous Glucose Monitors.

Medicare covers therapeutic continuous glucose monitors for beneficiaries with diabetes
who are treated with insulin or have a history of problematic hypoglycemia.
The treating practitioner must document an in-person or telehealth visit within six months
before ordering the monitor.
"""


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    csv = tmp_path / "raw" / "synthea" / "csv"
    csv.mkdir(parents=True)
    for name, text in {
        "patients": PATIENTS,
        "conditions": CONDITIONS,
        "medications": MEDICATIONS,
        "encounters": ENCOUNTERS,
        "allergies": ALLERGIES,
    }.items():
        (csv / f"{name}.csv").write_text(text)
    policies = tmp_path / "raw" / "policies"
    policies.mkdir(parents=True)
    (policies / "NCD-280.14-cgm.txt").write_text(POLICY)
    return tmp_path


@pytest.fixture
def cfg(data_dir: Path) -> Config:
    base = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text()
    )
    base["paths"] = {
        "synthea_dir": str(data_dir / "raw" / "synthea" / "csv"),
        "policy_dir": str(data_dir / "raw" / "policies"),
        "sqlite_db": str(data_dir / "processed" / "patients.db"),
        "index_dir": str(data_dir / "index"),
        "finetune_dir": str(data_dir / "finetune"),
        "adapter_dir": str(data_dir / "adapter"),
        "results_dir": str(data_dir / "results"),
    }
    base["vector_store"]["backend"] = "numpy"
    base["retrieval"]["rerank"] = False
    base["chunking"] = {"chunk_size": 400, "chunk_overlap": 60}
    base["evaluation"]["questions_file"] = str(data_dir / "eval" / "questions.jsonl")
    return Config(base)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


class FakeLLM:
    """Answers with the first sentence of passage [1], cited. Prompt-only mode gets a guess."""

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self.reply is not None:
            return self.reply
        user = messages[-1]["content"]
        match = re.search(r"\[1\] \(source: [^)]*\)\n(.+)", user)
        if not match:
            return "I think the patient has diabetes."
        return f"{match.group(1).strip()} [1]"


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
