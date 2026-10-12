"""Turn the Synthea CSV export into (1) a SQLite database and (2) one text summary per patient.

Synthea data is fully synthetic, so there is no real PHI. We still drop names and
addresses and refer to patients by their Synthea UUID, the habit you'd keep with real data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from cia.schema import Document

# Synthea CSV tables we load, and the columns we keep from each.
TABLES: dict[str, list[str]] = {
    "patients": ["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY", "STATE"],
    "conditions": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
    "medications": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION", "TOTALCOST"],
    "encounters": ["Id", "START", "PATIENT", "ENCOUNTERCLASS", "DESCRIPTION", "TOTAL_CLAIM_COST"],
    "procedures": ["START", "PATIENT", "CODE", "DESCRIPTION"],
    "allergies": ["START", "PATIENT", "DESCRIPTION"],
}


def load_tables(csv_dir: str | Path, max_patients: int | None = None) -> dict[str, pd.DataFrame]:
    """Read the Synthea CSVs we need, optionally limited to the first ``max_patients``."""
    csv_dir = Path(csv_dir)
    frames: dict[str, pd.DataFrame] = {}
    for table, columns in TABLES.items():
        path = csv_dir / f"{table}.csv"
        if not path.exists():
            frames[table] = pd.DataFrame(columns=columns)
            continue
        df = pd.read_csv(path, dtype=str)
        frames[table] = df[[c for c in columns if c in df.columns]].copy()

    if max_patients:
        keep = set(frames["patients"]["Id"].head(max_patients))
        frames["patients"] = frames["patients"][frames["patients"]["Id"].isin(keep)]
        for table, df in frames.items():
            if table != "patients" and "PATIENT" in df.columns:
                frames[table] = df[df["PATIENT"].isin(keep)]
    return frames


def write_sqlite(frames: dict[str, pd.DataFrame], db_path: str | Path) -> Path:
    """Persist the tables to SQLite so they can be queried with SQL later."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        for table, df in frames.items():
            df.to_sql(table, conn, if_exists="replace", index=False)
    return db_path


def _is_active(stop: object) -> bool:
    return stop is None or (isinstance(stop, float) and pd.isna(stop)) or str(stop).strip() == ""


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def patient_documents(frames: dict[str, pd.DataFrame]) -> list[Document]:
    """Build one plain-text clinical summary per patient for retrieval."""
    by_patient = {
        table: df.groupby("PATIENT") for table, df in frames.items() if "PATIENT" in df.columns
    }

    def rows(table: str, pid: str) -> pd.DataFrame:
        grouped = by_patient.get(table)
        if grouped is None or pid not in grouped.groups:
            return pd.DataFrame(columns=TABLES[table])
        return grouped.get_group(pid)

    docs: list[Document] = []
    for _, p in frames["patients"].iterrows():
        pid = p["Id"]
        cond = rows("conditions", pid)
        meds = rows("medications", pid)
        enc = rows("encounters", pid)
        proc = rows("procedures", pid)
        allg = rows("allergies", pid)

        active_conditions = _unique(
            [r.DESCRIPTION for r in cond.itertuples() if _is_active(r.STOP)]
        )
        past_conditions = _unique(
            [r.DESCRIPTION for r in cond.itertuples() if not _is_active(r.STOP)]
        )
        active_meds = _unique([r.DESCRIPTION for r in meds.itertuples() if _is_active(r.STOP)])
        procedures = _unique(list(proc["DESCRIPTION"]))[:15]
        allergies = _unique(list(allg["DESCRIPTION"]))
        enc_counts = enc["ENCOUNTERCLASS"].value_counts().to_dict() if len(enc) else {}
        total_cost = pd.to_numeric(enc.get("TOTAL_CLAIM_COST"), errors="coerce").sum()

        lines = [
            f"Patient {pid}",
            f"Gender: {p.get('GENDER', '')}. Birth date: {p.get('BIRTHDATE', '')}. "
            f"State: {p.get('STATE', '')}.",
            "Active conditions: " + ("; ".join(active_conditions) or "none recorded") + ".",
            "Resolved conditions: " + ("; ".join(past_conditions) or "none recorded") + ".",
            "Current medications: " + ("; ".join(active_meds) or "none recorded") + ".",
            "Allergies: " + ("; ".join(allergies) or "none recorded") + ".",
            "Procedures: " + ("; ".join(procedures) or "none recorded") + ".",
            "Encounters by type: "
            + (", ".join(f"{k} {v}" for k, v in sorted(enc_counts.items())) or "none")
            + f". Total claim cost: ${total_cost:,.2f}.",
        ]
        docs.append(
            Document(
                doc_id=f"patient-{pid}",
                text="\n".join(lines),
                source=f"patient:{pid}",
                metadata={"type": "patient", "patient_id": pid, "title": f"Patient {pid}"},
            )
        )
    return docs
