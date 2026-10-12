"""Load unstructured policy documents (CMS coverage determinations) from disk."""

from __future__ import annotations

import re
from pathlib import Path

from cia.schema import Document

SUPPORTED = {".pdf", ".txt", ".md"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily: only needed for PDFs

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_policy_documents(policy_dir: str | Path) -> list[Document]:
    """Read every supported file in ``policy_dir`` into a :class:`Document`."""
    policy_dir = Path(policy_dir)
    docs: list[Document] = []
    if not policy_dir.exists():
        return docs
    for path in sorted(policy_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        raw = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(errors="ignore")
        text = clean_text(raw)
        if not text:
            continue
        docs.append(
            Document(
                doc_id=f"policy-{path.stem}",
                text=text,
                source=f"policy:{path.name}",
                metadata={"type": "policy", "file": path.name, "title": f"Policy {path.stem}"},
            )
        )
    return docs
