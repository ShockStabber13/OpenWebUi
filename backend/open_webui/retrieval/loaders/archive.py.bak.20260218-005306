import os
import re
import shutil
import tempfile
import zipfile
import mimetypes
from pathlib import Path
from typing import Callable, Optional, List

from langchain_core.documents import Document

SKIP_PREFIXES_DEFAULT = [
    "__MACOSX/",
    ".git/",
    "node_modules/",
    "dist/",
    "build/",
    ".next/",
    ".venv/",
    "venv/",
    "__pycache__/",
]

ZIP_MAX_FILES = int(os.environ.get("RAG_ZIP_MAX_FILES", "2000"))
ZIP_MAX_TOTAL_UNCOMPRESSED_MB = int(os.environ.get("RAG_ZIP_MAX_TOTAL_UNCOMPRESSED_MB", "100"))
ZIP_MAX_MEMBER_MB = int(os.environ.get("RAG_ZIP_MAX_MEMBER_MB", "10"))

def _safe_dest_path(base_dir: str, member_name: str) -> Optional[Path]:
    member_name = member_name.replace("\\", "/")
    if member_name.startswith("/") or re.match(r"^[A-Za-z]:/", member_name):
        return None
    base = Path(base_dir).resolve()
    dest = (base / member_name).resolve()
    try:
        dest.relative_to(base)
    except Exception:
        return None
    return dest

def _looks_binary(path: Path, sample_bytes: int = 8192) -> bool:
    try:
        data = path.read_bytes()[:sample_bytes]
    except Exception:
        return True
    if b"\x00" in data:
        return True
    printable = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
    return len(data) > 0 and (printable / len(data)) < 0.70

class ZipArchiveLoader:
    def __init__(
        self,
        file_path: str,
        zip_filename: str,
        file_content_type: str,
        get_loader: Callable[[str, str, str], object],
        skip_prefixes: Optional[List[str]] = None,
    ):
        self.file_path = file_path
        self.zip_filename = zip_filename
        self.file_content_type = file_content_type
        self.get_loader = get_loader
        self.skip_prefixes = skip_prefixes or SKIP_PREFIXES_DEFAULT

    def load(self) -> list[Document]:
        docs: list[Document] = []

        max_total = ZIP_MAX_TOTAL_UNCOMPRESSED_MB * 1024 * 1024
        max_member = ZIP_MAX_MEMBER_MB * 1024 * 1024

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(self.file_path) as zf:
                infos = [i for i in zf.infolist() if not i.is_dir()]
                if len(infos) > ZIP_MAX_FILES:
                    infos = infos[:ZIP_MAX_FILES]

                total_uncompressed = 0

                for info in infos:
                    member_name = info.filename.replace("\\", "/")

                    if any(member_name.startswith(p) for p in self.skip_prefixes):
                        continue

                    if member_name.lower().endswith(".zip"):
                        continue

                    if info.file_size > max_member:
                        continue

                    total_uncompressed += info.file_size
                    if total_uncompressed > max_total:
                        break

                    dest = _safe_dest_path(tmp_dir, member_name)
                    if dest is None:
                        continue

                    dest.parent.mkdir(parents=True, exist_ok=True)

                    with zf.open(info) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)

                    if _looks_binary(dest):
                        continue

                    guessed_mime, _ = mimetypes.guess_type(member_name)
                    guessed_mime = guessed_mime or ""

                    try:
                        loader = self.get_loader(
                            os.path.basename(member_name) or member_name,
                            guessed_mime,
                            str(dest),
                        )
                        sub_docs = loader.load()
                    except Exception:
                        continue

                    for d in sub_docs:
                        md = dict(d.metadata or {})
                        md.setdefault("archive", self.zip_filename)
                        md.setdefault("archive_member", member_name)
                        md.setdefault("source", f"{self.zip_filename}::{member_name}")
                        d.metadata = md
                        docs.append(d)

        return docs
