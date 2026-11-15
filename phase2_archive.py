"""Verified, restartable extraction of archived Phase 2 inputs."""
import hashlib
import json
import os
import shutil
import stat
import zlib
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crc_matches(path, member):
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    crc = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc == member.CRC


def extract_verified(archive_path, destination, expected_sha256=None):
    archive_path, destination = Path(archive_path), Path(destination).resolve()
    digest = sha256(archive_path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"Archive provenance SHA-256 mismatch: {archive_path}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symlink is not supported: {member.filename}")
        restored = 0
        for member in members:
            target = (destination / member.filename).resolve()
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if _crc_matches(target, member):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".phase2_tmp")
            try:
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if not _crc_matches(temporary, member):
                    raise ValueError(f"Extracted member failed size/CRC validation: {member.filename}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            restored += 1
    record = {"archive": str(archive_path), "size_bytes": archive_path.stat().st_size,
              "sha256": digest, "trusted_expected_sha256": expected_sha256 or None,
              "members": len(members), "restored_files": restored}
    (destination / "phase2_archive_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return destination, record
