"""Rewrite already-finished run archives from the old `scalpel` name to `PACT`.

The method was renamed (`sampling/scalpel/` -> `sampling/pact/`, sampler name
`"scalpel"` -> `"pact"`). Runs finished before the rename carry the old name in
three places, and all three matter:

* the archive filename           `histoset_scalpel_disagreement_seed42.zip`
* the member paths inside it     `histoset/scalpel_disagreement_results.pt`
* fields inside the payloads     `sampler`, `run_name`, and the `run_name`
                                 recorded again in each probe's `metadata`

Leaving any one of them stale breaks something specific: `evaluation/
results_io.py` groups methods by `run_name`, so a mixed set would report
`scalpel_*` and `pact_*` as two different methods; and a reader comparing an
old archive against a new one would see two names for one method.

**This rewrites metadata only.** No metric, probability, index or weight is
touched -- the script asserts that afterwards by comparing every numeric field
before and after. A renamed archive therefore stays the same experiment, and
nothing needs re-running.

Usage (from the repository root):

    python scripts/rename_scalpel_to_pact.py --root ../weights ../new_exp
    python scripts/rename_scalpel_to_pact.py --root ../weights --dry-run

The original archive is kept as `<name>.zip.bak` unless `--no-backup` is given.
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path
from typing import Any, List, Tuple

import torch

OLD = "scalpel"
NEW = "pact"

# Payload fields that carry the sampler or run name as a bare string. Nested
# dicts (a probe's `metadata`, a selection's `sampler_config`) are walked too.
_NAME_KEYS = ("sampler", "run_name")


def _rename(text: str) -> str:
    return text.replace(OLD.upper(), NEW.upper()).replace(OLD, NEW)


def _rewrite_payload(value: Any) -> Any:
    """Rename the sampler/run-name strings anywhere in a loaded payload.

    Renames only values reached through a name-bearing KEY, never every string
    it finds: a class name, a sample id or a file path that happened to contain
    the word must not be rewritten by a metadata pass.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in _NAME_KEYS and isinstance(item, str):
                out[key] = _rename(item)
            else:
                out[key] = _rewrite_payload(item)
        return out
    if isinstance(value, list):
        return [_rewrite_payload(item) for item in value]
    return value


def _numeric_fingerprint(payload: Any) -> List[float]:
    """Every number in a payload, in traversal order.

    Compared before and after the rewrite: if the rename touched a metric, a
    probability, an index or a weight, this list changes and the script
    refuses to write the archive.
    """
    found: List[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                walk(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, torch.Tensor):
            found.append(float(value.double().sum().item()))
            found.extend(float(dimension) for dimension in value.shape)
        elif isinstance(value, bool):
            found.append(float(value))
        elif isinstance(value, (int, float)):
            found.append(float(value))
        else:
            numpy_array = getattr(value, "shape", None)
            if numpy_array is not None and hasattr(value, "dtype"):
                try:
                    found.append(float(value.astype("float64").sum()))
                    found.extend(float(dimension) for dimension in value.shape)
                except (TypeError, ValueError):
                    pass

    walk(payload)
    return found


def rename_archive(archive: Path, dry_run: bool = False, backup: bool = True) -> Tuple[Path, int]:
    """Rewrite one archive in place. Returns the new path and members changed."""
    new_path = archive.with_name(_rename(archive.name))
    changed = 0
    buffer = io.BytesIO()

    with zipfile.ZipFile(archive) as source:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                raw = source.read(info.filename)
                member = _rename(info.filename)
                if member != info.filename:
                    changed += 1

                if info.filename.endswith(".pt"):
                    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
                    before = _numeric_fingerprint(payload)
                    rewritten = _rewrite_payload(payload)
                    after = _numeric_fingerprint(rewritten)
                    if before != after:
                        raise RuntimeError(
                            f"{archive.name}:{info.filename} -- the rename changed a NUMBER, "
                            "not just a name. Refusing to write; nothing has been modified."
                        )
                    out = io.BytesIO()
                    torch.save(rewritten, out)
                    raw = out.getvalue()
                elif info.filename.endswith(".log"):
                    raw = _rename(raw.decode("utf-8", "replace")).encode("utf-8")

                target.writestr(member, raw)

    if dry_run:
        return new_path, changed

    if backup:
        shutil.copy2(archive, archive.with_suffix(archive.suffix + ".bak"))
    new_path.write_bytes(buffer.getvalue())
    if new_path != archive:
        archive.unlink()
    return new_path, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", nargs="+", required=True,
                        help="directories to search recursively for *.zip")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing anything")
    parser.add_argument("--no-backup", action="store_true",
                        help="do not keep the original as <name>.zip.bak")
    args = parser.parse_args()

    archives: List[Path] = []
    for root in args.root:
        base = Path(root)
        if not base.exists():
            raise FileNotFoundError(f"no such directory: {base}")
        archives.extend(sorted(p for p in base.rglob("*.zip") if OLD in p.name))

    if not archives:
        print(f"nothing to do: no *.zip containing {OLD!r}")
        return

    print(f"{len(archives)} archive(s) to rename{' (dry run)' if args.dry_run else ''}\n")
    for archive in archives:
        new_path, changed = rename_archive(archive, args.dry_run, not args.no_backup)
        print(f"  {archive.name}\n    -> {new_path.name}   ({changed} member paths renamed)")

    if not args.dry_run:
        print("\ndone. Originals kept as *.zip.bak"
              if not args.no_backup else "\ndone. No backups kept.")


if __name__ == "__main__":
    main()
