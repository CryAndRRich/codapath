"""LLM-written class descriptions -- the round-1 cold-start text prior.

This module writes and reads `config/descriptions/{dataset}_{style}.json`
(PLAN_IMPLEMENT.md §3). The file is generated **once**, committed to the
repo, and every later run just reads it -- `generate_descriptions` is not
meant to be called from `main.py` or any run notebook, only from
`generate_class_description.ipynb`.

**Frozen-artifact framing, not a reproducible computation.** A hosted model
call (`gemini-2.5-flash`) is not bit-for-bit reproducible even at
`temperature=0.0` -- Google's own docs note determinism is not guaranteed
across requests, let alone across months as the served weights are updated
behind a fixed model name. The reproducible unit here is the **written JSON
file**, not "re-run the prompt" -- exactly the same relationship
`features/vlm.py`'s vendored `config/prompts/` file has to a live download.
`load_descriptions` only ever reads a file already on disk; it never calls
the API.

**`gemini-3.x` is deliberately excluded** (`assert not MODEL.startswith(...)`
in the notebook, and mirrored here for anything using `generate_descriptions`
directly): `temperature`/`topK`/`topP` are documented as deprecated and
silently ignored on `gemini-3.7-flash`, `3.6-flash`, `3.5-flash-lite`. Passing
`temperature=0.0` against one of those models is a silent no-op, not an
error -- the call still succeeds, it just is not doing what the docstring of
this module (and the paper text built from it) claims.

**`load_descriptions(dataset, "manual")` is the control arm** and needs no
file at all: it reads `datasets.<dataset>.descriptions` straight out of
`config/config.yaml`, the same block `config.yaml` has always had. `"manual"`
therefore behaves like every other style from a caller's point of view
(same return shape) without this module ever writing a `manual` file.

The `google-genai` package is not installed in this environment, so
`generate_descriptions` imports it lazily, inside the function body --
the same pattern `features/vlm.py` uses for `conch`.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from typing import Dict, List, Optional

import yaml

__all__ = [
    "description_path",
    "load_descriptions",
    "generate_descriptions",
]

DESCRIPTIONS_DIR = "config/descriptions"

# Prompt templates by style -- kept here (not inline in the notebook) so a
# test can assert on the exact string sent to the API without executing the
# notebook.
_PROMPT_TEMPLATES = {
    "llm_short": (
        "You are a pathologist writing a single, dense sentence that a "
        "vision-language model's text encoder will compare against "
        "histopathology image patches. Describe the class {class_name!r} "
        "(dataset: {dataset}) in ONE sentence, focused on visually "
        "distinctive tissue morphology (cell shape, arrangement, texture) "
        "-- not clinical significance or diagnosis. Output only the "
        "sentence, no preamble, no quotes."
    ),
    "llm_morphology": (
        "You are a pathologist writing a detailed morphological description "
        "for the class {class_name!r} (dataset: {dataset}), to be encoded by "
        "a vision-language model's text tower and compared against "
        "histopathology image patches. Cover: cell/nuclear shape, "
        "arrangement or architecture, texture, and any distinctive staining "
        "pattern. 2-4 sentences. Output only the description, no preamble, "
        "no quotes."
    ),
    "llm_multi": (
        "You are a pathologist writing ONE short, visually distinctive "
        "sentence describing the class {class_name!r} (dataset: {dataset}), "
        "for a vision-language model's text encoder. This is variant "
        "{variant_index} of {num_variants} -- phrase it differently from "
        "the other variants (different visual emphasis or wording) while "
        "staying accurate. Output only the sentence, no preamble, no quotes."
    ),
}

VALID_STYLES = frozenset({"llm_short", "llm_morphology", "llm_multi"})


def description_path(dataset: str, style: str) -> str:
    """Path to the frozen description file for one (dataset, style) pair.

    Not keyed by seed or model: the description text does not depend on the
    train/test split, and this project generates it once, deliberately, not
    per-run.
    """
    return os.path.join(DESCRIPTIONS_DIR, f"{dataset}_{style}.json")


def _sha256_descriptions(descriptions: Dict[str, str]) -> str:
    """Same hashing convention as `features/vlm.py::description_sha256`
    (sorted-key JSON, so the hash is independent of dict insertion order) --
    duplicated rather than imported, because `features/vlm.py` importing
    `features/descriptions.py` (or vice versa) would create a cross-module
    dependency neither side otherwise needs; both modules hash the same way
    on purpose so a description file's own `sha256` field and a downstream
    manifest's `description_sha256` field are directly comparable.
    """
    ordered = {key: descriptions[key] for key in sorted(descriptions)}
    blob = json.dumps(ordered, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_descriptions(dataset: str, style: str, config: Optional[dict] = None) -> Dict[str, str]:
    """Return `{class_name: description}` for one (dataset, style) pair.

    `style="manual"` reads `config/config.yaml`'s `datasets.<dataset>.descriptions`
    directly -- no file, always available, the control every LLM style must
    beat to justify itself. Any other style reads the frozen JSON file
    `generate_class_description.ipynb` wrote; raises `FileNotFoundError` with
    an actionable message if that notebook has not been run yet for this
    (dataset, style).

    `config`, if given, is the already-parsed `config.yaml` dict (avoids a
    second parse when the caller has one already); otherwise this reads and
    parses it itself.
    """
    if config is None:
        with open("config/config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    if style == "manual":
        return dict(config["datasets"][dataset]["descriptions"])

    path = description_path(dataset, style)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No description file at {path!r}. Run generate_class_description.ipynb "
            f"with DATASET={dataset!r}, STYLE={style!r} first -- descriptions are "
            "generated once and committed, not produced on the fly by a run notebook."
        )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    descriptions = dict(payload["descriptions"])

    expected_class_names = list(config["datasets"][dataset]["descriptions"])
    if list(descriptions) != expected_class_names:
        raise ValueError(
            f"{path} class order {list(descriptions)} does not match "
            f"config.yaml's {expected_class_names} for dataset {dataset!r} -- "
            "regenerate the description file against the current config."
        )
    return descriptions


def generate_descriptions(
    dataset: str,
    style: str,
    class_names: List[str],
    model: str,
    temperature: float = 0.0,
    seed: int = 42,
    num_per_class: int = 1,
    api_key: Optional[str] = None,
) -> dict:
    """Call the Gemini API once per class (or `num_per_class` times, for
    `style="llm_multi"`) and return the full payload
    `generate_class_description.ipynb` writes to disk.

    Does NOT write the file itself -- the notebook owns the "refuse to
    overwrite unless OVERWRITE=True" check, which belongs at the call site,
    not buried in a library function a future caller might invoke without
    meaning to overwrite a frozen artifact.

    Raises `ValueError` for `style not in VALID_STYLES` (this function never
    handles `"manual"` -- that style has no API call, see `load_descriptions`)
    and for `model.startswith("gemini-3")` (temperature/topK/topP are
    deprecated and silently ignored on that family -- see module docstring).
    """
    if style not in VALID_STYLES:
        raise ValueError(f"style must be one of {sorted(VALID_STYLES)}, got {style!r}")
    if model.startswith("gemini-3"):
        raise ValueError(
            f"model={model!r}: temperature/topK/topP are deprecated and silently "
            "ignored on the gemini-3.x family (3.7-flash, 3.6-flash, 3.5-flash-lite) "
            "-- generating with temperature=0.0 against one of these is a silent "
            "no-op, not an error. Use a gemini-2.x model."
        )
    if style != "llm_multi" and num_per_class != 1:
        raise ValueError(
            f"num_per_class={num_per_class} only applies to style='llm_multi' "
            f"(got style={style!r})"
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or None)
    generation_config = types.GenerateContentConfig(temperature=temperature)

    descriptions: Dict[str, object] = {}
    for class_name in class_names:
        if style == "llm_multi":
            variants = []
            for i in range(num_per_class):
                prompt = _PROMPT_TEMPLATES[style].format(
                    class_name=class_name, dataset=dataset,
                    variant_index=i + 1, num_variants=num_per_class,
                )
                response = client.models.generate_content(
                    model=model, contents=prompt, config=generation_config,
                )
                variants.append(response.text.strip())
            descriptions[class_name] = variants
        else:
            prompt = _PROMPT_TEMPLATES[style].format(class_name=class_name, dataset=dataset)
            response = client.models.generate_content(
                model=model, contents=prompt, config=generation_config,
            )
            descriptions[class_name] = response.text.strip()

    return {
        "dataset": dataset,
        "model": model,
        "style": style,
        "temperature": temperature,
        "seed": seed,
        "num_per_class": num_per_class,
        "prompt_template": _PROMPT_TEMPLATES[style],
        "generated_at": date.today().isoformat(),
        "descriptions": descriptions,
        "sha256": _sha256_descriptions(
            {k: (v if isinstance(v, str) else json.dumps(v, sort_keys=True)) for k, v in descriptions.items()}
        ),
    }
