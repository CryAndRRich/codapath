"""LLM-written class descriptions -- the round-1 cold-start text prior.

This module writes and reads `config/descriptions/{dataset}_{style}.json`
(PLAN_IMPLEMENT.md §3). The file is generated **once**, committed to the
repo, and every later run just reads it -- `generate_descriptions` is not
meant to be called from `main.py` or any run notebook, only from
`generate_class_description.ipynb`.

**Frozen-artifact framing, not a reproducible computation.** A hosted model
call is not bit-for-bit reproducible even at `temperature=0.0` -- Google's
own docs note determinism is not guaranteed across requests, let alone across
months as the served weights are updated behind a fixed model name, and a
pinned model name can itself be retired (`gemini-2.5-flash`, this module's
original pin, returned `404 NOT_FOUND ... no longer available to new users`
as of 2026-08). The reproducible unit here is the **written JSON file**, not
"re-run the prompt and expect the same text" -- exactly the same relationship
`features/vlm.py`'s vendored `config/prompts/` file has to a live download.
`load_descriptions` only ever reads a file already on disk; it never calls
the API. `generate_descriptions`'s returned payload records `model`,
`temperature`, `seed` and `generated_at` alongside the text and its
`sha256` -- proof that the named model produced exactly this text on this
date, not a promise it will again.

**No model family is blocked.** This module calls the model through
**OpenRouter** (`https://openrouter.ai/api/v1/chat/completions`, an
OpenAI-compatible REST endpoint that proxies many providers under one API
key) rather than a provider-specific SDK, so `MODEL` is whatever
OpenRouter-format id the caller picks (e.g. `"google/gemini-2.5-flash"`,
`"anthropic/claude-3.5-haiku"`, `"openai/gpt-4o-mini"`) -- nothing here needs
to be kept in sync with it, and nothing here is tied to one vendor's model
family.

**`load_descriptions(dataset, "manual")` is the control arm** and needs no
file at all: it reads `datasets.<dataset>.descriptions` straight out of
`config/config.yaml`, the same block `config.yaml` has always had. `"manual"`
therefore behaves like every other style from a caller's point of view
(same return shape) without this module ever writing a `manual` file.

`generate_descriptions` talks to OpenRouter with the stdlib-adjacent
`requests` package (already a transitive dependency of `transformers`) --
no provider SDK needed, since OpenRouter's endpoint is plain REST.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import date
from typing import Callable, Dict, List, Optional

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
}

VALID_STYLES = frozenset({"llm_short", "llm_morphology"})


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


# A hosted model is shared infrastructure, so a single call can fail for
# reasons that have nothing to do with this caller. Retried here rather than
# left to the notebook, because the sweep is 78 sequential calls (9+14+16
# classes x 2 styles) and ONE transient failure anywhere in it used to abort
# the whole run, discarding every description already generated.
#
# Which statuses are worth retrying:
#   503 UNAVAILABLE       - "model is currently experiencing high demand",
#                           explicitly described by Google as temporary.
#   429 RESOURCE_EXHAUSTED - free-tier rate limit; a per-minute quota clears
#                           on its own, so waiting is the correct response.
#   500 / 504             - server-side error and gateway timeout.
# Everything else (401 bad key, 404 retired model name, 400 malformed
# prompt) is a deterministic error that would fail identically on a retry, so
# it is raised immediately instead of burning six delays first.
_RETRYABLE_STATUS = (429, 500, 503, 504)

# Exponential backoff with full jitter. Jitter matters even for a single
# sequential client: without it, a run that hits a busy model retries on a
# fixed schedule, and every client doing the same converges on the same
# retry instants -- the thundering-herd pattern that keeps an overloaded
# service overloaded. Delays are 2, 4, 8, 16, 32, 64s (jittered), so six
# attempts span up to ~2 minutes per class before giving up.
_MAX_ATTEMPTS = 6
_BASE_DELAY_SECONDS = 2.0
_MAX_DELAY_SECONDS = 64.0


def _status_code(error: BaseException) -> Optional[int]:
    """HTTP status of an OpenRouter request error, or None if it is not one.

    `requests.HTTPError` carries the status on `error.response.status_code`;
    read defensively (also checking bare `.code`/`.status_code` attributes)
    so a `requests` version bump can't silently turn "retry the 503" into
    "never retry anything" -- which would look exactly like the bug this
    code fixes.
    """
    response = getattr(error, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    text = str(error)
    for status in _RETRYABLE_STATUS:
        if text.startswith(f"{status} ") or f" {status} " in text:
            return status
    return None


def _call_with_retry(call: Callable[[], object], label: str, verbose: bool = True):
    """Run `call`, retrying transient server-side failures with backoff.

    `call` is a zero-argument closure rather than the client plus arguments,
    so this function needs to know nothing about the genai API surface and
    can be unit-tested without it.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - re-raised below unless retryable
            status = _status_code(error)
            if status not in _RETRYABLE_STATUS or attempt == _MAX_ATTEMPTS:
                raise
            delay = min(_BASE_DELAY_SECONDS * 2 ** (attempt - 1), _MAX_DELAY_SECONDS)
            delay = random.uniform(0.0, delay)  # full jitter
            if verbose:
                print(
                    f"    [retry] {label}: {status} on attempt {attempt}/"
                    f"{_MAX_ATTEMPTS}, waiting {delay:.1f}s"
                )
            time.sleep(delay)
    raise AssertionError("unreachable: the loop either returns or raises")


def generate_descriptions(
    dataset: str,
    style: str,
    class_names: List[str],
    model: str,
    temperature: float = 0.0,
    seed: int = 42,
    api_key: Optional[str] = None,
    request_interval: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Call the OpenRouter API once per class and return the full payload
    `generate_class_description.ipynb` writes to disk.

    Does NOT write the file itself -- the notebook owns the "refuse to
    overwrite unless OVERWRITE=True" check, which belongs at the call site,
    not buried in a library function a future caller might invoke without
    meaning to overwrite a frozen artifact.

    Raises `ValueError` for `style not in VALID_STYLES` (this function never
    handles `"manual"` -- that style has no API call, see `load_descriptions`).
    No model family is rejected -- see module docstring.

    **Transient server errors are retried, not fatal.** The calls here are
    strictly sequential -- one blocking request at a time, never concurrent --
    so a `503 UNAVAILABLE` ("this model is currently experiencing high
    demand") reflects load from every OTHER caller of a shared hosted model,
    not anything this sweep is doing. Retried with exponential backoff and
    full jitter; `request_interval` additionally spaces successive classes so
    a 16-class sweep does not fire 16 back-to-back requests at a model that
    just said it was busy. Set `request_interval=0.0` to disable the spacing
    (retries still apply).

    Without this, one transient failure anywhere in a 78-call sweep aborted
    the run and discarded every description already generated -- the whole
    sweep had to start over, into the same busy model.
    """
    if style not in VALID_STYLES:
        raise ValueError(f"style must be one of {sorted(VALID_STYLES)}, got {style!r}")

    import requests

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "No OpenRouter API key -- pass api_key= or set the OPENROUTER_API_KEY "
            "environment variable."
        )
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    descriptions: Dict[str, str] = {}
    for index, class_name in enumerate(class_names):
        prompt = _PROMPT_TEMPLATES[style].format(class_name=class_name, dataset=dataset)
        # A small fixed gap between classes. The calls were already strictly
        # sequential (one blocking request at a time, never concurrent), so
        # this does not fix a self-inflicted burst -- it just stops a 16-class
        # sweep from issuing 16 back-to-back requests into a model that is
        # already reporting high demand, which is what turns one 503 into a
        # run of them. Skipped before the first call, where there is nothing
        # to space out.
        if index > 0 and request_interval > 0.0:
            time.sleep(request_interval)

        def _call(prompt=prompt, class_name=class_name):
            resp = session.post(
                endpoint,
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()

        payload_json = _call_with_retry(
            _call,
            label=f"{dataset}/{style}/{class_name}",
            verbose=verbose,
        )
        choices = payload_json.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message", {}).get("content") or "").strip()
        if not text:
            # An empty completion is not a transport failure, so the retry
            # above never sees it -- but writing it would produce a frozen
            # artifact with a blank description that the notebook's own
            # assert would only catch after every remaining class was
            # generated. Fail on the class that actually failed.
            raise RuntimeError(
                f"{dataset}/{style}/{class_name}: the model returned an empty "
                "description (no text in the response). Re-run this pair; if it "
                "persists, the prompt may be tripping a safety filter."
            )
        descriptions[class_name] = text
        if verbose:
            print(f"    [{index + 1}/{len(class_names)}] {class_name}")

    return {
        "dataset": dataset,
        "model": model,
        "style": style,
        "temperature": temperature,
        "seed": seed,
        "prompt_template": _PROMPT_TEMPLATES[style],
        "generated_at": date.today().isoformat(),
        "descriptions": descriptions,
        "sha256": _sha256_descriptions(descriptions),
    }
