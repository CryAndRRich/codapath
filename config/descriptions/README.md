# Generated class descriptions

Each `{dataset}_{style}.json` file here is a **frozen artifact** written once
by `notebooks/generate_class_description.ipynb` and committed to the repo --
not regenerated per run. `style` is one of `llm_short` | `llm_morphology`.

There is no `manual` style any more: `config.yaml` used to carry a
hand-written `datasets.<dataset>.descriptions` map that doubled as a control
arm, and it now carries only `class_names` (the canonical class order every
stage of the pipeline uses). The weakest-text-prior control is
`extract_vlm_features.ipynb`'s class-name fallback, which derives the prompt
from the class name itself and needs no file. `llm_multi` was removed
earlier -- recoverable from git history.

Read them with `features/descriptions.py::load_descriptions(dataset, style)`,
never by parsing the JSON directly -- that function also validates the class
order matches `config.yaml`'s current order.

A hosted LLM call is not bit-for-bit reproducible even at `temperature=0.0`
(see `features/descriptions.py` module docstring); the reproducible unit is
this committed JSON file, not "run the notebook again and expect the same
text." If a description genuinely needs to change, regenerate with
`OVERWRITE=True` and commit the new file -- do not hand-edit the JSON, since
the file's own `sha256` field would then no longer match its content.

Note the `sha256` is computed over the descriptions with keys SORTED, so it
identifies the TEXT regardless of key order -- a matching hash does NOT prove
the file is loadable. The class order in the JSON must separately match
`config.yaml`'s `class_names`, which is what `load_descriptions` validates.
(`json.dump(..., sort_keys=True)` once alphabetized the nested `descriptions`
dict on the way to disk, producing files with a correct hash that
`load_descriptions` refused; the notebook now writes with `sort_keys=False`.)
