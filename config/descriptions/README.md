# Generated class descriptions

Each `{dataset}_{style}.json` file here is a **frozen artifact** written once
by `notebooks/generate_class_description.ipynb` and committed to the repo --
not regenerated per run. `style` is one of `llm_short` | `llm_morphology` |
`llm_multi` (`manual` needs no file: it reads `datasets.<dataset>.descriptions`
straight out of `config/config.yaml`).

Read them with `features/descriptions.py::load_descriptions(dataset, style)`,
never by parsing the JSON directly -- that function also validates the class
order matches `config.yaml`'s current order.

A hosted LLM call is not bit-for-bit reproducible even at `temperature=0.0`
(see `features/descriptions.py` module docstring); the reproducible unit is
this committed JSON file, not "run the notebook again and expect the same
text." If a description genuinely needs to change, regenerate with
`OVERWRITE=True` and commit the new file -- do not hand-edit the JSON, since
the file's own `sha256` field would then no longer match its content.
