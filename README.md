# Cold-start Active Learning for H&E Pathology

Selecting which histopathology patches to have a pathologist label, when almost
nothing is labelled yet. Every method here is scored under one protocol — frozen
DINOv2 ViT-B/14 features, a single `nn.Linear(768, C)` probe, the same test
metrics — so a difference in accuracy can only come from *which* patches were
selected.

Datasets: PathMNIST (9 classes), HistoSet (14), SkinTissue (16). Budgets
25 → 200 in steps of 25.

---

## Research direction

Three contributions, at three different points in the pipeline:

**1. Remove the cold start.** Every published sampler begins either at random
(margin, entropy, BADGE) or on pure coverage (Uncertainty Herding, TypiClust).
Both are label-blind. Instead: have an LLM write a rich description per class,
encode those with a pathology VLM's text tower, and score the pool against them
zero-shot — so round 1 already carries semantic signal. Implemented as
`sampling/pact/uncertainty.py::text_prior_weights`, reached through
`USE_TEXT=True`; not yet run at full scale.

**2. Cell-vs-visual disagreement as the acquisition signal.** Implemented, and
the current PACT. Keep Uncertainty Herding's objective exactly as published
and swap only its per-point weight: instead of "how close is the classifier to a
boundary here", use "how much do a tissue-level view and a cell-level view
disagree here". Two probes per round — one on DINOv2 patch features, one on
pooled CellViT cell embeddings — compared by Jensen-Shannon divergence after
temperature calibration. A patch both views read identically is uninformative
even when neither is confident; a patch where tissue and cell evidence point at
different classes is where a label buys the most. Pathology-specific: it needs a
nucleus segmenter, and it does not transfer to natural images.

**3. A training framework with an auxiliary loss.** Partly implemented. There
are two independent auxiliary losses, at two different stages. During
SELECTION, `training/probe.py::train_dual_probe` couples the two probes on the
UNLABELED pool (`pool_consistency_weight`, **code default 0 = off, but the
method as published uses 5** — see the note below), masked to the
region both heads already call confidently. That mask is what resolves the
tension with contribution 2: a coupling loss makes the two probes agree, which
shrinks exactly the divergence 2 selects on, so it is kept off the contested
region and applied only where they already agree. (A labeled-slice variant,
`consistency_weight`, was removed — it coupled the heads on the rows they were
already fitting and had no such mask.) After selection, the final-training pass
adds a center/contrastive loss on the encoder's features; see below. Keep the
frozen-DINOv2 protocol for comparing samplers and report the training framework
as a separate claim, or the gain cannot be attributed.

---

## Quick start

```bash
pip install -r requirements.txt

# One sampler, full budget sweep, one dataset.
python main.py --dataset pathmnist --sampler pact

# The controlled ablation: same machinery, Uncertainty Herding's own weight.
python main.py --dataset pathmnist --sampler pact --set uncertainty_mode=visual_margin

# A baseline.
python main.py --dataset pathmnist --sampler uncertainty_herding
```

`--set KEY=VALUE` overrides any sampler field from `config/config.yaml`; values
are parsed as YAML, so `true`, `0.5`, `null` and `[1,2]` arrive typed.

`pact` needs the CellViT cache — see [CellViT extraction](#cellvit-extraction).

On Kaggle, use the notebooks instead (below).

---

## Layout

```
main.py                  single entry point; owns the budget sweep
config/config.yaml       datasets, budgets, per-sampler hyperparameters

data/                    dataset loading, stable sample fingerprints
features/
  visual.py              frozen DINOv2 + on-disk feature cache
  cellvit/               nucleus segmentation, per-cell embeddings, cache, pooling
sampling/
  registry.py            name -> sampler function
  specs.py               how each sampler is swept over budgets  <- read this
  kernels.py             Gaussian coverage kernel + greedy facility location
  calibration.py         ECE-minimising temperature scaling
  uncertainty.py         margin, Jensen-Shannon divergence
  baselines/             published methods, one file each
  pact/               this project's method
training/                linear probe, dual probe, checkpoint IO
evaluation/
  metrics.py             test metrics, computed once during a run
  rescore.py             re-score a run's SAVED WEIGHTS on the test set
  sanity.py              degenerate-selection checks, run during acquisition
  visualize/             figure scripts (run, not imported) -> assets/img/
scripts/                 CellViT extraction and its preflight
notebooks/               Kaggle notebooks, one per pipeline stage
```

`tests/` exists locally but is untracked (see `.gitignore`). Run it with
`python -m pytest tests/ -q` from the repository root.

---

## How a sampler is swept over budgets

The single most error-prone part of this project. **Two independent properties**
decide how a sampler is run, and they must never be inferred from each other.
Both are declared per sampler in [`sampling/specs.py`](sampling/specs.py).

**Axis 1 — `passes`: how many selection passes the algorithm makes for one
budget.** A property of the paper. `single` = one pass produces the whole batch.
`multi` = selection interleaves with retraining a probe, so one budget takes
several internal rounds.

**Axis 2 — `prefix_exact`: whether running once at `max(budgets)` and taking the
first B picks equals running directly at B.** A property of the implementation's
dependence on the target budget. True lets the whole sweep share one run.

The axes are orthogonal, and every combination is populated:

|                | prefix-exact | not prefix-exact |
|---|---|---|
| **single pass** | `random`, `coreset` | `typiclust`, `activeft` |
| **multi round** | *(none currently)* | `margin`, `entropy`, `badge`, `uncertainty_herding`, `refine`, `pact` |

`typiclust` and `activeft` make exactly one pass yet are *not* prefix-exact,
because that pass reads B directly (`typiclust` derives `num_clusters` from it,
`activeft` parameterises its optimisation by it) — so a large-B run is not a
superset of a small-B run. The multi-round *and* prefix-exact cell is currently
empty: `tcm` was its only member and has been removed. The two fields stay
separate anyway, because inferring one from the other is the mistake this table
exists to prevent.

**The rule that catches most cases:** anything that scales an internal threshold
by B is not prefix-exact, however one-shot it looks. That is why
`uncertainty_herding` is False (its phase switch sits at `0.2·B`) and why
`refine` is False too — its stage-2 head *is* Uncertainty Herding.

A third field, `needs`, lists the extra pool arrays a sampler wants (the cell
view, for `pact`). It is **not** a classification axis. An earlier version
of this file split samplers into three sets where two differed only by this
field, which read as a third category that does not exist — and led to real
bugs.

---

## Samplers

Each baseline's module docstring records the paper and the reference
implementation it was verified against, and every deliberate deviation.

| Sampler | Paper | Verified against |
|---|---|---|
| `random`, `margin`, `entropy` | — | standard formulations |
| `coreset` | Sener & Savarese, ICLR 2018 | `repos/badge/.../kcenter_greedy.py` |
| `badge` | Ash et al., ICLR 2020 | `repos/badge/.../badge_sampling.py` |
| `typiclust` | Hacohen et al., ICML 2022 | `repos/typiclust/.../typiclust.py` |
| `activeft` | Xie et al., CVPR 2023 | `repos/activeft/.../ActiveFT_CIFAR.py` |
| `uncertainty_herding` | Bae et al., ICLR 2025 | `repos/uherding/.../uherding.py` + paper |
| `refine` | CVPR 2026 | `repos/refine/.../strategies.py` + official run config |
| `pact` | this project | — |

`dropquery` (TMLR 2024, verified against `repos/dropquery/ALFM/.../dropout.py`)
is no longer a runnable baseline and has no entry in `sampling/specs.py`, but
its module stays: `refine`'s candidate-generation ensemble calls it by name
through the registry, and dropping it would silently change `refine` away from
its published five-strategy configuration.

Baselines are the foundation of every claim, so fidelity beats convenience:
where the reference does something expensive, this code does the expensive
thing and documents the cost rather than quietly cutting it. `refine` is the
clearest case — the official total of 100 candidate batches per round is the
default, and lowering `num_candidate_batches` is a fidelity trade that must be
reported.

---

## PACT

One objective, evaluated identically in every round — Uncertainty Herding's
weighted facility location on the DINOv2 space the probe also lives in:

```
x* = argmax_i  sum_n  U_n · max( k_sigma(x_n, x_i) − K_n , 0 )
```

Unchanged from the paper: the Gaussian kernel, the running-max update `K_n`, the
radius adaptation `sigma = min pairwise distance over the labeled set`
recomputed every round (its Proposition 4 — the reason one method works across
budgets), and ECE-minimising temperature scaling before any margin is read.

The one substitution is `U`:

- **round 1** has no labels, so `U = 1` and the objective is exactly MaxHerding.
  This is where contribution 1 will plug in.
- **later rounds** set `U` from calibrated visual/cell disagreement, blended by
  patch reliability: `U = rho·JSD + (1 − rho)·visual_margin`. A patch where
  CellViT found no nucleus has `rho = 0` and falls back entirely to the visual
  margin.

Axes (all in `config/config.yaml` under `samplers.pact`):

| Field | Meaning |
|---|---|
| `uncertainty_mode` | `disagreement` (the method) or `visual_margin` (ablation: plain Uncertainty Herding) |
| `cell_pooling` | `mean`, `rff` (random-Fourier kernel mean embedding ≈ the patch's cell-distribution KDE) or `moments` |
| `missing_impute` | `mean` or `zero` for patches with no detected nucleus |
| `pool_consistency_weight` | **PACT as published sets this to 5** (`SEL_AUX_WEIGHT=5` in `run_al_main.ipynb`, whose own default is 0.0 — so a default run reproduces the ABLATION, not the method, and nothing warns you). Positive on all three datasets at seed 42 (+0.011 histoset, +0.011 pathmnist, +0.003 skintissue mean accuracy). > 0 adds a semi-supervised loss coupling the two probes on the UNLABELED pool, restricted to points both heads already call confidently. The mask keeps it off the contested region — pool-wide `JS(visual, cell)` *is* the acquisition weight, so an unmasked term would minimise the signal the sampler ranks on. Watch `pool_mask_fraction` in the trace for how much of the pool the term actually saw. |
| `pool_confidence_quantile` | share of each head's own confidence tail admitted (default 0.5). A quantile, not an absolute cut: an absolute 0.9 admitted 67% / 49% / 42% of pathmnist / histoset / skintissue (9 / 14 / 16 classes), so one number meant three different experiments. |
| — | How many pool rows the term uses is derived, not configured: half the pool capped at 20k. The pools differ 4.6x (22,400 vs 100,000 and 103,495), so a fixed row count and a fixed fraction each break on one end of that spread. |

---

## CellViT extraction

Nucleus extraction is deliberately separate from active-learning runs: it needs
a heavy, version-sensitive stack (`requirements-cellvit.txt`) that the sampling
and training paths never import. `main.py` only reads the cache it writes.

The adapter is locked to the official `cellvit==1.0.9` wheel, whose
model/checkpoint contract matches [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus).

**Do not** install CellViT's full dependency tree over Kaggle's PyTorch stack —
it pins NumPy, OpenCV, Pydantic, Ray and WSI packages this patch-level adapter
never imports. Kaggle already provides PyTorch, torchvision, NumPy, pandas,
SciPy, scikit-image, Numba and OpenCV. `scripts/preflight_cellvit.py` imports
each one and runs the real CellViT postprocessor, so an ABI problem surfaces
before the full extraction rather than hours into it.

Kaggle's Python 3.12 image cannot JIT-compile the mixed-dtype `stack_pred_maps`
helper in `cellvit==1.0.9`. The adapter bypasses only that four-channel
`argmax + stack` helper with an equivalent NumPy float32 buffer, and still calls
the official per-image HoVer/morphology/watershed post-processing unchanged. Do
not downgrade the whole Kaggle stack for this.

**Scale is guarded in code.** An x20 checkpoint must use 20× and ≈0.5 MPP; an
x40 checkpoint must use 40× and ≈0.25 MPP. Use a current official checkpoint —
the authors warn that older ones predate a corrected training release. For
PathMNIST (`input_mpp=0.5`) the native-scale choice is an x20 checkpoint
(`model_mpp=0.5, magnification=20`); with only the x40 AMP checkpoint, set
`model_mpp=0.25, magnification=40` and the adapter upsamples 2×. That is
scale-consistent but interpolation cannot recreate detail absent from a 0.5-MPP
source, so record the checkpoint choice as a limitation.

**Memory and time.** The official docs recommend ≥24 GB VRAM for the full WSI
pipeline; this project only does CellViT256 patch inference and defaults to
CellViT batch 2 + DINO crop batch 32 for a Kaggle T4/P100. Preflight executes
both configured batch sizes, so an OOM happens before the full job — drop to 1/8
if it does. An uncapped PathMNIST pilot on a T4 measured ≈67.9 nuclei/patch and
estimated a 14.7 GiB final cache, 44.1 GiB resumable peak, and 30 hours. The
notebook therefore defaults to `MAX_CELLS_PER_PATCH=16`. **Keep that cap
identical across every variant** — it changes the protocol and must be recorded
in results and manifests.

---

## Class descriptions

`generate_class_description.ipynb` writes `config/descriptions/{dataset}_{style}.json`
**once**, committed to the repo — the text prior `extract_vlm_features.ipynb`'s
`DESCRIPTION_STYLES` reads. Calls go through
[OpenRouter](https://openrouter.ai), a single OpenAI-compatible endpoint that
proxies many providers, so `MODEL` in the EDIT cell is an OpenRouter-format id
(`"<provider>/<model>"`, e.g. `"google/gemini-2.5-flash"`) — needs no GPU and
no dataset image, only an OpenRouter API key (Kaggle Secret
`OPENROUTER_API_KEY`, same pattern as `extract_vlm_features.ipynb`'s
`HF_TOKEN`).

**A hosted model call is not bit-for-bit reproducible**, even at
`temperature=0.0` — no provider guarantees determinism across requests, let
alone across months as the weights served behind a fixed model name change,
and a pinned model id can itself stop resolving without notice. The
reproducible unit is the **committed JSON file**, not "re-run
the notebook and expect the same text": the notebook refuses to overwrite an
existing file unless `OVERWRITE=True`, and the payload records `model`,
`temperature`, `seed`, `generated_at` and a `sha256` of the generated text —
proof of what that model actually returned once, not a promise it will again.

**No model family is blocked.** An earlier version rejected every
`gemini-3.x` model because `temperature`/`topK`/`topP` were documented as
deprecated and silently ignored there. That restriction is gone: pin whatever
model is current, and if temperature genuinely has no effect on it, the
written file is still exactly what that model returned — just potentially
harder to reproduce on a later call, the same tradeoff already accepted above.

Two styles (`STYLE` in the EDIT cell, `DESCRIPTION_STYLES` downstream):
`llm_short` (one dense sentence),
`llm_morphology` (2–4 sentences on cell shape/arrangement/texture/staining).
A third style, `llm_multi` (`NUM_PER_CLASS` differently-phrased variants per
class), was removed entirely — recoverable from git history if a
multi-variant ensemble is needed again. There is no `"manual"` style any
more: `config.yaml` used to carry a hand-written
`datasets.<dataset>.descriptions` map that doubled as a control arm, and it
now carries only `class_names` (the canonical class order). The
weakest-text-prior control is `extract_vlm_features.ipynb`'s class-name
fallback, which derives the prompt from the class name itself
(`breast_malignant` → `"breast malignant"`) and needs no config block to
maintain — it also lets that notebook run before any description file
exists.

**Read them with `features/descriptions.py::load_descriptions(dataset, style)`,
never by parsing the JSON directly** — that function also validates the class
order against `config.yaml`'s current `class_names`. Do not hand-edit a file
either: the payload's `sha256` would no longer match its content. Regenerate
with `OVERWRITE=True` and commit instead.

Note what that hash does and does not prove. It is computed over the
descriptions with keys **sorted**, so it identifies the TEXT regardless of key
order — **a matching hash does not mean the file is loadable.** The class order
inside the JSON must separately match `config.yaml`, which is what
`load_descriptions` checks. `json.dump(..., sort_keys=True)` once alphabetized
the nested `descriptions` dict on the way to disk, producing files whose hash
was correct and which `load_descriptions` refused; the notebook now writes with
`sort_keys=False`.

---

## CONCH extraction

`extract_vlm_features.ipynb` produces the second encoder Protocol B needs.
CONCH is CoCa-based (image + text tower; the public checkpoint has its
captioning decoder stripped), gated on Hugging Face, verified against
`repos/CONCH` and the paper directly.

**Two image embedding spaces, not interchangeable**
(`features/vlm.py::extract_vlm_image_features`):

| Space | Call | Used for |
|---|---|---|
| `RAW_SPACE` | `encode_image(x, proj_contrast=False, normalize=False)` | linear probe, coverage kernel, disagreement probes |
| `PROJ_SPACE` | `encode_image(x, proj_contrast=True, normalize=True)` | comparing an image against text (round-1 cold start) |

`encode_image`'s own defaults are `PROJ_SPACE`, not `RAW_SPACE` — the mistake
this module exists to make impossible is asking for one and silently getting
the other. Both are written from the same forward pass; 448×448 (CONCH's
resolution) is 4× the pixels of DINOv2's 224×224, so a second pass would
double the notebook's expensive part.

**448×448, OpenAI CLIP normalization, its own tokenizer** — never DINOv2's
224+ImageNet transform or a hand-rolled `Normalize`. The notebook always uses
the `preprocess` object `create_model_from_pretrained` returns.
`data/loaders.py::get_data_loaders` accepts a `transform=` override for
exactly this; not passing one still gives every other caller the original
224+ImageNet behavior, byte for byte.

**`logit_scale` is learned**, read off the loaded checkpoint
(`model.logit_scale.exp()`), never a hard-coded temperature — and saved into
the text manifest, because the AL round-1 text prior reads it and an AL run
never loads CONCH. It cannot change a zero-shot argmax (a positive scale on
the logits leaves the ordering intact), but it does change the top-2 gap the
acquisition margin reads, so a run that guessed it would select a different
set. `main.run` raises on a manifest that predates the field rather than
defaulting.

**Text prototypes** — one 512-d vector per class, cached separately by
`(dataset, description style)` since they don't depend on the seed or split.
Two styles, `llm_short` and `llm_morphology`, each reading the committed file
`generate_class_description.ipynb` wrote and falling back to the bare class
names when that file does not exist yet. `DESCRIPTION_STYLES` is a LIST: the
image features never see a prompt, so one extraction serves every style and
adding one costs 14 strings through the text tower rather than another 448x448
pass. The archive name lists the styles inside it.

**Zero-shot sanity check**, printed per style and never asserted. It exercises
transform, tokenizer, projection, `logit_scale` and the class ORDER end to end,
and a fault in any of them produces near-random accuracy with no other symptom
— but the extraction is already paid for by the time it runs, so a bad number
prints a warning naming the likely causes rather than throwing the cache away.
No accuracy threshold is applied: the paper's 79.1% is CRC100K-specific, and
histoset (14 classes) and skintissue (16) legitimately score far lower.

---

## Notebooks (Kaggle)

Seven, one per pipeline stage. Each clones and verifies the pinned branch and
installs its own dependencies; every notebook except
`generate_class_description.ipynb` zips its output so a session downloads as
one file (that one writes a single small JSON straight into the repo instead —
see "Class descriptions" above).

| Notebook | Purpose |
|---|---|
| `generate_class_description.ipynb` | frozen LLM class descriptions, generated once, committed |
| `extract_visual_features.ipynb` | DINOv2 features, once per (dataset, seed) |
| `extract_nucleus_features.ipynb` | CellViT cache; runs preflight first |
| `extract_vlm_features.ipynb` | CONCH image features (both embedding spaces) + text prototypes |
| `run_al_baseline.ipynb` | one of the 11 published baselines; no CellViT, no VLM |
| `run_al_main.ipynb` | `pact`, this project's own method — either image encoder |
| `evaluate_al_sampler.ipynb` | load each run's saved probe (+ LoRA adapter) and re-score it on the test set |

**The notebooks carry no prose.** No markdown cells, no explanatory comments —
only code, plus a single `# a | b | c` comment on each editable variable naming
the values it accepts. Anything that explains *why* lives in this README and in
`CLAUDE.md`, in one place rather than duplicated across seven notebooks where
it silently goes stale. Read the EDIT cell for what a run can be told to do;
read here for what the choice means.

`run_al_baseline.ipynb` and `run_al_main.ipynb` are deliberately separate
notebooks rather than one with a bigger menu: a baseline run only needs the
DINOv2 visual cache, and asserting that at the top of its own notebook
(`sampling.specs.BASELINE_SAMPLERS`) catches picking `pact` there before any
GPU time is spent, instead of failing deep inside `main.py` on the first
budget.

**`run_al_main.ipynb`'s `IMAGE_ENCODER` (`"dinov2"` | `"conch"`) decides the
ONE feature space the whole run uses** — the coverage kernel, the disagreement
probes inside `pact`, and the final evaluation probe.
Unlike DINOv2, a CONCH run never extracts its own
features: `main.py` only READS an already-published cache from
`extract_vlm_features.ipynb` (`_load_vlm_features`), and raises with an
actionable message if it is missing — loading the `conch` package, an HF
token and a slow 448×448 forward pass inside a 2-GPU AL sweep would duplicate
what the extraction notebook already does. `USE_TEXT=True` turns on the round-1 cold-start
text prior: `main.run` reads the prototypes from the same VLM cache and passes
them to `pact`, whose round 1 then weights by the margin of the zero-shot
distribution instead of running plain MaxHerding. It needs
`IMAGE_ENCODER="conch"` — the prototypes are compared to the image features by
a dot product, and DINOv2 has no text tower pointing into its space.

**There are TWO auxiliary losses, at two stages, and they are independent.**
`SEL_AUX_WEIGHT` acts on the two probes DURING selection (every round);
`AUX_LOSS` acts on the encoder's features AFTER selection. Confusing them is
easy and consequential, so the notebook names them apart.

**`SEL_AUX_WEIGHT` is the selection-stage term** — semi-supervised, computed
on the UNLABELED pool the probes never otherwise see (~22k–103k rows against
≤200 labeled ones), added to their cross-entropy in one backward pass:

```
loss = CE(visual) + CE(cell) + SEL_AUX_WEIGHT * JS(visual, cell | confident)
```

The confidence mask is load-bearing, not a knob: pool-wide `JS(visual, cell)`
*is* the acquisition weight, so an unmasked term would minimise the very
signal the sampler ranks on. Masking to the region both heads already call
confidently leaves the contested region — what the sampler wants to sample —
out of the loss. Watch `pool_mask_fraction` in the trace for how much of the
pool the term actually saw. Everything else about it adapts to the dataset on
its own (a confidence QUANTILE rather than an absolute cut; half the pool
capped at 20k rather than a fixed count), because the three datasets differ
by 9/14/16 classes and 4.6x in pool size.

**`USE_LORA`/`AUX_LOSS`/`AUGMENT` are the final-training pass** (§6.4/§6.5),
run AFTER a budget's points are already selected. **Every budget in the sweep
gets its own final-training pass**, not just the largest (the confirmed
full-curve choice: the research question is whether LoRA helps more at low or
high budgets). The encoder is loaded exactly ONCE per run and reused across
every budget's pass — so `reset_lora_parameters` restores a clean adapter
before each one, or budget k+1 would inherit budget k's fine-tuned weights.

Three rules govern how these axes combine:

- **`AUX_LOSS` requires `USE_LORA`.** Every loss in `training/losses.py`
  reads only `features`; with the encoder frozen those come from `no_grad`,
  so the term has no `grad_fn` and adds a constant that changes nothing. The
  combination is refused rather than run as a mislabeled baseline.
- **No axis reaches into selection.** `AUGMENT` used to: the per-round
  uncertainty probe trained on freshly augmented pixels
  (`make_augmented_feature_provider`). Measured on histoset seed 42 that
  changed the *selected set* — only ~55% overlap with the un-augmented run at
  every budget — because the provider can only augment the probe's training
  rows and the CellViT cell probe cannot be augmented at all (its embeddings
  come from a pixel-less cache), so `JS(visual, cell)` read the injected noise
  as disagreement (1.65x the frozen run's on average). `AUGMENT` now applies
  only to the final probe, like `USE_LORA` and `AUX_LOSS`.
- **The adapter has its own learning rate.** `LORA_LR` defaults to `1e-4`,
  separate from `PROBE_LR` (`1e-3`). One shared rate collapsed a real run —
  2849 of 5600 test rows predicted as one class at budget 200 — because a rate
  sized for a linear probe moves a rank-8 adapter's features 116.9% in L2
  (cosine 0.318 with the frozen output) against 82.9% / 0.652 at `1e-4`.
- **Test scoring follows the encoder.** A LoRA run re-encodes the test set
  through the adapted encoder; the frozen-encoder paths keep using the
  embedding cache, which describes exactly that encoder.

| Piece | What it does |
|---|---|
| `training/lora.py` | Hand-rolled LoRA (no `peft` locally). `LinearLoRA` wraps DINOv2's separate `query`/`value` `nn.Linear` layers; `FusedQKVLoRA` wraps the fused `qkv` `nn.Linear` in CONCH's **timm** vision trunk (`model.visual.trunk.blocks[i].attn.qkv`), adding a delta to the Q and V row-blocks only. Verified against the real modules they wrap: `r=0` is bit-for-bit identical to the frozen original. `reset_lora_parameters` restores the freshly-wrapped state (zero delta) so each budget in a sweep starts from a clean adapter — `main.run` reuses one encoder object across all budgets and trains it in place. |
| `training/losses.py` | `center_loss`, `supcon_loss`, `triplet_loss`, one shared `(features, logits, labels) -> scalar` signature. `supcon`/`triplet` **raise** on any batch with fewer than 2 samples of a present class — a silent near-zero loss on a thin batch is exactly the failure mode this project has already lost debugging time to once (see the minmax-on-a-constant-vector lesson). |
| `data/augment.py` | `flip_rotate` only — deliberately no color jitter, since this project already lost a method to a stain-shortcut failure from color augmentation on H&E tiles. |
| `training/finetune.py` | `finetune_and_evaluate` — reads raw pixels for just the selected indices (`RawRGBDataset`, no dataset-wide decode), trains encoder + probe end-to-end when `use_lora=True`, or just the probe (encoder frozen under `inference_mode`) when only `augment` is active. **Test-set scoring follows the encoder**: `use_lora=True` re-encodes the test set through the adapted encoder (`encode_dataset`, requires `test_dataset`), because a probe trained on adapted features cannot be scored against the frozen cache; the augment-only path leaves the encoder frozen and so keeps using the cache directly. |

A probe from this pass records `metadata["final_train_cfg"]` (and
`results["final_train_cfg"]`) — present only when the pass actually ran, so a
reader can tell "this run never had a final-training axis" apart from "this
run's final-training axes were all off." No second results key: §2.0 already
settled this — `USE_LORA=False` and `USE_LORA=True` are two notebook runs
with two distinct zip names, so nothing can collide inside `results["linear"]`.

Publishing `/kaggle/working/<name>` as a Kaggle Dataset remounts it one level
deeper (`.../<name>/<name>`), so the notebooks *search* for their caches instead
of trusting a hard-coded path, and print what they found.

Every notebook that produces something ends the same way: **one zip at the top
of `/kaggle/working`, with the loose files deleted.** A "Save & Run All" session
has no terminal and no kaggle CLI, so the Output tab is the only way a file
leaves it — and leaving the originals beside the zip doubles the download and
can push the session over the ~20 GB Output quota, at which point the tab shows
*nothing at all*, including the files that were fine. The name carries the axes
that make two archives non-interchangeable (`utils/archive.py`):

| Notebook | Archive name |
|---|---|
| `extract_visual_features.ipynb` | `visual-dinov2_{dataset}_seed{seed}_{backbone}` |
| `extract_nucleus_features.ipynb` | `cellvit-nucleus_{dataset}_seed{seed}_{ckpt}_{encoder}_{cap}` |
| `extract_vlm_features.ipynb` | `vlm_{dataset}_seed{seed}_{vlm}_{description_style}` |
| `run_al_baseline.ipynb` | `{dataset}_{sampler}_seed{seed}` |
| `run_al_main.ipynb` | `{dataset}_{sampler}_seed{seed}[_{encoder}]` — `encoder` appended only when not `dinov2` |

**A run needs two Kaggle Datasets attached, and the feature cache cannot stand
in for the raw images.** `DATA_ROOT` is the raw dataset (`pathmnist_224.npz`,
HistoSet, SkinTissue); `FEATURE_DIR` is the `.npy` cache
`extract_visual_features.ipynb` published. The cache holds DINOv2 matrices and
nothing else — the oracle labels a sampler selects on, the labels the probe
trains against, and the sample-order fingerprint that *validates the cache
itself* all come from the dataset, so `main.run` opens it either way. Attaching
only the feature dataset fails inside `get_data_loaders`, before the cache is
ever consulted. What the cache does buy is the backbone forward pass over ~90k
images, which is the expensive part.

Seeds matter: ImageFolder datasets are split by a seeded generator, so a feature
cache is only valid for the seed it was built with.

### Two GPUs

**One run, one configuration, one zip.** Every notebook takes a single value
per axis — one dataset, one seed, one sampler, one config — and ends in one
archive whose name states all of it. A run covering several configurations
would put them under a single archive name that cannot say which result is
which, so sweeping is done by running the notebook again.

That costs nothing in speed, because both cards of a Kaggle **T4 x2** session
are used *within* one configuration: the run notebooks split the **budget
list** (`SPLIT_BUDGETS`), and the extraction notebooks shard one extraction
(`utils/parallel.py`, one worker process per GPU). Set `PARALLEL = False` for
serial.

Budget splitting is sound precisely for the samplers where every budget is
already an independent run: `spec.prefix_exact == False`. The flag itself
decides, so a new sampler cannot drift out of sync with a hand-kept list. A
prefix-exact sampler (`random`, `coreset`) derives its whole sweep from
one selection pass, so sharding it would repeat that pass per shard; `main.run`
refuses a `shard_tag` for those rather than silently doing more work.

Budgets are dealt round-robin, not contiguously, because cost grows with the
budget — a contiguous split would hand one worker every expensive budget. Each
shard writes its own `<run>_<tag>_results.pt` and log (the per-budget files are
already named by budget and never collide), and `main.merge_budget_shards`
folds them into the single `<run>_results.pt` an unsharded run would have
written, so nothing downstream needs to know a run was split.

Processes, not threads: each run calls `set_seed`, which mutates global RNG
state and an environment variable, so two threads doing it in one interpreter
would interleave and destroy reproducibility. Each worker pins one card via
`CUDA_VISIBLE_DEVICES` before torch initialises, and therefore always uses
`cuda:0` internally — which is why jobs pass `device_string` to
`main.run_on_worker` rather than a `torch.device`.

Work is dealt out round-robin up front and a worker that finishes early does
not steal from a slower one. A crashing job is reported with its traceback and
does not take the others down.

Both GPUs share one output stream, so their progress lines interleave and a
budget's numbers can appear anywhere in the log. Each run notebook therefore
ends with a summary cell that re-reads the saved `<run>_results.pt` and prints
one ordered row per budget — accuracy, precision, recall, macro-F1 — so the
table is correct however the log came out, and re-running that cell alone
reprints it without recomputing anything. Feature-cache writes go through a
temporary file and an atomic rename, because two workers can miss the cache and
extract simultaneously.

### Progress and ETA

Long loops report elapsed time, time remaining and an absolute finish time —
the number that actually answers "will this fit in the session". `tqdm` drives
the console; a periodic plain line carries the same information into the log,
where carriage-return bars are useless. Reports are rate-limited and suppressed
inside nested sampler calls (`utils.progress.quiet_progress`), since `refine`
invokes other samplers thousands of times per budget.

---

## Outputs

Per budget, under `checkpoints/<dataset>/`:

| File | Contents |
|---|---|
| `<run>_selected_budget_<B>.pt` | selected indices, sample ids, labels, per-class counts, sampler config, per-step trace, sanity report, timings, fingerprints |
| `<run>_probe_budget_<B>.pt` | the probe's linear weights, plus run/budget/seed/**encoder** metadata |
| `<run>_predictions_budget_<B>.pt` | test-set class probabilities and true labels, **scored in whatever feature space that budget's probe was trained in** |
| `<run>_results.pt` | accuracy / precision / recall / macro-F1 and timings for every budget |
| `<run>.log` | everything printed during the run |

The per-step trace inside `<run>_selected_budget_<B>.pt` carries each pick's
acquisition `score` and, where the method computes them as separate factors,
its `uncertainty` and `coverage` terms — so a later plot can ask which of the
two actually drove a selection. A sampler records only what it genuinely
computes: `coreset`, `typiclust` and `activeft` fit no classifier and so have
no uncertainty, and `random` has no score at all.

A run reports **accuracy, precision, recall and macro-F1 only**. Those numbers
are the run's report of ITSELF, which is why `evaluate_al_sampler.ipynb` does
not read them: it re-scores each saved probe against the test set, so the table
says what the shipped weights actually do rather than what a finished run said
they did.

`<run>` defaults to the sampler name, extended for `pact` with the config
axes that would otherwise overwrite each other
(`pact_disagreement`, `pact_visual_margin`, …), and suffixed `_s<seed>`
for any seed other than the config default. `_default_run_name` also accepts
`encoder` (default `"dinov2"`, a non-default value appends e.g. `_conch`) and
`use_text` with `description_style` (appends e.g. `_text-llm_short`), so a
CONCH run of the same sampler+config gets a name distinct from its DINOv2
counterpart, and two description styles get names distinct from each other.
Without this the runs would silently overwrite every file the first one wrote,
and the notebook's resume check would skip the later ones entirely, mistaking
a leftover results file for a finished run. The style is only appended when
`use_text` is on, so a run that read no text does not claim an axis it never
exercised, and calling `_default_run_name` with neither — as `run_al_baseline`
does — produces exactly the name it always has.

**A probe checkpoint states which feature space it was trained on.**
`metadata["encoder"]` (the backbone name) and `metadata["encoder_kind"]`
(`"dinov2"` today) let `evaluate_al_sampler.ipynb` load the matching test
features for each run instead of assuming one encoder for everything —
without this, a checkpoint trained on a different feature space either
crashes on a shape mismatch or, if the two widths ever coincided, would be
silently scored against the wrong features. A checkpoint written before this
field existed has no `encoder_kind` key and is read as DINOv2, so nothing
already published needs to be regenerated.

**The saved probabilities follow the encoder, like the metrics do.** For a
LoRA run they come from the adapted encoder, not the frozen cache — `main.py`
takes them from `finetune_and_evaluate`'s return value rather than recomputing
them. It used to recompute, which raised nothing (both spaces are 768-d) and
silently wrote a predictions file that disagreed with the same run's own
reported accuracy by up to 0.37. Archives written before 2026-09-05 by a LoRA
run still carry those wrong probabilities: their `_results.pt` is correct, but
anything derived from `_predictions_*.pt` (confusion matrix, per-class F1,
calibration) is not valid for those runs.

Everything a later plot needs is written **during** the run, because none of it
survives otherwise: the per-step acquisition score exists only inside the greedy
loop, and rebuilding the test predictions costs a full backbone pass.

### The per-step trace

`trace` in the selection file records, for each pick, its `round_index`, `rank`,
winning `score` and `margin_to_runner_up`; and for each round, the `sigma` in
force, a distribution summary of the weight and score vectors, and the round's
wall-clock. Two uses:

* **Visualisation** — selection order, per-round acquisition distributions, how
  `sigma` contracts as labels accumulate.
* **Auditing** — see below.

### The sanity report

Every budget is checked and the result printed and stored under `sanity`. This
exists because the failures that matter here all return the right number of
distinct in-range indices, so neither a smoke test nor a plausible accuracy
reveals them. `evaluation/sanity.py` flags:

| Finding | What it means |
|---|---|
| `index_ordered` | picks are ≥0.9 concordant with plain increasing index order — the signature of `argmax` over a constant score |
| `constant_step_score` / `constant_weight` | the objective could not tell candidates apart |
| `sigma_zero` | the kernel has collapsed into a duplicate indicator |
| `nonmonotone_gain` | coverage gains rose *within* a round, so the running max may not be updating |
| `single_class` / `classes_missing` | the selection cannot train a probe, or is missing classes |
| `duplicates` / `out_of_range` | the sampler returned an invalid index set |

A round may declare its weights uniform *by design* (`weight_uniform_by_design`),
which is how round 1 of a coverage method legitimately runs plain MaxHerding
without being flagged. Nothing raises: an alarm at budget 200 must not throw
away the hours already spent, so findings are reported, not enforced.

### Re-scoring a finished run

`evaluation/rescore.py` loads `<run>_probe_budget_<b>.pt` — and
`<run>_lora_budget_<b>.pt` when the run trained an adapter — and computes the
metrics against real test features. No results file is read, and none is
available to read: **`PACT.zip` and `baselines.zip` contain weights and
nothing else**, so there is no recorded metric to fall back on by
construction. What each run WAS — its encoder, dataset, seed, run name — comes
from the `metadata` dict `save_probe` writes inside each checkpoint, which is
why that dict exists.

`<run>_results.pt` still exists, under `data_upload/selected/`, alongside the
selections. It is a run's report of itself: it loads and prints the same
values whether the probe beside it is usable or corrupt, so it is kept for
provenance and deliberately kept OUT of the weight archives.

The cost is real: test labels come from the image dataset, so the notebook
needs it mounted, and a LoRA run needs a GPU and one forward pass over the test
set per budget.

Three things re-scoring has to guard, each tested by constructing the
situation it exists to catch:

* **feature space.** The cache is derived from the run's own
  `visual_backbone`, never named by hand, and `load_test_features` refuses a
  CONCH manifest whose `space` is not RAW — PROJ is the same 512 width, so the
  wrong one would score and report silently. It also refuses a cache whose
  `test_fingerprint` disagrees with the run's: same width, same row count,
  every label against the wrong patch.
* **the adapted space.** A run that saved `*_lora_budget_*.pt` trained its
  encoder, so its probe indexes a space no cache holds. `rescore_run` REFUSES
  to score it from a cache and requires an `encode_test` callable that
  re-encodes the test set through that budget's adapter — the same mistake
  once measured 0.10–0.21 accuracy against a 0.071 floor, invisible because
  the widths matched. `SCORE_LORA_RUNS=False` skips those runs instead, since
  the honest alternative costs a forward pass per budget.
* **the adapter itself.** `make_lora_encoder` reads `lora_alpha` from the file
  rather than assuming it (alpha scales the delta at forward time and leaves no
  trace in the weights), rejects tensor names that do not resolve against the
  wrapped module, and refuses an adapter whose every `lora_B` is zero — that
  state rebuilds the FROZEN encoder with correct shapes and no error.

---

## Papers

Full per-paper notes live in `references/*.md` and the PDFs in `pdfs/` (both
local-only, outside this repository).

**Uncertainty** — CEC (WACV 2025) · SaE (CVPR 2026)

**Coverage / diversity** — BADGE (ICLR 2020) · BAIT (NeurIPS 2021) ·
TypiClust (ICML 2022) · ActiveFT (CVPR 2023) · MaxHerding (ECCV 2024) ·
UncertaintyHerding (ICLR 2025)

**Hybrid** — CB+SQ (TMLR 2025) · REFINE (CVPR 2026)

**Medical / pathology AL** — PEAL (CVPR 2024) · OpenPath (MICCAI 2025)

**Evaluation** — PALM (ICCV 2025), ALDA (EMA4MICCAI 2026) and
Mechanism-Driven Phase Transitions (ECCV 2026) are **not implemented**; the
PALM/ALDA curve-fitting code was removed on 2026-09-12 (see CLAUDE.md)

Reference implementations are cloned read-only under `repos/` for line-by-line
comparison: `badge`, `coreset`, `typiclust`, `activeft`, `uherding`,
`dropquery`, `refine`, `PALM`. The `PALM` clone carries all three of that
group's papers — PALM itself, ALDA (`deep-al/tools/alda/`), and the
mechanism-driven phase analysis (`deep-al/tools/mechanistic/`). The third is
**not implemented here**: four of its six operational proxies could be computed
from what a run already saves, but empirical-risk reduction is measured inside
the AL loop and is not recoverable afterwards.

---
