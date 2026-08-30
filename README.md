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
zero-shot — so round 1 already carries semantic signal. *Not implemented yet.*

**2. Cell-vs-visual disagreement as the acquisition signal.** Implemented, and
the current SCALPEL. Keep Uncertainty Herding's objective exactly as published
and swap only its per-point weight: instead of "how close is the classifier to a
boundary here", use "how much do a tissue-level view and a cell-level view
disagree here". Two probes per round — one on DINOv2 patch features, one on
pooled CellViT cell embeddings — compared by Jensen-Shannon divergence after
temperature calibration. A patch both views read identically is uninformative
even when neither is confident; a patch where tissue and cell evidence point at
different classes is where a label buys the most. Pathology-specific: it needs a
nucleus segmenter, and it does not transfer to natural images.

**3. A training framework with an auxiliary loss.** Partly implemented. The
probe-to-probe consistency term is in `training/probe.py::train_dual_probe`
(`consistency_weight`, default 0 = off). LoRA on a pathology backbone plus a
center/contrastive loss is not implemented. Note the tension: a consistency loss
that makes the two probes agree shrinks exactly the divergence contribution 2
selects on, so the two must not be the same quantity in the same place. Keep the
frozen-DINOv2 protocol for comparing samplers and report the training framework
as a separate claim, or the gain cannot be attributed.

---

## Quick start

```bash
pip install -r requirements.txt

# One sampler, full budget sweep, one dataset.
python main.py --dataset pathmnist --sampler scalpel

# The controlled ablation: same machinery, Uncertainty Herding's own weight.
python main.py --dataset pathmnist --sampler scalpel --set uncertainty_mode=visual_margin

# A baseline.
python main.py --dataset pathmnist --sampler uncertainty_herding
```

`--set KEY=VALUE` overrides any sampler field from `config/config.yaml`; values
are parsed as YAML, so `true`, `0.5`, `null` and `[1,2]` arrive typed.

`scalpel` needs the CellViT cache — see [CellViT extraction](#cellvit-extraction).

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
  scalpel/               this project's method
training/                linear probe, dual probe, checkpoint IO
evaluation/              test metrics, PALM curve fitting, plots
scripts/                 CellViT extraction and its preflight
notebooks/               four Kaggle notebooks, one per pipeline stage
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
| **multi round** | `tcm` (once B ≥ 3·C) | `margin`, `entropy`, `badge`, `dropquery`, `uncertainty_herding`, `refine`, `scalpel` |

`typiclust` and `activeft` make exactly one pass yet are *not* prefix-exact,
because that pass reads B directly (`typiclust` derives `num_clusters` from it,
`activeft` parameterises its optimisation by it). Conversely `tcm` runs several
rounds yet *is* prefix-exact, because its phase boundary is a multiple of the
class count rather than of B.

**The rule that catches most cases:** anything that scales an internal threshold
by B is not prefix-exact, however one-shot it looks. That is why
`uncertainty_herding` is False (its phase switch sits at `0.2·B`) and why
`refine` is False too — its stage-2 head *is* Uncertainty Herding.

A third field, `needs`, lists the extra pool arrays a sampler wants (the cell
view, for `scalpel`). It is **not** a classification axis. An earlier version
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
| `tcm` | ICLR 2024 Workshop | paper text (no public code) |
| `dropquery` | TMLR 2024 | `repos/dropquery/ALFM/.../dropout.py` |
| `uncertainty_herding` | Bae et al., ICLR 2025 | `repos/uherding/.../uherding.py` + paper |
| `refine` | CVPR 2026 | `repos/refine/.../strategies.py` + official run config |
| `scalpel` | this project | — |

Baselines are the foundation of every claim, so fidelity beats convenience:
where the reference does something expensive, this code does the expensive
thing and documents the cost rather than quietly cutting it. `refine` is the
clearest case — the official total of 100 candidate batches per round is the
default, and lowering `num_candidate_batches` is a fidelity trade that must be
reported.

---

## SCALPEL

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

Axes (all in `config/config.yaml` under `samplers.scalpel`):

| Field | Meaning |
|---|---|
| `uncertainty_mode` | `disagreement` (the method) or `visual_margin` (ablation: plain Uncertainty Herding) |
| `cell_pooling` | `mean`, `rff` (random-Fourier kernel mean embedding ≈ the patch's cell-distribution KDE) or `moments` |
| `missing_impute` | `mean` or `zero` for patches with no detected nucleus |
| `consistency_weight` | > 0 couples the two probes while training; works *against* acquisition |

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
`DESCRIPTION_STYLE="llm_*"` reads. It calls `gemini-2.5-flash` (pinned; see
below) and needs no GPU and no dataset image, only a Gemini API key (Kaggle
Secret `GEMINI_API_KEY`, same pattern as `extract_vlm_features.ipynb`'s
`HF_TOKEN`).

**A hosted model call is not bit-for-bit reproducible**, even at
`temperature=0.0` — Google does not guarantee determinism across requests, let
alone across months as the weights served behind a fixed model name change.
The reproducible unit is the **committed JSON file**, not "re-run the
notebook and expect the same text": the notebook refuses to overwrite an
existing file unless `OVERWRITE=True`, and prints this caveat before calling
the API.

**`gemini-3.x` is rejected in code**, not just avoided by default —
`temperature`/`topK`/`topP` are documented as deprecated and silently ignored
on `gemini-3.7-flash`, `3.6-flash`, `3.5-flash-lite`. Using one of those would
make `TEMPERATURE=0.0` a silent no-op: the call still succeeds, so nothing
would flag that the setting this project's reproducibility claim depends on
was never applied.

Three styles (`STYLE` in the EDIT cell): `llm_short` (one dense sentence),
`llm_morphology` (2–4 sentences on cell shape/arrangement/texture/staining),
`llm_multi` (`NUM_PER_CLASS` differently-phrased variants per class, encoded
as an ensemble the same way `conch_official`'s templates are). `"manual"`
needs no file — `features/descriptions.py::load_descriptions` reads
`datasets.<dataset>.descriptions` from `config.yaml` directly, and is the
control every LLM style has to beat.

---

## CONCH extraction

`extract_vlm_features.ipynb` produces the second encoder Protocol B needs.
CONCH is CoCa-based (image + text tower; the public checkpoint has its
captioning decoder stripped), gated on Hugging Face, verified against
`repos/CONCH` and the paper directly (`PLAN_IMPLEMENT.md` §10).

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
(`model.logit_scale.exp()`), never a hard-coded temperature.

**Text prototypes** — one 512-d vector per class, cached separately by
`(dataset, description style)` since they don't depend on the seed or split.
`DESCRIPTION_STYLE = "conch_official"` uses the paper authors' own 22-template
× 4–5-classname CRC100K prompt ensemble (vendored, CC BY-NC-ND 4.0, in
`config/prompts/`; PathMNIST only — its 9 classes match `config.yaml`'s
`datasets.pathmnist.descriptions` order 1:1, checked in code via
`assert_class_order_matches_prompts`). `"manual"` reads `config.yaml`
directly; `"llm_*"` reads a file `generate_class_description.ipynb` wrote.

**Zero-shot sanity check**, printed and asserted `> 0.70` at the end of the
notebook — not a reproduction of the paper's 79.1% on CRC100K (PathMNIST is
224-native, resized up to 448 here), but a check that transform,
normalization, tokenizer and projection are all correct. A wrong one of those
does not crash — it silently produces near-random accuracy with no other
symptom.

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
| `run_al_main.ipynb` | `scalpel`, this project's own method — either image encoder |
| `evaluate_al_sampler.ipynb` | reload saved probes and rebuild the comparison table |

`run_al_baseline.ipynb` and `run_al_main.ipynb` are deliberately separate
notebooks rather than one with a bigger menu: a baseline run only needs the
DINOv2 visual cache, and asserting that at the top of its own notebook
(`sampling.specs.BASELINE_SAMPLERS`) catches picking `scalpel` there before any
GPU time is spent, instead of failing deep inside `main.py` on the first
budget.

**`run_al_main.ipynb`'s `IMAGE_ENCODER` (`"dinov2"` | `"conch"`) decides the
ONE feature space the whole run uses** — the coverage kernel, the disagreement
probes inside `scalpel`, and the final evaluation probe
(`PLAN_IMPLEMENT.md` §6.2). Unlike DINOv2, a CONCH run never extracts its own
features: `main.py` only READS an already-published cache from
`extract_vlm_features.ipynb` (`_load_vlm_features`), and raises with an
actionable message if it is missing — loading the `conch` package, an HF
token and a slow 448×448 forward pass inside a 2-GPU AL sweep would duplicate
what the extraction notebook already does. `USE_TEXT` (the round-1 cold-start
text prior) is present in the EDIT cell but **not yet implemented** — setting
it True raises `NotImplementedError` naming exactly what is missing
(`sampling/scalpel/sampler.py` round 1 is still plain MaxHerding, `U=1`; see
`CLAUDE.md`'s contribution #1) rather than silently running without it.

**`USE_LORA`/`AUX_LOSS`/`AUGMENT` are the final-training pass** (§6.4/§6.5),
run AFTER a budget's points are already selected — never inside selection
itself, which still always trains on the frozen embedding cache. **Every
budget in the sweep gets its own final-training pass**, not just the largest
(the confirmed full-curve choice: the research question is whether LoRA
helps more at low or high budgets). The encoder is loaded exactly ONCE per
run, reused across every budget's pass.

| Piece | What it does |
|---|---|
| `training/lora.py` | Hand-rolled LoRA (no `peft` locally). `LinearLoRA` wraps DINOv2's separate `query`/`value` `nn.Linear` layers; `MultiheadAttentionLoRA` replaces CONCH's fused `nn.MultiheadAttention` with a hand-written forward that re-derives Q/K/V from the same `in_proj_weight` plus a low-rank delta on Q and V (K is never adapted). Both verified against the real modules they wrap: `r=0` is bit-for-bit identical to the frozen original. |
| `training/losses.py` | `center_loss`, `supcon_loss`, `triplet_loss`, one shared `(features, logits, labels) -> scalar` signature. `supcon`/`triplet` **raise** on any batch with fewer than 2 samples of a present class — a silent near-zero loss on a thin batch is exactly the failure mode this project has already lost debugging time to once (see the minmax-on-a-constant-vector lesson). |
| `data/augment.py` | `flip_rotate` only — deliberately no color jitter, since this project already lost a method to a stain-shortcut failure from color augmentation on H&E tiles. |
| `training/finetune.py` | `finetune_and_evaluate` — reads raw pixels for just the selected indices (`RawRGBDataset`, no dataset-wide decode), trains encoder + probe end-to-end when `use_lora=True`, or just the probe (encoder frozen under `inference_mode`) when only `augment` is active. Test-set scoring always uses the frozen embedding cache, never a re-extracted adapted-encoder cache. |

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
prefix-exact sampler (`random`, `coreset`, `tcm`) derives its whole sweep from
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
| `<run>_predictions_budget_<B>.pt` | test-set class probabilities and true labels |
| `<run>_results.pt` | accuracy / precision / recall / macro-F1 and timings for every budget |
| `<run>.log` | everything printed during the run |

The per-step trace inside `<run>_selected_budget_<B>.pt` carries each pick's
acquisition `score` and, where the method computes them as separate factors,
its `uncertainty` and `coverage` terms — so a later plot can ask which of the
two actually drove a selection. A sampler records only what it genuinely
computes: `coreset`, `typiclust` and `activeft` fit no classifier and so have
no uncertainty, and `random` has no score at all.

A run reports **accuracy, precision, recall and macro-F1 only**. PALM and every
other curve-level metric are fitted by `evaluate_al_sampler.ipynb` from these
files: the fit needs the whole sweep to have finished, which a resumed or
GPU-split run cannot guarantee mid-sweep, and re-fitting costs seconds against
re-running the sweep.

`<run>` defaults to the sampler name, extended for `scalpel` with the config
axes that would otherwise overwrite each other
(`scalpel_disagreement`, `scalpel_visual_margin`, …), and suffixed `_s<seed>`
for any seed other than the config default. `_default_run_name` also accepts
`encoder` (default `"dinov2"`, a non-default value appends e.g. `_conch`) and
`use_text` (appends `_text` when true) so a future CONCH run of the same
sampler+config gets a name distinct from its DINOv2 counterpart — without
this, the two protocols would silently overwrite every file the first one
wrote, and the notebook's resume check would skip the second protocol's run
entirely, mistaking the first protocol's leftover results file for a
finished run of the second. Neither parameter is wired into `run()` yet
(that is `run_al_main.ipynb`'s job); calling `_default_run_name` with no
encoder/use_text — as every current notebook does — produces exactly the
same name it always has.

**A probe checkpoint states which feature space it was trained on.**
`metadata["encoder"]` (the backbone name) and `metadata["encoder_kind"]`
(`"dinov2"` today) let `evaluate_al_sampler.ipynb` build the matching test
features for each run instead of assuming one encoder for everything —
without this, a checkpoint trained on a different feature space either
crashes on a shape mismatch or, if the two widths ever coincided, would be
silently scored against the wrong features. A checkpoint written before this
field existed has no `encoder_kind` key and is read as DINOv2, so nothing
already published needs to be regenerated.

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

Accuracy at a handful of budgets is a noisy ranking: a method can win at one
budget and lose at the next. PALM fits the whole learning curve and reports
interpretable parameters instead. It is fitted in `evaluate_al_sampler.ipynb`,
not during a run — the fit needs every budget of the sweep to be present.

| PALM metric | Meaning | Better |
|---|---|---|
| `Amax` | accuracy ceiling | higher |
| `delta` | coverage efficiency per label | higher |
| `alpha` | cold-start offset | lower |
| `beta` | budget-scaling exponent | higher |
| `AUC` | overall curve area | higher |
| budget-to-90 | labels needed to reach 90% of `Amax` | lower |
| `RMSE` | fit reliability of the five above | lower |

---

## Papers

Full per-paper notes live in `references/*.md` and the PDFs in `pdfs/` (both
local-only, outside this repository).

**Uncertainty** — CEC (WACV 2025) · SaE (CVPR 2026)

**Coverage / diversity** — BADGE (ICLR 2020) · BAIT (NeurIPS 2021) ·
TypiClust (ICML 2022) · ActiveFT (CVPR 2023) · MaxHerding (ECCV 2024) ·
UncertaintyHerding (ICLR 2025)

**Hybrid** — TCM (ICLR 2024 Workshop) · DropQuery (TMLR 2024) ·
CB+SQ (TMLR 2025) · REFINE (CVPR 2026)

**Medical / pathology AL** — PEAL (CVPR 2024) · OpenPath (MICCAI 2025)

**Evaluation** — PALM (ICCV 2025)

Reference implementations are cloned read-only under `repos/` for line-by-line
comparison: `badge`, `coreset`, `typiclust`, `activeft`, `uherding`,
`dropquery`, `refine`, `PALM`.

---

## Appendix — results measured before the 2026-08-22 audit

**These numbers are not reproducible with the current code.** The baseline audit
of 2026-08-22 changed selection behaviour in `typiclust` and `tcm` (k-means
restarts pinned to the defaults the references relied on), `activeft` (the
detached factor in the diversity term), `tcm` (transition at 3·C with step C,
per the paper's own regime) and `refine` (official ensemble, candidate-batch
total and pool cap). They are kept as a regression reference only — for "did the
rewrite move this baseline, and by how much", not as reported results.

`SCALPEL` in these tables is the retired stain-shortcut method (v9), unrelated
to the current disagreement-based sampler of the same name. `CODAPath`, this
project's own earlier dual-VLM sampler, has since been **deleted from the
code** (`sampling/baselines/codapath.py`); it appears in these tables only as
historical measurement, never as a runnable sampler.

### HistoSet

#### Accuracy (linear) theo budget

| Phương pháp | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 | 325 | 350 | 375 | 400 | 425 | 450 | 475 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.3736 | 0.5138 | 0.6852 | 0.7312 | 0.7523 | 0.7557 | 0.7800 | 0.8036 | 0.8159 | 0.8216 | 0.8280 | 0.8316 | 0.8361 | 0.8436 | 0.8461 | 0.8466 | 0.8504 | 0.8514 | 0.8600 | 0.8611 |
| Coreset | 0.1982 | 0.2743 | 0.3680 | 0.4152 | 0.4455 | 0.4945 | 0.5414 | 0.5909 | 0.6088 | 0.5902 | 0.6252 | 0.6250 | 0.6484 | 0.6809 | 0.6789 | 0.7034 | 0.7173 | 0.7023 | 0.7205 | 0.7200 |
| UHerding | 0.5591 | 0.6625 | 0.6893 | 0.7400 | 0.7502 | 0.7752 | 0.7857 | 0.7862 | 0.7920 | 0.7977 | 0.7884 | 0.7879 | 0.7932 | 0.8102 | 0.8241 | 0.8161 | 0.8212 | 0.8257 | 0.8423 | 0.8520 |
| TCM | 0.4770 | 0.5877 | 0.6443 | 0.6930 | 0.7348 | 0.7530 | 0.7602 | 0.7629 | 0.7771 | 0.7873 | 0.7905 | 0.8016 | 0.8134 | 0.8218 | 0.8375 | 0.8388 | 0.8438 | 0.8404 | 0.8471 | 0.8523 |
| REFINE | 0.4980 | 0.6089 | 0.6641 | 0.7175 | 0.7488 | 0.7655 | 0.7848 | 0.8014 | 0.8205 | 0.8370 | 0.8455 | 0.8461 | 0.8502 | 0.8636 | 0.8693 | 0.8730 | 0.8721 | 0.8770 | 0.8836 | 0.8918 |
| TypiClust | 0.5271 | 0.6695 | 0.7070 | 0.7280 | 0.7630 | 0.7929 | 0.7896 | 0.8045 | 0.8139 | 0.8159 | 0.8216 | 0.8454 | 0.8407 | 0.8525 | 0.8570 | 0.8648 | 0.8754 | 0.8686 | 0.8855 | 0.8839 |
| ActiveFT | 0.4454 | 0.5586 | 0.6493 | 0.7214 | 0.7538 | 0.7679 | 0.7989 | 0.8077 | 0.8136 | 0.8189 | 0.8248 | 0.8295 | 0.8379 | 0.8529 | 0.8312 | 0.8338 | 0.8471 | 0.8505 | 0.8586 | 0.8573 |
| DropQuery | 0.6648 | 0.7209 | 0.7111 | 0.7902 | 0.7864 | 0.7832 | 0.8161 | 0.8189 | 0.8261 | 0.8321 | 0.8373 | 0.8371 | 0.8471 | 0.8423 | 0.8632 | 0.8811 | 0.8820 | 0.8586 | 0.8832 | 0.8818 |
| Entropy | 0.3682 | 0.4627 | 0.5504 | 0.6191 | 0.6786 | 0.6588 | 0.7145 | 0.7179 | 0.7316 | 0.7364 | 0.7902 | 0.7834 | 0.7961 | 0.7882 | 0.8227 | 0.8073 | 0.8470 | 0.8439 | 0.8504 | 0.8589 |
| Margin | 0.4729 | 0.5477 | 0.6750 | 0.7491 | 0.7780 | 0.8030 | 0.8129 | 0.8168 | 0.8434 | 0.8405 | 0.8573 | 0.8621 | 0.8582 | 0.8836 | 0.8825 | 0.8820 | 0.8895 | 0.8862 | 0.9027 | 0.8954 |
| BADGE | 0.5091 | 0.6118 | 0.6396 | 0.7109 | 0.7704 | 0.7846 | 0.8134 | 0.8152 | 0.8288 | 0.8429 | 0.8495 | 0.8609 | 0.8454 | 0.8725 | 0.8761 | 0.8632 | 0.8821 | 0.8779 | 0.8857 | 0.8962 |
| **SCALPEL** | 0.5625 | **0.7293** | 0.7598 | 0.7621 | 0.7898 | 0.8057 | 0.8146 | 0.8346 | 0.8386 | 0.8480 | 0.8532 | 0.8607 | 0.8529 | 0.8702 | 0.8800 | 0.8814 | 0.8812 | 0.8741 | 0.8854 | 0.8912 |

#### PALM (linear)

| Phương pháp | Amax | delta | alpha | beta | AUC | Budget to 90 | RMSE |
|---|---|---|---|---|---|---|---|
| Random | 0.8569 | 0.5321 | -0.3770 | 0.6489 | 0.7843 | 147.6 | 0.0145 |
| Coreset | 0.7784 | 0.2102 | 0.2797 | 0.8032 | 0.5736 | 419.4 | 0.0127 |
| UHerding | 0.9171 | 0.6986 | -0.6500 | 0.2335 | 0.7800 | 424.7 | 0.0104 |
| TCM | 0.9255 | 0.5805 | -0.4079 | 0.3564 | 0.7693 | 395.4 | 0.0073 |
| REFINE | 0.9299 | 0.5314 | 0.0281 | 0.4684 | 0.8020 | 267.4 | 0.0039 |
| TypiClust | 1.0000 | 0.6920 | -0.6609 | 0.2537 | 0.8062 | — | 0.0075 |
| ActiveFT | 0.8504 | 0.4210 | 0.4426 | 0.7945 | 0.7857 | 141.8 | 0.0078 |
| DropQuery | 1.0000 | 0.6249 | 0.5385 | 0.2556 | 0.8208 | — | 0.0122 |
| Entropy | 0.9797 | 0.4112 | -0.2603 | 0.4531 | 0.7280 | None | 0.0145 |
| Margin | 0.8952 | 0.4277 | 0.4233 | 0.7306 | 0.8146 | 163.3 | 0.0136 |
| BADGE | 0.8905 | 0.3495 | 1.3583 | 0.7912 | 0.8078 | 174.5 | 0.0104 |
| **SCALPEL** | 1.0000 | **0.7013** | **-0.8495** | 0.1983 | 0.8304 | — | 0.0079 |

### SkinTissue

#### Accuracy (linear) theo budget

| Phương pháp | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 | 325 | 350 | 375 | 400 | 425 | 450 | 475 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.3815 | 0.6089 | 0.6878 | 0.6980 | 0.7172 | 0.7475 | 0.7592 | 0.7512 | 0.7786 | 0.7834 | 0.7871 | 0.8012 | 0.8037 | 0.8204 | 0.8231 | 0.8218 | 0.8265 | 0.8295 | 0.8362 | 0.8420 |
| Coreset | 0.1795 | 0.2281 | 0.2963 | 0.3221 | 0.3405 | 0.3919 | 0.4049 | 0.4264 | 0.4400 | 0.4616 | 0.4617 | 0.4804 | 0.4664 | 0.4996 | 0.4876 | 0.4997 | 0.4900 | 0.4883 | 0.4998 | 0.5288 |
| UHerding | 0.5200 | 0.5871 | 0.7179 | 0.7348 | 0.7421 | 0.7417 | 0.7649 | 0.7637 | 0.7706 | | | | | | | | | | | |
| TCM | 0.5712 | 0.6462 | 0.6713 | 0.6948 | 0.7278 | 0.7428 | 0.7686 | 0.7810 | 0.7944 | 0.8035 | 0.8093 | 0.8185 | 0.7977 | 0.8203 | 0.8277 | 0.8295 | 0.8341 | 0.8264 | 0.8332 | 0.8406 |
| REFINE | 0.5577 | 0.6335 | 0.6923 | 0.7200 | 0.7488 | 0.7645 | 0.7819 | 0.7795 | 0.8003 | 0.8058 | 0.8157 | 0.8241 | 0.8193 | 0.8261 | 0.8309 | 0.8317 | 0.8306 | 0.8214 | 0.8353 | 0.8383 |
| TypiClust | 0.5596 | 0.6612 | 0.6758 | 0.7020 | 0.7404 | 0.7329 | 0.7520 | 0.7608 | 0.7802 | 0.7815 | 0.7955 | 0.7912 | 0.8009 | 0.8090 | 0.8092 | 0.8092 | 0.8119 | 0.8172 | 0.8202 | 0.8244 |
| ActiveFT | 0.5053 | 0.6374 | 0.6455 | 0.6990 | 0.7240 | 0.7275 | 0.7338 | 0.7481 | 0.7591 | 0.7659 | 0.7734 | 0.7853 | 0.7951 | 0.7954 | 0.7974 | 0.7879 | 0.7971 | 0.7941 | 0.8104 | 0.8112 |
| DropQuery | 0.6559 | 0.6895 | 0.7189 | 0.7381 | 0.7456 | 0.7512 | 0.7756 | 0.7785 | 0.7847 | 0.7807 | 0.7940 | 0.8067 | 0.8010 | 0.8225 | 0.8296 | 0.8372 | 0.8279 | 0.8223 | 0.8321 | 0.8466 |
| Entropy | 0.3447 | 0.5659 | 0.4964 | 0.5020 | 0.5974 | 0.6130 | 0.6397 | 0.7487 | 0.7564 | 0.7479 | 0.7465 | 0.7610 | 0.7676 | 0.7951 | 0.7993 | 0.8026 | 0.8158 | 0.8122 | 0.8188 | 0.8217 |
| Margin | 0.4677 | 0.5785 | 0.6883 | 0.7075 | 0.7507 | 0.7719 | 0.7804 | 0.7974 | 0.8123 | 0.8154 | 0.8219 | 0.8286 | 0.8117 | 0.8387 | 0.8453 | 0.8471 | 0.8531 | 0.8325 | 0.8569 | 0.8578 |
| BADGE | 0.4308 | 0.5647 | 0.6689 | 0.7182 | 0.7298 | 0.7458 | 0.7739 | 0.7856 | 0.7993 | 0.7998 | 0.8125 | 0.8180 | 0.8139 | 0.8352 | 0.8420 | 0.8420 | 0.8475 | 0.8492 | 0.8589 | 0.8515 |
| SCALPEL | 0.6278 | 0.6896 | 0.7164 | 0.7169 | 0.7321 | 0.7594 | 0.7800 | | | | | | | | | | | | | |

#### PALM (linear)

| Phương pháp | Amax | delta | alpha | beta | AUC | Budget to 90 | RMSE |
|---|---|---|---|---|---|---|---|
| Random | 1.0000 | 0.6108 | -0.9515 | 0.2230 | 0.7657 | None | 0.0068 |
| Coreset | 0.5172 | 0.1723 | 1.2384 | 0.9938 | 0.4240 | 278.1 | 0.0097 |
| UHerding | | | | | | | |
| TCM | 0.8479 | 0.3758 | 2.4630 | 0.7116 | 0.7757 | 170.8 | 0.0070 |
| REFINE | 0.8442 | 0.5373 | 0.7731 | 0.5875 | 0.7826 | 141.8 | 0.0045 |
| TypiClust | 1.0000 | 0.6229 | -0.5779 | 0.1965 | 0.7660 | None | 0.0065 |
| ActiveFT | 0.9381 | 0.6472 | -0.7395 | 0.2196 | 0.7502 | None | 0.0080 |
| DropQuery | 1.0000 | 0.5941 | 1.1110 | 0.2318 | 0.7838 | None | 0.0068 |
| Entropy | 0.8246 | 0.0198 | 7.1284 | 1.6698 | 0.7026 | 250.7 | 0.0347 |
| Margin | 0.8612 | 0.5626 | -0.1096 | 0.5577 | 0.7854 | 159.6 | 0.0094 |
| BADGE | 0.8839 | 0.5952 | -0.5062 | 0.4414 | 0.7773 | 220.3 | 0.0081 |
| SCALPEL | | | | | | | |

### PathMNIST

#### Accuracy (linear) theo budget

| Phương pháp | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 | 325 | 350 | 375 | 400 | 425 | 450 | 475 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.6812 | 0.8162 | 0.8373 | 0.8673 | 0.8766 | 0.8829 | 0.8877 | 0.9047 | 0.9070 | 0.9120 | 0.9110 | 0.9156 | 0.9095 | 0.9131 | 0.9159 | 0.9181 | 0.9184 | 0.9116 | 0.9181 | 0.9206 |
| Coreset | 0.5731 | 0.5153 | 0.5189 | 0.5812 | 0.6333 | 0.6543 | 0.6716 | 0.6696 | 0.6753 | 0.6613 | 0.6571 | 0.6524 | 0.6535 | 0.6818 | 0.6866 | 0.6900 | 0.6890 | 0.6964 | 0.6907 | 0.6883 |
| UHerding | 0.4922 | 0.7614 | 0.8341 | 0.8383 | 0.6312 | 0.7990 | 0.7982 | 0.8487 | 0.8525 | 0.8575 | 0.8787 | 0.8805 | | | | | | | | |
| TCM | 0.6280 | 0.7518 | 0.8159 | 0.8430 | 0.8457 | 0.8758 | 0.8799 | 0.8834 | 0.8903 | 0.8958 | 0.8894 | 0.8976 | 0.9077 | 0.8976 | 0.9015 | 0.9013 | 0.9093 | 0.9039 | 0.9058 | 0.9149 |
| REFINE | 0.7088 | 0.8046 | 0.8351 | 0.8536 | 0.8499 | 0.8742 | 0.8937 | 0.8894 | 0.8974 | 0.9070 | 0.9187 | 0.9019 | 0.9000 | 0.9095 | 0.9166 | 0.9219 | 0.9205 | 0.9010 | 0.9192 | 0.9240 |
| TypiClust | 0.7302 | 0.8206 | 0.8093 | 0.8797 | 0.8727 | 0.8745 | 0.8968 | 0.8903 | 0.8916 | 0.9102 | 0.9093 | 0.8930 | 0.9011 | 0.9035 | 0.9224 | 0.9024 | 0.8937 | 0.9052 | 0.8915 | 0.9033 |
| ActiveFT | 0.6515 | 0.8280 | 0.8290 | 0.8558 | 0.8642 | 0.8825 | 0.8864 | 0.8968 | 0.8808 | 0.9093 | 0.9064 | 0.8907 | 0.8943 | 0.9097 | 0.9043 | 0.9035 | 0.9054 | 0.8919 | 0.9121 | 0.9063 |
| DropQuery | 0.7827 | 0.8258 | 0.8338 | 0.8500 | 0.8825 | 0.8898 | 0.8791 | 0.9025 | 0.9058 | 0.8859 | 0.9109 | 0.9159 | 0.9141 | 0.9196 | 0.8923 | 0.9104 | 0.9045 | 0.9046 | 0.8955 | 0.9116 |
| Entropy | 0.6407 | 0.7085 | 0.7350 | 0.8049 | 0.8276 | 0.8458 | 0.8372 | 0.8730 | 0.8648 | 0.8421 | 0.8880 | 0.8791 | 0.8890 | 0.8759 | 0.8915 | 0.9102 | 0.9088 | 0.8982 | 0.9033 | 0.8990 |
| Margin | 0.6302 | 0.6377 | 0.8462 | 0.8706 | 0.8717 | 0.9199 | 0.8950 | 0.9058 | 0.9208 | 0.9022 | 0.9116 | 0.9139 | 0.9040 | 0.8968 | 0.9199 | 0.9036 | 0.9235 | 0.9167 | 0.9330 | 0.9149 |
| BADGE | 0.6054 | 0.7556 | 0.8123 | 0.8405 | 0.8404 | 0.9099 | 0.8905 | 0.8813 | 0.9019 | 0.9175 | 0.9156 | 0.9191 | 0.9078 | 0.9104 | 0.9032 | 0.9174 | 0.9263 | 0.9124 | 0.9214 | 0.9153 |
| SCALPEL | 0.7790 | 0.8326 | 0.8639 | 0.8799 | 0.8841 | 0.8840 | 0.8825 | 0.8861 | 0.8923 | 0.8939 | 0.9003 | 0.9004 | 0.9035 | | | | | | | |

#### PALM (linear)

| Phương pháp | Amax | delta | alpha | beta | AUC | Budget to 90 | RMSE |
|---|---|---|---|---|---|---|---|
| Random | 0.9335 | 0.8494 | -0.7355 | 0.2771 | 0.8919 | 69.0 | 0.0044 |
| Coreset | 0.6884 | 0.0091 | 13.6785 | 1.8789 | 0.6493 | 132.6 | 0.0237 |
| UHerding | | | | | | | |
| TCM | 0.9119 | 0.7569 | -0.3516 | 0.4476 | 0.8731 | 83.1 | 0.0049 |
| REFINE | 0.9474 | 0.8171 | -0.5611 | 0.2514 | 0.8866 | 97.8 | 0.0077 |
| TypiClust | 0.9046 | 0.6823 | 0.8253 | 0.6194 | 0.8837 | 56.4 | 0.0116 |
| ActiveFT | 0.9348 | 0.8714 | -0.9424 | 0.1895 | 0.8827 | 69.6 | 0.0086 |
| DropQuery | 0.9088 | 0.0087 | 13.7856 | 2.0158 | 0.8881 | 51.9 | 0.0089 |
| Entropy | 0.9094 | 0.5289 | 1.1834 | 0.6050 | 0.8505 | 129.1 | 0.0123 |
| Margin | 0.9314 | 0.0055 | 6.4602 | 2.6047 | 0.8837 | 91.7 | 0.0234 |
| BADGE | 0.9224 | 0.7297 | -0.3341 | 0.4926 | 0.8822 | 87.1 | 0.0110 |
| SCALPEL | | | | | | | |
