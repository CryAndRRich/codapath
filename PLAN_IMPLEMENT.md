# PLAN IMPLEMENT — notebook và code để chạy các variant

> Kế hoạch thực thi. Không phải mô tả code đã có (`README.md`), không phải quy
> tắc chung (`CLAUDE.md`), không phải không gian trục thử nghiệm
> (`PLAN_VARIANTS.md`). Xoá file này khi làm xong.
>
> Trạng thái nền: 145 test pass, pyflakes sạch, `main.run` đã verify end-to-end.

---

## 0. Hợp đồng cache — thứ TUYỆT ĐỐI không được đổi

Hai notebook `extract_visual_features.ipynb` và `extract_nucleus_features.ipynb`
**đóng băng**. Mọi notebook khác đọc output của chúng qua Kaggle Dataset. Vì vậy
cấu trúc dưới đây là hợp đồng, và code phía đọc phải khớp chính xác.

### 0.1 Visual features

Notebook ghi vào `FEATURE_DIR` (mặc định `/kaggle/working/features`), zip thành
`/kaggle/working/archive/visual_features.zip`. Bên trong **phẳng**, không có
thư mục con:

```
{dataset}_seed{seed}_{backbone_safe}_train.npy       (N_train, D) float32
{dataset}_seed{seed}_{backbone_safe}_test.npy        (N_test,  D) float32
{dataset}_seed{seed}_{backbone_safe}_manifest.json
```

`backbone_safe` = tên HF thay `/` bằng `_` (vd `facebook_dinov2-base`).
Manifest chứa `dataset`, `seed`, `backbone`, `train_fingerprint`,
`test_fingerprint`, `train_shape`, `test_shape`. **Cache không có manifest bị
từ chối** — không xác minh được thứ tự hàng.

### 0.2 CellViT features

Notebook ghi vào `CACHE_DIR/{dataset}_seed{seed}/`, zip thành
`cellvit_features.zip`. Bên trong **có một cấp thư mục**:

```
{dataset}_seed{seed}/
    offsets.npy              (num_patches+1,) int64   prefix sum kiểu ragged
    confidence.npy           (num_cells,)     float32
    sample_ids.npy           (num_patches,)   str
    cellvit_embeddings.npy   (num_cells, d)   float16
    cell_dino_features.npy   (num_cells, d)   float16
    bboxes.npy               (num_cells, 4)   int32
    manifest.json
    qc/*.png
```

`manifest.json` chứa `dataset`, `seed`, `sample_fingerprint`,
`checkpoint_sha256`, `input_mpp`, `model_mpp`, `magnification`,
`dino_backbone`, `max_cells_per_patch`, `has_*` flags.

### 0.3 Kaggle làm sâu thêm một cấp

Publish `/kaggle/working/<name>` thành Dataset thì Kaggle mount lại thành
`/kaggle/input/<slug>/<name>/<name>` — sâu hơn một cấp so với đường dẫn ai cũng
viết ra. Đây là lý do có `dir_containing()` trong notebook: nó **tìm** thư mục
chứa file mốc thay vì tin đường dẫn cứng, và in ra cái tìm được.

**Việc cần làm:** `dir_containing` hiện chỉ nằm trong `run_al_sampler.ipynb`.
Chuyển vào `utils/kaggle.py` để cả 4 notebook chạy dùng chung một bản. Notebook
chỉ cần khai `DATASET` + `SEED`, không điền đường dẫn tay.

Hàm cần có trong `utils/kaggle.py`:

| Hàm | Trả về |
|---|---|
| `find_dir_containing(probe, hint, roots)` | thư mục `D` sao cho `D/probe` tồn tại |
| `find_visual_cache(dataset, seed, backbone)` | thư mục chứa `_train.npy` **và** `_manifest.json` |
| `find_cellvit_cache(dataset, seed)` | thư mục chứa `{dataset}_seed{seed}/manifest.json` |
| `find_data_root()` | gốc dataset ảnh (đang hard-code 2 candidate) |
| `write_archive(source, name, slug)` | zip + `dataset-metadata.json`, in lệnh publish |

---

## 1. Xoá CODAPath

`codapath` là method đề xuất **cũ**, không còn cần.

### 1.1 Xoá hẳn

| Đường dẫn | Ghi chú |
|---|---|
| `sampling/baselines/codapath.py` | cả sampler lẫn `DualVLMExtractor` |
| `assets/` (8.9 MB) | paper cũ + slide + hình ablation |
| `config/config.yaml` → khối `samplers.codapath` | |
| `sampling/specs.py` → entry `"codapath"` | |
| `sampling/baselines/__init__.py` → import + `__all__` | |
| `evaluation/plots.py:113` → nhánh `if m.lower() in {"codapath", ...}` | giữ lại `scalpel` |
| `README.md` → 4 chỗ (dòng 23, 120, 135, 158) | |

### 1.2 KHÔNG chuyển sang chỗ khác — viết mới

`main.py::_load_vlm_inputs` đang import `extract_text_features` **từ**
`codapath.py`. Hàm đó là **dual-VLM**: nối embedding của PLIP text tower với
BiomedBERT, hai model độc lập, hai không gian khác nhau ghép cạnh nhau.

CONCH là CLIP-style **một** model: image tower và text tower **đã cùng một không
gian**, cosine giữa chúng có nghĩa. Logic dual-VLM không dùng được, nên đây là
viết mới chứ không phải di chuyển.

`features/vlm.py` (mới):

| Hàm | Việc |
|---|---|
| `load_vlm(name, device)` | `open_clip.create_model_and_transforms`, trả về `(model, preprocess)` |
| `encode_images(loader, model, device)` | (N, D) float32, đã L2-normalize |
| `encode_class_texts(descriptions, templates, class_names, model, device)` | (C, D) float32, trung bình trên template rồi normalize |
| `text_margin_uncertainty(image_emb, text_emb, tau)` | `1 − (p₁ − p₂)` trên `softmax(cos/τ)` |

`text_margin_uncertainty` là **margin thuần** — không phải `(1−margin)(1+JSD)`
của CODAPath (xem `PLAN_VARIANTS.md` §2.3 để biết vì sao).

### 1.3 Sau khi xoá

- `sampling/specs.py` còn **12** sampler.
- `spec.needs` không còn ai khai `"text_embeddings"` → nhánh đó trong `main.py`
  đổi thành: đọc từ **config của run**, không phải từ `spec.needs`. Lý do:
  cold-start text là một *trục cấu hình* của `scalpel`, không phải yêu cầu tĩnh
  của một sampler.
- `tests/test_sampler_specs.py:52` **có** dùng `spec_for("codapath")` để chứng
  minh `needs` và `prefix_exact` độc lập nhau. Phải viết lại bằng `scalpel`
  (cũng có `needs` nhưng `prefix_exact=False`) — chọn cặp khác chứ không xoá
  test, vì nó khoá đúng chỗ dự án từng nhầm hai lần.
- Số liệu cũ của codapath trong README appendix: **giữ**, đã có banner "không
  tái lập được". Nhưng thêm một dòng: sampler này đã bị xoá khỏi code.

---

## 2. Bảy notebook

| Notebook | Trạng thái | Việc |
|---|---|---|
| `extract_visual_features.ipynb` | **đóng băng** | không đụng |
| `extract_nucleus_features.ipynb` | **đóng băng** | không đụng |
| `generate_class_description.ipynb` | **mới** | §3 |
| `extract_vlm_features.ipynb` | **mới** | §4 |
| `run_al_baseline.ipynb` | **mới** (tách từ `run_al_sampler`) | §5 |
| `run_al_main.ipynb` | **mới** (tách từ `run_al_sampler`) | §6 |
| `run_al_sampler.ipynb` | **xoá** | bị hai cái trên thay thế |
| `evaluate_al_sampler.ipynb` | giữ | bỏ `codapath` khỏi `RUN_NAMES` ví dụ |

Mọi notebook mới đều theo đúng khung đã có: clone + verify branch → `%cd` →
pip install → cell EDIT → import → resolve cache → chạy → archive.

---

## 3. `generate_class_description.ipynb`

Sinh **một lần**, commit file vào repo, mọi lần chạy sau đọc file. Chạy được cả
local (không cần GPU) — nhưng để dạng notebook cho nhất quán.

Biến ở cell EDIT:

```python
DATASET = "pathmnist"          # pathmnist | histoset | skintissue
MODEL = "gemini-2.5-flash"     # PIN CỨNG. Không dùng nhánh 3.x (xem dưới)
STYLE = "llm_short"            # llm_short | llm_morphology | llm_multi
TEMPERATURE = 0.0
SEED = 42
NUM_PER_CLASS = 1              # >1 chỉ với STYLE="llm_multi"
OVERWRITE = False
API_KEY = ""                   # Kaggle Secrets hoặc dán tay
```

Ghi ra `config/descriptions/{dataset}_{style}.json`:

```json
{
  "dataset": "pathmnist",
  "model": "gemini-2.5-flash",
  "style": "llm_short",
  "temperature": 0.0,
  "seed": 42,
  "prompt_template": "<nguyên văn>",
  "generated_at": "2026-08-28",
  "descriptions": {"adipose": "...", "...": "..."},
  "sha256": "<hash của khối descriptions>"
}
```

**Bắt buộc trong notebook:**

- `assert not MODEL.startswith("gemini-3")` — `temperature`/`topK`/`topP` bị
  **deprecated và BỎ QUA** ở `gemini-3.7-flash`, `3.6-flash`, `3.5-flash-lite`.
  Dùng nhầm là `temperature=0` thành no-op và ta viết một câu sai vào paper.
- Refuse ghi đè khi file đã tồn tại, trừ `OVERWRITE=True`.
- In cảnh báo: hosted inference **không** bit-for-bit tái lập; file đã đóng băng
  mới là artifact tái lập, không phải lời gọi API.
- Không cần archive/zip: file nhỏ, commit thẳng vào repo.

Code mới: `features/descriptions.py` với `generate_descriptions(...)` và
`load_descriptions(dataset, style)`. `load_descriptions("...", "manual")` đọc
khối `datasets.<name>.descriptions` đang có trong `config.yaml` → `manual` là
control, không cần file.

---

## 4. `extract_vlm_features.ipynb`

Vì hai notebook extract đóng băng và chỉ biết DINOv2, CONCH cần notebook riêng.

Cell EDIT:

```python
DATASET = "pathmnist"
SEEDS = [42]
# CONCH | Astaxanthin/KEEP | wisdomik/QuiltNet-B-16-PMB | vinid/plip
VLM = "MahmoodLab/CONCH"
DESCRIPTION_STYLE = "llm_short"   # cái nào ở §3; "manual" = control
HF_TOKEN = ""                  # CONCH gated -> cần token
FEATURE_DIR = "/kaggle/working/vlm_features"
```

Đầu notebook, trước mọi download:

```python
import os, subprocess, sys
from huggingface_hub import login

# open_clip_torch is NOT enough: CONCH has its own tokenizer + factory.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "git+https://github.com/mahmoodlab/CONCH.git"])
login(HF_TOKEN)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

```python
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
model, preprocess = create_model_from_pretrained(
    "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=HF_TOKEN,
)
```

Ghi ra **đúng cùng quy ước đặt tên** với visual cache (§0.1) để `main.py` đọc
lại bằng chính `get_or_extract_features`, không cần đường code thứ hai:

```
# Không gian KHÔNG projection — cho coverage kernel, probe_v, linear head cuối
{dataset}_seed{seed}_{vlm_safe}_train.npy
{dataset}_seed{seed}_{vlm_safe}_test.npy
{dataset}_seed{seed}_{vlm_safe}_manifest.json

# Không gian ĐÃ projection + normalize — CHỈ để so với text (vòng 1)
{dataset}_seed{seed}_{vlm_safe}_proj_train.npy
{dataset}_seed{seed}_{vlm_safe}_proj_manifest.json

# Text prototype, cùng không gian với *_proj_*
{dataset}_{style}_text.npy            (C, 512) float32
{dataset}_{style}_text_manifest.json
```

Text manifest phải ghi `vlm`, `style`, `class_names` **theo đúng thứ tự**, và
`description_sha256` — nếu description đổi mà text embedding không đổi thì
mọi kết quả cold-start thành vô nghĩa mà không có gì báo.

**Việc cần làm trong code:** `features/visual.py::get_or_extract_features` hiện
hard-code `DINOv2Extractor`. Thêm tham số chọn extractor (`dinov2` | `vlm`) để
tái dùng toàn bộ logic cache/manifest/atomic-write đã có. Không copy hàm.

**Hai không gian, hai file** (xem §10.1): `proj_contrast=False` cho linear
probe/coverage, `proj_contrast=True, normalize=True` để so với text. Ghi cả hai
trong **một** forward pass — chạy lại vì thiếu một cái là mất hàng giờ GPU.

**CONCH là 448×448** (`conch_ViT-B-16.json`), khác 224 của DINOv2, dùng **OpenAI
CLIP** normalization chứ không phải ImageNet, và có **tokenizer riêng**. Notebook
này **phải** dùng `model, preprocess = create_model_from_pretrained(...)` của
package `conch` và dùng đúng `preprocess` đó — xem §10.2/10.3/10.4 để biết ba
cái bẫy. 4× pixel nên `batch_size` phải nhỏ hơn notebook DINOv2 nhiều.

Archive: `vlm_features.zip`, slug `{dataset}-seed{seed}-vlm-features`.

---

## 5. `run_al_baseline.ipynb`

Chỉ chạy baseline. **Không** CellViT, **không** VLM, **không** text.

Cell EDIT — đúng ba biến như hai notebook extract, cộng chọn sampler:

```python
DATASET = "pathmnist"
SEEDS = [42]

# 11 baseline (sau khi xoá codapath). ONE per run:
#   random | coreset | typiclust | activeft | tcm
#   margin | entropy | badge | dropquery | uncertainty_herding | refine
SAMPLER = "uncertainty_herding"

VARIANTS = [{}]          # [{}] = dùng nguyên config.yaml
PARALLEL = True
```

Mọi thứ khác lấy từ `config/config.yaml`.

Assert bắt buộc:

```python
assert SAMPLER in BASELINE_SAMPLERS, (
    f"{SAMPLER} không phải baseline. scalpel chạy ở run_al_main.ipynb"
)
assert "cell_embeddings" not in spec.needs      # baseline không cần CellViT
assert "text_embeddings" not in spec.needs      # baseline không cần VLM
```

Thêm `BASELINE_SAMPLERS` vào `sampling/specs.py` (một frozenset, dẫn xuất từ
`SAMPLER_SPECS` bằng cách loại `scalpel`) để notebook không tự liệt kê tay rồi
lệch khỏi code.

Chỉ cần visual cache DINOv2. Archive slug: `{dataset}-baseline-{sampler}`.

---

## 6. `run_al_main.ipynb`

Pipeline chính. Cell EDIT:

```python
DATASET = "pathmnist"
SEEDS = [42]

# --- biến thể ---
IMAGE_ENCODER = "dinov2"        # dinov2 | conch
USE_TEXT = False                # True = cold-start bằng LLM text (vòng 1)
# manual | conch_official | llm_short | llm_morphology   (xem §10.5)
DESCRIPTION_STYLE = "llm_short" # chỉ dùng khi USE_TEXT
CELL_POOLING = "mean"           # mean | weighted_mean
USE_LORA = False                # chỉ ở lần train cuối
AUX_LOSS = "none"               # none | center | supcon | triplet
AUX_WEIGHT = 0.5                # ĐÃ CHỐT: 0.5 cho mọi loại loss phụ
AUGMENT = "none"                # none | flip_rotate
PARALLEL = True

# Đã chốt, không thành trục:
#   - vòng 1 lấy B/5 ảnh, chia đều như hiện tại
#   - aux_weight = 0.5 bất kể center/supcon/triplet
```

### 6.1 Assert — quan hệ một chiều

```python
if USE_TEXT:
    assert IMAGE_ENCODER == "conch", (
        "USE_TEXT cần text tower cùng không gian với image tower; "
        "DINOv2 không có text tower. Đặt IMAGE_ENCODER='conch'."
    )
# Chiều ngược lại KHÔNG assert: CONCH mà USE_TEXT=False là hợp lệ —
# đó chính là ablation 'CONCH visual, coverage thuần vòng 1'.
```

Assert phụ:

```python
if AUX_LOSS in ("supcon", "triplet"):
    # cần >= 2 mẫu/lớp trong batch; budget nhỏ không đủ
    assert min(config["cumulative_budget"]) >= 2 * num_classes, (
        f"{AUX_LOSS} cần >=2 mẫu/lớp; budget {min(...)} với {num_classes} lớp "
        "thì loss sẽ âm thầm thành ~0. Dùng AUX_LOSS='center'."
    )
if AUGMENT != "none" or USE_LORA:
    assert DATA_ROOT is not None, (
        "augment/LoRA cần PIXEL, không chạy được từ embedding cache. "
        "Attach dataset ảnh gốc."
    )
```

### 6.2 Encoder xuyên suốt

`IMAGE_ENCODER` quyết định **cả ba** chỗ (xem `PLAN_VARIANTS.md` §1.1):
coverage kernel, `probe_v` tính disagreement, và linear head lúc đánh giá.
Không có chuyện vòng 1 CONCH rồi vòng 2 về DINOv2.

Trong code: `main.run` nhận `visual_backbone` từ config `models.vit`. Đổi thành
nhận tham số tường minh, và `_default_run_name` **phải** encode nó, nếu không
hai protocol ghi đè lên nhau.

### 6.3 cell_source cố định

Protocol B luôn dùng `cellvit_embedding` (token CellViT gốc), **không** dùng
`crop_dino`. Lý do: `cell_dino_features` được tạo bằng DINOv2 và manifest ghi
`dino_backbone`; dùng nó với CONCH là trộn hai không gian. Thêm assert:

```python
assert not (IMAGE_ENCODER == "conch" and CELL_SOURCE == "crop_dino"), (
    "crop_dino được extract bằng DINOv2; dùng với CONCH là trộn hai không gian"
)
```

### 6.4 Train cuối đọc lại ảnh gốc

Augment và LoRA cần **pixel**; cache chỉ có embedding đã tính qua transform cố
định 224×224. Nên:

- **Chọn mẫu** dùng embedding cache (nhanh, không đổi).
- **Train cuối** load ảnh gốc cho ~B ảnh đã chọn. `RawRGBDataset` đã có sẵn và
  trả đúng thứ tự split + `sample_id` khớp, nên không cần viết dataset mới —
  chỉ cần `Subset` theo `selected_indices`.
- B ≤ 200 ảnh nên chi phí không đáng kể.

Khi `USE_LORA=False` **và** `AUGMENT="none"`: train linear thẳng trên embedding
cache, không load ảnh — giữ đúng đường chạy nhanh hiện tại. Đây là control
`lora_r=0, aux_loss=none`.

### 6.5 Code mới cho §6

| File | Việc |
|---|---|
| `training/lora.py` | LoRA cho ViT. **`peft` KHÔNG có ở local** (đã kiểm tra: peft/timm/open_clip đều thiếu, torch 2.2.2). Tự implement ~60-80 dòng wrap `nn.Linear` của attention. Test: `lora_r=0` ≡ frozen, bit-for-bit |
| `training/losses.py` | `center_loss`, `supcon_loss`, `triplet_loss`. Chữ ký chung `(features, logits, labels) -> scalar`. `supcon`/`triplet` phải **raise có thông báo** khi batch không đủ 2 mẫu/lớp, không được lặng lẽ trả 0 |
| `training/finetune.py` | Vòng train cuối: nhận `train_mode`, `aux_loss`, `aux_weight`, `augment`; đọc pixel khi cần |
| `data/augment.py` | `flip_rotate` (an toàn sinh học với H&E: lật/quay 90° không đổi nhãn). **Không** color jitter — dự án từng có method chết vì stain shortcut |

Archive slug: `{dataset}-main-{encoder}{-text}{-lora}{-aux}`.

---

## 7. Đổi trong code lõi

| File | Đổi gì | Vì sao |
|---|---|---|
| `main.py` | bỏ `_load_vlm_inputs` cũ, dùng `features/vlm.py` | codapath xoá |
| | `visual_backbone` thành tham số tường minh của `run()` | encoder là trục thật |
| | `_default_run_name` encode encoder + text + lora + aux + augment | không thì ghi đè nhau |
| | nhánh text đọc từ config run, không từ `spec.needs` | text là trục của scalpel, không phải yêu cầu tĩnh |
| | thêm bước train cuối (§6.4) sau khi hết budget sweep | LoRA/augment |
| `features/visual.py` | `get_or_extract_features` nhận loại extractor | tái dùng cache logic cho CONCH |
| `data/loaders.py` | `get_data_loaders` nhận `transform` (hoặc `image_size`+`normalize`) thay vì hard-code | **CONCH là 448×448 và có normalize riêng**; hiện hàm này khoá cứng `Resize((224,224))` + ImageNet mean/std ở dòng 193-197. Dùng `preprocess` mà `open_clip` trả về, đừng tự viết lại |
| `features/vlm.py` | **mới** | §1.2 |
| `features/descriptions.py` | **mới** | §3 |
| `sampling/specs.py` | bỏ `codapath`; thêm `BASELINE_SAMPLERS` | §5 |
| `sampling/scalpel/sampler.py` | vòng 1 nhận `round1_weight` + text uncertainty | cold start |
| `utils/kaggle.py` | **mới**: `dir_containing` & bạn bè | §0.3 |
| `evaluation/plots.py` | bỏ nhánh codapath | §1.1 |
| `config/config.yaml` | bỏ `samplers.codapath`; thêm `vlm`, `descriptions`, `final_training` | |

---

## 8. Test phải thêm

Theo đúng nguyên tắc của dự án: assert vào **cơ chế**, không phải shape.

| Test | Bắt cái gì |
|---|---|
| `test_no_codapath_left` | grep toàn repo: không còn `codapath` ở `*.py`/`*.yaml`/`*.ipynb` |
| `test_lora_zero_equals_frozen` | `lora_r=0` cho weight **giống bit-for-bit** linear probe |
| `test_aux_loss_zero_weight_is_identity` | `aux_weight=0` ≡ `aux_loss="none"` |
| `test_supcon_raises_on_thin_batch` | không đủ 2 mẫu/lớp thì **raise**, không trả 0 |
| `test_text_requires_conch` | `USE_TEXT` + `dinov2` → assert nổ |
| `test_run_name_encodes_every_axis` | hai biến thể khác nhau **không** ra cùng tên |
| `test_notebook_set_is_exactly_seven` | cập nhật từ 4 → **7** notebook |
| `test_conch_zeroshot_matches_paper` | zero-shot CONCH trên test PathMNIST ≈ **79.1%** (paper). Lệch nhiều = sai transform/normalize/tokenizer/projection |
| `test_conch_uses_openai_normalization` | không phải ImageNet mean/std — bẫy ở §10.3 |
| `test_vlm_cache_has_both_spaces` | có cả `_train.npy` và `_proj_train.npy` |
| `test_baseline_notebook_rejects_scalpel` | `run_al_baseline` có assert chặn |
| `test_vlm_cache_naming_matches_visual` | cùng quy ước để `main.py` đọc một đường |
| `test_description_file_roundtrip` | sha256 khớp; đổi description mà không đổi hash thì fail |
| `test_augment_preserves_label` | flip/rotate không đổi nhãn |

---

## 9. Thứ tự làm

1. **Xoá codapath + assets** (§1.1) — độc lập, làm sạch trước khi thêm.
2. **`utils/kaggle.py`** (§0.3) — mọi notebook sau dùng chung.
3. **Tách `run_al_baseline`** (§5) — chạy được ngay, không cần gì mới. Đây là
   thứ cho bạn số liệu Protocol A sớm nhất.
4. **`features/vlm.py` + `extract_vlm_features`** (§1.2, §4).
5. **`generate_class_description`** (§3) — cần trước khi text chạy được.
6. **`run_al_main` phần chọn mẫu** (§6.1–6.3) — encoder + text, chưa train cuối.
7. **Train cuối** (§6.4, §6.5) — LoRA, loss phụ, augment. Phần nhiều code nhất.

Mục 1–3 nên làm một lượt: đều là dọn dẹp và tách, không thêm khái niệm mới.

---

## 10. CONCH — đã đọc code và paper, mọi thứ dưới đây đã xác nhận

Nguồn: `repos/CONCH` (clone `mahmoodlab/CONCH`) và
`pdfs/CONCH_2307.12914.pdf` (Lu et al., *Towards a Visual-Language Foundation
Model for Computational Pathology*).

### 10.1 Kiến trúc

CONCH là **CoCa**, không phải CLIP thuần: image encoder + text encoder +
multimodal decoder, train bằng contrastive (i2t + t2i) **cộng** captioning loss.
Bản public **đã xoá weight của decoder** để không rò dữ liệu/PHI — image tower
và text tower nguyên vẹn, nên phân loại và retrieval không bị ảnh hưởng. Ta chỉ
cần hai tower đó.

Config `conch_ViT-B-16.json`: `embed_dim 512`, `embed_dim_caption 768`,
`vision_cfg.image_size 448`, `patch_size 16`, `context_length 128`,
`vocab_size 32007`, `attentional_pool_contrast true`.

**Hai không gian embedding khác nhau — chọn sai là sai hết:**

| Gọi thế nào | Ra cái gì | Dùng khi nào |
|---|---|---|
| `encode_image(x, proj_contrast=False, normalize=False)` | trước projection head, **512-d** sau `attn_pool_contrast` + `ln_contrast` | **linear probe** (README nói rõ: "suitable for linear probe") |
| `encode_image(x, proj_contrast=True, normalize=True)` | sau `@ proj_contrast`, đã L2-normalize | **so với text** (image-text retrieval, zero-shot) |

Vì thế trong pipeline của ta, **cùng một ảnh cần hai vector khác nhau**:

- `U_n` vòng 1 (so với text) → `proj_contrast=True, normalize=True`
- coverage kernel + `probe_v` + linear head cuối → `proj_contrast=False`

Hai cái này KHÔNG thay thế được nhau. `extract_vlm_features` phải ghi **cả hai**
(hai file `.npy` riêng), nếu không sẽ phải chạy lại forward pass.

### 10.2 Resolution — 448, nhưng có một điểm tinh tế

- Pretrain: **448×448**, "larger images are first resized along the shorter edge
  and center-cropped and smaller images are **zero-padded** as needed".
- Zero-shot / retrieval trong paper: "we enforce the **maximum** image size to be
  448 × 448 ... similar to its pretraining configuration".
- Nhưng ở mục so sánh encoder (Methods, dòng 668): "For all experiments, we
  standardize the image input size to **224 × 224**" — đây là khi so CONCH với
  PLIP/BiomedCLIP/ResNet50 như **feature extractor**.
- Và CRC100K (chính là PathMNIST của ta) vốn là **224×224 @ 0.5 mpp**.

Nghĩa là: ảnh gốc của ta 224, `image_transform(448)` sẽ `Resize(448, BICUBIC)`
rồi `CenterCrop(448)` → **upscale 2×**. Đó là hành vi `preprocess` mặc định mà
`create_model_from_pretrained` trả về, và là cấu hình paper dùng cho zero-shot
(79.1% trên CRC100K). **Dùng đúng `preprocess` đó**, đừng tự viết transform.

`force_image_size=224` có tồn tại và `resize_pos_embed` sẽ nội suy pos-embed cho
khớp — hợp lệ về mặt kỹ thuật và khớp con số "224 standardize" ở Methods. Nhưng
đây là **một trục thử nghiệm**, không phải mặc định: 448 là cấu hình cho ra
79.1%. Nếu chạy 224 thì phải ghi rõ.

Chi phí: 448² = **4× pixel** so với 224² → `batch_size` phải giảm, forward pass
chậm hơn nhiều. PathMNIST 90k ảnh, đây là lý do `extract_vlm_features` phải là
notebook riêng chạy một lần.

### 10.3 Normalize — có một bẫy thật

`transform.py::image_transform` mặc định dùng **ImageNet** mean/std. Nhưng
`factory.py::create_model` ghi đè:

```python
model.visual.image_mean = OPENAI_DATASET_MEAN   # (0.48145466, 0.4578275, 0.40821073)
model.visual.image_std  = OPENAI_DATASET_STD    # (0.26862954, 0.26130258, 0.27577711)
```

và `create_model_from_pretrained` lấy `getattr(model.visual, 'image_mean')` để
xây `preprocess`. Nên thực tế CONCH dùng **OpenAI CLIP** normalization, KHÁC
ImageNet mean/std mà `data/loaders.py` đang hard-code cho DINOv2.

Tự viết `Normalize(ImageNet)` cho CONCH là sai âm thầm — không crash, chỉ ra
embedding lệch. **Luôn dùng `preprocess` do factory trả về.**

### 10.4 Tokenizer riêng

CONCH **không** dùng tokenizer của `open_clip`. Nó có
`conch_byte_level_bpe_uncased.json` riêng, `vocab_size 32007`, và hàm
`tokenize(tokenizer, texts)` cắt ở `max_length=127` rồi pad thêm 1 slot cho
`<cls>` (tổng 128 = `context_length`).

Hệ quả: **`pip install open_clip_torch` là KHÔNG đủ.** Phải cài package `conch`
từ GitHub (`pip install git+https://github.com/mahmoodlab/CONCH.git`) hoặc vendor
thư mục `conch/open_clip_custom/`. Thêm vào `requirements` của notebook VLM
(notebook khác không cần).

### 10.5 Prompt chính thức cho ĐÚNG task của ta

`prompts/crc100k_prompts_all_per_class.json` là NCT-CRC-HE-100K — **chính xác 9
lớp của PathMNIST**: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM. Gồm:

- **22 template** (`"CLASSNAME."`, `"a photomicrograph showing CLASSNAME."`,
  `"an H&E stained image of CLASSNAME."`, ...)
- **4–5 classname mỗi lớp** (vd TUM: "colorectal adenocarcinoma epithelium",
  "colorectal adenocarcinoma", "tumor", "adenocarcinoma", "malignant epithelium")

Cách ensemble chính thức (`zeroshot_path.py::zero_shot_classifier`): với mỗi
lớp, encode **mọi** (classname × template), L2-normalize **từng cái**, rồi
`mean` trên cả hai chiều, rồi normalize lại. Không phải mean rồi normalize một
lần.

**Đây thay đổi thiết kế §3:** ta có một baseline prompt **chính thức, mạnh, do
tác giả CONCH tinh chỉnh cho đúng dataset này**. Vậy `description_style` nên có:

| Option | Nguồn |
|---|---|
| `manual` | bản viết tay trong `config.yaml` (control cũ) |
| **`conch_official`** | 22 template × 4-5 classname từ file trên — **baseline mạnh nhất, phải có** |
| `llm_short` | LLM sinh, 1 câu |
| `llm_morphology` | LLM sinh, tả hình thái |

Nếu LLM description không thắng được `conch_official` thì đó là kết quả thật và
phải báo cáo — contribution "LLM làm giàu nhãn" chỉ có giá trị nếu nó hơn prompt
người viết. Bỏ qua baseline này là tự làm yếu đối thủ.

Lưu ý: chỉ PathMNIST có prompt chính thức. HistoSet/SkinTissue không có, nên
`conch_official` chỉ áp dụng được cho PathMNIST.

### 10.6 τ — không phải 0.05

Bạn chốt `0.05`. Nhưng CONCH có `logit_scale` **học được**, khởi tạo
`log(1/0.07)`, và code chính thức dùng chính nó:

```python
probs = F.softmax(logits * model.logit_scale.exp(), dim=1)   # zeroshot_path.py:175
```

Giá trị trong checkpoint là kết quả train, không phải 0.07 nữa. Đề xuất:
`tau_mode = "learned"` (mặc định, dùng `model.logit_scale.exp()` — đúng như tác
giả) và `tau_mode = "fixed"` với `tau=0.05` làm trục so sánh. Dùng 0.05 cứng
trong khi checkpoint mang scale khác là bỏ đi thông tin đã được train.

**Cần bạn xác nhận:** dùng `learned` làm mặc định (tôi khuyến nghị) hay giữ
`0.05` cứng như bạn nói ban đầu?

### 10.7 Đo baseline zero-shot của chính CONCH

Paper: CONCH zero-shot **79.1%** accuracy trên CRC100K test (7,180 ảnh), hơn
PLIP 11.7 điểm. Con số này là mốc kiểm tra: sau khi implement, chạy zero-shot
trên test set PathMNIST phải ra **xấp xỉ 79%**. Lệch nhiều nghĩa là transform,
normalize, tokenizer hay projection đã sai — bắt được lỗi trước khi nó âm thầm
làm hỏng toàn bộ kết quả cold-start.

Đây là loại test dự án này cần: assert vào **cơ chế** (khớp số đã công bố), chứ
không phải shape.
