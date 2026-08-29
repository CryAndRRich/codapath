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
| `find_data_root(candidates)` | gốc dataset ảnh; candidate ƯU TIÊN, 2 mount mặc định vẫn là fallback |

~~`write_archive(source, name, slug)`~~ — **bỏ**. Phía GHI archive không thuộc
module này: `utils/kaggle.py` là phía ĐỌC (dò cache đã mount), còn đặt tên +
đóng gói thuộc `utils/archive.py` (đã có `visual_archive_stem`,
`nucleus_archive_stem`, `results_archive_stem`). Và `dataset-metadata.json` +
lệnh CLI đã bị bỏ hẳn ở bước 3b: session "Save & Run All" không có terminal.

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
| `evaluate_al_sampler.ipynb` | **sửa** | §2.1 — hiện hard-code DINOv2, không đánh giá được run CONCH |

### 2.0 Quy ước chung: MỘT lần chạy = MỘT cấu hình = MỘT zip

Đây là nguyên tắc bao trùm mọi notebook, chốt sau khi rà bước 4. Trước đó code
để lẫn hai mô hình và gây ra đúng loại lỗi §6.2 nói:

- **Một lần chạy notebook nhận đúng MỘT giá trị cho mỗi trục**: một dataset,
  một seed, một sampler, một encoder, một `USE_LORA`, một `AUX_LOSS`… Không
  `SEEDS = [...]`, không `VARIANTS = [...]`, không `DATASETS = [...]`.
- **Kết thúc là đúng MỘT zip**, tên chứa đủ mọi trục để phân biệt. Nhiều
  cấu hình = nhiều lần chạy = nhiều zip khác tên.
- Muốn quét nhiều seed/variant: chạy notebook nhiều lần, đổi cell EDIT.

**Vì sao đây là quyết định đúng chứ không phải bớt tính năng:** song song 2 GPU
giờ đến từ **chia budget** (`SPLIT_BUDGETS`, bước 3b), không đến từ việc gộp
nhiều variant vào một lần chạy. Đã kiểm: với 1 sampler + 1 seed, 8/11 baseline
vẫn dùng được cả 2 GPU (3 cái còn lại là prefix-exact, vốn không chia được dù
có gộp variant hay không). Nên bỏ multiplicity **không mất gì**.

Ngược lại, giữ multiplicity thì hỏng thật: `VARIANTS=[{}, {...}]` chạy 2 biến
thể nhưng `results_archive_stem` chỉ sinh **một** tên zip, nên hai biến thể
nằm chung một archive không phân biệt được — đúng thứ §6.2 cảnh báo, chỉ là ở
tầng archive.

**Hệ quả cho §6.4 (train cuối):** câu hỏi "train cuối có ghi đè `results["linear"]`
không" **tự tan**. `USE_LORA=False` và `USE_LORA=True` là hai lần chạy, hai zip
khác tên, nên không thể đè nhau. Train cuối cứ ghi `"linear"` như thường; thứ
phân biệt hai cấu hình là **tên zip**, không phải một khoá thứ hai trong file.

**Trạng thái hiện tại (phải sửa):**

| Notebook | Hiện tại | Cần |
|---|---|---|
| `extract_nucleus_features` | `SEED` số ít, 1 dataset | ✅ đã đúng |
| `extract_visual_features` | `SEEDS=[...]`, `DATASETS=[...]` | → số ít |
| `run_al_baseline` | `SEEDS=[...]`, `VARIANTS=[...]` | → số ít, bỏ `VARIANTS` |
| `run_al_sampler` | `SEEDS=[...]`, `VARIANTS=[...]` | sẽ bị xoá ở bước 9 |
| `run_al_main` | chưa có | viết theo quy ước này ngay từ đầu |

`*_archive_stem` đổi theo: nhận `seed` số ít thay vì `seeds`. `visual_archive_stem`
cũng vậy nếu sửa `extract_visual_features`.

Mọi notebook mới đều theo đúng khung đã có: clone + verify branch → `%cd` →
pip install → cell EDIT → import → resolve cache → chạy → archive.

### 2.1 `evaluate_al_sampler.ipynb` — lỗ hổng phát hiện khi rà plan

Notebook này **hard-code `DINOv2Extractor`** và tự extract lại test feature mỗi
lần chạy. Hai vấn đề:

1. Run của Protocol B nằm trong không gian CONCH, nhưng notebook sẽ chấm chúng
   bằng feature DINOv2 → sai hoàn toàn mà **không có gì báo lỗi**: probe 512-d
   gặp feature 768-d thì crash, nhưng nếu số chiều tình cờ khớp thì ra con số
   vô nghĩa.
2. Nó extract lại toàn bộ test set mỗi lần, trong khi cache đã có sẵn.

Sửa: đọc `metadata` trong `<run>_probe_budget_<B>.pt` (đã có từ lần trước) để
biết run đó dùng encoder nào, rồi load đúng cache tương ứng. Thêm assert số
chiều probe khớp số chiều feature. Bỏ `codapath` khỏi `RUN_NAMES` ví dụ.

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
SEED = 42                         # MỘT seed, MỘT zip (§2.0)
# CONCH | Astaxanthin/KEEP | wisdomik/QuiltNet-B-16-PMB | vinid/plip
VLM = "MahmoodLab/CONCH"
DESCRIPTION_STYLE = "conch_official"  # cái nào ở §3; "manual" = control
HF_TOKEN = ""                  # CONCH gated -> cần token
DATA_ROOT = "..."              # ảnh gốc — cần cả ở đây (§5, hai dataset)
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

Archive: theo đúng khuôn 4 notebook hiện có — **một** zip ở đỉnh
`/kaggle/working`, xoá file rời, tên từ helper trong `utils/archive.py`. Cần
thêm `vlm_archive_stem(dataset, seed, vlm_name, style)`: `style` phải nằm
trong tên vì text prototype của hai style khác nhau là hai artifact khác nhau
và không thay thế nhau được.

#### 4.1 Chặn đường: `data/loaders.py` khoá cứng 224 + ImageNet norm

Đây là việc **phải làm trước** notebook, không phải chi tiết phụ.
`get_data_loaders` (dòng 193–197) tạo transform cố định:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

CONCH cần 448 + OpenAI mean/std, nên phải cho `get_data_loaders` nhận
`transform=None` và chỉ dựng transform mặc định khi không được truyền. Truyền
thẳng `preprocess` mà factory trả về — **không** tự viết lại (§10.3).

**Bẫy đi kèm, phải kiểm tra khi làm:** `main.py` và notebook `evaluate` đều gọi
`get_data_loaders`, và fingerprint thứ tự mẫu được tính từ `sample_id` chứ
không từ pixel, nên đổi transform **không** làm hỏng fingerprint hay cache
đang có. Cần một test ghim đúng điều này, vì nếu sai thì mọi cache DINOv2 đã
publish thành vô giá trị.

#### 4.2 Zero-shot 79.1% — mốc kiểm tra, nhưng KHÔNG phải test tự động

§10.7 đề xuất `test_conch_zeroshot_matches_paper`. Xem lại thì **không đưa vào
`tests/`**:

- Nó cần tải checkpoint CONCH (gated, cần HF token) + toàn bộ test set
  PathMNIST + một GPU. `tests/` hiện chạy trong ~4 phút không cần mạng.
- 79.1% của paper đo trên **CRC100K test (n=7,180)**, còn ta có **PathMNIST
  test (n=7,180)** — cùng nguồn NCT-CRC-HE nhưng PathMNIST đã resize về 224 rồi
  ta lại resize lên 448, nên con số **không bắt buộc trùng khít**.

Thay bằng: một **cell kiểm tra trong chính `extract_vlm_features.ipynb`**, chạy
zero-shot bằng prompt chính thức ngay sau khi extract xong, in accuracy và
**assert > 0.70**. Ngưỡng lỏng vì lý do resize ở trên; mục đích là bắt lỗi
transform/normalize/tokenizer/projection (sẽ cho ~11% = random), không phải tái
lập chính xác con số paper. Ghi accuracy đo được vào manifest.

#### 4.3 Thứ tự lớp — ĐÃ KIỂM TRA, khớp

`repos/CONCH/prompts/crc100k_prompts_all_per_class.json` có 9 lớp theo thứ tự
`ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`. Khối
`datasets.pathmnist.descriptions` trong `config.yaml` theo thứ tự
`adipose, background, debris, lymphocytes, mucus, smooth_muscle,
normal_colon_mucosa, cancer_associated_stroma, colorectal_adenocarcinoma` —
**khớp 1-1 đúng thứ tự**. Đây là điều kiện để so zero-shot với nhãn số của
PathMNIST mà không cần bảng ánh xạ.

**Nhưng file JSON lồng một cấp**: nội dung nằm dưới khoá `"0"`, tức
`json.load(f)["0"]["classnames"]` và `["0"]["templates"]`, **không** phải
`["classnames"]` ở mức cao nhất. Đọc sai là `KeyError` ngay — ghi ra đây để
khỏi mất thời gian dò.

Phải có assert trong code: thứ tự lớp lấy từ `config.yaml` và thứ tự lớp của
text prototype **phải khớp**, vì nếu lệch thì mọi thứ vẫn chạy và cho ra con số
sai hoàn toàn.

---

## 5. `run_al_baseline.ipynb`

Chỉ chạy baseline. **Không** CellViT, **không** VLM, **không** text.

Cell EDIT — đúng ba biến như hai notebook extract, cộng chọn sampler:

```python
DATASET = "pathmnist"
SEED = 42                # MỘT seed. Quét nhiều seed = chạy lại notebook (§2.0)

# 11 baseline (sau khi xoá codapath). ONE per run:
#   random | coreset | typiclust | activeft | tcm
#   margin | entropy | badge | dropquery | uncertainty_herding | refine
SAMPLER = "uncertainty_herding"

PARALLEL = True          # 2 GPU qua chia budget, không qua gộp variant
SPLIT_BUDGETS = True
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

**§5 ĐÃ XONG** (bước 3 + 3b). Ba điểm khác với bản kế hoạch gốc, đã làm và đã
test — ghi lại vì §6 phải theo cùng khuôn:

- **Cần HAI Kaggle Dataset**, không phải một. Cache đặc trưng không chứa nhãn,
  và fingerprint xác thực chính cache đó cũng phải tính từ ảnh gốc, nên
  `main.run` luôn mở dataset ảnh. `DATA_ROOT` (ảnh gốc) và `FEATURE_DIR`
  (`.npy` đã extract) là hai biến riêng ở cell EDIT.
- **Chia 2 GPU theo budget** (`SPLIT_BUDGETS`), điều kiện là chính
  `spec.prefix_exact`. Chia round-robin. Shard ghi
  `<run>_<tag>_results.pt` riêng rồi `main.merge_budget_shards` gộp lại.
- **Archive**: một zip ở đỉnh `/kaggle/working`, tên
  `{dataset}_{sampler}_seed{seed}` từ `results_archive_stem`, xoá file rời.
  Không dùng `dataset-metadata.json` + lệnh CLI (session "Save & Run All"
  không có terminal).

---

## 6. `run_al_main.ipynb`

Pipeline chính. Cell EDIT:

```python
DATASET = "pathmnist"
SEED = 42                       # MỘT seed, MỘT cấu hình, MỘT zip (§2.0)

# --- biến thể: mỗi lần chạy chọn MỘT giá trị cho mỗi dòng ---
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
nhận tham số tường minh, và `_default_run_name` **phải** encode nó.

**Đã đo mức nghiêm trọng, nặng hơn mô tả cũ.** `_default_run_name` hiện có chữ
ký `(sampler_name, sampler_cfg)` — encoder **không phải là input**, nên không
thể encode kể cả nếu muốn:

```
_default_run_name('uncertainty_herding', {})   # protocol A (DINOv2)
_default_run_name('uncertainty_herding', {})   # protocol B (CONCH)
-> cả hai đều ra 'uncertainty_herding'
```

Hai protocol sẽ ghi đè **toàn bộ** file của nhau: `_results.pt`,
`_selected_budget_*.pt`, `_probe_budget_*.pt`, `_predictions_budget_*.pt`, và
log. Bảng kết quả trông đầy đủ và là của protocol chạy sau.

Điểm nguy hiểm: cách resume của notebook là kiểm tra `<name>_results.pt` tồn
tại. Nếu protocol A chạy trước, protocol B sẽ bị **skip lặng lẽ** như đã xong
— không lỗi, không cảnh báo, và bảng cuối là số của A dán nhãn B.

Ngược lại, **cache đặc trưng thì an toàn**: `_feature_cache_paths` đã đưa
`vit_name` vào tên file, nên cache CONCH và DINOv2 không đụng nhau. Chỉ có tên
**run** là hỏng.

Vậy việc phải làm, theo thứ tự:

1. `_default_run_name(sampler_name, sampler_cfg, *, encoder, use_text, ...)` —
   thêm tham số, và encoder mặc định `"dinov2"` để mọi tên hiện có **không
   đổi** (các run baseline đã chạy vẫn resume được).
2. Test `test_run_name_encodes_every_axis`: sinh tên cho mọi tổ hợp trục của
   §6, assert **không có hai tổ hợp nào trùng tên**. Đây là test cơ chế đúng
   nghĩa — nó bắt được chính xác lỗi trên.
3. Test riêng: tên của cấu hình mặc định (dinov2, không text, không lora)
   **phải bằng đúng tên cũ**, nếu không mọi kết quả baseline đã có thành mồ côi.

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

**Đây là pha SAU khi đã chọn đủ budget, KHÔNG phải trong vòng lặp AL.** Trong
vòng lặp, probe vẫn train trên feature đóng băng như hiện tại — LoRA và loss
phụ không xuất hiện ở đó. (Đã xác nhận lại với bạn; ghi rõ ở đây vì đọc lướt
§6.5 rất dễ hiểu ngược.)

Augment và LoRA cần **pixel**; cache chỉ có embedding đã tính qua transform cố
định 224×224. Nên:

- **Chọn mẫu** dùng embedding cache (nhanh, không đổi).
- **Train cuối** load ảnh gốc cho ~B ảnh đã chọn. `RawRGBDataset` đã có sẵn
  (`data/loaders.py:100`, export trong `data/__init__.py`) và trả đúng thứ tự
  split + `sample_id` khớp, nên không cần viết dataset mới — chỉ cần `Subset`
  theo `selected_indices`.
- B ≤ 200 ảnh nên chi phí không đáng kể.

Khi `USE_LORA=False` **và** `AUGMENT="none"`: train linear thẳng trên embedding
cache, không load ảnh — giữ đúng đường chạy nhanh hiện tại. Đây là control
`lora_r=0, aux_loss=none`.

**Ghi kết quả — đã chốt theo §2.0, KHÔNG cần khoá riêng.** Một lần chạy có
đúng một `USE_LORA`/`AUX_LOSS`, nên `USE_LORA=False` và `USE_LORA=True` là hai
lần chạy, hai zip khác tên. Chúng không thể đè nhau, và train cuối cứ ghi vào
`results["linear"]` như bình thường. Thứ phân biệt hai cấu hình là **tên zip +
`run_name`**, không phải một khoá thứ hai trong file.

(Bản nháp trước của mục này đề xuất `results["final"]` riêng. Đó là hệ quả của
việc giả định một lần chạy ôm nhiều cấu hình — giả định đã bị §2.0 bác bỏ.)

**Còn phải quyết: train cuối chạy ở budget nào.** Hai lựa chọn, khác nhau về
chi phí chứ không về tính đúng đắn:

- **Chỉ budget lớn nhất** — trả lời "pipeline đầy đủ có thắng kết quả tốt nhất
  của Protocol A không". Rẻ.
- **Mọi budget** — cho một đường cong LoRA đầy đủ, so được với đường cong
  frozen ở từng mức nhãn. Đắt gấp ~8 lần.

Đường cong đầy đủ chỉ đáng nếu câu hỏi nghiên cứu là "LoRA giúp nhiều hơn ở
budget thấp hay cao"; nếu chỉ cần một con số cuối để so bảng thì budget lớn
nhất là đủ. **Cần bạn chốt.**

### 6.5 Code mới cho §6

| File | Việc |
|---|---|
| `training/lora.py` | LoRA cho ViT. **`peft` KHÔNG có ở local** (đã kiểm tra: peft/timm/open_clip đều thiếu, torch 2.2.2). Tự implement ~60-80 dòng wrap `nn.Linear` của attention. Test: `lora_r=0` ≡ frozen, bit-for-bit |
| `training/losses.py` | `center_loss`, `supcon_loss`, `triplet_loss`. Chữ ký chung `(features, logits, labels) -> scalar`. `supcon`/`triplet` phải **raise có thông báo** khi batch không đủ 2 mẫu/lớp, không được lặng lẽ trả 0 |
| `training/finetune.py` | Vòng train cuối: nhận `train_mode`, `aux_loss`, `aux_weight`, `augment`; đọc pixel khi cần |
| `data/augment.py` | `flip_rotate` (an toàn sinh học với H&E: lật/quay 90° không đổi nhãn). **Không** color jitter — dự án từng có method chết vì stain shortcut |
| `utils/archive.py` | thêm `main_archive_stem(...)` — cùng khuôn `results_archive_stem` đã có |

Archive: một zip ở đỉnh `/kaggle/working` như 4 notebook kia. Tên phải encode
**đúng những trục mà `_default_run_name` encode** (§6.2), nếu không hai biến
thể ra hai run_name khác nhau nhưng cùng một tên zip, và cái publish sau đè
cái trước — đúng lỗi §6.2 nhưng ở tầng archive.

---

## 7. Đổi trong code lõi

Bảng dưới **chỉ còn việc chưa làm**. Những gì bước 1–3b đã xong (xoá
`_load_vlm_inputs`, `BASELINE_SAMPLERS`, `utils/kaggle.py`, bỏ nhánh codapath
khỏi `evaluation/plots.py`, bỏ `samplers.codapath`) đã gỡ khỏi bảng để không
đọc nhầm là còn tồn.

| File | Đổi gì | Vì sao |
|---|---|---|
| `data/loaders.py` | `get_data_loaders` nhận `transform=None` thay vì hard-code | **Chặn đường cho §4.** CONCH 448 + OpenAI norm; hiện khoá cứng `Resize((224,224))` + ImageNet mean/std ở dòng 193-197. Truyền `preprocess` của factory, đừng tự viết lại |
| `features/visual.py` | `get_or_extract_features` nhận loại extractor | tái dùng cache logic cho CONCH, không copy hàm |
| `features/vlm.py` | **mới** | §1.2, §4 |
| `features/descriptions.py` | **mới** | §3 |
| `utils/archive.py` | thêm `vlm_archive_stem`, `main_archive_stem` | §4, §6.5 |
| `main.py` | `visual_backbone` thành tham số tường minh của `run()` | encoder là trục thật |
| | `_default_run_name` **thêm tham số** encoder/text/lora/aux/augment | §6.2 — hiện chữ ký không có encoder nên **không thể** encode; hai protocol ghi đè nhau |
| | nhánh text đọc từ config run, không từ `spec.needs` | text là trục của scalpel, không phải yêu cầu tĩnh |
| | thêm pha train cuối sau budget sweep | §6.4 — ghi `results["linear"]` như thường; phân biệt cấu hình bằng run_name/zip (§2.0) |
| `sampling/scalpel/sampler.py` | vòng 1 (`round_index == 0`, chỗ `weights_np = None`) nhận `round1_weight` từ text | cold start; hook đã có sẵn đúng chỗ |
| `evaluate_al_sampler.ipynb` | đọc encoder từ metadata probe, bỏ hard-code `DINOv2Extractor` | §2.1 — **bắt buộc trước khi có run CONCH nào**, nếu không không chấm được |
| `config/config.yaml` | thêm `vlm`, `descriptions`, `final_training` | |

---

## 8. Test phải thêm

Theo đúng nguyên tắc của dự án: assert vào **cơ chế**, không phải shape.

Ràng buộc bắt buộc: **`tests/` phải chạy được không mạng, không GPU, không HF
token**, trong ~vài phút (hiện 317 test / ~3.5 phút). Test nào cần checkpoint
CONCH thì **không** vào `tests/` — đưa thành cell kiểm tra trong notebook (xem
§4.2).

Đã xong ở bước 1–3b (giữ lại để khỏi viết trùng): `test_no_codapath_left`,
`test_baseline_notebook_rejects_scalpel`, và bộ notebook/archive/trace/shard.

| Test | Bắt cái gì |
|---|---|
| `test_loaders_accept_a_custom_transform` | truyền `transform` vào `get_data_loaders` thì nó được dùng; không truyền thì vẫn đúng 224+ImageNet như cũ |
| `test_custom_transform_does_not_change_sample_order` | **quan trọng nhất của §4.1**: fingerprint tính từ `sample_id`, nên đổi transform KHÔNG được làm hỏng cache DINOv2 đã publish |
| `test_conch_uses_openai_normalization` | đọc `image_mean/std` mà factory gán, assert là OpenAI chứ không phải ImageNet — bẫy §10.3. Không cần tải model nếu chỉ đọc hằng số |
| `test_conch_prompt_file_is_nested_under_zero` | ghim cấu trúc `["0"]["classnames"]` của file JSON (§4.3) — đọc sai là `KeyError` |
| `test_prompt_class_order_matches_config` | 9 lớp CONCH khớp thứ tự `datasets.pathmnist.descriptions`; lệch = số sai mà không lỗi |
| `test_vlm_cache_naming_matches_visual` | cùng quy ước để `main.py` đọc một đường |
| `test_vlm_cache_has_both_spaces` | có cả `_train.npy` và `_proj_train.npy` |
| `test_description_file_roundtrip` | sha256 khớp; đổi description mà không đổi hash thì fail |
| `test_run_name_encodes_every_axis` | **§6.2**: sinh tên cho mọi tổ hợp trục, assert không tổ hợp nào trùng tên |
| `test_default_run_name_is_unchanged` | cấu hình mặc định phải ra **đúng tên cũ**, nếu không mọi kết quả baseline đã chạy thành mồ côi |
| `test_text_requires_conch` | `USE_TEXT` + `dinov2` → assert nổ |
| `test_lora_zero_equals_frozen` | `lora_r=0` cho weight **giống bit-for-bit** linear probe |
| `test_aux_loss_zero_weight_is_identity` | `aux_weight=0` ≡ `aux_loss="none"` |
| `test_supcon_raises_on_thin_batch` | không đủ 2 mẫu/lớp thì **raise**, không trả 0 |
| `test_augment_preserves_label` | flip/rotate không đổi nhãn |
| `test_one_config_per_run` | **§2.0**: cell EDIT không có `SEEDS`/`VARIANTS`/`DATASETS` dạng list — một lần chạy một cấu hình |
| `test_archive_stem_separates_every_axis` | hai cấu hình khác nhau **không** ra cùng tên zip (cùng lỗi §6.2, ở tầng archive) |
| `test_notebook_set_is_exactly_seven` | cập nhật từ 5 → **7** notebook |

---

## 9. Thứ tự làm

1. ~~Xoá codapath + assets~~ (§1.1) — **XONG**. `codapath.py`, `assets/` (12
   file, hồi được từ `4b6280f`), entry trong `specs.py`/`__init__.py`/
   `config.yaml`, nhánh `_load_vlm_inputs` trong `main.py`, 5 chỗ trong
   README, 1 test viết lại (`test_needs_is_not_an_axis`, giờ chứng minh bằng
   `scalpel`/`random`/`margin` vì `scalpel` là sampler duy nhất còn `needs`
   khác rỗng). `BASELINE_SAMPLERS` đã thêm vào `specs.py` (dùng ở bước 3).
2. ~~`utils/kaggle.py`~~ (§0.3) — **XONG**. 4 hàm (`find_dir_containing`,
   `find_data_root`, `find_visual_cache`, `find_cellvit_cache`); **không**
   gồm archive-writing. Đã wire vào `run_al_sampler.ipynb` (cell 6–8), thay
   `dir_containing` cục bộ. 13 test mới.
   *Cập nhật ở bước 3b:* lúc đó module cố ý không gộp archive vì 4 notebook
   dùng 2 pattern zip khác nhau. Sau bước 3b **đã gộp** — cả 4 notebook giờ
   dùng chung khuôn "một zip ở đỉnh `/kaggle/working`, xoá file rời, tên từ
   `utils/archive.py`". Việc đặt tên vẫn ở `utils/archive.py` (nơi đã có
   `visual_archive_stem`/`nucleus_archive_stem`), **không** ở `utils/kaggle.py`
   — `kaggle.py` là phía ĐỌC, `archive.py` là phía GHI.
   Cũng ở bước 3b: sửa `find_data_root` từ "candidate THAY THẾ default" thành
   "candidate ƯU TIÊN, default vẫn là fallback" — bản cũ khiến gõ sai
   `DATA_ROOT` một ký tự là mất luôn cơ chế dò remount.
   (`test_kaggle_cache_lookup.py`), bắt được 1 lỗi thật lúc viết: test kỳ vọng
   sai hướng trả về của `find_cellvit_cache` (hàm đúng, trả về **parent** của
   `pathmnist_seed42/` để khớp `main.py::_load_cell_view` tự nối path — test
   ban đầu viết ngược, đã sửa). 256 test pass, pyflakes sạch.
3. ~~Tách `run_al_baseline`~~ (§5) — **XONG**. Notebook mới 11 cell, cùng khung
   với `run_al_sampler.ipynb` nhưng: không nhắc CellViT/VLM ở đâu trong code
   (chỉ ở prose giải thích phạm vi), assert `SAMPLER in BASELINE_SAMPLERS` +
   `spec.needs` rỗng cả hai loại ngay trong cell cấu hình (nổ trước khi tốn
   GPU, không phải giữa chừng vòng lặp budget), archive slug
   `{dataset}-baseline-{sampler}` *(đã thay ở bước 3b bằng zip
   `{dataset}_{sampler}_seed{seed}`)*. `run_al_sampler.ipynb` giữ nguyên (cần
   cho `scalpel`) — sẽ xoá ở bước 9 khi `run_al_main` thay thế nó.
   17 test mới (8 test riêng cho notebook này trong
   `test_kaggle_notebooks.py`, cộng `EXPECTED_NOTEBOOKS` +1). Một lần viết
   test sai — kỳ vọng "không được nhắc CellViT" chặn cả comment/assert-message
   giải thích lý do reject `scalpel`, phải sửa lại để chỉ chặn *cơ chế* (import
   `find_cellvit_cache`, biến `CELLVIT_DIR`, `cellvit_cache_dir=`), không chặn
   *nhắc tên*. 265 test pass (kể cả bộ `test_kaggle_cache_lookup.py` của bước
   2), pyflakes sạch. Verify thêm bằng mô phỏng thủ công: dựng cache DINOv2
   giả lập đúng như `extract_visual_features.ipynb` sẽ publish, gọi
   `find_visual_cache` với `hint` không phải đường dẫn chính xác (mô phỏng
   Kaggle remount), rồi `main.run_on_worker` **không** truyền
   `cellvit_cache_dir` — chạy xong ghi đủ file kết quả.
3b. ~~Rà soát luồng baseline sau bước 3~~ — **XONG**. Audit toàn luồng phát hiện
   4 vấn đề, 3 cái phải sửa:
   - **Không sampler baseline nào ghi trace** — hạ tầng `SelectionTrace` có sẵn
     và `main.py` truyền `trace=` vào mọi sampler, nhưng `grep` trên
     `sampling/baselines/*.py` không khớp dòng nào, nên `main.py` rơi vào nhánh
     backfill và mọi `score` đều `None`. Đã thêm ghi trace cho cả 10 sampler có
     điểm số (`random` không có score thật nên vẫn để backfill). Mỗi sampler
     chỉ ghi thứ nó *thực sự tính*: `coreset`/`typiclust`/`activeft` không fit
     classifier nên không có `uncertainty`; `tcm` gắn nhãn `phase` 1/2 vì hai
     pha tối ưu hai đại lượng khác nhau; `uncertainty_herding`/`refine` tách
     riêng `uncertainty` và `coverage` khỏi `score` (tích của chúng) để sau này
     trả lời được "yếu tố nào thực sự quyết định lượt chọn".
   - **PALM chạy trong lúc run** — đã bỏ `_fit_palm` khỏi `main.py`. Không tạo
     lỗ hổng: `evaluate_al_sampler.ipynb` vốn tự tính lại accuracy từ probe đã
     lưu rồi fit PALM độc lập, chưa từng đọc `_palm.pt`.
   - **2 GPU không hoạt động với cấu hình mặc định** — `utils/parallel.py` có
     `workers = min(workers, len(variants))`, mà mặc định `SEEDS=[42]` ×
     `VARIANTS=[{}]` = 1 job → 1 GPU chạy, 1 GPU ngồi không. Đã thêm chia theo
     **budget** (`SPLIT_BUDGETS`), chia round-robin (không chia liền khối, vì
     chi phí tăng theo budget). Điều kiện chia là chính `spec.prefix_exact`
     chứ không phải danh sách tên tự giữ — sampler prefix-exact
     (`random`/`coreset`/`tcm`) dùng chung một lượt chọn cho cả sweep nên chia
     ra sẽ lặp lại lượt đó ở mỗi shard; `main.run` raise thẳng nếu nhận
     `shard_tag` cho loại này. Mỗi shard ghi `<run>_<tag>_results.pt` + log
     riêng (file theo budget vốn đã không đụng nhau), rồi
     `main.merge_budget_shards` gộp lại đúng một `<run>_results.pt` như chạy
     không chia — downstream không cần biết đã chia.
   - **Archive sai khuôn** — notebook baseline ghi vào thư mục `archive/`, GIỮ
     file gốc, rồi in lệnh `kaggle datasets create` mà session "Save & Run All"
     không chạy được (không terminal, không CLI); tên zip cứng
     `al_checkpoints.zip`. Đã đổi sang đúng khuôn 2 notebook extract: một zip ở
     đỉnh `/kaggle/working`, tên `{dataset}_{sampler}_seed{seed}` từ
     `results_archive_stem` mới, xoá file rời. Sửa cùng lỗi ở
     `run_al_sampler.ipynb`. Giờ cả 4 notebook cùng một khuôn.
   - **Hai Kaggle Dataset, không phải một** — cache đặc trưng không chứa nhãn,
     và fingerprint xác thực chính cache đó cũng tính từ ảnh gốc, nên
     `main.run` luôn mở dataset ảnh. Tách `DATA_ROOT` / `FEATURE_DIR` ở cell
     EDIT, kèm sửa `find_data_root` (xem bước 2).
   50 test mới (35 `test_baseline_trace.py` + 8 `test_budget_shards.py` + 7
   archive/notebook/cache). Test quan trọng nhất là *fidelity*: chạy mỗi
   sampler hai lần cùng seed, có và không có trace, danh sách index phải bằng
   nhau — đã xác nhận **cả 11 sampler không đổi lựa chọn**. Và sharded vs
   unsharded cho **cùng accuracy và cùng selected indices ở mọi budget**.
   Tổng: **317 test pass**, pyflakes sạch.

---

**Từ đây trở đi là việc chưa làm.** Thứ tự dưới có ràng buộc phụ thuộc thật,
không phải sắp xếp tuỳ ý — mỗi bước ghi rõ nó chặn cái gì.

4. ~~**Một cấu hình một lần chạy**~~ (§2.0) — **XONG**. `SEEDS`/`VARIANTS`/
   `DATASETS` đã bỏ khỏi cả ba notebook (`run_al_baseline`, `run_al_sampler`,
   `extract_visual_features`); `visual_archive_stem` và `results_archive_stem`
   nhận `seed` số ít.
   - **Kiểm trước khi đổi**: mọi zip đã publish (`../data/*.zip`) đều là
     một-dataset-một-seed, và dạng số ít sinh ra **đúng tên cũ** — không
     artifact nào bị mồ côi. Có test ghim điều này
     (`test_results_stem_matches_the_already_published_naming`).
   - `run_al_sampler` mặc định có **2 variant** (`{}` + ablation
     `visual_margin`), nên nó thật sự dùng multiplicity. Giờ ablation là lần
     chạy thứ hai, có comment ghi sẵn giá trị `OVERRIDES` để dán.
   - Cũng phải sửa `run_al_sampler` chứ không để đó chờ bước 10: nó gọi
     `results_archive_stem(..., SEEDS)`, đổi chữ ký là nó vỡ ngay.
   - **Thêm cell tóm tắt kết quả** cho cả hai notebook run: hai GPU dùng chung
     một luồng stdout nên log xen kẽ và số của một budget rơi lung tung. Cell
     mới đọc lại `<run>_results.pt` đã lưu (nguồn đáng tin duy nhất) và in
     đúng thứ tự budget, **chỉ metric** (accuracy/precision/recall/macro-F1 +
     thời gian chọn), không in score/sigma/log. Chạy lại riêng cell này là in
     lại được, không tính toán gì.
   - Verify bằng chạy thật: shard in ra budget theo thứ tự **8, 24, 16, 32**
     (xen kẽ) nhưng bảng tóm tắt ra **8, 16, 24, 32** đúng thứ tự.
   - Một lỗi tự gây khi sửa: dùng regex đổi `dataset`→`DATASET` làm hỏng
     `loader.dataset` thành `loader.DATASET` và `manifest["dataset"]` thành
     `manifest["DATASET"]`. `ast.parse` không bắt được vì vẫn đúng cú pháp.
     Đã rà lại từng cell và sửa; bài học: đừng regex trên code, viết lại cell.
5. ~~**`data/loaders.py` nhận `transform`**~~ (§4.1) — **XONG**.
   `get_data_loaders` thêm tham số `transform=None`, mặc định gọi
   `default_transform()` (hàm mới, đúng transform cũ 224+ImageNet, tách ra để
   gọi tên được tường minh). Không truyền = hành vi y hệt trước, đã verify
   bit-for-bit (shape, giá trị pixel). `default_transform` export qua
   `data/__init__.py`.
   **Không sửa bất kỳ notebook/script nào khác** — `main.py`, cả 4 notebook,
   3 script trong `scripts/` đều gọi `get_data_loaders(...)` không có
   `transform=`, nên tất cả rơi vào default, không cần đổi.
   7 test mới (`tests/test_loaders_transform.py`), quan trọng nhất:
   `test_custom_transform_does_not_change_sample_order` — đổi transform (448 +
   OpenAI norm, mô phỏng CONCH) **không** làm đổi `sample_id`, fingerprint,
   nhãn, hay `class_names` so với transform mặc định. Đây là test bảo vệ toàn
   bộ cache DINOv2 đã publish: `features/visual.py` xác thực cache bằng
   fingerprint tính từ `sample_id`, không phải từ pixel.
6. ~~**`features/vlm.py` + `extract_vlm_features.ipynb`**~~ (§1.2, §4) —
   **XONG**.
   - `features/vlm.py` mới: hai không gian embedding đặt tên rõ (`RAW_SPACE`/
     `PROJ_SPACE`), naming cache đúng quy ước `features/visual.py`, ensemble
     prompt chính thức (normalize từng cái rồi mean rồi normalize lại — đúng
     công thức gốc, không phải mean-rồi-normalize-một-lần), `zero_shot_logits`
     thuần numpy (test được không cần `conch`). **Import `conch` trễ (bên
     trong hàm)**, giống hệt pattern `_load_cell_view` dùng cho `cellvit` —
     module tự import sạch dù máy local không có `conch`/`open_clip`.
   - `get_or_extract_vlm_features` nhận `model=` thay vì tự load lại: notebook
     đã phải load model trước để lấy `preprocess` (dataloader cần transform
     đúng của CONCH), nên truyền model đó vào tránh load 2 lần trên cache
     miss. Cache hit thì hoàn toàn không đụng tới model.
   - **File prompt CONCH chính thức đã vendor vào repo**
     (`config/prompts/crc100k_prompts_all_per_class.json`, license CC
     BY-NC-ND 4.0 — hỏi và được xác nhận trước khi copy, ghi rõ nguồn +
     license trong `config/prompts/README.md`). Cần thiết vì notebook Kaggle
     không có sẵn `repos/CONCH` (nó nằm ngoài git repo, chỉ có trên máy local).
   - `extract_vlm_features.ipynb` (14 cell): EDIT cell không có list nào
     (đúng §2.0), assert thứ tự lớp trước khi dùng `conch_official`, assert
     `DATASET=="pathmnist"` khi dùng `conch_official`, cell zero-shot in
     accuracy + `assert > 0.70` (không phải test tự động — xem §4.2), archive
     đúng khuôn 4 notebook kia.
   - `vlm_archive_stem(dataset, seed, vlm_name, style)` thêm vào
     `utils/archive.py`.
   - **Dọn theo đường**: `open_clip_torch` trong `requirements.txt` là rác từ
     thời BiomedCLIP/CODAPath cũ — không có `import open_clip` nào thật trong
     repo (`conch.open_clip_custom` là module khác, thuộc package `conch`).
     Đã xoá, và sửa comment sai tương ứng trong
     `extract_nucleus_features.ipynb`.
   - **Verify bằng chạy thật, không chỉ đọc code**: dựng package `conch` giả
     (CoCa deterministic, đúng chữ ký `encode_image(normalize, proj_contrast)`,
     `logit_scale`, tokenizer) và chạy **cả 14 cell của notebook thật** (bỏ
     qua cell 1-3 git/pip vì không chạy được local) trên dữ liệu PathMNIST giả
     lập — extract ra đúng shape 2 không gian, cache resume đúng (lần 2 load
     lại, không gọi `encode_image`), text prototype resume đúng, zero-shot
     assert bắn đúng khi model giả không có tín hiệu thật, archive ra đúng 8
     file tên đúng.
   29 test mới (`test_vlm_features.py` 18 mới hoàn toàn + `test_kaggle_notebooks.py`
   9 + `test_archive_naming.py` 2). `test_import_graph.py` không cần sửa vì
   import trễ đã sạch từ đầu — tự chạy qua, không cần test riêng. 366 test
   pass, pyflakes sạch.
7. **`generate_class_description.ipynb`** (§3) — cần trước khi `USE_TEXT` với
   style `llm_*` chạy được. Độc lập với bước 5, có thể đổi chỗ.
8. **`evaluate_al_sampler.ipynb` đọc encoder từ metadata** (§2.1) — **phải xong
   TRƯỚC khi chạy run CONCH đầu tiên**, không phải sau. Hiện nó hard-code
   `DINOv2Extractor`, nên một run Protocol B sẽ bị chấm bằng feature DINOv2:
   probe 512-d gặp feature 768-d thì crash (còn may), nhưng nếu chiều tình cờ
   khớp thì ra số vô nghĩa mà không có gì báo. Rẻ, và bỏ quên thì tốn cả session
   GPU mới phát hiện.
9. **`_default_run_name` + encoder tường minh** (§6.2) — cũng nên làm sớm và
   độc lập: nó chỉ là đổi chữ ký + test, nhưng bỏ quên thì Protocol B ghi đè
   Protocol A **và bị resume skip lặng lẽ**. Làm trước bước 9 để `run_al_main`
   sinh tên đúng ngay từ đầu.
10. **`run_al_main` phần chọn mẫu** (§6.1–6.3) — encoder + text, chưa train
   cuối. Xoá `run_al_sampler.ipynb` ở bước này (nó bị thay thế), cập nhật
   `EXPECTED_NOTEBOOKS`.
11. **Train cuối** (§6.4, §6.5) — LoRA, loss phụ, augment. Phần nhiều code
    nhất, và là phần duy nhất phụ thuộc vào tất cả những bước trên.

**Ràng buộc thứ tự, tóm tắt:**

```
4 (một cấu hình / lần chạy) ──┐
5 (loaders transform) ────────┴──> 6 (vlm.py + extract) ──┐
7 (descriptions) ─────────────────────────────────────────┼──> 10 (run_al_main) ──> 11 (train cuối)
8 (evaluate đọc encoder) ─────────────────────────────────┤
9 (run_name encode encoder) ──────────────────────────────┘
```

4 phải trước 6 và 10 để hai notebook mới viết đúng khuôn §2.0 ngay từ đầu.
8 và 9 độc lập nhau và độc lập với 4–7; cả hai đều rẻ và cả hai đều là loại lỗi
"chạy xong cả session GPU mới biết sai". Làm chúng bất cứ lúc nào trước 10,
đừng để sau.

**Chưa quyết, cần chốt trước bước 10:** train cuối chạy ở **budget lớn nhất**
hay **mọi budget** (§6.4). Khác nhau ~8 lần chi phí LoRA, không khác về tính
đúng đắn. Không còn là câu hỏi về ghi đè — §2.0 đã giải quyết chuyện đó.

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

Giá trị trong checkpoint là kết quả train, không phải 0.07 nữa.

**ĐÃ CHỐT: dùng `learned`** — `model.logit_scale.exp()`, đúng như code chính
thức. Không hard-code 0.05. `tau_mode="fixed"` giữ lại làm trục ablation tuỳ
chọn, không phải mặc định.

### 10.7 Đo baseline zero-shot của chính CONCH

Paper: CONCH zero-shot **79.1%** accuracy trên CRC100K test (7,180 ảnh), hơn
PLIP 11.7 điểm. Con số này là mốc kiểm tra: sau khi implement, chạy zero-shot
trên test set PathMNIST phải ra **xấp xỉ 79%**. Lệch nhiều nghĩa là transform,
normalize, tokenizer hay projection đã sai — bắt được lỗi trước khi nó âm thầm
làm hỏng toàn bộ kết quả cold-start.

Đây là loại test dự án này cần: assert vào **cơ chế** (khớp số đã công bố), chứ
không phải shape.

**Đã sửa lại ở §4.2:** nó là một **cell trong notebook**, không phải test trong
`tests/` (cần HF token + GPU + test set, còn `tests/` phải chạy offline trong
vài phút), và ngưỡng là **>0.70** chứ không phải bằng 79.1% — PathMNIST đã
resize về 224 rồi ta up lên 448 nên không bắt buộc trùng khít con số paper.

---

## 11. Rà soát trước khi code — những gì ĐÃ kiểm tra

Kiểm tra bằng code thật, không suy đoán. Ghi lại để không phải rà lại.

| Câu hỏi | Kết quả |
|---|---|
| 11 baseline có cái nào cần VLM/CellViT không? | **Không** — tất cả `needs=()`. Tách `run_al_baseline` là sạch |
| CellViT cache có khoá theo encoder không? | **Không** — chỉ theo `dataset+seed`. Nên **dùng lại được** cho cả DINOv2 và CONCH, không phải extract lại |
| Check `dino_backbone` có chặn CONCH không? | **Không** — chỉ fire khi `cell_source="crop_dino"`. CONCH + `cellvit_embedding` chạy được ngay |
| Probe và test feature có cùng backbone không? | **Có** — cùng một lời gọi `get_or_extract_features`, nên đổi `visual_backbone` là đổi cả hai, nhất quán |
| `evaluate_al_sampler` có đánh giá được run CONCH không? | **KHÔNG** — hard-code DINOv2. Xem §2.1 |
| Test nào tham chiếu codapath? | `tests/test_sampler_specs.py:52`. Viết lại bằng `scalpel` |

**Rà lại vòng 2 (trước bước 4), kiểm bằng code thật:**

| Câu hỏi | Kết quả |
|---|---|
| `_default_run_name` có encode được encoder không? | **KHÔNG** — encoder không nằm trong chữ ký `(sampler_name, sampler_cfg)`. Hai protocol ra **cùng một tên**, ghi đè toàn bộ file của nhau, và bị resume skip lặng lẽ. Xem §6.2 |
| Cache đặc trưng có đụng nhau giữa 2 encoder không? | **Không** — `_feature_cache_paths` đã có `vit_name` trong tên. Chỉ tên **run** hỏng, không phải cache |
| `RawRGBDataset` (§6.4) có thật không? | **Có** — `data/loaders.py:100`, đã export ở `data/__init__.py`. Dùng được ngay |
| Hook vòng 1 của scalpel có đúng chỗ không? | **Có** — `sampling/scalpel/sampler.py:110`, `round_index == 0` → `weights_np = None`. Đúng chỗ cắm text prior |
| `data/loaders.py` khoá cứng 224 + ImageNet thật không? | **Đúng** — dòng 193–197. Chặn đường CONCH 448. Xem §4.1 |
| File prompt CONCH có cấu trúc như §10.5 mô tả không? | **KHÔNG hẳn** — nội dung đúng (9 lớp, 22 template) nhưng **lồng dưới khoá `"0"`**. Đọc ở mức cao nhất là `KeyError`. Xem §4.3 |
| Thứ tự 9 lớp CONCH có khớp `config.yaml` không? | **Khớp 1-1 đúng thứ tự** — ADI→adipose … TUM→colorectal_adenocarcinoma. Không cần bảng ánh xạ |
| Test zero-shot 79.1% đưa vào `tests/` được không? | **Không nên** — cần HF token + GPU + test set; `tests/` hiện chạy 3.5 phút không mạng. Chuyển thành cell trong notebook, ngưỡng lỏng (>0.70) vì PathMNIST đã resize 224 rồi ta up lên 448. Xem §4.2 |

### 11.1 Rủi ro còn lại, đã biết và chấp nhận

- ~~`assets/` có hồi được không~~ — **ĐÃ KIỂM TRA: có.** Đã commit ở
  `4b6280f Add figs and pdf files` và hai commit trước đó, `git ls-files` thấy
  đủ file. Xoá an toàn, lấy lại bằng `git checkout <sha> -- assets/`.
- **Chưa commit gì cả.** Toàn bộ thay đổi từ các lượt trước (trace, sanity,
  parallel, progress) vẫn nằm ở working tree.
- **CONCH 448 chưa đo thời gian thật.** 4x pixel so với DINOv2 224 là suy ra từ
  số pixel, chưa benchmark. Phải chạy thử một batch trước khi cam kết cả session
  cho PathMNIST 90k ảnh.
