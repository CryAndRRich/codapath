# PLAN — luồng chạy và các trục thử nghiệm

> Chỉ có hai thứ trong file này: (1) luồng chính chạy như thế nào, (2) mỗi trục
> có những option nào để thử. Không có quy tắc code chung (ở `CLAUDE.md`),
> không có việc sửa notebook (đã làm xong, xem `README.md`).
>
> Chỉ liệt kê option **dùng được**. Những thứ đã tìm hiểu rồi loại (model gated,
> model không có text tower, API hết free tier) không có ở đây.

---

## 1. Luồng chính

```
                        ┌─────────────── VÒNG 1 (cold start) ───────────────┐

   Class name ──▶ LLM ──▶ Class description ──┐
                                              ├──▶ VLM ──▶ text embedding ──┐
   Image ─────────────────────────────────────┘     │                       │
                                                    └──▶ visual embedding ──┤
                                                                            ▼
                                                                      uncertainty
                                                                            │
                                                                            ▼
                                                                       UHerding
                                                                     (coverage × U)
                                                                            │
                                                                            ▼
                                                                    batch 1 (B/5 ảnh)
                        └───────────────────────────────────────────────────┘
                                                                            │
                        ┌──────────── VÒNG 2..5 (có nhãn rồi) ──────────────┤
                                                                            ▼
   Image ──▶ visual encoder ──▶ visual embedding ──▶ probe_v ────────┐
                                                                     ├──▶ disagreement
   Image ──▶ CellViT ──▶ nhiều cell embedding ──▶ pool ──▶ probe_c ──┘  = U
                                                                            │
                                                                            ▼
                                                                       UHerding
                                                                            │
                                                                            ▼
                                                                     batch 2..5
                        └───────────────────────────────────────────────────┘
                                                                            │
                                                                            ▼
                                              ĐỦ BUDGET ──▶ TRAINING classifier
                                                     (± finetune backbone, ± loss phụ)
                                                                            │
                                                                            ▼
                                                              accuracy / F1 / PALM
```

**Điều giữ nguyên xuyên suốt:** công thức acquisition luôn là Uncertainty
Herding

```
x* = argmax_i  Σ_n  U_n · max( k_σ(x_n, x_i) − K_n , 0 )
```

Coverage kernel, running-max `K_n`, và σ = min pairwise distance trên tập đã
nhãn (tính lại mỗi vòng) **không đổi** ở bất kỳ trục nào. Chỉ `U_n` đổi:

| Vòng | `U_n` là gì |
|---|---|
| 1 | từ text–image similarity của VLM (§2), hoặc U≡1 nếu không cold-start (§3) |
| 2..5 | disagreement giữa probe visual và probe cell (§4, §5) |

**Điều KHÔNG được nhập nhằng:** text embedding sống trong không gian VLM, còn
coverage kernel chạy trên không gian visual encoder. Text chỉ dùng làm **trọng
số vô hướng `U_n`**, không đi vào kernel. Đổi luôn coverage sang không gian VLM
là chuyện khác và sẽ phá so sánh với baseline.

### 1.1 Hai protocol — hai claim, không bao giờ trộn một bảng

**Đây là quyết định kiến trúc quan trọng nhất của cả dự án.** Có hai câu hỏi
khác nhau, mỗi câu hỏi một bảng riêng.

| | **Protocol A — so sánh chọn mẫu** | **Protocol B — so sánh pipeline** |
|---|---|---|
| Encoder | DINOv2, đóng băng | CONCH, **xuyên suốt** |
| Coverage kernel | DINOv2 | CONCH visual |
| `probe_v` (disagreement) | DINOv2 | CONCH visual |
| Linear head đánh giá | DINOv2 | **CONCH** |
| Vòng 1 | `U ≡ 1` (coverage thuần) | text embedding từ LLM description |
| Training | linear probe | linear + LoRA + center loss (+ augment) |
| Chạy cho | **mọi** sampler (13 cái) | chỉ pipeline đầy đủ |
| Trả lời | "sampler nào chọn mẫu tốt hơn" | "pipeline đầy đủ có hơn DINO+linear không" |

**Quy tắc một encoder xuyên suốt:** vòng 1 dùng encoder nào thì dùng encoder đó
đến hết — kể cả linear head lúc đánh giá. Không có chuyện vòng 1 CONCH rồi vòng
2 nhảy về DINOv2: `σ` và `K_n` đo ở không gian khác thì coverage tích luỹ từ
vòng trước thành vô nghĩa.

**Protocol B so với cái gì:** so với **kết quả tốt nhất của Protocol A**, tức
là so pipeline-với-pipeline. KHÔNG so với một biến thể bị làm yếu của chính B.
Nhờ vậy B thắng là thắng thật, và A vẫn là bảng xếp hạng sampler sạch (chỉ khác
nhau đúng một biến: chọn mẫu nào).

Hệ quả bắt buộc: `_default_run_name` phải encode backbone, nếu không hai
protocol ghi đè lên nhau (xem `CLAUDE.md`).

### 1.2 Control cho từng trục

| Câu hỏi | Trục | Control |
|---|---|---|
| Xoá cold start có lợi không | §2 vs §3 | `round1_weight=uniform` |
| Cell embedding có thêm gì không | §4, §5 | `uncertainty_mode=visual_margin` |
| Framework training có hơn linear probe không | §6, §7 | `lora_r=0, aux_loss=none` |

Mỗi câu hỏi phải có control tắt hẳn cơ chế đang xét. Không có control thì con
số không nói lên điều gì.

---

## 2. Xoá cold start — LLM + VLM (vòng 1)

### 2.1 LLM sinh class description

Sinh **một lần**, commit file, mọi lần chạy sau đọc file. Cách phát biểu về tái
lập phải đúng: `temperature=0` **không** cho bit-for-bit trên hosted inference
(kernel batch làm thứ tự cộng float phụ thuộc request của người khác cùng
batch; MoE routing theo capacity từng batch). **Artifact mới là bảo đảm tái
lập, không phải tham số sampling.**

| Option | Free tier | seed | Ghi chú |
|---|---|---|---|
| **`gemini-2.5-flash`** | có, rộng | có (best-effort) | Mặc định. Pin cứng id này |
| `groq` (llama/qwen) | 30 RPM / 14.4k RPD, không cần thẻ | có + `system_fingerprint` | Chọn nếu muốn dấu vết phát hiện backend đổi |
| `mistral` free | ~1 rps | `random_seed` | Cần xác minh SĐT + opt-in data training |
| `openrouter` `:free` | 20 RPM / 50 RPD | có | Doc ghi rõ "determinism is not guaranteed" |

**Cảnh báo bắt buộc:** tuyệt đối không dùng nhánh `gemini-3.x-flash` —
`temperature`/`topK`/`topP` bị **deprecated và bị BỎ QUA** ở
`gemini-3.7-flash`, `3.6-flash`, `3.5-flash-lite`. Dùng nhầm là `temperature=0`
thành no-op và ta viết một câu sai vào paper.

Trục nội dung description (`description_style`):

| Option | Ghi chú |
|---|---|
| `manual` | Bản viết tay đang có trong `config/config.yaml`. **Control bắt buộc** |
| `llm_short` | 1 câu, giống độ dài bản manual — so sánh công bằng nhất |
| `llm_morphology` | Ép LLM tả đặc điểm hình thái (nhân, mật độ, kiến trúc mô) |
| `llm_multi` | Nhiều description/lớp rồi lấy trung bình text embedding |

### 2.2 VLM (cần có **cả** image encoder và text encoder)

Ở Protocol B, VLM này là encoder **xuyên suốt** — coverage kernel, `probe_v`, và
cả linear head lúc đánh giá đều dùng nó (§1.1).

| Option | HF repo | License | dim | Gated | Ghi chú |
|---|---|---|---|---|---|
| **CONCH** | `MahmoodLab/CONCH` | CC-BY-NC-ND-4.0 | 512 | **có** | Mặc định đã chọn |
| KEEP | `Astaxanthin/KEEP` | MIT | 768 | không | Zero-shot CRC tốt nhất nhóm ungated, init từ UNI |
| QuiltNet-B-16-PMB | `wisdomik/QuiltNet-B-16-PMB` | MIT | 512 | không | Bản B-32 đạt 88.38% NCT-CRC-100K |
| PLIP | `vinid/plip` | — | 512 | không | 87.88% NCT-CRC-100K, baseline patho-CLIP được cite nhiều nhất |

**Về gated:** không phải rào cản — xin quyền trên HF rồi thêm vào đầu notebook:

```python
import os
from huggingface_hub import login
login(HF_TOKEN)                              # token cá nhân, set khi import lên Kaggle
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

Ba điều cần biết về CONCH — không ảnh hưởng việc chạy, nhưng ảnh hưởng lúc
viết bài và lúc set transform:

- License **CC-BY-NC-ND**: non-commercial, no-derivatives. Ràng buộc lúc publish
  weight/model dẫn xuất, không ràng buộc việc dùng để chạy thí nghiệm. LoRA
  adapter trên CONCH có thể bị coi là derivative — kiểm tra trước khi định
  release adapter.
- HF yêu cầu email cơ quan là email **chính** của account, không nhận gmail.
- **Input resolution của CONCH không tra được** từ model card, GitHub README hay
  abstract arXiv — chỉ chắc chắn embedding 512-d. Phải đọc `open_clip` config
  `conch_ViT-B-16` hoặc phần methods của bài Nature Medicine trước khi set
  transform. Đừng đoán 224 rồi để nó âm thầm resize sai.

Các model ungated còn lại giữ trong bảng làm phương án thay thế nếu CONCH gặp
vấn đề license hoặc access, không phải để chạy hết.

**Không đổi `vlm_primary`/`vlm_secondary` trong config** — hai khoá đó là của
baseline CODAPath, đổi là làm baseline khác paper. Thêm khoá riêng cho pipeline
này.

### 2.3 Biến text+image thành `U_n`

| Option (`round1_weight`) | Công thức | Trực giác |
|---|---|---|
| `uniform` | `U ≡ 1` | Control = MaxHerding thuần (§3) |
| `text_maxsim` | `1 − minmax(max_c cos(img, text_c))` | Không khớp rõ lớp nào ⇒ đáng gán nhãn |
| `text_margin` | `1 − minmax(cos_top1 − cos_top2)` | Nằm giữa hai lớp ⇒ đáng gán nhãn |
| `text_entropy` | entropy của `softmax(cos / τ)` | Cần thêm trục `τ` |

`text_maxsim` và `text_margin` **không tương đương**, và trực giác không quyết
định được cái nào tốt hơn — phải chạy cả hai rồi báo cáo cái nào thắng.

**Margin ở đây là margin thuần** `1 − (p₁ − p₂)` trên `softmax(cos/τ)`, KHÔNG
phải công thức `(1−margin)·(1+JSD)` của baseline CODAPath. Lý do: JSD trong
CODAPath là giữa `probs` và **one-hot của chính argmax của nó** — đó là thước đo
độ "nhọn" của phân phối, không liên quan gì đến JSD visual↔cell ở vòng 2–5 dù
trùng tên. Dùng margin thuần thì cả pipeline có đúng một định nghĩa margin,
giống hệt `visual_margin` ở vòng 2–5, và hệ số `(1+JSD)` không âm thầm quay lại
dưới một cái tên giờ đã mang nghĩa khác.

Cái "tương tự CODAPath" chỉ là **phần lấy text embedding** (và loss phụ ở §6).
Objective vẫn là UHerding: CODAPath dùng cosine trần làm kernel (không Gaussian,
không σ) và kết hợp bằng `(1−α)·coverage + α·U` — cộng, không phải
`Σ U_n · gain`. Đừng để người đọc tưởng ta mượn cả objective.

Trục phụ: `prompt_templates` (đã có 5 mẫu trong config) — dùng 1 mẫu hay
ensemble trung bình cả 5.

---

## 3. Không cold start — chỉ coverage vòng 1

Vòng 1 `U ≡ 1`, objective thành MaxHerding thuần. Lúc này chỉ cần **một model
encode ảnh**, không cần text tower.

| Option (`visual_backbone`) | HF repo | dim | Ghi chú |
|---|---|---|---|
| **DINOv2-base** | `facebook/dinov2-base` | 768 | Mặc định hiện tại. Là protocol chung để so sánh sampler ⇒ **giữ nguyên cho bảng baseline** |
| DINOv2-large | `facebook/dinov2-large` | 1024 | Cùng họ, kiểm tra kết quả có phụ thuộc kích thước backbone |
| KEEP (image tower) | `Astaxanthin/KEEP` | 768 | Pathology; dùng được cả ở §2 nên tiết kiệm 1 lần extract |
| QuiltNet-B-16 (image tower) | `wisdomik/QuiltNet-B-16-PMB` | 512 | Pathology |
| PLIP (image tower) | `vinid/plip` | 512 | Pathology |

Lưu ý: đổi backbone là đổi không gian feature ⇒ **không so trực tiếp** với bảng
đang có. Muốn claim "backbone pathology tốt hơn DINOv2" thì phải chạy lại **tất
cả** sampler trên backbone mới, không chỉ scalpel.

Trục feature lấy ra từ backbone (`visual_pool`): `cls` (đang dùng) /
`mean_patch` / `cls + mean_patch` nối lại (nhớ normalize từng nửa).

---

## 4. CellViT → một vector đại diện cho mỗi patch

CellViT trả về `n_i` cell embedding cho patch `i` (`n_i` = 0 với patch không có
nhân). Cần gộp thành 1 vector.

### 4.1 Cách gộp (`cell_pooling`)

| Option | Công thức | Phân biệt được gì |
|---|---|---|
| **`mean`** | trung bình có trọng số confidence | Mặc định. Chỉ moment 1 |
| `mean_unweighted` | trung bình đơn thuần | Ablation: confidence có tác dụng gì |
| `rff` | **KDE** — kernel mean embedding xấp xỉ bằng random Fourier features | Phân biệt 2 bag **cùng moment 1 nhưng khác mode** — cái `mean` không thấy |
| `moments` | mean ⊕ std ⊕ log(count) ⊕ mean-confidence | Rẻ, có thêm độ phân tán và mật độ tế bào |
| `max` | max-pool từng chiều | Nhấn cell cực trị thay vì cell điển hình |
| `attention` | trọng số học được | Cần tham số học ⇒ chỉ hợp với §6, không dùng cho sampling thuần |

Trục con của `rff`: `rff_dim ∈ {32, 64, 128}`, `rff_bandwidth` (null = median
heuristic). **Bandwidth này là của KDE trên cell, khác hoàn toàn `σ` của
coverage kernel** — hai thứ trùng tên khái niệm, đừng lẫn.

### 4.2 Trọng số cho từng cell (`cell_weight`)

| Option | Trọng số của cell j |
|---|---|
| **`confidence`** | `type_prob` từ CellViT (mặc định) |
| `uniform` | 1 |
| `area` | diện tích nhân — nhân lớn đóng góp nhiều hơn |
| `confidence × area` | kết hợp |

### 4.3 Nguồn embedding (`cell_source`)

| Option | Là gì |
|---|---|
| **`cellvit_embedding`** | token CellViT tại vị trí nhân (mặc định) |
| `crop_dino` | crop từng nhân, mask nền, cho qua DINO. Cache đã có sẵn cả hai |

### 4.4 Patch không có nhân nào (`missing_impute`)

| Option | Ghi chú |
|---|---|
| **`mean`** | hướng trung bình của mọi cell view hợp lệ ⇒ patch thành "điển hình", không thắng coverage nhờ việc lạ |
| `zero` | Ablation. Vector 0 có cosine 0 với **mọi** hàng kể cả hàng 0 khác ⇒ greedy chọn thừa chúng. Luôn đọc `missing=` trong log trước khi tin |

Trục `reliability_mode`: `valid` (0/1) hay `mean_confidence` — quyết định `ρ`
trong công thức blend ở §5.

---

## 5. Disagreement thành `U_n` (vòng 2..5)

### 5.1 Đo độ bất đồng (`divergence`)

| Option | Chặn [0,1] | Ghi chú |
|---|---|---|
| **`jsd`** | có | Mặc định, đối xứng |
| `kl_visual_cell` | không | **Không đối xứng** ⇒ đây và dòng dưới là 2 option khác nhau. Phải minmax |
| `kl_cell_visual` | không | |
| `tv` | có | Total variation, ít nhạy với đuôi phân phối |
| `hellinger` | có | Nhạy hơn TV |
| `top1_disagree` | có | 0/1 khi argmax khác nhau. Thô nhưng dễ giải thích ⇒ control tốt |

Cái nào không chặn trên thì minmax, và **ghi rõ đã minmax** (minmax trên vector
hằng số trả toàn 0 — xem `CLAUDE.md`).

Trục `calibrate`: có/không temperature-scale từng probe trước khi so. Nếu không
calibrate thì divergence phần lớn phản ánh độ over-confident khác nhau của hai
head chứ không phải bất đồng thật.

### 5.2 Kết hợp với margin (`weight_combine`)

| Option | `U_n` |
|---|---|
| **`rho_blend`** | `ρ·D + (1−ρ)·margin_v` (mặc định) |
| `product` | `D · margin_v` |
| `mean_margin` | `ρ·D + (1−ρ)·(margin_v + margin_c)/2` — dùng cả `margin_c`, hiện **đang bỏ không** |
| `max_margin` | `ρ·D + (1−ρ)·max(margin_v, margin_c)` |
| `sum_all` | `a·D + b·margin_v + c·margin_c`, 3 trọng số |
| `visual_margin` | `margin_v` thuần = **ablation UHerding, control bắt buộc** |

`product` khác `rho_blend` về bản chất: product bằng 0 khi **một trong hai**
bằng 0, blend thì không. Ở budget nhỏ điều này đổi hẳn tập chọn.

---

## 6. Training sau khi đủ budget

Xem §1.1 cho định nghĩa hai protocol. Nhắc lại phần liên quan đến training:

- **Protocol A**: DINOv2 đóng băng + linear probe, chạy cho mọi sampler. Đây là
  bảng so sánh sampler và **giữ nguyên không đổi**.
- **Protocol B**: CONCH xuyên suốt, **chọn mẫu lại bằng chính CONCH** (không tái
  dùng chỉ số của A — chọn trong không gian CONCH là một phần của pipeline đang
  được đánh giá), rồi train linear + LoRA + center loss trên CONCH.

Vì B chọn mẫu lại, nó **không** là ablation của A và không được đặt cùng bảng.
B so với **kết quả tốt nhất của A**: pipeline-với-pipeline.

### 6.1 Train cái gì (`train_mode`)

| Option | Ghi chú |
|---|---|
| **`linear_only`** | Backbone đóng băng, chỉ linear head. **Control bắt buộc** = Protocol A |
| `lora` | Backbone + LoRA adapter + linear head |
| `last_block` | Mở khoá vài block cuối, không LoRA |
| `full` | Finetune toàn bộ. Ở 200 ảnh gần như chắc chắn tệ hơn — chạy để có một điểm "thua" |

Trục LoRA: `lora_target ∈ {qv, qkvo, mlp}`, `lora_blocks ∈ {last4, last8, all}`,
`lora_r ∈ {2,4,8,16}`, `lora_alpha ∈ {r, 2r}`, `lora_dropout ∈ {0, 0.1}`.
`lora_r=0` ≡ `linear_only` và phải cho kết quả **giống hệt** — đó là một test.

Trục backbone cho Protocol B: giống bảng §3. Muốn claim "framework tốt hơn
DINO" thì bắt buộc có cấu hình `backbone=dinov2-base` để cô lập biến.

### 6.2 Loss phụ ngoài cross-entropy (`aux_loss`)

| Option | Cần gì | Ghi chú |
|---|---|---|
| **`none`** | — | **Control bắt buộc** |
| **`center`** ⬅ | 1 center/lớp học cùng | **Đã chốt: thử cái này trước.** Không cần ≥2 mẫu/lớp trong batch nên chạy được ở mọi budget |
| `supcon` | nhãn + augmentation | Cần **≥2 mẫu/lớp trong batch** — ở budget 25 với 9 lớp là không đủ, phải guard chứ không được lặng lẽ sai |
| `triplet` | mining + `margin` | Dễ collapse ở ít mẫu |
| `arcface` / `cosface` | margin góc | Ổn định hơn triplet |
| `probe_consistency` | 2 probe visual/cell | Chính ý "loss giữa hai probe". Dùng lại `training/probe.py::_consistency_penalty` |
| `center + supcon` | | Cho phép cộng nhiều loss, mỗi cái một weight |

**Loss phụ tính trên đâu** (`aux_feature`) — trục dễ bỏ sót nhất, ba cái này
**không tương đương**: `backbone_output` / `projection` (thêm MLP 2 lớp rồi mới
tính, đúng cách SupCon làm) / `logits`.

**Trọng số** `aux_weight ∈ {0, 0.01, 0.1, 0.5, 1.0}`. `aux_weight=0` phải cho
kết quả giống hệt `aux_loss=none` — cũng là một test.

**LoRA và loss phụ CHỈ ở lần train cuối.** Trong 5 vòng AL, hai probe
visual/cell là linear probe trên feature **đóng băng**, train xong dùng để tính
`U_n` rồi bỏ đi — không LoRA, không loss phụ, không cập nhật backbone. Backbone
chỉ được finetune **một lần duy nhất**, sau khi đã chọn đủ B ảnh.

Vì thế `probe_consistency` không xung đột với disagreement: lúc loss phụ chạy
thì việc chọn mẫu đã xong hẳn, embedding dùng để chọn mẫu không bao giờ bị loss
phụ làm thay đổi. Điều kiện để giữ tính chất này là **thứ tự**: chọn hết → mới
train. Nếu sau này ai đó cập nhật backbone giữa các vòng (LoRA rồi vòng sau chọn
trên embedding mới) thì xung đột kích hoạt, và khi đó `probe_consistency` với
`disagreement` không được dùng cùng lúc. Với `center` (đã chốt) thì không có vấn
đề này trong cả hai trường hợp.

### 6.3 Điều phải báo cáo dù kết quả xấu

Ở 25–50 ảnh, LoRA **rất có thể tệ hơn** linear probe. Đó là kết quả thật và
phải nằm trong bảng đủ 8 budget, không phải lý do bỏ cấu hình. Chỉ báo cáo
budget mà LoRA thắng là cherry-pick.

---

## 7. Data khi training

| Trục | Option | Ghi chú |
|---|---|---|
| `augment` | **`none`** | Control bắt buộc |
| | `flip_rotate` | H&E không có chiều "đúng" ⇒ flip/rotate 90° là vô hại về mặt sinh học. An toàn nhất |
| | `flip_rotate_scale` | Thêm random resized crop |
| | `color_jitter` | Cẩn thận: dự án này từng có một phương pháp chết vì **stain shortcut** |
| | `stain_augment` | Jitter trong không gian H&E (Macenko/Vahadane) thay vì RGB — đúng hơn về mặt miền, nhưng thêm dependency |
| `class_balance` | **`weighted_ce`** | Đang dùng: `class_weights` nghịch đảo tần suất |
| | `none` | Ablation |
| | `oversample` | Lặp mẫu lớp thiểu số |
| `val_split` | **`none`** | Không early-stop. Ở budget 25 thì val chỉ ~5 ảnh, gần như vô nghĩa |
| | `holdout_20` | Tách 20% **từ chính budget** để early-stop. **Tuyệt đối không early-stop trên test set** — đó là rò rỉ |
| `epochs` | 100 (mặc định) / 50 / 200 | Tương tác mạnh với `augment` và `train_mode` |

---

## 8. Cách quét — không chạy tích Descartes

Tổng số tổ hợp là hàng chục nghìn. **Không chạy hết được và không nên.**

Quét **một trục mỗi lần**, mọi trục khác để default, so với chính default đó.
Đây là ablation one-factor-at-a-time: không tìm được tương tác giữa các trục,
nhưng là thứ duy nhất khả thi với GPU Kaggle và là thứ reviewer đọc được.

Chỉ quét 2D cho cặp thật sự nghi có tương tác:

| Cặp | Vì sao |
|---|---|
| `lora_r` × `aux_weight` | Cả hai điều tiết capacity; tối ưu của cái này phụ thuộc cái kia |
| `augment` × `train_mode` | Augment chỉ có ý nghĩa khi backbone được train |
| `cell_pooling` × `divergence` | Pooling đổi phân phối logits của probe cell ⇒ đổi thang divergence |

Bắt buộc có trong mọi bảng: cấu hình default làm mốc, và control tắt hẳn cơ chế
đang xét (`round1_weight=uniform`, `uncertainty_mode=visual_margin`,
`lora_r=0`, `aux_loss=none`, `augment=none`).

Mỗi trục mới phải vào `_default_run_name` và `sampling/specs.py` — xem
`CLAUDE.md`, đó là quy tắc chung chứ không phải việc của plan này.

---

## 9. Thứ tự đề xuất

Xếp theo (giá trị / chi phí):

1. **§5** divergence + weight_combine — chi phí ~0, chỉ là công thức trên logits
   đã có. Không cần extract lại gì.
2. **§4** cell_pooling + cell_weight — đọc lại cache có sẵn, không chạy CellViT.
3. **§2** LLM + VLM vòng 1 — contribution 1, chưa implement gì. Cần 1 lần
   extract VLM feature, không cần CellViT.
4. **§6 + §7** Protocol B — nhiều trục nhất nên cần nhiều thời gian quét nhất,
   nhưng chỉ chạy trên run đã có.
5. **§3** đổi backbone — đắt vì phải chạy lại **mọi** sampler để so sánh còn
   công bằng.
