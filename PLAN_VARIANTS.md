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
   Image ──▶ visual encoder ──▶ visual embedding ──▶ probe_v ──┐
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

**Ba trục lớn ⇒ ba câu hỏi tách biệt:**

| Câu hỏi | Trục | So với cái gì |
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

| Option | HF repo | License | dim | Ghi chú |
|---|---|---|---|---|
| **KEEP** | `Astaxanthin/KEEP` | MIT | 768 | Mặc định. Zero-shot CRC tốt nhất nhóm ungated, init từ UNI |
| **QuiltNet-B-16-PMB** | `wisdomik/QuiltNet-B-16-PMB` | MIT | 512 | Bản B-32 đạt 88.38% NCT-CRC-100K |
| **PLIP** | `vinid/plip` | — | 512 | 87.88% NCT-CRC-100K. Baseline pathology-CLIP được cite nhiều nhất |
| **BiomedCLIP** | `microsoft/BiomedCLIP-...` | MIT | 512 | General-biomedical chứ không thuần patho; yếu nhất nhóm |

Tất cả đều ungated, 224×224, tải được trên Kaggle không cần xin quyền.

**Không đổi `vlm_primary`/`vlm_secondary` trong config** — hai khoá đó là của
baseline CODAPath, đổi là làm baseline khác paper. Thêm khoá riêng `vlm_round1`.

### 2.3 Biến text+image thành `U_n`

| Option (`round1_weight`) | Công thức | Trực giác |
|---|---|---|
| `uniform` | `U ≡ 1` | Control = MaxHerding thuần (§3) |
| `text_maxsim` | `1 − minmax(max_c cos(img, text_c))` | Không khớp rõ lớp nào ⇒ đáng gán nhãn |
| `text_margin` | `1 − minmax(cos_top1 − cos_top2)` | Nằm giữa hai lớp ⇒ đáng gán nhãn |
| `text_entropy` | entropy của `softmax(cos / τ)` | Cần thêm trục `τ` |

`text_maxsim` và `text_margin` **không tương đương**, và trực giác không quyết
định được cái nào tốt hơn — phải chạy cả hai rồi báo cáo cái nào thắng.

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

Hai protocol tách biệt, **không trộn bảng**:

- **Protocol A** (đang có): visual encoder đóng băng + linear probe. Chạy cho
  mọi sampler. Đây là bảng so sánh sampler, giữ nguyên không đổi.
- **Protocol B** (mới): ± finetune backbone + loss phụ. Chạy trên **tập chỉ số
  đã chọn sẵn** từ Protocol A (đọc `<run>_selected_budget_<B>.pt`), **không
  chọn mẫu lại**. Nhờ vậy B không ảnh hưởng câu hỏi "sampler nào tốt hơn".

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
| `center` | 1 center/lớp học cùng | Rẻ và ổn định nhất ở ít mẫu |
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

**Xung đột phải nhớ:** `probe_consistency` kéo hai probe đồng ý sẽ triệt tiêu
đúng cái divergence mà §5 chọn mẫu dựa trên. Trong Protocol B thì hai việc ở
hai thời điểm khác nhau (chọn mẫu xong trước khi train) nên **không xung đột** —
nhưng chỉ vì B không chọn mẫu lại. Gộp vào vòng AL là xung đột quay lại.

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
