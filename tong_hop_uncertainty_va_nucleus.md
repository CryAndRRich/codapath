# Tổng hợp: Uncertainty & Nucleus Contribution

---

## Phần 1: Cách tính Uncertainty

### 1.1 SCALPEL (dự án hiện tại)

**File**: [scalpel.py](file:///d:/codapath/codapath/sampling/scalpel.py)

SCALPEL dùng **2 probe** huấn luyện mỗi round trên dữ liệu đã gán nhãn:
- **Morphology probe `p_M`**: linear probe trên DINOv2 features (768d)
- **Stain probe `p_S`**: linear probe trên stain descriptor (14d)

**Các thành phần uncertainty:**

#### a) Vacuity (explore signal — tốt cho cold-start)
```
λ = κ × L / (H + 0.1)       # H = entropy của p_M
vacuity = L / (λ + L)
```
- Cao khi entropy `p_M` lớn → model thiếu evidence → cần explore thêm
- Dùng công thức Dirichlet đơn giản, **không có module learnable**

#### b) Margin uncertainty (exploit signal — tốt ở mid/high budget)
```
margin_uncertainty = 1 - (top1_prob - top2_prob)
```
- Cao khi model phân vân giữa 2 lớp → mẫu gần biên quyết định
- Tính từ output `p_M` (morphology probe)

#### c) Stain shortcut discount
```
s_shortcut = p_S(x)[argmax p_M(x)]     # stain probe ủng hộ dự đoán bao nhiêu?
reconcile = margin_uncertainty × (1 - s_shortcut)
```
- Nếu stain probe cũng đồng ý với morphology probe → mẫu "dễ" theo shortcut → giảm ưu tiên
- **Đây là phần đang bị dừng** vì stain phụ thuộc scanner, không phải nhãn

#### d) Kết hợp qua schedule
```
W = (1 - t) × normalize(vacuity) + t × normalize(reconcile)
```
- `t` tăng từ 0→1 qua các round: ban đầu explore (vacuity) → sau exploit (reconcile)
- Có adaptive cap: nếu stain gap không có ý nghĩa → cap `t` ở 0.4

**Code cụ thể** ([dòng 314-363](file:///d:/codapath/codapath/sampling/scalpel.py#L314-L363)):
```python
# Round 0: pure coverage (chưa có probe)
if r == 0:
    W = torch.ones(N)

# Round 1+:
else:
    # Train 2 probe
    probe_m = train_linear(feat_np[sel], y, ...)      # morphology
    probe_s = train_linear(stain_z[sel], y, ...)      # stain

    # Tính uncertainty
    vac = _vacuity(p_m, kappa)                         # explore
    unc = _margin_uncertainty(p_m)                     # margin
    s_shortcut = p_s[np.arange(N), pred_m]             # stain discount
    reconcile = unc * (1.0 - s_shortcut)               # discounted margin

    # Kết hợp
    W = (1-t) * normalize(vac) + t * normalize(reconcile)
```

---

### 1.2 CEC (WACV 2025) — "Entropy bị lệch, phải sửa trước khi dùng"

CEC dùng VLM zero-shot (CLIP) và phát hiện **entropy thô bị sai** (miscalibrated):

#### Bước 1: Hiệu chỉnh entropy bằng contextual prior
- VLM hay tự tin quá mức vào lớp phổ biến → entropy thấp giả → bỏ sót mẫu uncertain thật
- **Cách sửa**: Mỗi lớp tính "mức tự tin trung bình" từ top-N mẫu confident nhất → chia xác suất cho prior → tính lại entropy

```
P(c) = trung bình p(c|x) trên top-N mẫu confident nhất của lớp c
p̃(c|x) = [p(c|x) / P(c)] / Σ [p(c'|x) / P(c')]     # chuẩn hoá lại
H_cal(x) = -Σ p̃(c|x) × log p̃(c|x)                   # entropy đã hiệu chỉnh
```

#### Bước 2: Neighbor uncertainty (lọc outlier)
- Mẫu uncertain nhưng xung quanh toàn mẫu confident → có thể là outlier/noise → không nên chọn
```
U_neighbor(x) = trung bình có trọng số entropy_cal của k láng giềng gần nhất
U_total(x) = H_cal(x) + β × U_neighbor(x)
```

#### Bước 3: Weighted K-Means để đảm bảo diversity
- K-Means với trọng số = `U_total`, chọn mẫu gần centroid mỗi cluster

**So với SCALPEL:**
| | SCALPEL | CEC |
|---|---|---|
| Uncertainty thô | Margin (top1 - top2) | Entropy |
| Hiệu chỉnh bias | ❌ Không | ✅ Chia cho contextual prior |
| Lọc outlier | ❌ Không (dựa vào coverage) | ✅ Neighbor uncertainty (k-NN) |
| Nguồn | Linear probe (cần nhãn) | VLM zero-shot (không cần nhãn) |

---

### 1.3 SaE (CVPR 2026) — "Không chỉ hỏi có chắc không, mà hỏi TẠI SAO không chắc"

SaE tách uncertainty thành **2 loại khác nhau** bằng phân phối Dirichlet:

#### Vacuity — "chưa thấy bao giờ"
```
u_vac = K / S = K / (Σ e_k + K)
```
- Cao khi tổng evidence `E_total` rất nhỏ → model mù tịt → dữ liệu hiếm, OOD
- Ví dụ: Đưa ảnh UFO cho model phân loại Chó/Mèo → evidence = 0 cho cả 2 lớp

#### Dissonance — "bằng chứng xung đột"
```
u_diss = Σ_k [b_k / Σ_{j≠k} b_j] × Bal(b_k, b_j)
```
- Cao khi evidence mạnh nhưng chia đều cho nhiều lớp → model phân vân
- Ví dụ: Ảnh Chó lai Sói → evidence Chó = 10, evidence Sói = 10

#### Similarity Evidence Head (SEH)
- MLP nhẹ nhận (image feature, similarity vector) → dự đoán evidence strength `e_k`
- Train bằng loss kép: khớp `1/λ` với CE loss, khớp `λ` với `1/entropy`
- **Hoạt động từ round 0** (không cần nhãn, chỉ cần VLM similarity)

**So với SCALPEL:**
| | SCALPEL | SaE |
|---|---|---|
| Vacuity | Công thức cố định `L/(κL/(H+0.1)+L)` | Dirichlet đầy đủ với learnable SEH |
| Dissonance | ❌ Không tách — gộp chung vào margin | ✅ Tách riêng bằng belief balance |
| Cold-start | Round 0 = pure coverage (không có uncertainty) | Round 0 đã có uncertainty từ SEH |
| Ước evidence | Dùng entropy thô | MLP học từ dữ liệu |

---

### 1.4 Có nên update cách tính uncertainty trong SCALPEL?

> [!IMPORTANT]
> **Không nên.** Giữ nguyên margin + vacuity hiện tại.

**Lý do:**

1. **Margin đã competitive**: Nhìn bảng kết quả — Margin sampling thuần đứng top 1-3 ở mid-high budget trên cả 3 dataset, thắng cả BADGE, DropQuery, UHerding

2. **Contribution bị pha loãng**: Nếu vừa đổi uncertainty formula vừa đổi stain→nucleus → reviewer hỏi "improvement đến từ đâu?" → phải ablation thêm, khó justify

3. **Hyperparameter nổ tổ hợp**: Mỗi thành phần thêm = thêm hyperparameter. SCALPEL đã có `kappa`, `t_schedule`, `gap_thresh`, `t_cap`... Thêm calibration (N, β) hoặc Dirichlet (SEH architecture, loss weights) → rất khó tune

4. **Đơn giản hơn = dễ giải thích**: Paper contribution rõ ràng: "thay stain bằng nucleus" — 1 ý tưởng, 1 ablation, 1 story

**Kết luận:**
```
Giữ nguyên:  margin = 1 - (top1 - top2)         ← đã competitive
Giữ nguyên:  vacuity = L/(κL/(H+0.1) + L)       ← explore signal, đơn giản
Thay đổi:    stain_discount → nucleus_discount    ← contribution chính
```

CEC và SaE nên để trong **Related Work** để so sánh, không cần tích hợp vào method.

---

## Phần 2: Contribution — Thay Stain bằng Nucleus

### 2.1 Tại sao bỏ stain?

Từ [PLAN.md](file:///d:/codapath/codapath/PLAN.md#L92):
> Tín hiệu stain-shortcut phụ thuộc **scanner/máy chụp** hơn là nhãn ảnh — biến thiên phản ánh khác biệt thiết bị/lô nhuộm, không phải nội dung chẩn đoán. Dùng nó làm discount tức là đang tối ưu theo trục thiết bị chứ không phải trục nhãn — **sai mục tiêu**.

### 2.2 Tại sao chọn nhân?

Từ [PLAN.md](file:///d:/codapath/codapath/PLAN.md#L83) và [AL note](file:///d:/codapath/codapath/AL%20note%206_8.md#L12-L16):
- Hình thái nhân (kích thước, mật độ, pleomorphism) là tín hiệu chẩn đoán mà **pathologist dùng trực tiếp**
- DINOv2 (backbone ảnh tự nhiên) không được huấn luyện ưu tiên đặc trưng nhân
- Nucleus probe bổ sung **thông tin mà DINOv2 không encode** — đó là argument

Về câu hỏi trong note: *"Attention cũng đã tập trung vào phần này rồi?"*
→ DINOv2 attention tập trung vì nhân salient, nhưng **không phân biệt** loại nhân, không đo pleomorphism, không đếm số nhân. Nucleus probe bổ sung thông tin **có cấu trúc** mà attention map không capture.

### 2.3 Cách thay đổi trong code

Trong SCALPEL hiện tại ([dòng 324-326](file:///d:/codapath/codapath/sampling/scalpel.py#L324-L326)):
```python
# HIỆN TẠI: stain probe
probe_s = train_linear(stain_z[sel], y, L, ...)
p_s = probe_s.predict_proba(stain_z, device)
s_shortcut = p_s[np.arange(N), pred_m]
```

Thay thành:
```python
# MỚI: nucleus probe
probe_n = train_linear(nucleus_z[sel], y, L, ...)
p_n = probe_n.predict_proba(nucleus_z, device)
n_shortcut = p_n[np.arange(N), pred_m]
```

Trong đó `nucleus_z` là nucleus features (z-scored), thay thế `stain_z`.

**Ý nghĩa discount đổi từ:**
- Cũ: "stain giải thích được dự đoán → mẫu dễ theo shortcut màu"
- Mới: "nucleus morphology đơn giản đã giải thích được dự đoán → DINOv2 không cần thêm mẫu này"

### 2.4 Cách segment nhân

Có 3 model phổ biến cho nucleus segmentation trong pathology:

| Model | Đặc điểm | Output | Phân loại nhân |
|---|---|---|---|
| **HoVer-Net** | Dự đoán horizontal/vertical map để tách instance | Instance mask + type | ✅ 5 loại (epithelial, inflammatory, spindle, misc, necrotic) |
| **Cellpose** | Dựa trên gradient flow, generalizable | Instance mask | ❌ Không phân loại |
| **StarDist** | Dự đoán star-convex polygon, nhanh | Instance mask | ❌ Không phân loại |

> [!TIP]
> **Gợi ý dùng HoVer-Net** vì có phân loại loại nhân → thêm được feature "tỷ lệ epithelial/inflammatory/..." — pathologist dùng thông tin này để chẩn đoán.

### 2.5 Nucleus features — dùng gì?

#### Cách 1: Handcrafted features (GỢI Ý BẮT ĐẦU VỚI CÁI NÀY)

Từ instance mask, tính **đặc trưng thống kê** mỗi patch:

| Feature | Ý nghĩa lâm sàng | Số chiều |
|---|---|---|
| Số lượng nhân | Mật độ tế bào (cellularity) | 1 |
| Diện tích trung bình | Kích thước nhân | 1 |
| Std diện tích | Pleomorphism/anisonucleosis — dấu hiệu ác tính | 1 |
| CV diện tích (std/mean) | Hệ số biến thiên — đo đa hình nhân | 1 |
| Tỷ lệ loại nhân (nếu HoVer-Net) | Thành phần epithelial/inflammatory/stromal/necrotic | 4-5 |
| Eccentricity trung bình | Hình dạng nhân (tròn vs dài) | 1 |
| Mật độ nhân (count/area) | Mật độ tương đối | 1 |

→ Vector **~10-15 chiều**, tương đương stain descriptor 14 chiều → **drop-in replacement**, code gần như không đổi.

**Ưu điểm:**
- ✅ Interpretable — giải thích được cho reviewer
- ✅ Chi phí thấp — segment offline 1 lần, tính stats nhanh
- ✅ Drop-in — chỉ đổi `extract_stain_features` → `extract_nucleus_features`

#### Cách 2: Intermediate features từ model segment

Lấy feature map từ layer trung gian HoVer-Net/StarDist → global average pooling → vector ~256-512d.

**Ưu điểm:** Giàu thông tin hơn handcrafted
**Nhược điểm:** Black box, khó giải thích, cần đổi input dim

#### Cách 3: Crop nhân → extract DINOv2 riêng

Segment → crop từng nhân → DINOv2 feature mỗi nhân → aggregate thành 1 vector/patch.

**Ưu điểm:** Tận dụng DINOv2 ở mức nhân
**Nhược điểm:** Chi phí cao nhất, pipeline phức tạp, khó giải thích

### 2.6 So sánh 3 cách

| | Cách 1: Handcrafted | Cách 2: Intermediate | Cách 3: Crop nhân |
|---|---|---|---|
| **Drop-in thay stain** | ✅ Dễ nhất | ⚠️ Cần đổi dim | ⚠️ Pipeline phức tạp |
| **Interpretable** | ✅ Rất rõ | ❌ Black box | ❌ Black box |
| **Chi phí** | Thấp | Trung bình | Cao |
| **Paper argument** | Dễ: "pathologist dùng đặc điểm này" | Khó giải thích | Khó giải thích |

> [!IMPORTANT]
> **Gợi ý**: Bắt đầu với **Cách 1** (handcrafted). Nếu kết quả tốt → đã đủ. Nếu không → thử Cách 2 sau. Cách 1 có lợi thế lớn nhất về interpretability và dễ justify contribution trong paper.

### 2.7 Pipeline tổng thể

```
[Ảnh H&E patch]
       │
       ├──→ DINOv2 ──→ features 768d ──→ morphology probe p_M ──→ margin + vacuity
       │                                                              │
       └──→ HoVer-Net ──→ instance mask + type ──→ handcrafted       │
                            │                       features ~15d    │
                            │                           │            │
                            │                    nucleus probe p_N   │
                            │                           │            │
                            │                    n_shortcut score    │
                            │                           │            │
                            └───────────────────────────┘            │
                                                                     │
                                    reconcile = margin × (1 - n_shortcut)
                                                                     │
                                    W = (1-t) × vacuity + t × reconcile
                                                                     │
                                              greedy coverage selection
```
