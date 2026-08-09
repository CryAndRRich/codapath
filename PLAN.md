## Tổng hợp papers

### Nhóm 1: Phương pháp đề xuất

| Method | Mô tả | Venue |
|---|---|---|
| Hiện tại| Phương pháp AL cold-start cho ảnh pathology tận dụng thông tin về stain. Mỗi round huấn luyện hai lớp linear: hình thái `p_M` (đặc trưng DINOv2) cho margin uncertainty, và màu nhuộm H&E `p_S` (đặc trưng stain, 14 chiều). Kết hợp margin uncertainty với thông tin stain rồi chọn mẫu bằng greedy submodular coverage trên kernel DINOv2. Khác biệt so với các phương pháp khác: là AL đầu tiên khai thác stain (nuisance confound đặc thù ảnh H&E) làm tín hiệu chọn mẫu | - |
| **CODAPath** | Giải quyết cold-start bằng cách dùng hai Med-VLM (PLIP + BiomedCLIP) làm zero-shot prior thay random initialization. Tính uncertainty kết hợp margin và Jensen-Shannon Divergence so với nhãn one-hot, rồi chọn mẫu bằng greedy tối đa hoá tổ hợp cộng có trọng số α giữa marginal coverage và uncertainty chuẩn hoá. Sau khi có batch nhãn, fine-tune backbone kép bằng LoRA kết hợp Center Loss để chống collapse biểu diễn khi dữ liệu ít | - |

### Nhóm 2: Uncertainty Estimation

| Method | Mô tả | Venue |
|---|---|---|
| **CEC** (Calibrated Entropy-weighted Clustering) | Giải quyết vấn đề CLIP zero-shot bị entropy lệch (miscalibrated, thiên vị lớp phổ biến) bằng cách hiệu chỉnh entropy trước khi dùng làm tín hiệu uncertainty: tính prior theo ngữ cảnh cho mỗi lớp từ top-N ảnh có xác suất cao nhất của lớp đó, rồi chia xác suất dự đoán cho prior này trước khi tính lại entropy. Để tránh chọn outlier có entropy giả cao, cộng thêm "neighbor uncertainty" - trung bình có trọng số khoảng cách của entropy hiệu chỉnh của các k-nearest-neighbor trong không gian đặc trưng CLIP. Cuối cùng, để đảm bảo đa dạng trong batch, thực hiện weighted K-Means (trọng số = uncertainty tổng hợp) và chọn mẫu gần centroid mỗi cluster (không chọn mẫu uncertain nhất) để tránh outlier| WACV 2025 |
| **SaE** (Similarity-as-Evidence) | VLM zero-shot bị overconfident khi ép qua softmax cứng, nên thay vào đó dùng một Similarity Evidence Head (SEH) - MLP nhẹ nhận cả ảnh và vector similarity - để dự đoán một hệ số evidence-strength, từ đó dựng tham số Dirichlet thay cho phân phối categorical cứng. Từ Dirichlet này tách uncertainty thành vacuity (thiếu evidence tổng, ứng với lớp hiếm/chưa biết) và dissonance (evidence xung đột giữa các lớp, ứng với ranh giới quyết định mơ hồ). SEH được huấn luyện bằng loss kép: khớp 1/λ với cross-entropy loss thực nghiệm và khớp λ với nghịch đảo entropy của VLM. Đặc trưng nổi bật: hoàn toàn không cần model đã huấn luyện ở vòng đầu, khác các baseline entropy/margin/BADGE dựa trên classifier đã huấn luyện | CVPR 2026 |

### Nhóm 3: Coverage / Diversity

| Method | Mô tả | Venue |
|---|---|---|
| **BADGE** (Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds) | Gộp uncertainty và diversity vào một không gian nhúng gradient duy nhất: với mỗi mẫu chưa nhãn, tính gradient của cross-entropy loss theo trọng số lớp cuối, dùng nhãn dự đoán của model hiện tại làm nhãn giả - gradient này là tích ngoài giữa biểu diễn tầng áp cuối (đại diện/diversity) và phần dư xác suất (uncertainty: chuẩn gradient nhỏ khi model tự tin, lớn khi không chắc). Nhãn model dự đoán chính là nhãn làm cực tiểu chuẩn gradient, nên gradient thu được là chặn dưới bảo toàn cho gradient thật. Để chọn batch vừa uncertain vừa đa dạng, dùng khởi tạo k-MEANS++ trên các nhúng gradient này. | ICLR 2020 |
| **BAIT** (Gone Fishing: Neural Active Learning with Fisher Embeddings) | Nhìn active learning qua lý thuyết ước lượng (Fisher information/A-optimality): chọn batch mẫu để tối thiểu hoá vết của nghịch đảo ma trận Fisher tích lũy trên tập đã chọn nhân với ma trận Fisher trên toàn bộ pool chưa nhãn - tổng quát và mạnh hơn cách BADGE chỉ dùng gradient hạng-1 và bỏ qua thông tin Fisher toàn cục. Giải bài toán tối ưu bằng forward-backward greedy: mở rộng lên 2B ứng viên rồi cắt về B, dùng biến đổi Woodbury/trace-rotation để tính song song trên GPU mà không cần lưu ma trận Fisher đầy đủ cho mỗi mẫu. | NeurIPS 2021 |
| **TypiClust** (Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets) | Ở ngân sách thấp nên oversample vùng dễ/điển hình để học nhanh, ở ngân sách cao nên oversample vùng khó/atypical (kiểu uncertainty sampling). Ở chế độ ít nhãn, dùng biểu diễn self-supervised (SimCLR/DINO) để phân cụm K-means với số cụm = số mẫu đã label + budget (đảm bảo luôn có ít nhất budget cụm chưa phủ), rồi từ mỗi cụm chưa phủ lớn nhất chọn mẫu có typicality cao nhất - nghịch đảo khoảng cách trung bình tới k-nearest-neighbor trong cụm. Kết hợp "chọn từ cụm chưa phủ" (đa dạng) và "chọn điểm điển hình nhất trong cụm" (mật độ cao, đại diện) | ICML 2022 |
| **ActiveFT** (Active Finetuning: Exploiting Annotation Budget in the Pretraining-Finetuning Paradigm) | Tối ưu trực tiếp một mô hình tham số hoá liên tục gồm B "tâm" trong không gian đặc trưng đã chuẩn hoá, sao cho phân phối tạo bởi các tâm này gần với phân phối của toàn bộ tập dữ liệu. Sau khi tối ưu, mẫu thực tế được chọn là điểm dữ liệu gần nhất với mỗi tâm đã hội tụ | CVPR 2023 |
| **MaxHerding** (Generalized Coverage for More Robust Low-Budget Active Learning) | Tổng quát hoá ProbCover thành "generalized coverage" với kernel liên tục bất kỳ (thường Gaussian). Khi dùng kernel k-means thì quy về CoreSet/TypiClust, cho thấy các phương pháp low-budget khác chỉ là trường hợp đặc biệt. Dùng Gaussian kernel với phương sai cố định nên ổn định qua hyperparameter | ECCV 2024 |
| **UncertaintyHerding / UHerding** (One Active Learning Method for All Label Budgets) | Tổng quát hoá "generalized coverage" (MaxHerding) thành "uncertainty coverage" bằng cách nhân trọng số uncertainty của mô hình hiện tại vào hàm coverage kernel. Chọn mẫu là greedy tối đa hoá marginal gain của UCoverage (đảm bảo (1-1/e)-approximation nhờ submodularity), với hai cơ chế tự thích ứng: temperature scaling làm uncertainty gần hằng số khi label ít (tự lùi về MaxHerding ở low-budget), và radius adaptation (co lại theo số label tăng, đẩy về gần pure uncertainty sampling ở high-budget). Nhờ đó tự nội suy giữa hành vi coverage-based và uncertainty-based mà không cần biết trước budget. Bài báo còn chứng minh weighted k-means, ALFA-Mix, BADGE đều là trường hợp đặc biệt/gần đúng của UHerding | ICLR 2025 |

### Nhóm 4: Hybrid (Uncertainty + Diversity)

| Method | Mô tả | Venue |
|---|---|---|
| **TCM** (Bridging Diversity and Uncertainty in Active Learning with Self-Supervised Pre-Training: TypiClust -> Margin) | Heuristic đơn giản dựa trên quan sát: với backbone self-supervised pretrained đã đóng băng, điểm chuyển giao tối ưu giữa chiến lược diversity (TypiClust) và uncertainty (Margin) xảy ra rất sớm trong quá trình AL. Chạy TypiClust (K-means + chọn điểm điển hình nhất mỗi cụm) trong vài vòng đầu để đảm bảo phủ đa dạng, sau đó chuyển hẳn sang Margin sampling cho các vòng còn lại - số vòng dùng TypiClust đặt cố định theo tổng budget. Thực nghiệm cho thấy vượt trội cả TypiClust và Margin đứng riêng, đặc biệt trên dữ liệu long-tail, và không nhạy với việc chọn chính xác số vòng chuyển giao - đơn giản hơn hẳn các phương pháp thích ứng phức tạp như SelectAL mà không mất hiệu năng khi dùng backbone pretrained. | ICLR 2024 |
| **DropQuery** (Revisiting Active Learning in the Era of Vision Foundation Models) | Xuất phát từ khảo sát cho thấy với đặc trưng foundation model (DINOv2, OpenCLIP) chất lượng cao, không còn phase-transition từ diversity sang uncertainty - uncertainty sampling thô đã cạnh tranh được ngay từ vòng 2 nếu có thêm diversity. Gồm 3 bước: khởi tạo cold-start bằng K-means trên đặc trưng foundation model (chọn mẫu gần centroid); đo uncertainty bằng áp dropout ngẫu nhiên lên vector đặc trưng M lần (feature-level dropout) và giữ lại mẫu có trên 50% lượt dự đoán không nhất quán với dự đoán gốc (dấu hiệu gần biên quyết định); trên tập ứng viên uncertain này, chạy K-means với K=budget và chọn mẫu gần mỗi centroid để đảm bảo đa dạng. Vì chỉ cần một pass forward để trích đặc trưng (backbone đóng băng hoàn toàn), rất rẻ và không cần tinh chỉnh hyperparameter, vượt trội các baseline classic trên cả ảnh tự nhiên và ảnh y sinh out-of-domain. | TMLR 2024 |
| **CB+SQ** (Active Prompt Learning with Vision-Language Model Priors - Class-guided clustering + Selective Querying) | Giải quyết cold-start và lãng phí ngân sách của active prompt learning cho CLIP bằng hai cơ chế: class-guided clustering - nối đặc trưng ảnh với đặc trưng văn bản, rồi K-means trên đặc trưng nối này để có cụm cân bằng theo lớp ngay từ vòng đầu (không cần nhãn nào), chọn mẫu gần centroid; và selective querying - dùng ngưỡng tin cậy thích ứng theo từng lớp để tự động gán pseudo-label cho mẫu CLIP đã tự tin đúng, chỉ gửi mẫu tin cậy thấp cho annotator, cho warm-start tốt hơn hẳn đồng thời tiết kiệm đáng kể ngân sách nhãn thật. | TMLR 2025 |
| **REFINE** (Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning) | Ensemble AL hai giai đoạn: giai đoạn "progressive filtering" chạy nhiều round, mỗi round mỗi chiến lược thành viên (Margin, BADGE, TypiClust, MaxHerding, BAIT, AlfaMix, DropQuery, UHerding...) đề xuất nhiều batch trên subsample ngẫu nhiên của pool hiện tại, rồi lấy hợp của tất cả batch để tạo pool con mới nhỏ hơn - mẫu vô giá trị hiếm khi được chọn liên tục nên bị loại dần theo cấp số nhân, mẫu giá trị cao sống sót với xác suất cao. Sau nhiều round lọc (pool co lại rất mạnh), giai đoạn 2 dùng UHerding để chọn batch cuối từ pool đã lọc, đảm bảo phủ đa dạng trên chính các mẫu giá trị đã xác định. Thiết kế tách rời "đánh giá giá trị mẫu" khỏi "đảm bảo đa dạng batch" nên không chỉ đánh bại từng chiến lược đơn lẻ và các ensemble khác (SelectAL, TAILOR, AutoAL, TCM) mà còn dùng được làm bước tiền xử lý cải thiện bất kỳ chiến lược AL nào. | CVPR 2026 |

### Nhóm 5: Medical / Pathology AL

| Method | Mô tả | Venue |
|---|---|---|
| **PEAL** (Parameter-Efficient Active Learning for Foundational Models) | Chỉ ra rằng khi dùng linear probing (backbone đóng băng hoàn toàn), các phương pháp diversity dựa trên khoảng cách đặc trưng mất hiệu quả vì đặc trưng không đổi theo nhãn mới. Giải pháp là chèn LoRA (rank thấp, ~0.03% tham số) vào các lớp attention của DINOv2 và cập nhật LoRA sau mỗi round AL cùng head phân loại, làm đặc trưng "tiến hoá" theo dữ liệu đã nhãn. Thực nghiệm trên ảnh y tế (Histology, APTOS) và EuroSAT cho thấy cần ít mẫu nhãn hơn đáng kể để đạt cùng độ chính xác so với linear probing. | CVPR 2024 |
| **OpenPath** (Open-Set Active Learning for Pathology Image Classification via Pre-trained Vision-Language Models) | Mở rộng AL sang bối cảnh open-set trong pathology, nơi pool chưa nhãn chứa cả mẫu thuộc lớp mục tiêu (ID) và mẫu không liên quan (OOD). Ở vòng đầu, dùng VLM y sinh (BioMedCLIP) với prompt lớp ID và các lớp OOD do GPT-4 tự sinh để phân loại zero-shot mọi mẫu, giữ lại mẫu dự đoán thuộc ID, phân cụm K-means++ và chọn mẫu gần centroid - tránh random initialization thường gây ô nhiễm OOD nặng. Từ vòng 2, dùng Diverse Informative ID Sampling: tính prototype cho mỗi lớp ID từ dữ liệu đã nhãn, giữ mẫu chưa nhãn gần prototype nhất làm ứng viên ID, chia ngẫu nhiên thành nhiều batch nhỏ và trong mỗi batch chỉ chọn top-entropy - cân bằng giữa loại OOD, đa dạng và tính thông tin. Đạt độ tinh khiết truy vấn cao hơn baseline random hàng chục điểm phần trăm ngay vòng đầu, giải quyết đồng thời cold-start và ô nhiễm OOD. | MICCAI 2025 |

### Nhóm 6: Metric AL

| Method | Mô tả | Venue |
|---|---|---|
| **PALM** (A Predictive Model for Evaluating Sample Efficiency in Active Learning) | Mô hình toán học để đánh giá và so sánh các chiến lược AL: giả định accuracy là hàm của xác suất phủ (coverage) không gian dữ liệu bởi mẫu đã nhãn, suy ra công thức tường minh với 4 tham số có ý nghĩa - A_max (accuracy tối đa), δ (hiệu quả phủ một mẫu nhãn), α (điểm khởi đầu/cold-start), β (tốc độ tăng theo budget). Các tham số được fit bằng nonlinear least squares chỉ từ vài điểm quan sát thực nghiệm, cho phép dự đoán toàn bộ learning curve từ rất sớm mà không cần chạy hết quá trình AL. Nhờ đó, so sánh hai phương pháp AL không còn chỉ dừng ở "accuracy tại budget B cố định" mà phân tích có nguyên tắc theo từng khía cạnh riêng: coverage efficiency (δ), khả năng cold-start (α), khả năng mở rộng theo ngân sách (β) - ví dụ cho thấy pretrained embeddings (MoCov3/DINOv2) làm tăng δ đáng kể so với không có embedding. | ICCV 2025 |

---

## Code Implementation

| Method | Link |
|---|---|
| REFINE | [dal-toolbox/cleaning_the_pool/strategies.py](https://github.com/dhuseljic/dal-toolbox/blob/main/publications/cleaning_the_pool/strategies.py) |
| MedCALBench | [HiLab-git/MedCAL-Bench](https://github.com/HiLab-git/MedCAL-Bench) |
| OpenPath | [HiLab-git/OpenPath](https://github.com/HiLab-git/OpenPath) |
| UncertaintyHerding | [dal-toolbox/herding.py](https://github.com/dhuseljic/dal-toolbox/blob/main/dal_toolbox/active_learning/strategies/herding.py) |
| MaxHerding | [dal-toolbox/herding.py](https://github.com/dhuseljic/dal-toolbox/blob/main/dal_toolbox/active_learning/strategies/herding.py) |
| DCoM | [avihu111/TypiClust](https://github.com/avihu111/TypiClust) |
| ActiveLLM | [PEASEC/ActiveLLM](https://github.com/PEASEC/ActiveLLM) |
| CB+SQ | [ml-postech/active-prompt-learning](https://github.com/ml-postech/active-prompt-learning) |
| TCM | [dal-toolbox/cleaning_the_pool/strategies.py](https://github.com/dhuseljic/dal-toolbox/blob/main/publications/cleaning_the_pool/strategies.py) |
| DropQuery | [dal-toolbox/dropquery.py](https://github.com/dhuseljic/dal-toolbox/blob/main/dal_toolbox/active_learning/strategies/dropquery.py) |
| ActiveFT | [yichen928/ActiveFT](https://github.com/yichen928/ActiveFT) |
| TypiClust | [avihu111/TypiClust](https://github.com/avihu111/TypiClust) |
| BAIT | [JordanAsh/badge](https://github.com/JordanAsh/badge) |
| BADGE | [JordanAsh/badge](https://github.com/JordanAsh/badge) |
| PALM | [juliamachnio/PALM](https://github.com/juliamachnio/PALM) |

--- 
## Plan

Mathod hiện tại cần một đặc thù riêng của ảnh pathology (không transfer sang ảnh tự nhiên). Đã khảo sát và đang thử nghiệm 3 trục: **stain**, **nucleus segmentation**, **multi-scale**

## Đặc thù pathology đang khai thác

### 1. Stain (nhuộm H&E)
Màu nhuộm Hematoxylin & Eosin là nuisance variable - biến thiên theo lab/máy scan/lô hoá chất, không mang thông tin chẩn đoán trực tiếp. Độc nhất với pathology, không transfer sang ảnh tự nhiên.

### 2. Nucleus segmentation (nhân tế bào)
Hình thái nhân (kích thước, mật độ, đa hình/anisonucleosis, texture chromatin) là tín hiệu chẩn đoán mà pathologist dùng trực tiếp để đánh giá - khác biệt căn bản với DINOv2 (backbone ảnh tự nhiên, không được huấn luyện để ưu tiên đặc trưng này).

### 3. Multi-scale
Pathologist luôn quan sát đa tỉ lệ (context tổng quan ở độ phóng đại thấp + chi tiết nhân ở độ phóng đại cao) - đặc thù của ảnh mô bệnh học/WSI, khác hẳn ảnh tự nhiên vốn được chụp ở một tỉ lệ duy nhất.

## Các hướng đang thử nghiệm

### Stain - ĐÃ DỪNG

**Lý do dừng**: tín hiệu stain-shortcut đo được phụ thuộc nhiều vào **máy chụp/scanner** hơn là bản thân nhãn ảnh - biến thiên quan sát được chủ yếu phản ánh khác biệt thiết bị/lô nhuộm giữa các nguồn dữ liệu (đúng bản chất "nuisance" của stain), không phải tín hiệu gắn với nội dung chẩn đoán của từng mẫu. Dùng nó làm discount per-sample cho uncertainty tức là đang tối ưu theo trục thiết bị chứ không phải trục nhãn - sai mục tiêu.

### Nucleus segmentation - ĐANG THỬ NGHIỆM

**Ý tưởng**: segment nhân tế bào bằng một trong hai cách, rồi tận dụng thông tin thu được cho uncertainty và/hoặc coverage:

1. Dùng thẳng một **model Segment Pathology chuyên biệt** (vd HoVer-Net, Cellpose, StarDist...) -> lấy instance mask (+ type nhân nếu model hỗ trợ phân loại epithelial/inflammatory/stromal/necrotic) -> tính đặc trưng hình thái mỗi patch (số lượng nhân, diện tích trung bình/độ lệch chuẩn, hệ số biến thiên diện tích = pleomorphism/anisonucleosis, thành phần theo loại nhân...).
2. **Trích đặc trưng từ một layer trung gian** của model đó - dùng model segment song song với DINOv2, vì nó đã học sẵn ưu tiên cấu trúc nhân/mô thay vì đối tượng tự nhiên.

### Multi-scale - ĐANG THỬ NGHIỆM

**Ý tưởng**:
- Upscale ảnh gốc lên x2, x4.
- Crop nhiều vùng con theo 3 chiến lược: ngẫu nhiên, theo góc (corner crop), theo thuật toán chọn vùng (region-selection có định hướng).
- Mỗi tỉ lệ (gốc, x2, x4) đi qua một lớp linear độc lập.
- Kết hợp output của các tỉ lệ lại để tính vào uncertainty (disagreement giữa các scale = tín hiệu bất định).

---

## Điểm số hiện tại

### HistoSet (đầy đủ 20 budget)

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
| **-** | 0.5625 | 0.7293 | 0.7598 | 0.7621 | 0.7898 | 0.8057 | 0.8146 | 0.8346 | 0.8386 | 0.8480 | 0.8532 | 0.8607 | 0.8529 | 0.8702 | 0.8800 | 0.8814 | 0.8812 | 0.8741 | 0.8854 | 0.8912 |
| **Rank** | 2/12 | 1/12 | 1/12 | 2/12 | 1/12 | 1/12 | 2/12 | 1/12 | 2/12 | 1/12 | 2/12 | 3/12 | 2/12 | 3/12 | 2/12 | 2/12 | 4/12 | 4/12 | 4/12 | 4/12 |

### PathMNIST

| Phương pháp | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 | 325 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.6812 | 0.8162 | 0.8373 | 0.8673 | 0.8766 | 0.8829 | 0.8877 | 0.9047 | 0.9070 | 0.9120 | 0.9110 | 0.9156 | 0.9095 |
| Coreset | 0.5731 | 0.5153 | 0.5189 | 0.5812 | 0.6333 | 0.6543 | 0.6716 | 0.6696 | 0.6753 | 0.6613 | 0.6571 | 0.6524 | 0.6535 |
| UHerding | 0.4922 | 0.7614 | 0.8341 | 0.8383 | 0.6312 | 0.7990 | 0.7982 | 0.8487 | 0.8525 | 0.8575 | 0.8787 | 0.8805 | — |
| TCM | 0.6280 | 0.7518 | 0.8159 | 0.8430 | 0.8457 | 0.8758 | 0.8799 | 0.8834 | 0.8903 | 0.8958 | 0.8894 | 0.8976 | 0.9077 |
| REFINE | 0.7088 | 0.8046 | 0.8351 | 0.8536 | 0.8499 | 0.8742 | 0.8937 | 0.8894 | 0.8974 | 0.9070 | 0.9187 | 0.9019 | 0.9000 |
| TypiClust | 0.7302 | 0.8206 | 0.8093 | 0.8797 | 0.8727 | 0.8745 | 0.8968 | 0.8903 | 0.8916 | 0.9102 | 0.9093 | 0.8930 | 0.9011 |
| ActiveFT | 0.6515 | 0.8280 | 0.8290 | 0.8558 | 0.8642 | 0.8825 | 0.8864 | 0.8968 | 0.8808 | 0.9093 | 0.9064 | 0.8907 | 0.8943 |
| DropQuery | 0.7827 | 0.8258 | 0.8338 | 0.8500 | 0.8825 | 0.8898 | 0.8791 | 0.9025 | 0.9058 | 0.8859 | 0.9109 | 0.9159 | 0.9141 |
| Entropy | 0.6407 | 0.7085 | 0.7350 | 0.8049 | 0.8276 | 0.8458 | 0.8372 | 0.8730 | 0.8648 | 0.8421 | 0.8880 | 0.8791 | 0.8890 |
| Margin | 0.6302 | 0.6377 | 0.8462 | 0.8706 | 0.8717 | 0.9199 | 0.8950 | 0.9058 | 0.9208 | 0.9022 | 0.9116 | 0.9139 | 0.9040 |
| BADGE | 0.6054 | 0.7556 | 0.8123 | 0.8405 | 0.8404 | 0.9099 | 0.8905 | 0.8813 | 0.9019 | 0.9175 | 0.9156 | 0.9191 | 0.9078 |
| **-** | 0.7790 | 0.8326 | 0.8639 | 0.8799 | 0.8841 | 0.8840 | 0.8825 | 0.8861 | 0.8923 | 0.8939 | 0.9003 | 0.9004 | 0.9035 |
| **Rank** | 2/12 | 1/12 | 1/12 | 1/12 | 1/12 | 4/12 | 7/12 | 7/12 | 6/12 | 8/12 | 8/12 | 6/12 | 6/11* |

### SkinTissue
| Phương pháp | 25 | 50 | 75 | 100 | 125 | 150 | 175 |
|---|---|---|---|---|---|---|---|
| Random | 0.3815 | 0.6089 | 0.6878 | 0.6980 | 0.7172 | 0.7475 | 0.7592 |
| Coreset | 0.1795 | 0.2281 | 0.2963 | 0.3221 | 0.3405 | 0.3919 | 0.4049 |
| UHerding | 0.5200 | 0.5871 | 0.7179 | 0.7348 | 0.7421 | 0.7417 | 0.7649 |
| TCM | 0.5712 | 0.6462 | 0.6713 | 0.6948 | 0.7278 | 0.7428 | 0.7686 |
| REFINE | 0.5577 | 0.6335 | 0.6923 | 0.7200 | 0.7488 | 0.7645 | 0.7819 |
| TypiClust | 0.5596 | 0.6612 | 0.6758 | 0.7020 | 0.7404 | 0.7329 | 0.7520 |
| ActiveFT | 0.5053 | 0.6374 | 0.6455 | 0.6990 | 0.7240 | 0.7275 | 0.7338 |
| DropQuery | 0.6559 | 0.6895 | 0.7189 | 0.7381 | 0.7456 | 0.7512 | 0.7756 |
| Entropy | 0.3447 | 0.5659 | 0.4964 | 0.5020 | 0.5974 | 0.6130 | 0.6397 |
| Margin | 0.4677 | 0.5785 | 0.6883 | 0.7075 | 0.7507 | 0.7719 | 0.7804 |
| BADGE | 0.4308 | 0.5647 | 0.6689 | 0.7182 | 0.7298 | 0.7458 | 0.7739 |
| **-** | 0.6278 | 0.6896 | 0.7164 | 0.7169 | 0.7321 | 0.7594 | 0.7800 |
| **Rank** | 2/12 | 1/12 | 3/12 | 5/12 | 6/12 | 3/12 | 3/12 |