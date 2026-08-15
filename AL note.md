Phương pháp mới có thể chia \-\> nhiều gđ: có thể là mới ở quy trình chọn mẫu/quy trình huấn luyện

Nếu muốn đánh giá 1 cái mới của tôi thì ph cố định mọi thành phần kia

So sánh fair: 

- cùng tập training \-\> cluoc training tốt hay kh  
- thu thập thêm, tăng cường data, sinh thêm data sẽ giúp \-\> để cách học là giống nhau, eval dữ liệu đầu vào

→ Có thể có nhiều tầng đánh giá khác nhau 

Ý tưởng nhân:

- Chỉ lấy vùng nhân được crop/segment ra để giúp học phân loại  
- Lấy feature map cve hợp lý hơn → Vde/Câu hỏi đặt ra: Mình giúp mô hình học nhanh hơn th chứ về lý thuyết thì attention cũng đã tập trung vào phần này r; còn nếu mình thêm nhánh này thì đang như kiểu focus vào việc học nhanh hơn/tối ưu chi phí hơn về giai đoạn học. (nhấn mạnh các đặc điểm riêng đấy lúc đưa ra arguement chứ ko nhất thiết phải là tốt hơn tuyệt đối ở mọi phương diện)  
- có thể có nhiều cách dùng cái segment: dùng mỗi ảnh nhân dc cắt ra, kết hợp ảnh RGB ban đầu với cái seg mask,...

Ý tưởng multi-scale:

- lưu ý khi upscale, cẩn thận k vỡ ảnh. tìm hiểu về các thư viện/thuật toán lieen quan đến việc upscale ảnh sao cho ảnh sau khi upscale trông smooth hơn: nearest neighbor, bilinear interpolation, bicubic interpolation  
- có thể kết hợp thông tin đa phân giải theo kiểu: Vùng to chứa thông tin ngữ cảnh cho vùng nhỏ, vùng nhỏ học một cách chi tiết.   
- ý tưởng đa phân giải thì có thể trên ảnh hoặc trên cả kiến trúc nữa. 

Note 8/8
Tiến hành chạy thử nghiệm so sánh kết quả (Hạn 11/08)

Thống nhất chung:
- Coverage luôn dùng công thức của bài Uncertainty Herding
- Chạy các budget 25, 50, 75, 100, 125, 150, 175, 200 với dataset PathMNIST
- Chia quá trình chọn thành 5 vòng, sau mỗi vòng chọn xong thì train lại. Vòng đầu mặc định chọn bằng Coverage
- Score = Uncertainty x Coverage trong đó cả hai cái đều chuẩn hoá trước về [0, 1]

Luồng gốc: Ảnh đi vào DINO trả về feats dùng để tính Coverage (UHerding), feats cho qua 1 lớp Linear để tính Uncertainty

(Minh Hải) Thử nghiệm uncertainty: Thay công thức tính Uncertainty bằng công thức Margin, Entropy và CEC

(Kiên) Thử nghiệm multiscale: Ảnh upscale x4 x9, crop => 3 ảnh (gốc, x4, x9) đưa qua DINO và 3 lớp Linear riêng ứng với mỗi ảnh rồi concat/cộng đầu ra các lớp Linear. 

Về tính Uncertainty:
- Tính bằng disagreement giữa đầu ra của 3 lớp Linear ứng với 3 ảnh 
- Thêm 1 lớp Linear nhận vào concat/cộng của đầu ra 3 lớp Linear rồi dùng đầu ra để tính bằng bằng Margin (Hoặc có thể concat/cộng đầu ra mỗi ảnh từ DINO luôn, không cần 3 lớp Linear nữa)

Về tính Coverage:
- Dùng feats ảnh gốc
- Dùng kết hợp feats của 3 ảnh

(Dũng) Thử nghiệm nhân tế bào: Dùng model CellViT nhận vào ảnh RGB trả về segment map hoặc cell embedding
- Dùng segment map lấy ra phần nhân tế bào từ ảnh RGB và đưa mỗi phần đó vào DINO lấy feats, đi qua 1 lớp Linear lấy đầu ra dùng để tính Uncertainty bằng Margin
- Dùng cell embedding đi qua 1 lớp Linear, ảnh RGB đi qua DINO + 1 Linear. 

Về tính Uncertainty:
- Tính bằng disagreement giữa đầu ra hai lớp (1 lớp của cell và 1 lớp của ảnh)
- Thêm 1 lớp Linear nhận vào concat/cộng của đầu ra 2 lớp Linear rồi dùng đầu ra để tính bằng Margin

Về tính Coverage:
- Dùng full ảnh gốc đi qua DINO lấy feats

(Nam Hải) Thử nghiệm nhân tế bào: Dùng model CellViT nhận vào ảnh RGB trả về segment map hoặc cell embedding
- Dùng cell embedding đi qua 1 lớp Linear, ảnh RGB đi qua DINO + 1 Linear. 

Về tính Uncertainty:
- Tính bằng đầu ra lớp Linear ảnh RGB bằng Margin

Về tính Coverage:
- Dùng đầu ra của DINO
- Dùng cell embedding
- Tạo 1 lớp linear nhận vào cả đầu ra của DINO và cell embedding


Note 14/8
Tiến hành chạy thử nghiệm so sánh kết quả (Hạn 18/08)

Thống nhất chung:
- Chạy các budget 25, 50, 75, 100, 125, 150, 175, 200 với dataset PathMNIST
- Chia quá trình chọn thành 5 vòng, sau mỗi vòng chọn xong thì train lại
- Ai thử nghiệm chạy liên quan đến Uncertainty Herding check lại branch namhai của repo, file sampling/uncertainty_herding.py, tao có sửa lại code cho khớp với official repo

(Dũng) 
Chạy song song hai nhánh:
- Nhánh 1 dùng visual embedding DINO, dùng Uncertainty Herding tính score, rank
- Nhánh 2 dùng cell embedding CellViT (cộng trung bình, KDE), dùng Uncertainty Herding tính score, rank
- Tổng hợp hai nhánh lại, dùng score (cộng tổng, trung bình) hoặc dùng rank (RRF hoặc phương pháp nào khác)

Chạy vẫn chia 5 vòng, mỗi vòng cần chọn A ảnh để tổng hợp đủ thì mỗi nhánh chạy lấy 2A ảnh

Chạy lại cái Disagreement cũ nhưng bây giờ chạy theo chuẩn Uncertainty Herding hơn, method gốc dùng Margin thì thay bằng Disagreement

Tìm hiểu thêm có thể ngoài loss CE lúc train hai probe mỗi nhánh thì có thể bổ sung 1 loss phụ nào đấy không (cả trường hợp hai nhánh hợp song song với tuần tự)

(Minh Hải)
Chạy tuần từ hai nhánh 1 => 2:
- Nhánh 1 dùng cell embedding CellViT (cộng trung bình, KDE), dùng Uncertainty Herding tính score
- Nhánh 2 dùng visual embedding DINO, dùng Uncertainty Herding tính score

Chạy vẫn chia 5 vòng, nhánh 1 thì lọc ra 1/2 ảnh có điểm cao nhất, nhánh 2 từ 1/2 ảnh mới lọc từ nhánh 1 chọn ra A ảnh 

Tìm hiểu thêm có thể ngoài loss CE lúc train hai probe mỗi nhánh thì có thể bổ sung 1 loss phụ nào đấy không (cả trường hợp hai nhánh song song với tuần tự)

(Kiên)
Dùng VAE nhận đầu vào là visual embedding concat cell embedding (cộng trung bình, KDE)
Xây dựng graph y hệt bài SARGraphAL (paper, code), để ý bài này nó là sequential active learning, chọn 1 ảnh xong là update lại luôn (khác với mấy baseline chọn 1 lần cả budget hoặc chia budget thành 5 vòng)
Về tính score để chọn điểm thì thử nghiệm
- Chạy theo cách chọn của bài SARGraphAL (Laplace learning + Margin)
- Bài SARGraphAL chỉ tính uncertainty không tính coverage nên thử bổ sung 1 cơ chế tính coverage liên quan đến graph xem (Effective resistance/commute-time distance hoặc Personalized PageRank từ tập có nhãn)
- Chạy Uncertainty Herding nhưng giữ công thức coverage và tính uncertainty theo SARGraphAL
- Chạy Uncertainty Herding nhưng giữ công thức uncertainty và tính coverage bằng cơ chế liên quan đến graph tạo từ bài SARGraphAL (có thể là thay kernel Gaussian gốc của UHerding bằng kernel graph sẵn của SARGraphAL)

Ngoài ra tìm hiểu thêm về mục đích của VAE, nếu chỉ dùng để nén hay giảm chiều vector thì liệu có cách nào khác không. Đọc thêm bài AGCL, đại ý là chọn điểm chưa nhãn có representation khác biệt nhất so với tập đã nhãn sẽ tối đa hoá coverage, dùng attention, xem xem có áp dụng được gì không

(Nam Hải)
Dùng VAE nhận đầu vào là visual embedding và cell embedding (cộng trung bình, KDE), bổ sung một loss phụ align để representation visual và cell embedding cùng một ảnh tương đồng nhau
Hoặc tạo 2 VAE độc lập nhận vào visual hoặc cell embedding tạo thành 2 graph độc lập. Kết hợp 2 graph lại theo bài DEUCE

Về tính score để chọn điểm thì thử nghiệm
- Chạy theo cách chọn của bài SARGraphAL (Laplace learning + Margin)
- Chạy Uncertainty Herding nhưng giữ công thức coverage và tính uncertainty theo SARGraphAL
