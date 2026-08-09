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
