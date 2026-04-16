# CSC4005 - Lab 1 Report (NEU MLP)

## 1. Mục tiêu
Mục tiêu của Lab 1 là huấn luyện mô hình MLP cho bài toán phân loại lỗi bề mặt thép trên bộ dữ liệu NEU, đồng thời thực hiện thí nghiệm có kiểm soát để:
- Theo dõi learning curves (train/val loss, train/val accuracy).
- So sánh ít nhất 3 cấu hình hyperparameter.
- Chọn cấu hình tốt nhất dựa trên validation set, sau đó mới báo cáo test set.

## 2. Thiết lập thí nghiệm
- Môi trường: `csc4005-dl`
- Dữ liệu: thư mục `data`
- Số lớp: 6 (`Crazing`, `Inclusion`, `Patches`, `Pitted_Surface`, `Rolled-in_Scale`, `Scratches`)
- Kích thước ảnh: 64x64
- Batch size: 32
- Epoch: 20
- Patience: 5
- Augmentation: bật (`--augment`)

### 2.2 Kiến trúc mạng MLP
Mô hình được định nghĩa trong `src/model.py` với cấu hình mặc định:
- Input: ảnh grayscale resize về `64x64`, flatten thành vector `4096` chiều.
- Hidden layers: `4096 -> 512 -> 256 -> 64`.
- Mỗi hidden layer gồm: `Linear + ReLU + Dropout`.
- Output layer: `Linear(64 -> 6)` tương ứng 6 lớp lỗi.
- Hàm mất mát: `CrossEntropyLoss`.

Với batch đầu vào `x`, logits đầu ra được tính bởi chuỗi biến đổi:

$$
z_1 = W_1 x + b_1,\quad h_1 = \text{Dropout}(\text{ReLU}(z_1))
$$

$$
z_2 = W_2 h_1 + b_2,\quad h_2 = \text{Dropout}(\text{ReLU}(z_2))
$$

$$
z_3 = W_3 h_2 + b_3,\quad h_3 = \text{Dropout}(\text{ReLU}(z_3))
$$

$$
	ext{logits} = W_4 h_3 + b_4
$$

### 2.1 Các run đã chạy
1. **Run A (baseline_adamw)**
- optimizer: adamw
- lr: 0.001
- weight_decay: 0.0001
- dropout: 0.3

2. **Run B (run_b_sgd)**
- optimizer: sgd
- lr: 0.01
- weight_decay: 0.0001
- dropout: 0.3

3. **Run C (run_c_strong_reg)**
- optimizer: adamw
- lr: 0.0005
- weight_decay: 0.001
- dropout: 0.5

## 3. Kết quả thực nghiệm

### 3.1 Bảng so sánh chính
| Run | best_val_acc | best_val_loss | test_acc | test_loss |
|---|---:|---:|---:|---:|
| baseline_adamw | 0.4185 | 1.4993 | 0.3815 | 1.4957 |
| run_b_sgd | **0.4519** | **1.4405** | **0.4481** | **1.4158** |
| run_c_strong_reg | 0.3296 | 1.6286 | 0.3444 | 1.6021 |

Nguồn số liệu: `metrics.json` trong từng thư mục output.

### 3.2 Đường cong học tập và output
- Run A: `outputs/baseline_adamw/curves.png`
- Run B: `outputs/run_b_sgd/curves.png`
- Run C: `outputs/run_c_strong_reg/curves.png`

**Run A - Learning Curves**
![Run A Curves](outputs/baseline_adamw/curves.png)

**Run B - Learning Curves**
![Run B Curves](outputs/run_b_sgd/curves.png)

**Run C - Learning Curves**
![Run C Curves](outputs/run_c_strong_reg/curves.png)

### 3.3 W&B dashboard (3 run)
- Project: `csc4005-lab1-neu-mlp`
- Run ID baseline: `vqjsjtja`
- Run ID run_b_sgd: `kwo7wr5u`
- Run ID run_c_strong_reg: `s1z9tl2i`

Sau khi login W&B, đồng bộ bằng `wandb sync` để hiện đầy đủ trên dashboard online.

Các artifact đầy đủ:
- `best_model.pt`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `metrics.json`

## 4. Phân tích

### 4.1 So sánh theo tiêu chí validation
- **Run B (SGD)** đạt `best_val_acc` cao nhất và `best_val_loss` thấp nhất, cho thấy khả năng tổng quát hóa tốt nhất trên tập validation trong 3 cấu hình.
- **Run A (AdamW baseline)** cho kết quả trung bình, ổn định nhưng thấp hơn Run B.
- **Run C (regularization mạnh hơn)** giảm hiệu năng rõ rệt so với Run A/B.

### 4.2 Dấu hiệu overfitting / underfitting
- **Overfitting mạnh** chưa xuất hiện rõ ở Run B vì val metric vẫn cải thiện đáng kể trong giai đoạn cuối.
- **Run C có xu hướng underfitting nhẹ đến vừa**: train/val accuracy đều thấp hơn, val loss cao hơn 2 run còn lại. Nguyên nhân có thể do regularization quá mạnh (dropout 0.5 + weight_decay 0.001) kết hợp learning rate thấp.

### 4.3 AdamW và SGD trong thí nghiệm này
- Với cấu hình đã chọn, **SGD (Run B)** cho kết quả tốt hơn AdamW baseline.
- Kết luận này chỉ đúng trong phạm vi thí nghiệm hiện tại (kiến trúc MLP, dữ liệu hiện tại, số epoch và các hyperparameter đã cố định).

## 5. Chọn best model
Best config được chọn: **run_b_sgd**.

Lý do chọn:
1. `best_val_acc` cao nhất (0.4519).
2. `best_val_loss` thấp nhất (1.4405).
3. `test_acc` cũng cao nhất (0.4481) sau khi đã chọn theo validation.
4. Learning curve cho thấy tiến triển học tập tốt hơn so với baseline và cấu hình regularization mạnh.

## 6. Đánh giá test set (sau khi chọn best config)
Sử dụng model tốt nhất từ `outputs/run_b_sgd/best_model.pt`:
- test accuracy: **0.4481**
- classification report: `outputs/run_b_sgd/metrics.json`
- confusion matrix: `outputs/run_b_sgd/confusion_matrix.png`
- ví dụ dự đoán đúng: `outputs/run_b_sgd/examples_correct.png`
- ví dụ dự đoán sai: `outputs/run_b_sgd/examples_wrong.png`

**Confusion Matrix (Best Model - run_b_sgd)**
![Run B Confusion Matrix](outputs/run_b_sgd/confusion_matrix.png)

**Ví dụ dự đoán đúng (Best Model)**
![Run B Correct Examples](outputs/run_b_sgd/examples_correct.png)

**Ví dụ dự đoán sai (Best Model)**
![Run B Wrong Examples](outputs/run_b_sgd/examples_wrong.png)

## 7. Trả lời câu hỏi tự kiểm tra
1. **Vì sao cần tách train/validation/test?**
- Train để học tham số, validation để chọn mô hình/hyperparameter, test để ước lượng hiệu năng cuối cùng khách quan.

2. **Validation dùng để làm gì?**
- Dùng để theo dõi tổng quát hóa trong lúc huấn luyện, điều chỉnh hyperparameter, chọn best config mà không đụng vào test.

3. **Vì sao phân loại nhiều lớp dùng Cross-Entropy?**
- Vì Cross-Entropy đo chênh lệch giữa phân phối xác suất dự đoán và nhãn thật one-hot, phù hợp với bài toán softmax đa lớp.

4. **Khi nào AdamW có lợi hơn SGD?**
- Khi cần hội tụ nhanh ở giai đoạn đầu, ít phải tinh chỉnh learning rate thủ công, hoặc khi mô hình/dữ liệu khiến SGD khó tối ưu.

5. **Dấu hiệu overfitting là gì?**
- Train accuracy tăng cao, train loss giảm nhưng val loss tăng hoặc val accuracy ngừng tăng/giảm.

6. **W&B giúp ích gì khi so sánh nhiều cấu hình?**
- Lưu và đối chiếu toàn bộ cấu hình + metric + curves theo từng run, giúp kết luận dựa trên bằng chứng thay vì cảm tính.

## 8. Hướng cải thiện tiếp theo
1. Dùng scheduler `plateau` để giảm learning rate khi val loss chững.
2. Tinh chỉnh từng yếu tố theo kiểu controlled experiment (chỉ đổi 1 biến mỗi lần).
3. Thử điều chỉnh `hidden_dims` để tăng năng lực biểu diễn vừa phải, tránh tăng quá mạnh gây overfit.
