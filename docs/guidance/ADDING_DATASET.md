# Hướng dẫn Thêm Bộ Dữ Liệu Mới & Chạy Thực Nghiệm (Ví dụ: `Hepar2`)

Trong bài báo [SOURCE_PAPER.md](../papers/SOURCE_PAPER.md) (Mục V), tác giả thực nghiệm trên 8 mạng Bayes chuẩn từ kho **bnlearn**, tiêu biểu là **Hepar2** (70 nút, 123 cạnh) với 3 kích thước mẫu: `5.000`, `10.000` và `15.000`.

Dưới đây là 3 bước ngắn gọn để tải, tạo dữ liệu cho một mạng mới (ví dụ `Hepar2`) và chạy học cấu trúc / suy diễn trên **FastPGM**.

---

## Bước 1: Tạo thư mục và tải file mạng chuẩn (`.bif`)

Mọi dữ liệu trong FastPGM đều phải đặt trong một thư mục con của `dataset/`.

```bash
# 1. Tạo thư mục riêng cho bộ dữ liệu hepar2
mkdir -p /home/uyenuyen/c_project/FastPGM/dataset/hepar2
cd /home/uyenuyen/c_project/FastPGM/dataset/hepar2

# 2. Tải file mạng chuẩn hepar2.bif từ bnlearn và giải nén
wget https://www.bnlearn.com/bnrepository/hepar2/hepar2.bif.gz
gunzip hepar2.bif.gz
```
> **Mẹo:** Bạn có thể tải các mạng khác trong bài báo (`insurance`, `munin1`, `diabetes`, `link`, `munin2`, `munin3`) theo cùng cấu trúc link tại: https://www.bnlearn.com/bnrepository/

---

## Bước 2: Sinh dữ liệu mẫu CSV từ file `.bif`

FastPGM yêu cầu dữ liệu dạng CSV, trong đó **dòng đầu tiên là tên các biến theo đúng thứ tự trong file `.bif`**.
* **Tập Train** (`hepar2_s5000`, `hepar2_s10000`, `hepar2_s15000`): Dữ liệu đầy đủ, không khuyết giá trị.
* **Tập Test suy diễn** (`testing_hepar2_1k_p20_csv`): Giữ lại cột nhãn (cột 0) và 20% biến quan sát ngẫu nhiên (80% biến còn lại để trống `,,`).

Chạy đoạn script Python sau tại thư mục gốc `/home/uyenuyen/c_project/FastPGM` (cài đặt trước: `pip install pgmpy pandas`):

```python
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
import numpy as np

# 1. Đọc mạng hepar2.bif và lấy đúng thứ tự biến
reader = BIFReader("dataset/hepar2/hepar2.bif")
model = reader.get_model()
variables = list(reader.get_variables())
sampler = BayesianModelSampling(model)

# 2. Sinh tập Train: 5.000, 10.000 và 15.000 mẫu
for n in [5000, 10000, 15000]:
    df = sampler.forward_sample(size=n, show_progress=False)[variables]
    df.to_csv(f"dataset/hepar2/hepar2_s{n}", index=False)

# 3. Sinh tập Test suy diễn: 1.000 mẫu, quan sát 20% biến (p20)
test_df = sampler.forward_sample(size=1000, show_progress=False)[variables]
rng = np.random.default_rng(42)
mask = rng.random((len(test_df), len(variables) - 1)) > 0.20
test_df.iloc[:, 1:] = test_df.iloc[:, 1:].mask(mask, "")
test_df.to_csv("dataset/hepar2/testing_hepar2_1k_p20_csv", index=False)
```

Sau khi chạy xong, thư mục `dataset/hepar2/` sẽ có:
```text
dataset/hepar2/
├── hepar2.bif                   # Mạng gốc dùng để chấm điểm SHD
├── hepar2_s5000                 # Tập train 5.000 mẫu
├── hepar2_s10000                # Tập train 10.000 mẫu
├── hepar2_s15000                # Tập train 15.000 mẫu
└── testing_hepar2_1k_p20_csv    # Tập test suy diễn 1.000 mẫu (20% bằng chứng)
```

---

## Bước 3: Chạy thực nghiệm Học cấu trúc & Suy diễn

Đứng tại thư mục `build/`:
```bash
cd /home/uyenuyen/c_project/FastPGM/build
```

### 1. Học cấu trúc mạng (Tái hiện Mục V của bài báo)
* **Chạy tuần tự 1 luồng (`Fast-BNS-seq` - Bảng III):**
  ```bash
  ./fastpgm -j 0 -m 0 -t 1 -g 1 -a 0.05 -f1 hepar2/hepar2.bif -f2 hepar2/hepar2_s5000
  ```
* **Chạy song song 12 luồng (`Fast-BNS-par` - Bảng III):**
  ```bash
  ./fastpgm -j 0 -m 0 -t 12 -g 1 -a 0.05 -f1 hepar2/hepar2.bif -f2 hepar2/hepar2_s5000
  ```
* **Thử nghiệm theo kích thước mẫu 10.000 và 15.000 (Hình 3):**
  ```bash
  ./fastpgm -j 0 -m 0 -t 12 -g 1 -f1 hepar2/hepar2.bif -f2 hepar2/hepar2_s10000
  ./fastpgm -j 0 -m 0 -t 12 -g 1 -f1 hepar2/hepar2.bif -f2 hepar2/hepar2_s15000
  ```
* **Thử nghiệm `group_size` tối ưu của Hepar2 (`-g 8` - Hình 4):**
  ```bash
  ./fastpgm -j 0 -m 0 -t 12 -g 8 -f1 hepar2/hepar2.bif -f2 hepar2/hepar2_s10000
  ```

### 2. Chạy Học cấu trúc + Học tham số + Suy diễn Junction Tree (`-j 4 -m 1`)
Lệnh này tự động học cấu trúc từ `hepar2_s5000`, học bảng xác suất (MLE), sau đó xây cây **Junction Tree** để suy diễn trên `testing_hepar2_1k_p20_csv` và in ra độ chính xác (`accuracy`):

```bash
./fastpgm -j 4 -m 1 -t 12 -g 8 -f1 hepar2/hepar2.bif -f2 hepar2/hepar2_s5000 -f3 hepar2/testing_hepar2_1k_p20_csv
```

