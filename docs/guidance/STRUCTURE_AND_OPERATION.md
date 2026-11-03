# Hướng dẫn Cấu trúc và Vận hành FastPGM

**FastPGM** là thư viện C++ (hỗ trợ đa luồng bằng OpenMP và giao diện Python) dùng để **học cấu trúc (Structure Learning)**, **học tham số (Parameter Learning)**, **suy diễn (Inference)** và **phân lớp (Classification)** trên mạng Bayes (Bayesian Networks).

---

## 1. Cấu trúc thư mục của Repo

```text
FastPGM/
├── CMakeLists.txt                  # Cấu hình build CMake chính (tạo file chạy `fastpgm`)
├── dataset/                        # Chứa toàn bộ dữ liệu đầu vào/đầu ra (bắt buộc đặt ở đây)
│   └── alarm/                      # Bộ dữ liệu mẫu của mạng ALARM
│       ├── alarm.bif / alarm.xml   # Cấu trúc và bảng xác suất gốc của mạng ALARM
│       ├── alarm_s5000             # Tập dữ liệu huấn luyện (CSV, 5.000 mẫu)
│       ├── alarm_s10000            # Tập dữ liệu huấn luyện (CSV, 10.000 mẫu)
│       ├── alarm_s15000            # Tập dữ liệu huấn luyện (CSV, 15.000 mẫu)
│       ├── testing_alarm_1k_p20_csv# Tập dữ liệu kiểm thử (CSV, 1.000 mẫu, quan sát 20% biến)
│       └── alarm_1k_pt             # Bảng xác suất chuẩn để đối chiếu kết quả suy diễn
├── include/fastpgm/                # Các file Header (.h) khai báo lớp và hàm
├── src/
│   ├── fastpgm/                    # Mã nguồn chính (.cpp) của thư viện và hàm `main.cpp`
│   │   ├── main.cpp                # Điểm vào chương trình CLI (`./fastpgm`)
│   │   ├── Parameter.cpp           # Đọc và xử lý các tham số dòng lệnh (-j, -m, -t, -g...)
│   │   ├── Dataset.cpp             # Đọc/ghi/chuyển đổi dữ liệu định dạng CSV và LibSVM
│   │   ├── Network.cpp             # Biểu diễn đồ thị mạng Bayes (Node, Edge, DAG)
│   │   ├── CustomNetwork.cpp       # Đọc mạng chuẩn từ file .bif / .xml (XMLBIF)
│   │   ├── SampleSetGenerator.cpp  # Sinh dữ liệu mẫu ngẫu nhiên từ mạng Bayes có sẵn
│   │   ├── fastpgm_api/            # Các hàm API đóng gói sẵn từng tác vụ (gọi từ main & Python)
│   │   ├── structure_learning/     # Học cấu trúc mạng (PCStable.cpp, IndependenceTest.cpp...)
│   │   ├── parameter_learning/     # Học tham số mạng bằng hợp lý cực đại (MLE)
│   │   └── inference/              # Suy diễn xác suất (JunctionTree.cpp, BruteForce.cpp...)
│   └── test/                       # Các bài kiểm thử tự động bằng GoogleTest (`BayesianNetwork_test`)
├── lib/                            # Thư viện phụ trợ (tinyxml2, ARFF, stats, gcem, googletest)
├── python/                         # Giao diện Python (đóng gói qua pybind11)
└── docs/                           # Tài liệu hướng dẫn và bài báo khoa học (papers/)
```

---

## 2. Luồng hoạt động của chương trình (`./fastpgm`)

Khi bạn chạy `./fastpgm` từ dòng lệnh, chương trình đi theo luồng sau:

```text
[Dòng lệnh CLI]
      │
      ▼
1. Parameter::ParseParameters()  (src/fastpgm/Parameter.cpp)
   Đọc các cờ: -j (Job), -m (Method), -t (Threads), -g (Group size), -f0..f4 (Files)
      │
      ▼
2. Hàm main()                    (src/fastpgm/main.cpp)
   Ghép đường dẫn thư mục `dataset/` với tên file và điều hướng theo cờ `-j`
      │
      ├──► [-j 0] BNSL_PCStable()    ──► Học cấu trúc bằng PC-Stable (Fast-BNS)
      ├──► [-j 1] BNL_PCStable()     ──► Học cấu trúc (PC-Stable) + Học tham số (MLE)
      ├──► [-j 2] BNEI_JT()          ──► Suy diễn chính xác bằng Junction Tree
      ├──► [-j 4] C_PCStable_JT()    ──► Phân lớp (PC-Stable + MLE + Junction Tree)
      └──► [-j 5] Sample_Generator() ──► Sinh tập mẫu (CSV / LibSVM) từ file mạng .xml
```

---

## 3. Cách biên dịch (Build)

Từ thư mục gốc `FastPGM`:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Sau khi biên dịch xong:
* File chạy chính nằm tại: `build/fastpgm`
* File chạy Unit Test nằm tại: `build/src/test/BayesianNetwork_test`

---

## 4. Cách sử dụng chương trình (`./fastpgm`)

> **Quy tắc quan trọng:** Mọi file dữ liệu đầu vào phải đặt trong thư mục `dataset/`. Khi truyền đường dẫn qua tham số `-f0` đến `-f4`, bạn chỉ ghi **đường dẫn tương đối bên trong `dataset/`** (ví dụ: `alarm/alarm_s5000`).

Đứng tại thư mục `build/`:

```bash
cd /home/uyenuyen/c_project/FastPGM/build
```

### Bảng tra cứu nhanh các tham số dòng lệnh

| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `-h` | Hiển thị bảng trợ giúp các tham số | — |
| `-j <số>` | Chọn tác vụ (**0**: Học cấu trúc, **1**: Học cấu trúc + tham số, **2**: Suy diễn chính xác, **4**: Phân lớp, **5**: Sinh mẫu) | `0` |
| `-m <số>` | Chọn thuật toán con tương ứng với tác vụ `-j` | `0` |
| `-t <số>` | Số luồng CPU chạy song song (OpenMP) | `1` |
| `-g <số>` | Kích thước nhóm CI test (`group_size`) trong PC-Stable | `1` |
| `-a <số>` | Mức ý nghĩa thống kê $\alpha$ cho kiểm định $G^2$ trong PC-Stable | `0.05` |
| `-q <số>` | Số lượng mẫu cần sinh (khi dùng `-j 5`) | `10000` |
| `-f0 <path>` | File mạng Bayes đầu vào (`.xml`) cho suy diễn / sinh mẫu | `alarm/alarm.xml` |
| `-f1 <path>` | File mạng chuẩn tham chiếu (`.bif`) để tính độ lệch `SHD` khi học cấu trúc | *(trống)* |
| `-f2 <path>` | File dữ liệu huấn luyện (Training set CSV) | `alarm/alarm_s5000` |
| `-f3 <path>` | File dữ liệu kiểm thử (Testing set CSV) | `alarm/testing_alarm_1k_p20_csv` |
| `-f4 <path>` | File bảng xác suất tham chiếu cho suy diễn | `alarm/alarm_1k_pt` |

---

## 5. Các câu lệnh mẫu thường dùng

### 5.1. Học cấu trúc mạng (PC-Stable / Fast-BNS)
* **Chạy mặc định (1 luồng, 5.000 mẫu ALARM):**
  ```bash
  ./fastpgm
  ```
* **Chạy đa luồng (ví dụ 8 luồng) và chấm điểm độ lệch SHD so với mạng gốc `alarm.bif`:**
  ```bash
  ./fastpgm -j 0 -m 0 -t 8 -g 1 -a 0.05 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000
  ```
* **Thử nghiệm với kích thước mẫu lớn hơn (10.000 hoặc 15.000 mẫu) và `group_size = 6`:**
  ```bash
  ./fastpgm -j 0 -m 0 -t 8 -g 6 -f1 alarm/alarm.bif -f2 alarm/alarm_s10000
  ```

### 5.2. Học cả cấu trúc và tham số (PC-Stable + MLE)
```bash
./fastpgm -j 1 -m 0 -t 8 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000
```
*(Kết quả lưu tại `dataset/alarm/alarm_s5000_struct` và `dataset/alarm/alarm_s5000_param`).*

### 5.3. Suy diễn chính xác (Exact Inference với Junction Tree)
```bash
./fastpgm -j 2 -m 1 -t 8 -f0 alarm/alarm.xml -f3 alarm/testing_alarm_1k_p20_csv -f4 alarm/alarm_1k_pt
```

### 5.4. Phân lớp dữ liệu (Classification)
```bash
./fastpgm -j 4 -m 1 -t 8 -f2 alarm/alarm_s5000 -f3 alarm/testing_alarm_1k_p20_csv
```

### 5.5. Sinh dữ liệu mẫu từ mạng Bayes có sẵn
* Sinh 5.000 mẫu ra định dạng **CSV** (`-m 1`):
  ```bash
  ./fastpgm -j 5 -m 1 -q 5000 -f0 alarm/alarm.xml
  ```
  *(File kết quả sẽ được tạo tại `dataset/alarm/sample_5000_csv`).*

### 5.6. Chạy bộ kiểm thử tự động (Unit Tests)
Cần đứng tại thư mục `build/src/test` để đường dẫn tương đối `../../../dataset/` trỏ đúng vào thư mục `dataset/`:
```bash
cd /home/uyenuyen/c_project/FastPGM/build/src/test
./BayesianNetwork_test
```

---

## 6. Hướng dẫn đọc kết quả đầu ra khi chạy `./fastpgm`

Ví dụ khi chạy lệnh học cấu trúc đa luồng với mạng tham chiếu:
```text
./fastpgm -j 0 -m 0 -t 12 -g 1 -a 0.05 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000
==================================================
Job: PC-stable for structure learning, #threads = 12
        group size = 1
        reference BN: alarm/alarm.bif
        sample set: alarm/alarm_s5000
==================================================
Finish loading training data. Number of instances: 5000. Number of variables: 37.
--------------------------------------------------
Structure learning with PC-Stable...
--------------------------------------------------
Generating undirected complete graph...
--------------------------------------------------
Removing edges level by level...
Level 0... 
# of CI-tests is 654
# remaining edges = 262, time = 0.0131068
Level 1... 
# of CI-tests is 3979
# remaining edges = 67, time = 0.0732678
Level 2... 
# of CI-tests is 4650
# remaining edges = 46, time = 0.107328
Level 3... 
# of CI-tests is 4746
# remaining edges = 44, time = 0.111237
Level 4... 
# of CI-tests is 4761
# remaining edges = 44, time = 0.11241
--------------------------------------------------
Orienting v-structure
--------------------------------------------------
Orienting other undirected edges
Learned structure has been saved in /home/uyenuyen/c_project/FastPGM/dataset/alarm/alarm_s5000_struct
--------------------------------------------------
# of CI-tests = 4761
pc-stable step 0: 8.4533e-05 s 
 (0.0743616%)
pc-stable step 1: 0.112458 s 
 (98.9264%)
pc-stable: 0.113678 s 
SHD = 5
```

### Ý nghĩa từng phần trong kết quả:

1. **Thông tin cấu hình & Dữ liệu đầu vào:**
   * `#threads = 12`, `group size = 1`: Chạy thuật toán với 12 luồng OpenMP, mỗi lần mỗi luồng lấy ra nhóm 1 phép kiểm định độc lập có điều kiện (CI test) từ hàng đợi công việc (Dynamic Work Pool).
   * `Number of instances: 5000. Number of variables: 37`: Tập dữ liệu `alarm_s5000` có 5.000 mẫu (dòng) và 37 biến/nút (cột).

2. **Quá trình khởi tạo và cắt tỉa cạnh theo từng độ sâu (`Level 0` $\rightarrow$ `Level 4`):**
   * `Generating undirected complete graph...`: Khởi tạo đồ thị vô hướng đầy đủ gồm $\frac{37 \times 36}{2} = 666$ cạnh nối tất cả các cặp biến.
   * `Level d...` (Độ sâu $d = 0, 1, 2, 3, 4$): Kích thước tập biến điều kiện $\mathcal{S}$ trong kiểm định $I(X, Y \mid \mathcal{S})$ với $|\mathcal{S}| = d$:
     * **`Level 0`** ($|\mathcal{S}| = 0$): Kiểm định độc lập biên $I(X, Y)$. Từ 666 cạnh ban đầu, loại bỏ các cạnh độc lập, còn lại `# remaining edges = 262` sau `0.0131` giây.
     * **`Level 1` $\rightarrow$ `Level 4`** ($|\mathcal{S}| = 1 \dots 4$): Lần lượt kiểm định với tập điều kiện gồm 1, 2, 3, 4 biến lân cận. Số cạnh giữ lại giảm dần: `262` $\rightarrow$ `67` $\rightarrow$ `46` $\rightarrow$ `44` $\rightarrow$ `44` cạnh. Khi không còn đỉnh nào đủ số lân cận để xét độ sâu tiếp theo, vòng lặp dừng ở `Level 4`.
   * `# of CI-tests is ...`: Số phép kiểm định độc lập cộng dồn tính đến hết Level đó.
   * `time = ...`: Thời gian cộng dồn (giây) của bước tìm khung đồ thị (Step 1) tính đến hết Level đó.

3. **Định hướng cạnh và lưu kết quả:**
   * `Orienting v-structure` (Bước 2 của PC-Stable): Xác định các cấu trúc chữ V ($X \rightarrow Z \leftarrow Y$).
   * `Orienting other undirected edges` (Bước 3 của PC-Stable): Áp dụng bộ quy tắc Meek để định hướng các cạnh vô hướng còn lại mà không tạo chu trình.
   * `Learned structure has been saved in ...`: Đường dẫn file lưu cấu trúc mạng học được (`dataset/alarm/alarm_s5000_struct`).

4. **Tổng kết thời gian & Độ chính xác:**
   * `# of CI-tests = 4761`: Tổng số phép kiểm định $G^2$ được ghi nhận. *(Lưu ý: Biến đếm `num_ci_test` trong code không dùng khóa nguyên tử `atomic` để tránh làm giảm tốc độ đa luồng; nếu chạy 1 luồng `-t 1` con số chính xác tuyệt đối là `5206` phép kiểm định).*
   * `pc-stable step 0`: Thời gian tạo đồ thị đầy đủ ban đầu (`8.45e-05` giây, chiếm ~`0.07%`).
   * `pc-stable step 1`: Thời gian chạy các phép kiểm định CI để tìm khung đồ thị (`0.112458` giây, chiếm **`98.93%`** tổng thời gian — đây chính là bước nặng nhất mà bài báo Fast-BNS tập trung tối ưu).
   * `pc-stable`: **Tổng thời gian thực thi toàn bộ thuật toán PC-Stable** (`0.113678` giây). Đây là con số dùng để báo cáo trong các bảng thực nghiệm của bài báo.
   * `SHD = 5` (*Structural Hamming Distance*): Khoảng cách khác biệt cấu trúc giữa mạng học được và mạng chuẩn `alarm.bif` (số cạnh bị thiếu, thừa hoặc ngược chiều). **SHD càng nhỏ thì cấu trúc học được càng chính xác** (bằng `0` là giống hệt mạng gốc).


