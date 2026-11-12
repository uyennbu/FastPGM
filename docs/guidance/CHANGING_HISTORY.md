# Nhật Ký Thay Đổi Mã Nguồn C/C++ (Changing History)

Tài liệu này ghi lại ngắn gọn, đầy đủ các chỉnh sửa trên mã nguồn C/C++ (`.h` và `.cpp`) của dự án **FastPGM**.

---

## 1. Mục Đích Thay Đổi
Bổ sung tính năng chạy cấu trúc mạng với **3 mức độ song song hóa (Granularities of Parallelism)** để tái hiện thực nghiệm trong bài báo gốc (*IPDPS 2022, Mục IV-A & V-C, Fig. 2*):
1. **CI-level parallelism** (`-p 0`, mặc định gốc): Phân phối song song các nhóm phép kiểm định độc lập điều kiện (CI test) qua *dynamic work pool*.
2. **Edge-level parallelism** (`-p 1`): Phân phối song song theo từng cạnh ở mỗi độ sâu (*depth*) bằng OpenMP.
3. **Sample-level parallelism** (`-p 2`): Duyệt cạnh và CI tuần tự; song song hóa việc đếm tần số trên tập mẫu (*samples*) với chỉ thị `#pragma omp atomic`.

> **Tiêu chí thực hiện:** Giữ thay đổi ở mức tối thiểu (*minimal invasive*), bảo toàn 100% logic và hiệu năng mặc định ban đầu của tác giả khi không dùng cờ `-p`.

---

## 2. Bảng Tổng Hợp Các File Đã Chỉnh Sửa

| STT | File | Loại thay đổi | Chức năng chính |
| :---: | :--- | :---: | :--- |
| 1 | `include/fastpgm/Parameter.h` | Header | Thêm biến cấu hình `int parallelism_type;` |
| 2 | `src/fastpgm/Parameter.cpp` | Source | Khởi tạo mặc định và phân tích cú pháp cờ `-p` từ dòng lệnh |
| 3 | `src/fastpgm/main.cpp` | Source | In thông tin loại song song và truyền tham số xuống API gọi học cấu trúc |
| 4 | `include/fastpgm/fastpgm_api/fastpgm_api.h` | Header | Thêm tham số `int parallelism_type = 0` vào hàm API |
| 5 | `src/fastpgm/fastpgm_api/fastpgm_api.cpp` | Source | Truyền `parallelism_type` khi khởi tạo đối tượng `PCStable` |
| 6 | `include/fastpgm/structure_learning/PCStable.h` | Header | Bổ sung biến thành viên và cập nhật constructor của lớp `PCStable` |
| 7 | `src/fastpgm/structure_learning/PCStable.cpp` | Source | Cài đặt logic phân phối luồng cho Edge-level và Sample-level |
| 8 | `include/fastpgm/structure_learning/CellTable.h` | Header | Khai báo biến luồng toàn cục `extern int sample_level_threads;` |
| 9 | `src/fastpgm/structure_learning/CellTable.cpp` | Source | Cài đặt song song hóa duyệt mẫu với `#pragma omp atomic` khi đếm tần số |

---

## 3. Chi Tiết Thay Đổi Theo Từng Nhóm File

### A. Tham số dòng lệnh & Điểm vào (CLI & Main)
- **`include/fastpgm/Parameter.h`**:
  - Khai báo trường: `int parallelism_type; // 0: CI-level, 1: Edge-level, 2: Sample-level`.
- **`src/fastpgm/Parameter.cpp`**:
  - Gán giá trị mặc định trong constructor: `parallelism_type = 0;`.
  - Bổ sung case `'p'` trong `ParseParameters`: nhận giá trị `0`, `1` hoặc `2` từ tùy chọn `-p`.
- **`src/fastpgm/main.cpp`**:
  - In ra màn hình dòng trạng thái `parallelism type = ...` khi chạy Job 0 (PC-Stable).
  - Truyền `param.parallelism_type` vào lời gọi hàm `BNSL_PCStable` và `BNL_PCStable`.

### B. Tầng API Giao Tiếp (`fastpgm_api`)
- **`include/fastpgm/fastpgm_api/fastpgm_api.h`**:
  - Thêm tham số `int parallelism_type = 0` vào nguyên mẫu hàm `BNSL_PCStable` và `BNL_PCStable`.
- **`src/fastpgm/fastpgm_api/fastpgm_api.cpp`**:
  - Khởi tạo `PCStable` với tham số `parallelism_type`: `new PCStable(network, alpha, 1000, parallelism_type);`.

### C. Thuật Toán Cốt Lõi (`PCStable`)
- **`include/fastpgm/structure_learning/PCStable.h`**:
  - Thêm thuộc tính `int parallelism_type = 0;`.
  - Cập nhật hàm khởi tạo: `PCStable(Network *net, double a, int d = 1000, int par_type = 0);`.
- **`src/fastpgm/structure_learning/PCStable.cpp`**:
  - Trong constructor: lưu lại `parallelism_type = par_type;`.
  - Trong `StructLearnByPCStable`:
    - Gán `sample_level_threads = (parallelism_type == 2) ? num_threads : 1;`.
    - Ở Level 0: Nếu là Sample-level (`-p 2`), các cạnh chạy tuần tự (`num_threads_level0 = 1`) để việc song song hóa diễn ra bên trong phép kiểm định; nếu là CI-level hoặc Edge-level, vòng lặp Level 0 chạy đa luồng như cũ.
  - Trong `SearchAtDepth` (từ Level 1 trở đi):
    - **Edge-level (`-p 1`)**: Sử dụng `#pragma omp parallel for num_threads(num_threads) schedule(dynamic)` phân phối trực tiếp các cạnh cho các luồng. Mỗi luồng phụ trách kiểm tra toàn bộ điều kiện trên cạnh đó bằng vòng lặp `while (CheckEdge(...))`.
    - **Sample-level (`-p 2`)**: Duyệt cạnh tuần tự 1 luồng; các luồng được đẩy vào bên trong hàm đếm mẫu.
    - **CI-level (`-p 0`)**: Giữ nguyên cơ chế hàng đợi động *dynamic work pool* (stack kích thước 128 edges) của mã gốc.
  - Trong `StructLearnCompData`: Reset lại `sample_level_threads = 1` sau khi hoàn thành để tránh tác dụng phụ.

### D. Bảng Tần Số Contingency Table (`CellTable`)
- **`include/fastpgm/structure_learning/CellTable.h`**:
  - Thêm khai báo: `extern int sample_level_threads;`.
- **`src/fastpgm/structure_learning/CellTable.cpp`**:
  - Định nghĩa biến: `int sample_level_threads = 1;`.
  - Bổ sung nhánh xử lý trong 5 hàm duyệt dữ liệu mẫu:
    1. `Counts2D::FillTable` (kiểm định bậc 0)
    2. `Counts3D::CountLevel1` (kiểm định bậc 1 đơn lẻ)
    3. `Counts3D::CountLevelN` (kiểm định bậc $N$ đơn lẻ)
    4. `Counts3DGroup::CountLevel1` (kiểm định bậc 1 theo nhóm)
    5. `Counts3DGroup::CountLevelN` (kiểm định bậc $N$ theo nhóm)
  - Khi `sample_level_threads > 1`: Duyệt tập mẫu bằng `#pragma omp parallel for num_threads(sample_level_threads)` và tăng giá trị ô đếm bằng `#pragma omp atomic n[...]++`.
  - Khi `sample_level_threads == 1`: Chạy vòng lặp thông thường để đạt hiệu năng tối đa mà không chịu chi phí đồng bộ `atomic`.

---

## 4. Cách Sử Dụng Cờ `-p` Sau Khi Thay Đổi

```bash
# 1. CI-level (Mặc định gốc - Fast-BNS)
./build/fastpgm -j 0 -m 0 -t 4 -p 0 -g 1 -a 0.05 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000

# 2. Edge-level parallelism
./build/fastpgm -j 0 -m 0 -t 4 -p 1 -g 1 -a 0.05 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000

# 3. Sample-level parallelism
./build/fastpgm -j 0 -m 0 -t 4 -p 2 -g 1 -a 0.05 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000
```

