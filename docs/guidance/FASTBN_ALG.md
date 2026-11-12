# Thuật Toán Fast-BNS (FastBN): Mã Giả và Phân Tích Mã Nguồn

> **Tài liệu tham chiếu:**
> - Bài báo: *"Fast Parallel Bayesian Network Structure Learning"* (IEEE IPDPS 2022) — tác giả Jiantong Jiang, Zeyi Wen, Ajmal Mian.
> - Mã nguồn C++ trong repo: `src/fastpgm/structure_learning/` (`PCStable.cpp`, `IndependenceTest.cpp`, `CellTable.cpp`, `ChoiceGenerator.cpp`).

---

## 1. Tổng Quan Kiến Trúc Thuật Toán Fast-BNS

Thuật toán **Fast-BNS** (Fast Parallel Bayesian Network Structure Learning) là phiên bản song song hóa hiệu năng cao của thuật toán **PC-stable** (Colombo & Maathuis, 2014) dùng để học cấu trúc mạng Bayes (DAG / CPDAG) từ dữ liệu dạng bảng.

Quy trình gồm **3 giai đoạn**:
1. **Giai đoạn 1 — Tìm khung đồ thị vô hướng (Skeleton Identification):** Khởi tạo đồ thị đầy đủ $G$, loại bỏ các cạnh độc lập có điều kiện theo từng độ sâu $d = 0, 1, 2, \dots$. Giai đoạn này chiếm **hơn 90%** tổng thời gian chạy và là trọng tâm tối ưu của Fast-BNS.
2. **Giai đoạn 2 — Định hướng cấu trúc chữ V (v-structures):** Với mỗi bộ ba không liên kết trực tiếp $(V_i - V_k - V_j)$, nếu $V_k \notin SepSet(V_i, V_j)$ thì định hướng $V_i \rightarrow V_k \leftarrow V_j$.
3. **Giai đoạn 3 — Định hướng các cạnh vô hướng còn lại (Meek Rules):** Áp dụng 3 luật Meek (Rule 1, Rule 2, Rule 3) để tránh tạo thêm v-structure mới hoặc vòng lặp (cycles), thu được đồ thị dạng CPDAG.

---

## 2. Các Đổi Mới Cốt Lõi So Với PC-stable Truyền Thống

| Cơ chế | PC-stable truyền thống (bnlearn / pcalg) | Fast-BNS (FastPGM) | Lợi ích |
| :--- | :--- | :--- | :--- |
| **Mức độ song song** | Edge-level (chia cạnh) hoặc đơn luồng | **CI-level Parallelism** | Cân bằng tải hoàn hảo, không có atomic |
| **Quản lý công việc** | Vòng lặp tĩnh qua các cạnh | **Dynamic Work Pool** (Stack 128 edges) | Luồng không bị khóa khi cạnh hoàn thành sớm |
| **Nhóm kiểm định** | 1 CI test / lần duyệt dữ liệu | **CI Grouping** (nhóm $gs$ tests cùng lúc) | Tái sử dụng dữ liệu cột của $X, Y$, giảm đọc RAM |
| **Bố trí bộ nhớ** | Hàng dọc / Row-major | **Cache-friendly Column-major** | Tránh cache miss L1/LL khi duyệt dữ liệu mẫu |
| **Sinh tập điều kiện** | Sinh & lưu trước toàn bộ vào RAM | **On-the-fly Generation** (Thuật toán Buckles & Lybanon) | Không tốn bộ nhớ lưu chỉ số tập điều kiện |

---

## 3. Mã Giả Chi Tiết Thuật Toán Fast-BNS

### Thuật Toán 1: Khung Tổng Thể Fast-BNS (Main Algorithm)

```text
Algorithm 1: Fast-BNS Main Procedure
Input : Tập biến V = {V1, V2, ..., Vn}, Tập dữ liệu D (m mẫu), Mức ý nghĩa alpha, Kích thước nhóm gs, Số luồng t
Output: Đồ thị CPDAG G = (V, E), Bảng tập phân tách SepSet

1: Khởi tạo đồ thị vô hướng đầy đủ G = (V, E) với mọi cặp (Vi, Vj)
2: Khởi tạo bảng danh sách kề Adj[Vi] = V \ {Vi} với mọi Vi in V
3: Khởi tạo bảng tập phân tách SepSet(Vi, Vj) = rỗng

// --- BƯỚC 1: XỬ LÝ MỨC 0 (Độ sâu d = 0, tập điều kiện rỗng) ---
4: PARALLEL FOR mỗi cạnh e = (Vi, Vj) in E DO
5:     Thực hiện kiểm định độc lập biên I(Vi, Vj | rỗng) bằng thống kê G^2
6:     IF p_value(Vi, Vj) > alpha THEN
7:         Đánh dấu e.need_remove = TRUE
8:         SepSet(Vi, Vj) = rỗng
9:     END IF
10: END PARALLEL FOR
11: Xóa tất cả các cạnh có need_remove = TRUE khỏi G và cập nhật Adj

// --- BƯỚC 2: XỬ LÝ CÁC MỨC TIẾP THEO (d >= 1) BẰNG CI-LEVEL PARALLELISM ---
12: Cho độ sâu d = 1
13: REPEAT
14:     Adj_copy = Snapshot(Adj)  // Đảm bảo tính chất Order-Independent của PC-stable
15:     has_more_tests = SearchAtDepth(D, G, Adj, Adj_copy, d, t, gs, alpha, SepSet)
16:     d = d + 1
17: UNTIL has_more_tests == FALSE OR d > max_depth

// --- BƯỚC 3: ĐỊNH HƯỚNG CẤU TRÚC V-STRUCTURES ---
18: OrientVStructure(G, Adj, SepSet)

// --- BƯỚC 4: ĐỊNH HƯỚNG CÁC CẠNH VÔ HƯỚNG BẰNG LUẬT MEEK ---
19: OrientImplied(G, Adj)

20: RETURN G, SepSet
```

---

### Thuật Toán 2: Tìm Khung Tại Độ Sâu $d \ge 1$ (`SearchAtDepth`)

Sử dụng cơ chế **Dynamic Work Pool** dạng ngăn xếp (stack) để phân phối linh hoạt các nhóm kiểm định CI giữa $t$ luồng:

```text
Algorithm 2: SearchAtDepth (CI-level Parallelism with Dynamic Work Pool)
Input : Dữ liệu D, Đồ thị G, Danh sách kề Adj, Bản sao Adj_copy, Độ sâu d, Số luồng t, Group size gs, alpha, SepSet
Output: Boolean (True nếu đồ thị còn bậc tự do > d để tiếp tục lên mức d+1)

1: Khởi tạo Stack_Edge_ID chứa tất cả chỉ số cạnh e in G: e.process = NO
2: WHILE Stack_Edge_ID không rỗng DO
3:     // Lấy ra một lô tối đa B = 128 cạnh từ đỉnh Stack
4:     Lấy ra K cạnh từ Stack_Edge_ID (K = min(128, |Stack_Edge_ID|)) và gán vào Batch[0...K-1]
5:     
6:     PARALLEL FOR i = 0 TO K-1 WITH t THREADS DO
7:         need_to_push[i] = CheckEdge(D, G, Adj_copy, Batch[i], d, gs, alpha, SepSet)
8:     END PARALLEL FOR
9:     
10:    // Đẩy ngược lại các cạnh chưa duyệt xong vào Stack để các luồng xử lý tiếp
11:    FOR i = K-1 DOWNTO 0 DO
12:        IF need_to_push[i] == TRUE THEN
13:            Push Batch[i] vào Stack_Edge_ID
14:        END IF
15:    END FOR
16: END WHILE

// Xóa các cạnh đã phát hiện độc lập tại độ sâu d
17: FOR mỗi cạnh e = (Vi, Vj) in G DO
18:     IF e.need_remove == TRUE THEN
19:         Xóa e khỏi G
20:         Xóa Vj khỏi Adj[Vi], xóa Vi khỏi Adj[Vj]
21:     END IF
22: END FOR

23: max_deg = max_{Vi in V} (|Adj[Vi]| - 1)
24: RETURN (max_deg > d)
```

---

### Thuật Toán 3: Xử Lý Kiểm Định Từng Cạnh Theo Nhóm (`CheckEdge` & `Testing`)

Mỗi luồng kiểm tra một cạnh $e = (X, Y)$ bằng cách lấy ra tối đa $gs$ tập điều kiện tiếp theo, tính bảng tần số và quyết định:

```text
Algorithm 3: CheckEdge
Input : Dữ liệu D, Đồ thị G, Adj_copy, Chỉ số cạnh edge_id, Độ sâu d, Group size gs, alpha, SepSet
Output: Boolean (True: cạnh chưa xong, cần push lại vào Pool; False: cạnh đã xong)

1: X = G.edges[edge_id].Node1,  Y = G.edges[edge_id].Node2

// Khởi tạo máy sinh tổ hợp (ChoiceGenerator) nếu cạnh chưa bắt đầu
2: IF G.edges[edge_id].process == NO THEN
3:     Adj_X = Adj_copy[X] \ {Y}
4:     IF |Adj_X| >= d THEN
5:         G.edges[edge_id].cg = New ChoiceGenerator(|Adj_X|, d)
6:         G.edges[edge_id].process = NODE1
7:     ELSE
8:         Adj_Y = Adj_copy[Y] \ {X}
9:         IF |Adj_Y| >= d THEN
10:            G.edges[edge_id].cg = New ChoiceGenerator(|Adj_Y|, d)
11:            G.edges[edge_id].process = NODE2
12:        ELSE
13:            G.edges[edge_id].need_remove = FALSE
14:            RETURN FALSE  // Bậc kề < d, không có tập điều kiện nào thỏa mãn -> giữ nguyên cạnh
15:        END IF
16:    END IF
17: END IF

// Sinh trực tiếp gs tập điều kiện tiếp theo theo thứ tự từ điển (Lexicographical)
18: Choices = G.edges[edge_id].cg.NextN(gs)  // Choices có kích thước k <= gs
19: IF Choices rỗng THEN
20:     // Đã duyệt hết tập lân cận của node hiện tại
21:     IF G.edges[edge_id].process == NODE1 THEN
22:         // Chuyển sang duyệt lân cận của node 2
23:         G.edges[edge_id].process = ENODE1
24:         Adj_Y = Adj_copy[Y] \ {X}
25:         IF |Adj_Y| >= d THEN
26:             G.edges[edge_id].cg = New ChoiceGenerator(|Adj_Y|, d)
27:             G.edges[edge_id].process = NODE2
28:             RETURN TRUE // Push lại để lần tới duyệt NODE2
29:         ELSE
30:             RETURN FALSE // Không đủ lân cận -> giữ cạnh
31:         END IF
32:     ELSE
33:         RETURN FALSE // Đã duyệt hết cả Node 1 và Node 2 mà không độc lập -> giữ cạnh
34:     END IF
35: END IF

// --- THỰC HIỆN KIỂM ĐỊNH NHÓM (GROUP CI TESTS) ---
36: Xây dựng bảng tần số đồng thời Counts3DGroup cho k tập điều kiện S_1, ..., S_k:
    - Tận dụng m mẫu của cột X và cột Y (chỉ nạp 1 lần từ RAM vào L1-cache)
    - Duyệt qua từng mẫu: tính chỉ số ô z_1, ..., z_k cho từng tập điều kiện
    - Tăng tần số n[z_i, x, y] tương ứng

37: FOR m = 1 TO k DO
38:     Tính thống kê G^2_m và bậc tự do df_m từ bảng tần số
39:     p_value_m = 1 - ChiSquareCDF(G^2_m, df_m)
40:     IF p_value_m > alpha THEN
41:         // Phát hiện độc lập! Cạnh bị loại bỏ ngay lập tức
42:         G.edges[edge_id].need_remove = TRUE
43:         CRITICAL SECTION:
44:             SepSet(X, Y) = S_m
45:         END CRITICAL SECTION
46:         Hủy G.edges[edge_id].cg
47:         RETURN FALSE // Đã độc lập, kết thúc cạnh, không cần push lại
48:     END IF
49: END FOR

// Nếu tất cả k kiểm định đều phụ thuộc:
50: IF k == gs THEN
51:     RETURN TRUE  // Vẫn còn tổ hợp phía sau, push lại vào Pool
52: ELSE
53:     // Đã là nhóm cuối cùng của Node hiện tại
54:     IF G.edges[edge_id].process == NODE1 THEN
55:         G.edges[edge_id].process = ENODE1
56:         RETURN TRUE
57:     ELSE
58:         G.edges[edge_id].need_remove = FALSE
59:         RETURN FALSE
60:     END IF
61: END IF
```

---

### Thuật Toán 4: Định Hướng Cấu Trúc Chữ V và Luật Meek

```text
Algorithm 4: Orientations (v-structure & Meek Rules)

// 1. Định hướng v-structure
PROCEDURE OrientVStructure(G, Adj, SepSet):
1: FOR mỗi node Vk in V DO
2:     FOR mỗi cặp lân cận (Vi, Vj) của Vk (Vi != Vj) DO
3:         IF (Vi, Vj) KHÔNG kề nhau trong G THEN
4:             IF Vk KHÔNG THUỘC SepSet(Vi, Vj) THEN
5:                 Định hướng: Vi -> Vk và Vj -> Vk
6:             END IF
7:         END IF
8:     END FOR
9: END FOR

// 2. Định hướng theo 3 luật Meek
PROCEDURE OrientImplied(G, Adj):
10: REPEAT
11:     changed = FALSE
12:     FOR mỗi cạnh vô hướng b -- c trong G DO
13:         // Luật 1: a -> b -- c và a không kề c => b -> c
14:         FOR mỗi cha a của b (a -> b) DO
15:             IF a không kề c THEN
16:                 Đổi b -- c thành b -> c (nếu không tạo chu trình); changed = TRUE
17:             END IF
18:         END FOR
19:         
20:         // Luật 2: a -> b -> c và a -- c => a -> c
21:         FOR mỗi nút b là con của a và cha của c DO
22:             Đổi a -- c thành a -> c (nếu không tạo chu trình); changed = TRUE
23:         END FOR
24:         
25:         // Luật 3: d -- a, d -- b, d -- c, b -> a, c -> a, b không kề c => d -> a
26:         FOR mỗi cặp lân cận chung (b, c) của a và d DO
27:             IF b không kề c VÀ b -> a VÀ c -> a THEN
28:                 Đổi d -- a thành d -> a; changed = TRUE
29:             END IF
30:         END FOR
31:     END FOR
32: UNTIL changed == FALSE
```

---

## 4. Ánh Xạ Giữa Mã Giả Và Mã Nguồn C++ FastPGM

Để thuận tiện tra cứu và chỉnh sửa, bảng dưới đây đối chiếu chi tiết từng hàm mã giả với file C++ tương ứng trong thư mục `src/fastpgm/`:

| Tên thuật toán / Khối mã giả | Hàm C++ tương ứng | File mã nguồn C++ |
| :--- | :--- | :--- |
| **Quy trình chính (Alg 1)** | `PCStable::StructLearnCompData` | `src/fastpgm/structure_learning/PCStable.cpp` (dòng 25) |
| **Học khung đồ thị Mức 0** | Vòng lặp OpenMP tại Mức 0 | `src/fastpgm/structure_learning/PCStable.cpp` (dòng 133–176) |
| **Học khung Mức $d \ge 1$ (Alg 2)** | `PCStable::SearchAtDepth` | `src/fastpgm/structure_learning/PCStable.cpp` (dòng 265–388) |
| **Kiểm tra trạng thái cạnh (Alg 3)** | `PCStable::CheckEdge` | `src/fastpgm/structure_learning/PCStable.cpp` (dòng 399–494) |
| **Thực hiện nhóm kiểm định CI** | `PCStable::Testing` | `src/fastpgm/structure_learning/PCStable.cpp` (dòng 526–616) |
| **Sinh tổ hợp theo thứ tự từ điển** | `ChoiceGenerator::NextN` | `src/fastpgm/ChoiceGenerator.cpp` (dòng 63–105) |
| **Tính toán bảng tần số nhóm 3D** | `Counts3DGroup::CountLevel1`, `CountLevelN` | `src/fastpgm/structure_learning/CellTable.cpp` (dòng 339–427) |
| **Tính $G^2$, bậc tự do & p-value** | `IndependenceTest::ComputeGSquareXYZGroup` | `src/fastpgm/structure_learning/IndependenceTest.cpp` (dòng 163–290) |
| **Định hướng v-structure (Alg 4)** | `PCStable::OrientVStructure` | `src/fastpgm/structure_learning/PCStable.cpp` (dòng 715–749) |
| **Định hướng quy tắc Meek (Alg 4)** | `PCStable::OrientImplied`, `Rule1`, `Rule2`, `Rule3` | `src/fastpgm/structure_learning/PCStable.cpp` (dòng 751–848) |

---

## 5. Phân Tích Độ Phức Tạp và Hiệu Quả Tăng Tốc

### 1. Số lượng phép kiểm định CI
Tại độ sâu $d$, mỗi cạnh $E_i = (V_i, V_j)$ có số phép kiểm định tối đa là:
$$\binom{|Adj(V_i) \setminus \{V_j\}|}{d} + \binom{|Adj(V_j) \setminus \{V_i\}|}{d}$$
Nhờ cơ chế **Dynamic Work Pool**, ngay khi một nhóm kiểm định thỏa mãn điều kiện độc lập ($p\text{-value} > \alpha$), cạnh lập tức bị hủy bỏ và không cần duyệt các tổ hợp còn lại.

### 2. Tiết kiệm truy cập bộ nhớ nhờ Grouping ($gs$)
Khi kiểm định độc lập $gs$ tập điều kiện $S_1, S_2, \dots, S_{gs}$ cho cùng một cặp biến $(V_i, V_j)$:
- **Cách truyền thống ($gs = 1$):** Cần $gs \times (2 + d)$ lần nạp vector cột dữ liệu từ RAM.
- **Fast-BNS:** Cột dữ liệu của $V_i$ và $V_j$ chỉ được nạp **1 lần duy nhất** vào bộ nhớ đệm (L1 Cache). Chỉ có các cột trong tập điều kiện $S_k$ được thay đổi.
- **Tỉ lệ giảm truy cập bộ nhớ:**
$$\text{Memory Access Reduction} \approx \frac{2(gs - 1)}{gs(d + 2)}$$
Thực nghiệm cho thấy với $gs = 6 \div 8$, Fast-BNS đạt điểm cân bằng tối ưu giữa việc giảm truy cập bộ nhớ và số phép kiểm định CI dư thừa phát sinh.

