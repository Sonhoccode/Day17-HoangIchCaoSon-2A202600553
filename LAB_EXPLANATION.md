# Giải thích bài lab: Memory Systems for AI Agent

Tài liệu này tóm tắt nhanh yêu cầu của bài lab, đầu ra mong đợi, và cách tiếp cận triển khai hợp lý.

## 1. Mục tiêu bài lab

Mục tiêu chính của lab là xây dựng và so sánh hai loại agent:

- `Baseline Agent`: chỉ nhớ trong phạm vi một thread / một phiên
- `Advanced Agent`: có nhớ ngắn hạn, nhớ bền vững qua `User.md`, và compact memory cho hội thoại dài

Điểm quan trọng của lab không chỉ là “agent nhớ nhiều hơn”, mà là hiểu trade-off giữa:

- khả năng recall
- chi phí token
- kích thước memory file
- độ phức tạp của hệ thống

## 2. Yêu cầu bài lab

### 2.1. Cấu hình chung

File cần hoàn thiện:

- `src/config.py`

Yêu cầu:

- xác định `base_dir`, `data_dir`, `state_dir`
- đọc biến môi trường từ `.env`
- hỗ trợ các provider:
  - `openai`
  - `custom`
  - `gemini`
  - `anthropic`
  - `ollama`
  - `openrouter`
- cấu hình compact memory:
  - ngưỡng token để compact
  - số message cần giữ lại sau compact

### 2.2. Memory layer

File cần hoàn thiện:

- `src/memory_store.py`

Yêu cầu:

- ước lượng token bằng heuristic đơn giản
- lưu hồ sơ người dùng vào `User.md`
- hỗ trợ đọc / ghi / sửa nội dung profile
- trích xuất fact ổn định từ message
- compact hội thoại dài bằng summary + recent messages

### 2.3. Baseline Agent

File cần hoàn thiện:

- `src/agent_baseline.py`

Yêu cầu:

- chỉ có short-term memory trong cùng thread
- không có persistent memory
- không nhớ facts qua session mới

### 2.4. Advanced Agent

File cần hoàn thiện:

- `src/agent_advanced.py`

Yêu cầu:

- có short-term memory
- có persistent memory bằng `User.md`
- có compact memory khi hội thoại quá dài
- trả lời tốt hơn baseline trong các câu hỏi recall qua nhiều session

### 2.5. Benchmark

File cần hoàn thiện:

- `src/benchmark.py`

Yêu cầu:

- chạy 2 bộ dữ liệu:
  - `data/conversations.json`
  - `data/advanced_long_context.json`
- so sánh baseline và advanced
- xuất các chỉ số:
  - `Agent tokens only`
  - `Prompt tokens processed`
  - `Cross-session recall`
  - `Response quality`
  - `Memory growth (bytes)`
  - `Compactions`

### 2.6. Test

File cần hoàn thiện:

- `src/test_agents.py`

Yêu cầu tối thiểu:

- test đọc / ghi / sửa `User.md`
- test compact memory được kích hoạt
- test cross-session recall
- test advanced giảm prompt load trên thread dài

## 3. Đầu ra mong đợi

Sau khi hoàn thành, repo nên có:

- code chạy được cho baseline và advanced
- benchmark in ra bảng so sánh rõ ràng
- test pass
- `state/` được tạo để chứa profile / memory cục bộ

Về mặt hành vi, kết quả đúng nên cho thấy:

- baseline quên fact khi sang thread mới
- advanced giữ được fact qua session mới
- advanced tốn thêm chi phí lưu memory nhưng recall tốt hơn
- compact memory giúp giảm prompt load ở hội thoại dài

## 4. Hướng giải quyết đề xuất

### Bước 1: Hoàn thiện cấu hình

Làm `config.py` trước để toàn bộ code còn lại có một nguồn cấu hình thống nhất.

### Bước 2: Làm lớp memory

Triển khai `memory_store.py` trước vì đây là nền móng của advanced agent.

### Bước 3: Làm baseline

Baseline nên đơn giản, deterministic, và chỉ dùng dữ liệu trong thread hiện tại.

### Bước 4: Làm advanced

Advanced cần kết hợp:

- session memory
- `User.md`
- compact summary

### Bước 5: Làm benchmark

Benchmark nên dùng cùng input cho cả hai agent để so sánh công bằng.

### Bước 6: Làm test

Test các hành vi cốt lõi trước, rồi mới tối ưu thêm.

## 5. Logic thiết kế nên giữ

Khi triển khai, nên giữ ranh giới rõ:

- `short-term memory`: chỉ dùng cho hội thoại hiện tại
- `persistent memory`: lưu fact ổn định vào `User.md`
- `compact memory`: nén phần cũ của thread để giảm ngữ cảnh

Nếu làm đúng, bài lab sẽ thể hiện được câu chuyện sau:

1. baseline rẻ nhưng nhớ kém
2. advanced nhớ tốt hơn nhờ persistent memory
3. hội thoại dài làm chi phí token tăng
4. compact memory giúp kiểm soát chi phí đó

## 6. Ghi chú thực hành

- Nên ưu tiên triển khai offline deterministic trước
- Sau khi test ổn mới nối sang provider thật nếu cần
- Không nên lưu mọi câu chữ vào `User.md`; chỉ lưu fact ổn định
- Nên tránh compact quá sớm vì có thể làm mất ngữ cảnh quan trọng

## 7. Kết luận

Lab này chủ yếu dạy cách thiết kế memory system cho agent theo hướng thực dụng:

- nhớ đúng cái cần nhớ
- quên cái không cần nhớ
- giữ chi phí token trong tầm kiểm soát

Nếu baseline và advanced thể hiện được đúng khác biệt này qua code, test và benchmark, bài lab đạt mục tiêu.
