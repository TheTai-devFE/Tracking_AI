# Backlog PoC UniFace

## Thứ tự ưu tiên đã chốt

1. Hoàn thiện detection và tracking ổn định.
2. Hoàn thiện head pose, attention, dwell và viewer session.
3. Benchmark trên video thật và chốt provider tracking.
4. Sau đó mới bật lại age/gender như module tùy chọn.
5. Recommendation chỉ bắt đầu khi dữ liệu tracking và nhân khẩu học đã được kiểm chứng.

CLI PoC mặc định không tải hoặc chạy model age/gender. Chỉ bật thử nghiệm bằng
`--with-attributes`.

## Mốc 1 - Lõi analytics

- [x] Định nghĩa contract observation và session event.
- [x] Xây dựng session state machine độc lập với model.
- [x] Thêm hysteresis cho attention và đếm look-away.
- [x] Tổng hợp tuổi/giới tính theo nhiều frame.
- [x] Viết unit test deterministic.

## Mốc 2 - Provider

- [x] Tạo protocol chung cho vision provider.
- [x] Tạo adapter cho pipeline baseline.
- [x] Tạo UniFace provider không dùng recognition/embedding.
- [x] Chạy smoke test khởi tạo đầy đủ model và CLI trên video tổng hợp.
- [ ] Chạy thử UniFace trên video người thật ở môi trường standee.
- [ ] Hiệu chỉnh detector, tracking buffer và ngưỡng head pose.

## Mốc 3 - Runner và benchmark

- [x] Tạo CLI video/webcam có lựa chọn provider.
- [x] Xuất JSONL session event và video overlay.
- [x] Tạo công cụ tổng hợp kết quả A/B từ hai lần chạy cùng video.
- [x] Tạo benchmark age/gender UniFace và MiVOLO trên cùng face crop.
- [ ] Cung cấp hoặc tái tạo `models/mivolo_age_gender.onnx` từ checkpoint chính thức.
- [ ] Chuẩn bị manifest ảnh crop có ground truth tuổi/giới tính.
- [ ] Gắn nhãn 5-10 video đại diện.
- [ ] Xuất báo cáo accuracy, ID switch, FPS và latency.

## Mốc 4 - Tích hợp ứng dụng

- [ ] Chọn UniFace, hybrid hoặc baseline dựa trên benchmark.
- [ ] Tạo FastAPI ingest/query service.
- [ ] Thiết kế PostgreSQL schema và aggregate queries.
- [ ] Tạo Next.js CMS/dashboard.
- [ ] Tạo Expo development build cho standee.
- [ ] Benchmark inference trên phần cứng standee.

Không bắt đầu Mốc 4 trước khi hoàn thành decision gate ở Mốc 3.

## Kết quả kiểm tra hiện tại

- 8/8 unit test đạt.
- SCRFD 500M, MobileNetV3 head-pose và age/gender ONNX khởi tạo thành công.
- Model UniFace được lưu trong `models/uniface/` và không commit lên Git.
- CLI đã đọc video, xử lý frame, ghi JSONL và tạo video overlay thành công.
- `uv pip check` xác nhận các package trong môi trường tương thích.

## Dữ liệu còn thiếu để tiếp tục benchmark

- Video người thật đại diện cho góc đặt camera và khoảng cách của standee.
- Ground truth viewer/non-viewer và khoảng attention cho từng video.
- `models/yolov8n-face.pt` và `models/mivolo_age_gender.onnx` để chạy baseline A/B.
- Thông số phần cứng Android/standee dự kiến để đánh giá mục tiêu FPS.
