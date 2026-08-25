# Tracking AI

Hệ thống computer vision đo lường mức độ tiếp cận và chú ý của người xem đối với
nội dung quảng cáo trên màn hình standee.

Pipeline nhận frame từ video hoặc camera và trả về các thông tin ước lượng:

- số người xuất hiện trước màn hình;
- số người thực sự nhìn quảng cáo;
- thời gian hiện diện và thời gian chú ý;
- số lần quay mặt đi;
- hướng đầu gồm yaw, pitch và roll;
- nhóm tuổi và giới tính khi kết quả đủ tin cậy.

Dự án không huấn luyện model. Các bước xử lý sử dụng model pretrained, hình học
`solvePnP` và các kỹ thuật xử lý ảnh truyền thống.

## Trạng thái dự án

Hệ thống sử dụng pipeline **UniFace** (SCRFD cho Face Detection, ByteTrack cho Face Tracking, MobileNetV3 cho Head Pose và tùy chọn ước lượng tuổi/giới tính).

Mục tiêu hiện tại là hoàn thiện tracking, attention và dwell trên video đại diện
cho môi trường standee. Age/gender mặc định bị tắt và chỉ được bật để thử nghiệm
khi cần thiết.

Kế hoạch và backlog:

- [Kế hoạch PoC tiếng Việt](PLAN_UNIFACE_POC_VI.md)
- [Kế hoạch PoC tiếng Anh](PLAN_UNIFACE_POC.md)
- [Backlog triển khai](TASKS_UNIFACE_POC.md)

## Yêu cầu môi trường

- Python 3.12, được khóa trong `.python-version`.
- `uv` để quản lý Python, virtual environment và dependency.
- Webcam hoặc video đầu vào để chạy thử.

## Cài đặt

Từ thư mục `Tracking_AI`, chạy:

```bash
uv sync
```

Lệnh trên sẽ tạo `.venv` và cài dependency theo `uv.lock`.

### Model weights

UniFace tự tải các weights cần thiết trong lần chạy đầu tiên và lưu tại:

```text
models/uniface/
```

Thư mục `models/`, dữ liệu đầu vào và output đều được bỏ qua bởi Git.

## Chạy PoC UniFace

### Xử lý video

```bash
uv run python run_experiment.py \
  --provider uniface \
  --video data/test.mp4 \
  --standee-id ST-001 \
  --campaign-id CMP-001 \
  --creative-id AD-001 \
  --events-out outputs/uniface-events.jsonl \
  --render-out outputs/uniface-overlay.mp4
```

Thêm `--display` để hiển thị overlay trong lúc xử lý:

```bash
uv run python run_experiment.py \
  --provider uniface \
  --video data/test.mp4 \
  --events-out outputs/uniface-events.jsonl \
  --display
```

### Chạy webcam

```bash
uv run python run_experiment.py \
  --provider uniface \
  --webcam 0 \
  --events-out outputs/webcam-events.jsonl \
  --display
```

Nhấn `q` để kết thúc cửa sổ webcam.

### Bật thử nghiệm age/gender

```bash
uv run python run_experiment.py \
  --provider uniface \
  --video data/test.mp4 \
  --with-attributes \
  --events-out outputs/uniface-with-attributes.jsonl
```

Age/gender mặc định không chạy. Cờ `--with-attributes` tải thêm model nhân khẩu
học khi chạy thử riêng tính năng này.

## Benchmark tuổi và giới tính

Benchmark UniFace sử dụng bộ ảnh crop khuôn mặt. Tạo manifest CSV theo mẫu:

```csv
image,age,gender
../data/age_gender/person_001.jpg,27,female
../data/age_gender/person_002.jpg,42,male
```

Có thể tham khảo `eval/age_gender_manifest.example.csv`.

Chạy benchmark:

```bash
uv run python -m eval.eval_age_gender \
  --manifest eval/age_gender_manifest.csv \
  --models uniface \
  --output-dir outputs/age-gender-uniface
```

Benchmark tạo hai file:

- `predictions.csv`: dự đoán và sai số của từng ảnh;
- `summary.json`: MAE tuổi, median absolute error, CS@5, accuracy nhóm tuổi,
  accuracy giới tính, failure rate và latency.

## Định nghĩa chỉ số

| Chỉ số | Ý nghĩa |
|---|---|
| `impression` | Một người xuất hiện trước màn hình đủ lâu để tạo lượt tiếp cận. |
| `viewer` | Một impression có thời gian nhìn màn hình vượt ngưỡng tối thiểu. |
| `engaged_viewer` | Một viewer duy trì chú ý trong thời gian dài hơn. |
| `presence_seconds` | Tổng thời gian người đó hiện diện trước camera. |
| `attention_seconds` | Tổng thời gian được phân loại là đang chú ý. |
| `attention_ratio` | Tỷ lệ thời gian chú ý trên thời gian hiện diện. |
| `look_away_count` | Số lần chuyển từ trạng thái chú ý sang quay mặt đi. |

Kết quả được tính theo session tracking ẩn danh. `track_id` chỉ có ý nghĩa trong
một luồng camera liên tục, vì vậy `estimated_viewers` không phải số người duy nhất
tuyệt đối trong ngày.

## Dữ liệu output

Mỗi dòng trong file JSONL đại diện cho một audience session đã đóng:

```json
{
  "schema_version": 1,
  "session_id": "random-uuid",
  "standee_id": "ST-001",
  "campaign_id": "CMP-001",
  "creative_id": "AD-001",
  "provider": "uniface",
  "provider_track_id": 3,
  "started_at": 0.0,
  "ended_at": 8.0,
  "presence_seconds": 8.0,
  "attention_seconds": 5.4,
  "attention_ratio": 0.675,
  "look_away_count": 1,
  "is_impression": true,
  "is_viewer": true,
  "is_engaged": true,
  "age_group": "25-34",
  "age_confidence": 0.72,
  "gender": "unknown",
  "gender_confidence": 0.48
}
```

Confidence của tuổi/giới tính hiện phản ánh độ đồng thuận giữa nhiều lần dự đoán
trong cùng session. Nó không được xem là xác suất chính xác tuyệt đối của model.

## Quyền riêng tư

PoC được thiết kế theo hướng đo lường ẩn danh:

- không chạy face recognition;
- không tạo hoặc lưu face embedding;
- không gắn tên hay định danh người thật;
- mặc định không lưu ảnh crop khuôn mặt;
- session ID được tạo ngẫu nhiên;
- chỉ lưu các chỉ số tổng hợp cần thiết cho dashboard.

Video overlay chỉ nên được bật trong quá trình phát triển và đánh giá có kiểm soát.
Không nên lưu hoặc tải video camera lên server trong thiết kế production mặc định.

## Kiểm thử

Chạy toàn bộ unit test:

```powershell
uv run python -m unittest discover -s tests -v
```

Các test hiện tập trung vào:

- state machine của audience session;
- attention hysteresis và look-away;
- đóng session khi mất track;
- chuẩn hóa nhóm tuổi và giới tính;
- tổng hợp metric theo session.

## Cấu trúc thư mục

| Đường dẫn | Vai trò |
|---|---|
| `audience/contracts.py` | Contract observation, session event và cấu hình. |
| `audience/session_tracker.py` | State machine impression, viewer, dwell và look-away. |
| `audience/demographics.py` | Chuẩn hóa nhóm tuổi và giới tính. |
| `providers/base.py` | Protocol dùng chung cho vision provider. |
| `providers/factory.py` | Factory khởi tạo provider (`uniface`). |
| `providers/uniface_provider.py` | Pipeline UniFace (SCRFD + ByteTrack + MobileNetV3). |
| `run_experiment.py` | CLI video/webcam và xuất JSONL/overlay. |
| `eval/eval_age_gender.py` | Benchmark độ tuổi/giới tính UniFace. |
| `eval/metrics.py` | Các metric session-level. |
| `configs.py` | Cấu hình ngưỡng góc nhìn, kích thước xử lý và đường dẫn. |
| `preprocess.py` | Resize, CLAHE và chuyển đổi màu. |
| `server/` | FastAPI WebSocket server và database persistence. |
| `web/` | Ứng dụng Next.js client cho standee và dashboard. |
| `tests/` | Unit test cho PoC analytics và database. |

## Hướng phát triển tiếp theo

1. Thu thập video đại diện cho vị trí camera và khoảng cách standee thực tế.
2. Gắn ground truth viewer/non-viewer và attention duration để đánh giá accuracy.
3. Tinh chỉnh ngưỡng attention yaw/pitch và calibration cho từng vị trí camera.
4. Mở rộng tính năng CMS quản trị creative và dashboard analytics.

## Website thu thập tracking

PoC web gồm hai service:

- FastAPI cổng `8000`: nhận frame JPEG qua WebSocket, chạy UniFace tracking-only,
  lưu audience session vào PostgreSQL (hoặc SQLite khi chạy local) và trả report API.
- Next.js cổng `3000`: phát creative, đọc camera, hiển thị overlay/telemetry và báo
  cáo session.

### Cấu hình PostgreSQL

Sao chép `.env.example` thành `.env`, sau đó thay `DATABASE_URL` bằng chuỗi kết nối thật:

```dotenv
DATABASE_URL=postgresql+psycopg://tracking_user:password@localhost:5432/tracking_ai
TRACKING_CORS_ORIGINS=http://localhost:3000
```

Backend chấp nhận cả tiền tố `postgres://` và `postgresql://`, sau đó tự chọn driver Psycopg 3.
Mật khẩu có ký tự đặc biệt phải được percent-encode trong URL. Các bảng và index cần thiết
được tạo tự động khi FastAPI khởi động. File `.env` đã được bỏ qua bởi Git.

Nếu chưa khai báo `DATABASE_URL`, backend tiếp tục dùng SQLite tại
`data/tracking_poc.db` để phát triển local.

### Khởi động FastAPI

Tại thư mục `Tracking_AI`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app `
  --host 0.0.0.0 `
  --port 8000 `
  --env-file .env `
  --reload
```

Kiểm tra API:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

Endpoint `/health` trả về `database_backend` và URL đã ẩn mật khẩu để xác nhận backend
đang kết nối đúng database. Backend không lưu frame hoặc ảnh crop.

### Khởi động Next.js

Mở PowerShell thứ hai:

```powershell
cd D:\clonegit\Tracking_AI\web
npm.cmd install
npm.cmd run dev -- --hostname 0.0.0.0 --port 3000
```

Trên máy phát triển, mở:

```text
http://localhost:3000
```

Trên standee cùng mạng LAN, mở địa chỉ IP của máy chạy Next.js, ví dụ:

```text
http://192.168.1.20:3000
```

Trình duyệt chỉ cấp quyền camera cho HTTPS hoặc `localhost`. Khi thử bằng standee
qua địa chỉ LAN, cần HTTPS reverse proxy/tunnel hoặc kiosk browser đã cấu hình cho
phép camera trên origin nội bộ.

### API tracking và báo cáo

| Endpoint | Vai trò |
|---|---|
| `WS /ws/tracking/{standee_id}` | Nhận frame và trả observation theo thời gian thực. |
| `GET /api/v1/reports/overview` | Funnel impression, viewer, engaged và attention. |
| `GET /api/v1/reports/sessions` | Danh sách session gần nhất. |
| `GET /api/v1/reports/timeline` | Tổng hợp theo giờ. |
| `GET /health` | Trạng thái backend. |

Đây là collector PoC một standee, chưa có authentication hoặc quản trị creative
trên server. Không mở trực tiếp API này ra Internet trước khi bổ sung xác thực,
giới hạn kích thước/rate frame và chính sách CORS production.
