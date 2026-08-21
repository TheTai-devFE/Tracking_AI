# Kế hoạch PoC phân tích người xem bằng UniFace

## 1. Mục tiêu

Xây dựng pipeline đo lường người xem ẩn danh cho màn hình quảng cáo và so sánh
UniFace với pipeline hiện tại gồm YOLO, ByteTrack, MiVOLO và MediaPipe.

Trong mỗi khoảng thời gian phát quảng cáo, hệ thống cần ước lượng:

- có bao nhiêu người xuất hiện trước màn hình (`impression`);
- có bao nhiêu người thực sự nhìn vào màn hình (`viewer`);
- thời gian hiện diện, thời gian chú ý và số lần quay mặt đi;
- hướng đầu gồm yaw, pitch và roll;
- nhóm tuổi và giới tính ước lượng khi độ tin cậy đủ cao;
- kết quả theo standee, chiến dịch, nội dung quảng cáo và thời gian.

Đây là nhánh thử nghiệm. UniFace chỉ thay thế pipeline hiện tại khi kết quả trên
cùng tập video chứng minh được sự cải thiện rõ ràng.

## 2. Phạm vi

### Bao gồm trong PoC

- Xử lý video offline và webcam cục bộ.
- Kiến trúc provider hỗ trợ hai engine: `baseline` và `uniface`.
- Tổng hợp dữ liệu theo phiên người xem thay vì chỉ xuất từng khuôn mặt mỗi frame.
- Xuất JSONL/CSV phù hợp để FastAPI tiếp nhận trong tương lai.
- So sánh độ chính xác, độ ổn định, độ trễ và tốc độ xử lý.
- Xử lý ưu tiên quyền riêng tư: session ID ngẫu nhiên, mặc định không lưu ảnh mặt.

### Chưa thực hiện trong PoC đầu tiên

- Nhận dạng danh tính hoặc theo dõi một người qua nhiều lần ghé thăm.
- Khẳng định số người duy nhất trong ngày là tuyệt đối chính xác.
- Huấn luyện hoặc fine-tune model.
- Triển khai production, truyền video từ xa, CMS hoặc APK hoàn chỉnh.
- Tự động chọn quảng cáo trước khi chất lượng đo lường được xác thực.

## 3. Định nghĩa chỉ số

Các tên dưới đây phải được sử dụng thống nhất giữa pipeline CV, API, database và
CMS.

| Chỉ số | Định nghĩa |
|---|---|
| `impression` | Một khuôn mặt được track liên tục ít nhất `min_presence_seconds`. |
| `viewer` | Một impression có tổng thời gian chú ý ít nhất `min_attention_seconds`. |
| `engaged_viewer` | Một viewer có thời gian chú ý ít nhất `engaged_seconds`. |
| `presence_seconds` | Thời gian từ lần đầu đến lần cuối quan sát được người đó, không tính khoảng mất dấu vượt quá timeout. |
| `attention_seconds` | Tổng các khoảng thời gian được phân loại là chú ý sau khi làm mượt theo thời gian. |
| `attention_ratio` | `attention_seconds / presence_seconds`, giới hạn trong `[0, 1]`. |
| `look_away_count` | Số lần trạng thái chuyển từ chú ý sang không chú ý sau hysteresis. |
| `estimated_viewers` | Số phiên đạt điều kiện viewer, không phải số người duy nhất dựa trên sinh trắc học. |

Ngưỡng khởi tạo được đặt trong cấu hình:

- `min_presence_seconds`: 1,0 giây;
- `min_attention_seconds`: 0,7 giây;
- `engaged_seconds`: 5,0 giây;
- `track_gap_tolerance_seconds`: 0,5 giây;
- `session_expiry_seconds`: 3,0 giây;
- kết quả tuổi/giới tính dưới ngưỡng confidence được gán là `unknown`.

Các ngưỡng này phải được hiệu chỉnh bằng video thực tế, không xem là hằng số nghiệp
vụ cố định.

## 4. Kiến trúc mục tiêu

```text
Video/Webcam
    |
    v
Tiền xử lý frame
    |
    +--> Baseline: Ultralytics + MediaPipe + MiVOLO
    |
    +--> UniFace: detector + ByteTrack + pose/gaze + age/gender
             |
             v
PersonObservation chuẩn hóa
             |
             v
AudienceSessionTracker (state machine nghiệp vụ)
             |
             v
AudienceSession event (JSONL/CSV)
             |
             v
FastAPI tương lai -> PostgreSQL -> Next.js CMS
```

Code model chỉ tạo quan sát theo từng frame, không tự quyết định một người có phải
là viewer hay không. Cả hai provider phải dùng chung session tracker để benchmark
phản ánh khác biệt model thay vì khác biệt logic nghiệp vụ.

## 5. Contract dữ liệu

### Quan sát theo từng frame

```python
@dataclass
class PersonObservation:
    provider: str
    track_id: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    yaw: float | None
    pitch: float | None
    roll: float | None
    attentive: bool | None
    attention_confidence: float | None
    age: float | None
    age_group: str | None
    age_confidence: float | None
    gender: str | None
    gender_confidence: float | None
```

### Phiên người xem đã đóng

```json
{
  "schema_version": 1,
  "session_id": "random-uuid",
  "standee_id": "ST-001",
  "campaign_id": "CMP-001",
  "creative_id": "AD-001",
  "provider": "uniface",
  "started_at": "2026-08-14T03:30:00Z",
  "ended_at": "2026-08-14T03:30:08Z",
  "presence_seconds": 8.0,
  "attention_seconds": 5.4,
  "attention_ratio": 0.675,
  "look_away_count": 1,
  "is_viewer": true,
  "is_engaged": true,
  "age_group": "25-34",
  "age_confidence": 0.72,
  "gender": "unknown",
  "gender_confidence": 0.48
}
```

Event không chứa frame gốc, ảnh khuôn mặt, embedding, tên hoặc định danh sinh trắc
học lâu dài.

## 6. Cấu trúc source code dự kiến

Giữ các module hiện tại hoạt động và đặt phần thử nghiệm trong package riêng:

```text
audience/
  contracts.py          # observation và session event chuẩn hóa
  session_tracker.py    # state machine impression/viewer/dwell/look-away
  aggregation.py        # tổng hợp theo giờ/ngày/chiến dịch
providers/
  base.py               # protocol chung cho provider
  baseline.py           # adapter cho pipeline hiện tại
  uniface_provider.py   # detector/tracker/pose/gaze/attribute của UniFace
eval/
  run_comparison.py     # chạy hai provider trên cùng video
  metrics.py            # ID switch, sai số đếm, attention và FPS
  annotations/          # ground truth cục bộ, không commit lên Git
run_experiment.py       # chạy video/webcam và xuất JSONL/video overlay
```

## 7. Các giai đoạn triển khai

### Giai đoạn A - Contract và session state machine

1. Tạo các dataclass `PersonObservation`, `AudienceSession` và cấu hình.
2. Xây dựng chuyển trạng thái: detected, present, attentive, viewing và closed.
3. Cho phép mất dấu ngắn mà không đóng session ngay lập tức.
4. Tổng hợp nhiều dự đoán tuổi/giới tính bằng vote có confidence.
5. Viết unit test cho dwell, look-away, expiry, tái sử dụng track ID và kết quả
   `unknown` bằng timestamp xác định.

Điều kiện hoàn thành:

- Test session chạy độc lập, không tải model ML.
- Một track giả lập tạo đúng một event với duration đúng như mong đợi.

### Giai đoạn B - UniFace provider

1. Thử detector nhẹ nhất phù hợp với CPU mục tiêu. Bắt đầu bằng SCRFD, chỉ so sánh
   YOLOv8Face khi cần.
2. Chuyển output ByteTrack của UniFace sang `PersonObservation`.
3. Tích hợp head pose trước; đánh giá gaze riêng vì gaze cần khuôn mặt lớn và rõ hơn.
4. Chỉ chạy age/gender định kỳ, lấy nhiều mẫu theo session, không chạy mỗi frame.
5. Đặt cache/weights rõ ràng trong `models/uniface/`.
6. Lazy-load model và trả lỗi rõ ràng khi thiếu hoặc tải weights thất bại.

Điều kiện hoàn thành:

- Video cục bộ tạo được video overlay và JSONL chứa các session đã đóng.
- Không ghi ảnh mặt nếu chưa bật cờ debug một cách tường minh.
- Chọn provider bằng CLI mà không sửa code.

### Giai đoạn C - Baseline adapter và đánh giá A/B

1. Chuyển pipeline hiện tại sang cùng observation contract.
2. Chuẩn bị 5-10 video đại diện cho:
   - một người nhìn và quay đi;
   - nhiều người đi cắt ngang nhau;
   - che khuất một phần và xuất hiện lại;
   - mặt nhìn nghiêng;
   - ngược sáng và thiếu sáng;
   - khuôn mặt ở gần và xa camera.
3. Gắn ground truth ở mức session: số người xuất hiện, viewer/non-viewer, khoảng
   attention tương đối và tính liên tục của track.
4. Chạy hai provider trên cùng frame và timestamp.
5. Xuất báo cáo so sánh.

Các metric bắt buộc:

- sai số tuyệt đối khi đếm impression và viewer;
- precision/recall của viewer classification;
- MAE của attention duration;
- số track bị phân mảnh và ID switch quan sát được;
- độ chính xác nhóm tuổi và tỷ lệ `unknown` trên dữ liệu có đồng thuận;
- độ chính xác giới tính và tỷ lệ `unknown` trên dữ liệu có đồng thuận;
- FPS trung bình, độ trễ p95, CPU và RAM;
- cold-start time và dung lượng model.

Mục tiêu ban đầu cho PoC có kiểm soát:

- sai số viewer không quá 10% trên video đã gắn nhãn;
- viewer precision và recall đạt ít nhất 0,85;
- median sai số attention duration không quá 1 giây;
- không có ID switch có thể tránh được trong video chỉ có một người;
- duy trì ít nhất 15 FPS trên phần cứng standee dự kiến hoặc có cấu hình sampling
  được chứng minh vẫn giữ độ chính xác metric.

Các ngưỡng nghiệm thu sẽ được điều chỉnh sau tập video gắn nhãn đầu tiên.

### Giai đoạn D - FastAPI tiếp nhận event

Chỉ bắt đầu khi giai đoạn C đã chọn được provider.

1. Tạo FastAPI service với schema có version.
2. Tạo API đăng ký/cấu hình thiết bị, lịch quảng cáo, nhận event theo batch và truy
   vấn analytics tổng hợp.
3. Bảo đảm ingest idempotent bằng `session_id` kết hợp `standee_id`.
4. Lưu timestamp theo UTC và giữ timezone của thiết bị trong metadata.
5. Lưu event trong PostgreSQL; chỉ tạo bảng tổng hợp hoặc materialized view sau khi
   xác định được mẫu truy vấn thực tế.
6. Thêm xác thực cho từng standee và cơ chế retry batch.

Trong production, FastAPI không nên nhận một luồng camera full-frame liên tục.
Standee nên xử lý cục bộ và gửi event nhỏ gọn nếu phần cứng cho phép.

### Giai đoạn E - Next.js CMS

1. CRUD campaign và creative.
2. Cấu hình nhóm tuổi mục tiêu, lịch phát và các điều kiện nội dung.
3. Quản lý standee, trạng thái online/offline, creative hiện tại và lần đồng bộ cuối.
4. Dashboard funnel: impressions -> viewers -> engaged viewers.
5. Biểu đồ theo ngày/giờ, standee, campaign, creative, tuổi và giới tính.
6. Hiển thị confidence và tỷ lệ `unknown` để dữ liệu ước lượng không bị trình bày
   như dữ liệu tuyệt đối.

### Giai đoạn F - Ứng dụng standee bằng Expo

1. Phát creative đã tải xuống và hoạt động khi mất mạng.
2. Gắn mỗi audience event với đúng `creative_id` đang phát.
3. Cache cấu hình, creative và event chưa gửi.
4. Báo trạng thái thiết bị và cập nhật cấu hình từ xa.
5. Thử truy cập frame camera bằng Expo development build/native module.
6. Benchmark inference trên thiết bị trước khi chọn ONNX native, service cục bộ hoặc
   inference server.

Không chọn thiết kế mặc định là upload liên tục video camera. Cách này tốn băng
thông, tăng độ trễ, phụ thuộc mạng và tạo thêm rủi ro riêng tư.

### Giai đoạn G - Thử nghiệm đề xuất quảng cáo

Chỉ xây recommendation sau khi dữ liệu đo lường đáng tin cậy.

1. Bắt đầu bằng rule rõ ràng: lịch phát, trạng thái campaign, nhóm tuổi cho phép,
   an toàn nội dung và frequency cap.
2. Xếp hạng creative bằng phân bố người xem tổng hợp theo standee và khung giờ,
   không tạo hồ sơ danh tính cá nhân.
3. Giữ một tỷ lệ exploration để quảng cáo mới vẫn có cơ hội được phát.
4. Ghi lại lý do chọn và version của rule/model trong mỗi quyết định.
5. Đánh giá hiệu quả bằng A/B test ở cấp campaign.

## 8. Cấu hình và CLI dự kiến

```powershell
uv run python run_experiment.py `
  --provider uniface `
  --video data/test.mp4 `
  --standee-id ST-001 `
  --campaign-id CMP-001 `
  --creative-id AD-001 `
  --events-out outputs/uniface-events.jsonl `
  --render-out outputs/uniface-overlay.mp4
```

Toàn bộ ngưỡng, model, provider, chu kỳ sampling và cờ privacy/debug phải nằm trong
cấu hình, không yêu cầu sửa code của từng stage.

## 9. Rủi ro và biện pháp kiểm soát

| Rủi ro | Biện pháp kiểm soát |
|---|---|
| Đếm trùng sau khi che khuất hoặc xuất hiện lại | Cho phép khoảng mất dấu ngắn, đặt thời gian hết hạn session và báo cáo estimated sessions thay vì số người tuyệt đối. |
| Sai lệch nhân khẩu học hoặc ảnh quá nhỏ | Dùng confidence threshold, trả `unknown`, báo cáo theo nhóm và đánh giá trên dữ liệu tương tự môi trường thật có đồng thuận. |
| Head pose báo chú ý sai | Hiệu chỉnh theo vị trí camera/màn hình; chỉ dùng gaze khi chất lượng crop đáp ứng. |
| FPS thấp do chạy mọi model mỗi frame | Giảm tần suất age/gender, chọn detector nhẹ, batch khi có thể và đo latency p95. |
| Xung đột package OpenCV | Giải quyết `opencv-python` và `opencv-contrib-python` trước khi khóa môi trường production. |
| Không tải được model | Cài sẵn weights đã xác minh và cấu hình cache directory rõ ràng. |
| Rò rỉ dữ liệu riêng tư | Xử lý cục bộ, bỏ frame sau xử lý, dùng session ID ngẫu nhiên, không lưu embedding hoặc nhận dạng. |
| Kết quả khác nhau giữa thiết bị | Ghi version provider/cấu hình trong event và xây dựng quy trình calibration. |

## 10. Điểm quyết định

Chỉ áp dụng UniFace nếu giai đoạn C chứng minh nó cải thiện ít nhất một tiêu chí quan
trọng như độ chính xác viewer, attention, độ ổn định track hoặc độ đơn giản khi triển
khai mà không làm FPS, cold-start, dependency hoặc giấy phép trở nên không phù hợp.

Ba kết quả có thể xảy ra:

- **Áp dụng:** UniFace tốt hơn và trở thành provider mặc định.
- **Kết hợp:** chỉ dùng một số phần của UniFace như SCRFD hoặc gaze, vẫn giữ MiVOLO
  và head pose hình học hiện tại.
- **Không áp dụng:** giữ pipeline baseline và loại UniFace khỏi dependency production.

Phải lưu lại quyết định và kết quả benchmark trước khi triển khai tầng FastAPI,
Next.js và Expo.
