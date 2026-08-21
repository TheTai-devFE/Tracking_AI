"use client";

import {
  Activity,
  Camera,
  CircleStop,
  Expand,
  FileImage,
  MonitorPlay,
  Play,
  RefreshCw,
  Upload,
  Users,
  Wifi,
  WifiOff,
} from "lucide-react";
import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Observation = {
  track_id: number;
  bbox: [number, number, number, number];
  yaw: number | null;
  pitch: number | null;
  attentive: boolean | null;
};

type Overview = {
  sessions: number;
  impressions: number;
  viewers: number;
  engaged_viewers: number;
  average_presence_seconds: number;
  average_attention_seconds: number;
  attention_rate: number;
};

type Session = {
  session_id: string;
  provider_track_id: number;
  started_at: number;
  presence_seconds: number;
  attention_seconds: number;
  is_viewer: number;
};

const emptyOverview: Overview = {
  sessions: 0,
  impressions: 0,
  viewers: 0,
  engaged_viewers: 0,
  average_presence_seconds: 0,
  average_attention_seconds: 0,
  attention_rate: 0,
};

function deriveApiUrl() {
  if (process.env.NEXT_PUBLIC_TRACKING_API_URL) return process.env.NEXT_PUBLIC_TRACKING_API_URL;
  if (typeof window !== "undefined") return `${window.location.protocol}//${window.location.hostname}:8000`;
  return "http://localhost:8000";
}

export default function TrackingCollector() {
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [standeeId, setStandeeId] = useState("ST-001");
  const [campaignId, setCampaignId] = useState("CMP-LOCAL");
  const [creativeId, setCreativeId] = useState("AD-LOCAL");
  const [creative, setCreative] = useState<{ url: string; type: "image" | "video"; name: string } | null>(null);
  const [status, setStatus] = useState<"idle" | "connecting" | "running" | "error">("idle");
  const [statusMessage, setStatusMessage] = useState("Chưa kết nối");
  const [observations, setObservations] = useState<Observation[]>([]);
  const [overview, setOverview] = useState<Overview>(emptyOverview);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [processingMs, setProcessingMs] = useState(0);
  const [resultFps, setResultFps] = useState(0);
  const [sentFrames, setSentFrames] = useState(0);
  const [droppedFrames, setDroppedFrames] = useState(0);

  const cameraRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null);
  const adStageRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const resultCounterRef = useRef({ count: 0, started: performance.now() });

  useEffect(() => setApiUrl(deriveApiUrl()), []);

  const websocketUrl = useMemo(() => {
    const base = apiUrl.replace(/^http/, "ws").replace(/\/$/, "");
    const params = new URLSearchParams({ campaign_id: campaignId, creative_id: creativeId });
    return `${base}/ws/tracking/${encodeURIComponent(standeeId)}?${params}`;
  }, [apiUrl, standeeId, campaignId, creativeId]);

  const loadReport = useCallback(async () => {
    try {
      const query = new URLSearchParams({ standee_id: standeeId });
      const [overviewResponse, sessionsResponse] = await Promise.all([
        fetch(`${apiUrl}/api/v1/reports/overview?${query}`),
        fetch(`${apiUrl}/api/v1/reports/sessions?${query}&limit=8`),
      ]);
      if (!overviewResponse.ok || !sessionsResponse.ok) throw new Error("report request failed");
      setOverview(await overviewResponse.json());
      setSessions(await sessionsResponse.json());
    } catch {
      // The connection badge already communicates server availability.
    }
  }, [apiUrl, standeeId]);

  useEffect(() => {
    loadReport();
    const interval = setInterval(loadReport, 5000);
    return () => clearInterval(interval);
  }, [loadReport]);

  useEffect(() => {
    const canvas = overlayRef.current;
    const video = cameraRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.lineWidth = 3;
    context.font = "16px Arial";
    for (const observation of observations) {
      const [x1, y1, x2, y2] = observation.bbox;
      const color = observation.attentive ? "#18c46c" : "#9aa3ad";
      context.strokeStyle = color;
      context.fillStyle = color;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const pose = observation.yaw == null ? "" : ` Y:${observation.yaw.toFixed(0)} P:${observation.pitch?.toFixed(0)}`;
      context.fillText(`#${observation.track_id}${pose}`, x1, Math.max(18, y1 - 6));
    }
  }, [observations]);

  const stopTracking = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (cameraRef.current) cameraRef.current.srcObject = null;
    setStatus("idle");
    setStatusMessage("Đã dừng");
    setObservations([]);
    loadReport();
  }, [loadReport]);

  useEffect(() => stopTracking, [stopTracking]);

  const sendFrame = useCallback(() => {
    const video = cameraRef.current;
    const canvas = captureRef.current;
    const socket = socketRef.current;
    if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.readyState < 2) return;
    if (socket.bufferedAmount > 1_000_000) {
      setDroppedFrames((value) => value + 1);
      return;
    }
    const width = 640;
    const height = Math.max(1, Math.round((video.videoHeight / video.videoWidth) * width));
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d")?.drawImage(video, 0, 0, width, height);
    canvas.toBlob(async (blob) => {
      if (!blob || socket.readyState !== WebSocket.OPEN) return;
      socket.send(await blob.arrayBuffer());
      setSentFrames((value) => value + 1);
    }, "image/jpeg", 0.7);
  }, []);

  const startTracking = async () => {
    if (status === "running" || status === "connecting") return;
    setStatus("connecting");
    setStatusMessage("Đang mở camera");
    setSentFrames(0);
    setDroppedFrames(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 20 } },
        audio: false,
      });
      streamRef.current = stream;
      if (!cameraRef.current) throw new Error("camera element unavailable");
      cameraRef.current.srcObject = stream;
      await cameraRef.current.play();

      const socket = new WebSocket(websocketUrl);
      socketRef.current = socket;
      socket.onopen = () => {
        setStatusMessage("Đang khởi tạo model");
      };
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "initializing") {
          setStatusMessage("Đang nạp model tracking");
        }
        if (payload.type === "ready") {
          setStatus("running");
          setStatusMessage("Tracking đang chạy");
          timerRef.current = setInterval(sendFrame, 180);
          socket.send(JSON.stringify({
            type: "telemetry",
            browser: navigator.userAgent,
            screen_resolution: `${window.screen.width}x${window.screen.height}`,
            camera_resolution: `${cameraRef.current?.videoWidth}x${cameraRef.current?.videoHeight}`,
            target_sent_fps: 5.5,
          }));
        }
        if (payload.type === "frame_result") {
          setObservations(payload.observations);
          setProcessingMs(payload.processing_ms);
          const counter = resultCounterRef.current;
          counter.count += 1;
          const elapsed = performance.now() - counter.started;
          if (elapsed >= 1000) {
            setResultFps((counter.count * 1000) / elapsed);
            resultCounterRef.current = { count: 0, started: performance.now() };
          }
          if (payload.closed_sessions?.length) loadReport();
        }
      };
      socket.onerror = () => {
        setStatus("error");
        setStatusMessage("Không kết nối được server");
      };
      socket.onclose = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = null;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        if (cameraRef.current) cameraRef.current.srcObject = null;
        setStatus((current) => current === "error" ? "error" : "idle");
        setStatusMessage((current) => current === "Không kết nối được server" ? current : "Kết nối đã đóng");
      };
    } catch (error) {
      setStatus("error");
      setStatusMessage(error instanceof Error ? error.message : "Không mở được camera");
      streamRef.current?.getTracks().forEach((track) => track.stop());
    }
  };

  const chooseCreative = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (creative) URL.revokeObjectURL(creative.url);
    setCreative({
      url: URL.createObjectURL(file),
      type: file.type.startsWith("video/") ? "video" : "image",
      name: file.name,
    });
    if (creativeId === "AD-LOCAL") setCreativeId(file.name.replace(/\.[^.]+$/, ""));
  };

  const enterFullscreen = () => adStageRef.current?.requestFullscreen();
  const connected = status === "running";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><Activity size={19} /> Tracking AI</div>
        <div className={`connection ${connected ? "online" : ""}`}>
          {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
          {statusMessage}
        </div>
      </header>

      <section className="workspace">
        <div className="stage-column">
          <div className="section-heading">
            <div><span className="eyebrow">Creative</span><h1>{creative?.name ?? "Chưa chọn nội dung"}</h1></div>
            <div className="toolbar">
              <label className="button secondary"><Upload size={17} /> Chọn tệp<input type="file" accept="image/*,video/*" onChange={chooseCreative} /></label>
              <button className="icon-button" onClick={enterFullscreen} title="Toàn màn hình"><Expand size={18} /></button>
            </div>
          </div>
          <div className="ad-stage" ref={adStageRef}>
            {creative?.type === "video" && <video src={creative.url} autoPlay loop muted playsInline />}
            {creative?.type === "image" && <img src={creative.url} alt="Creative đang phát" />}
            {!creative && <div className="empty-stage"><MonitorPlay size={36} /><span>AD-LOCAL</span></div>}
          </div>

          <div className="metrics-strip">
            <Metric label="Impressions" value={overview.impressions} />
            <Metric label="Viewers" value={overview.viewers} />
            <Metric label="Engaged" value={overview.engaged_viewers} />
            <Metric label="Attention" value={`${Math.round(overview.attention_rate * 100)}%`} />
          </div>
        </div>

        <aside className="control-column">
          <section className="panel config-panel">
            <div className="panel-title"><MonitorPlay size={17} /><h2>Phiên đo</h2></div>
            <label>API server<input value={apiUrl} onChange={(event) => setApiUrl(event.target.value)} disabled={connected} /></label>
            <div className="field-grid">
              <label>Standee<input value={standeeId} onChange={(event) => setStandeeId(event.target.value)} disabled={connected} /></label>
              <label>Campaign<input value={campaignId} onChange={(event) => setCampaignId(event.target.value)} disabled={connected} /></label>
            </div>
            <label>Creative ID<input value={creativeId} onChange={(event) => setCreativeId(event.target.value)} disabled={connected} /></label>
            <div className="action-row">
              {!connected ? (
                <button className="button primary" onClick={startTracking} disabled={status === "connecting"}><Play size={17} /> Bắt đầu</button>
              ) : (
                <button className="button danger" onClick={stopTracking}><CircleStop size={17} /> Dừng</button>
              )}
              <button className="icon-button" onClick={loadReport} title="Làm mới báo cáo"><RefreshCw size={17} /></button>
            </div>
          </section>

          <section className="panel camera-panel">
            <div className="panel-title"><Camera size={17} /><h2>Camera debug</h2><span className="track-count">{observations.length} tracks</span></div>
            <div className="camera-stage">
              <video ref={cameraRef} muted playsInline />
              <canvas ref={overlayRef} />
              {!streamRef.current && <div className="camera-empty"><Camera size={28} /></div>}
            </div>
            <canvas ref={captureRef} className="capture-canvas" />
            <div className="telemetry-row">
              <span>{resultFps.toFixed(1)} FPS</span>
              <span>{processingMs.toFixed(0)} ms</span>
              <span>{sentFrames} sent</span>
              <span>{droppedFrames} dropped</span>
            </div>
          </section>

          <section className="panel sessions-panel">
            <div className="panel-title"><Users size={17} /><h2>Session gần nhất</h2></div>
            <div className="session-list">
              {sessions.length === 0 && <div className="empty-row">Chưa có session</div>}
              {sessions.map((session) => (
                <div className="session-row" key={session.session_id}>
                  <span className={`viewer-dot ${session.is_viewer ? "yes" : ""}`} />
                  <strong>#{session.provider_track_id}</strong>
                  <span>{session.presence_seconds.toFixed(1)}s hiện diện</span>
                  <span>{session.attention_seconds.toFixed(1)}s nhìn</span>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}
