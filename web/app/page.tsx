"use client";

import {
  Activity,
  Camera,
  CircleStop,
  Expand,
  FileImage,
  Film,
  Layers,
  LayoutDashboard,
  MonitorPlay,
  Pause,
  Play,
  RefreshCw,
  SkipBack,
  SkipForward,
  Smartphone,
  Tv,
  Upload,
  Users,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Observation = {
  track_id: number;
  bbox: [number, number, number, number];
  yaw: number | null;
  pitch: number | null;
  attentive: boolean | null;
  age?: number | null;
  age_group?: string | null;
  gender?: string | null;
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
  age_group?: string;
  estimated_age?: number;
  gender?: string;
};

type PlaylistItem = {
  id: string;
  name: string;
  file_name?: string;
  file_url: string;
  media_type: "video" | "image";
  duration_seconds: number;
  campaign_id: string;
  is_active: boolean;
  sort_order: number;
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

function detectMediaType(type?: string, filename?: string, url?: string): "video" | "image" {
  if (type === "video") return "video";
  if (type === "image") return "image";
  if (type && type.toLowerCase().startsWith("video")) return "video";
  if (type && type.toLowerCase().startsWith("image")) return "image";

  const str = `${filename || ""} ${url || ""}`.toLowerCase();
  if (/\.(mp4|webm|mov|mkv|avi|m4v|ogv|flv)\b/i.test(str)) {
    return "video";
  }
  return "image";
}

function deriveApiUrl() {
  if (process.env.NEXT_PUBLIC_TRACKING_API_URL) return process.env.NEXT_PUBLIC_TRACKING_API_URL;
  if (typeof window !== "undefined") return `${window.location.protocol}//${window.location.hostname}:8000`;
  return "http://localhost:8000";
}

export default function StandeePlayer() {
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [standeeId, setStandeeId] = useState("ST-001");
  const [campaignId, setCampaignId] = useState("CMP-LOCAL");

  // Playlist State
  const [playlist, setPlaylist] = useState<PlaylistItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [localCreative, setLocalCreative] = useState<{ url: string; type: "video" | "image"; name: string; id: string } | null>(null);

  // Playback Control States
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [aspectMode, setAspectMode] = useState<"landscape" | "portrait">("landscape");
  const [imageProgress, setImageProgress] = useState(0);

  // Tracking & Telemetry
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
  const videoPlayerRef = useRef<HTMLVideoElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const playlistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const imageAnimFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const resultCounterRef = useRef({ count: 0, started: performance.now() });

  useEffect(() => setApiUrl(deriveApiUrl()), []);

  // Fetch playlist from server
  const loadPlaylist = useCallback(async () => {
    try {
      const res = await fetch(`${apiUrl}/api/v1/creatives?only_active=true`);
      if (res.ok) {
        const items: PlaylistItem[] = await res.json();
        setPlaylist(items);
      }
    } catch (err) {
      console.error("Failed to load playlist:", err);
    }
  }, [apiUrl]);

  useEffect(() => {
    loadPlaylist();
    // Auto-poll playlist every 15s to pick up changes made in Admin without reloading
    const interval = setInterval(loadPlaylist, 15000);
    return () => clearInterval(interval);
  }, [loadPlaylist]);

  // Current active creative item
  const currentCreative = useMemo(() => {
    if (localCreative) return localCreative;
    if (playlist.length === 0) return null;
    const item = playlist[currentIndex % playlist.length];
    const resolvedType = detectMediaType(item.media_type, item.file_name, item.file_url);
    return {
      id: item.id,
      name: item.name,
      type: resolvedType,
      url: item.file_url.startsWith("http") ? item.file_url : `${apiUrl}${item.file_url}`,
      duration: item.duration_seconds || 10,
      campaign_id: item.campaign_id,
    };
  }, [playlist, currentIndex, localCreative, apiUrl]);

  // Advance to next playlist item
  const nextCreative = useCallback(() => {
    if (playlist.length <= 1) {
      if (videoPlayerRef.current) {
        videoPlayerRef.current.currentTime = 0;
        videoPlayerRef.current.play().catch(() => {});
      }
      return;
    }
    setCurrentIndex((prev) => (prev + 1) % playlist.length);
  }, [playlist.length]);

  const prevCreative = useCallback(() => {
    if (playlist.length <= 1) {
      if (videoPlayerRef.current) {
        videoPlayerRef.current.currentTime = 0;
        videoPlayerRef.current.play().catch(() => {});
      }
      return;
    }
    setCurrentIndex((prev) => (prev - 1 + playlist.length) % playlist.length);
  }, [playlist.length]);

  // Play / Pause Toggle
  const togglePlay = () => {
    if (!videoPlayerRef.current) return;
    if (videoPlayerRef.current.paused) {
      videoPlayerRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    } else {
      videoPlayerRef.current.pause();
      setIsPlaying(false);
    }
  };

  const toggleMute = () => {
    if (!videoPlayerRef.current) return;
    videoPlayerRef.current.muted = !videoPlayerRef.current.muted;
    setIsMuted(videoPlayerRef.current.muted);
  };

  // Auto-play video whenever URL changes
  useEffect(() => {
    if (currentCreative?.type === "video" && videoPlayerRef.current) {
      setCurrentTime(0);
      videoPlayerRef.current.currentTime = 0;
      videoPlayerRef.current.play().then(() => setIsPlaying(true)).catch((e) => {
        console.log("Video auto-play needs muted or interaction:", e);
      });
    }
  }, [currentCreative?.url, currentCreative?.type]);

  // Synchronize dynamic creative with AI backend WebSocket
  useEffect(() => {
    if (!currentCreative) return;
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(
        JSON.stringify({
          type: "set_creative",
          creative_id: currentCreative.id,
          campaign_id: ("campaign_id" in currentCreative ? currentCreative.campaign_id : campaignId) || campaignId,
        })
      );
    }
  }, [currentCreative, campaignId]);

  // Handle image timer duration & progress animation
  useEffect(() => {
    if (playlistTimerRef.current) clearTimeout(playlistTimerRef.current);
    if (imageAnimFrameRef.current) cancelAnimationFrame(imageAnimFrameRef.current);
    setImageProgress(0);

    if (!currentCreative || currentCreative.type !== "image" || (playlist.length <= 1 && !localCreative)) return;

    const durationSec = (("duration" in currentCreative ? currentCreative.duration : 10) || 10);
    const durationMs = durationSec * 1000;
    const startTime = performance.now();

    const updateProgress = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(100, (elapsed / durationMs) * 100);
      setImageProgress(progress);
      if (elapsed < durationMs) {
        imageAnimFrameRef.current = requestAnimationFrame(updateProgress);
      }
    };
    imageAnimFrameRef.current = requestAnimationFrame(updateProgress);

    playlistTimerRef.current = setTimeout(() => {
      nextCreative();
    }, durationMs);

    return () => {
      if (playlistTimerRef.current) clearTimeout(playlistTimerRef.current);
      if (imageAnimFrameRef.current) cancelAnimationFrame(imageAnimFrameRef.current);
    };
  }, [currentCreative, nextCreative, playlist.length, localCreative]);

  const loadReport = useCallback(async () => {
    try {
      const query = new URLSearchParams({ standee_id: standeeId });
      const [overviewResponse, sessionsResponse] = await Promise.all([
        fetch(`${apiUrl}/api/v1/reports/overview?${query}`),
        fetch(`${apiUrl}/api/v1/reports/sessions?${query}&limit=8`),
      ]);
      if (overviewResponse.ok) setOverview(await overviewResponse.json());
      if (sessionsResponse.ok) setSessions(await sessionsResponse.json());
    } catch (error) {
      console.error("Report fetch error:", error);
    }
  }, [apiUrl, standeeId]);

  const drawBoxes = useCallback((obsList: Observation[], frameWidth?: number, frameHeight?: number) => {
    const canvas = overlayRef.current;
    const video = cameraRef.current;
    if (!canvas || !video || video.videoWidth === 0) return;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    // sendFrame downscales the camera image to maxSide 640 before sending, so
    // the server reports boxes in that smaller space. The overlay canvas is the
    // camera's native size — without this scale a 1280px camera draws every box
    // at half size, offset toward the top-left.
    const scaleX = frameWidth ? width / frameWidth : 1;
    const scaleY = frameHeight ? height / frameHeight : 1;

    obsList.forEach((obs) => {
      const [rx1, ry1, rx2, ry2] = obs.bbox;
      const x1 = rx1 * scaleX;
      const y1 = ry1 * scaleY;
      const x2 = rx2 * scaleX;
      const y2 = ry2 * scaleY;
      const attentive = obs.attentive === true;
      ctx.strokeStyle = attentive ? "#18b566" : "#aab2ba";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const genderVi = obs.gender === "female" ? "Nữ" : obs.gender === "male" ? "Nam" : "";
      const ageText = obs.age ? `~${Math.round(obs.age)} tuổi` : obs.age_group && obs.age_group !== "unknown" ? obs.age_group : "";
      const demoPart = [genderVi, ageText].filter(Boolean).join(" • ");
      const anglePart = obs.yaw !== null ? `${Math.round(obs.yaw)}°` : "";
      const tag = `#${obs.track_id}${anglePart ? ` | ${anglePart}` : ""}${demoPart ? ` | ${demoPart}` : ""}`;

      ctx.fillStyle = attentive ? "#18b566" : "#4a5568";
      ctx.font = "bold 13px sans-serif";
      const textWidth = ctx.measureText(tag).width;
      ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 8, 20);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(tag, x1 + 4, Math.max(14, y1 - 7));
    });
  }, []);

  const sendFrame = useCallback(() => {
    const socket = socketRef.current;
    const video = cameraRef.current;
    const capture = captureRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN || !video || video.readyState < 2 || !capture) return;

    if (socket.bufferedAmount > 256 * 1024) {
      setDroppedFrames((count) => count + 1);
      return;
    }

    const maxSide = 640;
    const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
    capture.width = Math.round(video.videoWidth * scale);
    capture.height = Math.round(video.videoHeight * scale);

    const ctx = capture.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, capture.width, capture.height);

    capture.toBlob(
      (blob) => {
        if (!blob || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then((buffer) => {
          socketRef.current?.send(buffer);
          setSentFrames((count) => count + 1);
        });
      },
      "image/jpeg",
      0.7
    );
  }, []);

  const stopTracking = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (cameraRef.current) cameraRef.current.srcObject = null;
    setStatus("idle");
    setStatusMessage("Đã dừng");
    setObservations([]);
    loadReport();
  }, [loadReport]);

  const startTracking = async () => {
    setStatus("connecting");
    setStatusMessage("Đang khởi tạo camera...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      if (cameraRef.current) cameraRef.current.srcObject = stream;
      await cameraRef.current?.play();

      const activeCid = currentCreative?.id || "AD-LOCAL";
      const baseWs = apiUrl.replace(/^http/, "ws").replace(/\/$/, "");
      const wsUrl = `${baseWs}/ws/tracking/${encodeURIComponent(standeeId)}?campaign_id=${encodeURIComponent(campaignId)}&creative_id=${encodeURIComponent(activeCid)}&enable_attributes=true`;

      const socket = new WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => setStatusMessage("Đang kết nối AI backend...");
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "ready") {
          setStatus("running");
          setStatusMessage("AI Tracking đang hoạt động");
          timerRef.current = setInterval(sendFrame, 100);
        } else if (data.type === "frame_result") {
          setObservations(data.observations || []);
          drawBoxes(data.observations || [], data.frame_width, data.frame_height);
          setProcessingMs(data.processing_ms || 0);
          resultCounterRef.current.count += 1;
          const elapsed = (performance.now() - resultCounterRef.current.started) / 1000;
          if (elapsed >= 1.0) {
            setResultFps(resultCounterRef.current.count / elapsed);
            resultCounterRef.current = { count: 0, started: performance.now() };
          }
          if (data.closed_sessions?.length) loadReport();
        }
      };
      socket.onerror = () => {
        setStatus("error");
        setStatusMessage("Lỗi kết nối WebSocket");
      };
      socket.onclose = () => {
        if (status === "running") setStatusMessage("WebSocket đã ngắt");
      };
    } catch (error) {
      setStatus("error");
      setStatusMessage(`Lỗi: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const chooseLocalCreative = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const resolvedType = detectMediaType(file.type, file.name);
    setLocalCreative({
      id: `local_${file.name.replace(/\.[^.]+$/, "")}`,
      url: URL.createObjectURL(file),
      type: resolvedType,
      name: file.name,
    });
  };

  const clearLocalCreative = () => {
    setLocalCreative(null);
    loadPlaylist();
  };

  const enterFullscreen = () => adStageRef.current?.requestFullscreen();
  const connected = status === "running";

  return (
    <main className="app-shell">
      <header className="topbar">
        <Link href="/" className="brand">
          <Activity size={20} />
          <span>Tracking AI Standee</span>
        </Link>
        <nav className="nav-links">
          <Link href="/" className="nav-link active">
            <Tv size={16} /> Màn hình Standee
          </Link>
          <Link href="/admin" className="nav-link">
            <LayoutDashboard size={16} /> Quản trị Admin
          </Link>
        </nav>
        <div className={`connection ${connected ? "online" : ""}`}>
          {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
          {statusMessage}
        </div>
      </header>

      <section className="workspace">
        <div className="stage-column">
          <div className="section-heading">
            <div>
              <span className="eyebrow">
                {localCreative
                  ? "Tệp cục bộ (Thử nghiệm tạm)"
                  : playlist.length > 0
                  ? `Đang phát: ${currentIndex + 1} / ${playlist.length}`
                  : "Creative"}
              </span>
              <h1>{currentCreative?.name ?? "Chưa có nội dung trong playlist"}</h1>
            </div>
            <div className="toolbar">
              {/* Aspect Ratio Mode Switcher */}
              <button
                className="button secondary"
                onClick={() => setAspectMode((m) => (m === "landscape" ? "portrait" : "landscape"))}
                title="Đổi định dạng màn hình (16:9 Ngang hoặc 9:16 Dọc Standee)"
              >
                {aspectMode === "landscape" ? <Tv size={16} /> : <Smartphone size={16} />}
                <span>{aspectMode === "landscape" ? "Màn ngang (16:9)" : "Màn dọc Standee (9:16)"}</span>
              </button>

              {localCreative ? (
                <button className="button secondary" onClick={clearLocalCreative} title="Quay lại phát Playlist từ Admin">
                  <XCircle size={16} color="var(--red)" /> Trở về Playlist Server
                </button>
              ) : (
                <label className="button secondary" title="Tải file tạm từ máy">
                  <Upload size={16} /> Chọn tệp tạm
                  <input type="file" accept="video/*,image/*,.mp4,.webm,.mov,.jpg,.jpeg,.png" onChange={chooseLocalCreative} />
                </label>
              )}
              <Link href="/admin" className="button secondary">
                <Layers size={16} /> Quản lý Playlist ({playlist.length})
              </Link>
              <button className="icon-button" onClick={enterFullscreen} title="Toàn màn hình">
                <Expand size={18} />
              </button>
            </div>
          </div>

          {/* Ad Playback Stage */}
          <div className={`ad-stage ${aspectMode === "portrait" ? "portrait" : ""}`} ref={adStageRef}>
            {currentCreative && (
              <div className="playlist-badge">
                {currentCreative.type === "video" ? <Film size={13} /> : <FileImage size={13} />}
                <span>
                  {localCreative
                    ? `[Tệp tạm] ${currentCreative.name}`
                    : playlist.length > 0
                    ? `[${currentIndex + 1}/${playlist.length}] ${currentCreative.name}`
                    : currentCreative.name}
                </span>
              </div>
            )}

            {currentCreative?.type === "video" && (
              <>
                <video
                  ref={videoPlayerRef}
                  key={currentCreative.url}
                  src={currentCreative.url}
                  autoPlay
                  muted={isMuted}
                  playsInline
                  loop={playlist.length <= 1 && !localCreative}
                  onError={() => {
                    console.error("Lỗi tải video, tự động chuyển tiếp...");
                    setTimeout(nextCreative, 2000);
                  }}
                  onTimeUpdate={() => {
                    if (videoPlayerRef.current) {
                      setCurrentTime(videoPlayerRef.current.currentTime);
                      setDuration(videoPlayerRef.current.duration || 0);
                    }
                  }}
                  onLoadedMetadata={() => {
                    if (videoPlayerRef.current) {
                      setDuration(videoPlayerRef.current.duration || 0);
                    }
                  }}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={nextCreative}
                  style={{ width: "100%", height: "100%", objectFit: "contain" }}
                />

                {/* Custom Video Control Overlay */}
                <div className="stage-overlay-controls">
                  <div
                    className="progress-bar-container"
                    onClick={(e) => {
                      if (!videoPlayerRef.current || !duration) return;
                      const rect = e.currentTarget.getBoundingClientRect();
                      const pos = (e.clientX - rect.left) / rect.width;
                      videoPlayerRef.current.currentTime = pos * duration;
                    }}
                  >
                    <div
                      className="progress-bar-fill"
                      style={{ width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%" }}
                    />
                  </div>
                  <div className="controls-row">
                    <div className="controls-buttons">
                      <button className="ctrl-btn" onClick={prevCreative} title="Clip trước">
                        <SkipBack size={14} />
                      </button>
                      <button className="ctrl-btn" onClick={togglePlay} title={isPlaying ? "Tạm dừng" : "Phát"}>
                        {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                      </button>
                      <button className="ctrl-btn" onClick={nextCreative} title="Clip tiếp theo">
                        <SkipForward size={14} />
                      </button>
                      <button className="ctrl-btn" onClick={toggleMute} title={isMuted ? "Bật âm thanh" : "Tắt âm thanh"}>
                        {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
                      </button>
                    </div>
                    <div>
                      {Math.floor(currentTime)}s / {Math.floor(duration || 0)}s
                    </div>
                  </div>
                </div>
              </>
            )}

            {currentCreative?.type === "image" && (
              <>
                <img
                  key={currentCreative.url}
                  src={currentCreative.url}
                  alt={currentCreative.name}
                  style={{ width: "100%", height: "100%", objectFit: "contain" }}
                />
                {/* Image countdown indicator */}
                <div className="image-timer-bar" style={{ width: `${imageProgress}%` }} />
              </>
            )}

            {!currentCreative && (
              <div className="empty-stage">
                <MonitorPlay size={44} color="#88939e" />
                <strong>Chưa có nội dung nào trong Playlist</strong>
                <p>Hãy truy cập trang Quản trị Admin để upload video/ảnh quảng cáo.</p>
                <Link href="/admin" className="button primary" style={{ marginTop: "8px" }}>
                  <LayoutDashboard size={16} /> Đến trang Quản trị Admin
                </Link>
              </div>
            )}
          </div>

          {/* Playlist Carousel / Quick Switcher */}
          {playlist.length > 1 && !localCreative && (
            <div className="playlist-carousel-drawer">
              <div className="playlist-carousel-header">
                <span>Danh sách phát vòng lặp ({playlist.length} nội dung)</span>
                <span style={{ fontSize: "11px", color: "var(--muted)" }}>Bấm vào để chuyển ngay</span>
              </div>
              <div className="playlist-carousel-track">
                {playlist.map((item, idx) => {
                  const isActive = idx === currentIndex % playlist.length;
                  const fullMediaUrl = item.file_url.startsWith("http") ? item.file_url : `${apiUrl}${item.file_url}`;
                  const isVid = detectMediaType(item.media_type, item.file_name, item.file_url) === "video";
                  return (
                    <div
                      key={item.id}
                      className={`playlist-item-card ${isActive ? "active" : ""}`}
                      onClick={() => setCurrentIndex(idx)}
                    >
                      {isVid ? (
                        <video src={fullMediaUrl} className="playlist-card-thumb" muted preload="metadata" />
                      ) : (
                        <img src={fullMediaUrl} alt={item.name} className="playlist-card-thumb" />
                      )}
                      <div className="playlist-card-title">{item.name}</div>
                      <div className="playlist-card-sub">
                        <span>{isVid ? "VIDEO" : `${item.duration_seconds}s`}</span>
                        {isActive && <span style={{ color: "var(--green)", fontWeight: 700 }}>ĐANG PHÁT</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Realtime Metrics Strip */}
          <div className="metrics-strip">
            <Metric label="Lượt tiếp cận (Impressions)" value={overview.impressions} />
            <Metric label="Lượt xem thật (Viewers)" value={overview.viewers} />
            <Metric label="Xem chăm chú (Engaged)" value={overview.engaged_viewers} />
            <Metric label="Tỉ lệ chú ý (Attention)" value={`${Math.round(overview.attention_rate * 100)}%`} />
          </div>
        </div>

        {/* Right Sidebar: Controls & Camera Debug */}
        <aside className="control-column">
          <section className="panel config-panel">
            <div className="panel-title">
              <MonitorPlay size={17} />
              <h2>Điều khiển Camera Tracking</h2>
            </div>
            <div className="field-grid">
              <label>
                Mã Standee
                <input value={standeeId} onChange={(e) => setStandeeId(e.target.value)} disabled={connected} />
              </label>
              <label>
                Mã Chiến dịch
                <input value={campaignId} onChange={(e) => setCampaignId(e.target.value)} disabled={connected} />
              </label>
            </div>
            <div className="action-row" style={{ marginTop: "12px" }}>
              {!connected ? (
                <button className="button primary" onClick={startTracking} disabled={status === "connecting"}>
                  <Play size={17} /> Bắt đầu Tracking
                </button>
              ) : (
                <button className="button danger" onClick={stopTracking}>
                  <CircleStop size={17} /> Dừng Tracking
                </button>
              )}
              <button className="icon-button" onClick={loadReport} title="Làm mới báo cáo">
                <RefreshCw size={17} />
              </button>
            </div>
          </section>

          {/* Camera Debug Panel */}
          <section className="panel camera-panel">
            <div className="panel-title">
              <Camera size={17} />
              <h2>Camera AI Debug</h2>
              <span className="track-count">{observations.length} người đang xem</span>
            </div>
            <div className="camera-stage">
              <video ref={cameraRef} muted playsInline />
              <canvas ref={overlayRef} />
              {!streamRef.current && <div className="camera-empty"><Camera size={32} /></div>}
            </div>
            <canvas ref={captureRef} className="capture-canvas" />
            <div className="telemetry-row">
              <span>{resultFps.toFixed(1)} FPS</span>
              <span>{processingMs.toFixed(0)} ms</span>
              <span>{sentFrames} gửi</span>
              <span>{droppedFrames} drop</span>
            </div>
          </section>

          {/* Recent Sessions */}
          <section className="panel sessions-panel">
            <div className="panel-title">
              <Users size={17} />
              <h2>Session vừa xem xong</h2>
            </div>
            <div className="session-list">
              {sessions.length === 0 && <div className="empty-row">Chưa có session nào</div>}
              {sessions.map((session) => {
                const genderVi = session.gender === "female" ? "Nữ" : session.gender === "male" ? "Nam" : "";
                const ageText = session.estimated_age
                  ? `~${Math.round(session.estimated_age)} tuổi`
                  : session.age_group && session.age_group !== "unknown"
                  ? session.age_group
                  : "";
                const demo = [genderVi, ageText].filter(Boolean).join(" • ");
                return (
                  <div className="session-row" key={session.session_id} style={{ gridTemplateColumns: "8px 45px 1fr 1fr 1fr" }}>
                    <span className={`viewer-dot ${session.is_viewer ? "yes" : ""}`} />
                    <strong>#{session.provider_track_id}</strong>
                    <span>{session.presence_seconds.toFixed(1)}s đứng</span>
                    <span>{session.attention_seconds.toFixed(1)}s nhìn</span>
                    <span style={{ color: "var(--blue)", fontWeight: 600, fontSize: "10px", textAlign: "right" }}>
                      {demo || "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
