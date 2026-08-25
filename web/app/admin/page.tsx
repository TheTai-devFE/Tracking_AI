"use client";

import {
  Activity,
  ArrowDown,
  ArrowUp,
  BarChart3,
  CheckCircle2,
  Clock,
  Eye,
  FileImage,
  Film,
  Layers,
  LayoutDashboard,
  MonitorPlay,
  Play,
  Plus,
  RefreshCw,
  Trash2,
  Tv,
  Upload,
  Users,
  X,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Creative = {
  id: string;
  name: string;
  file_name: string;
  file_url: string;
  media_type: "video" | "image";
  duration_seconds: number;
  campaign_id: string;
  is_active: boolean;
  sort_order: number;
  created_at: string;
};

type CreativeReport = {
  creative_id: string;
  name: string;
  media_type: string;
  file_url: string;
  duration_seconds: number;
  is_active: boolean;
  sessions: number;
  impressions: number;
  viewers: number;
  engaged_viewers: number;
  total_presence_seconds: number;
  total_attention_seconds: number;
  average_presence_seconds: number;
  average_attention_seconds: number;
  attention_rate: number;
  look_away_count: number;
};

type Overview = {
  sessions: number;
  impressions: number;
  viewers: number;
  engaged_viewers: number;
  total_presence_seconds: number;
  total_attention_seconds: number;
  average_presence_seconds: number;
  average_attention_seconds: number;
  attention_rate: number;
};

type QueueItem = {
  id: string;
  file: File;
  previewUrl: string;
  mediaType: "video" | "image";
  name: string;
  duration: number;
  campaign: string;
  status: "pending" | "uploading" | "done" | "error";
  errorMsg?: string;
};

function deriveApiUrl() {
  if (process.env.NEXT_PUBLIC_TRACKING_API_URL) return process.env.NEXT_PUBLIC_TRACKING_API_URL;
  if (typeof window !== "undefined") return `${window.location.protocol}//${window.location.hostname}:8000`;
  return "http://localhost:8000";
}

function detectMediaType(file: File): "video" | "image" {
  if (file.type.startsWith("video/")) return "video";
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext && ["mp4", "webm", "mov", "mkv"].includes(ext)) return "video";
  return "image";
}

export default function AdminPage() {
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [activeTab, setActiveTab] = useState<"playlist" | "analytics">("playlist");
  const [creatives, setCreatives] = useState<Creative[]>([]);
  const [reports, setReports] = useState<CreativeReport[]>([]);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [loading, setLoading] = useState(false);

  // Upload Queue
  const [uploadQueue, setUploadQueue] = useState<QueueItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploadingAll, setIsUploadingAll] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Preview Modal
  const [previewMedia, setPreviewMedia] = useState<{ url: string; type: "video" | "image"; name: string } | null>(null);

  useEffect(() => setApiUrl(deriveApiUrl()), []);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [crRes, repRes, ovRes] = await Promise.all([
        fetch(`${apiUrl}/api/v1/creatives`),
        fetch(`${apiUrl}/api/v1/reports/creatives`),
        fetch(`${apiUrl}/api/v1/reports/overview`),
      ]);
      if (crRes.ok) setCreatives(await crRes.json());
      if (repRes.ok) setReports(await repRes.json());
      if (ovRes.ok) setOverview(await ovRes.json());
    } catch (err) {
      console.error("Failed to load admin data:", err);
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 8000);
    return () => clearInterval(interval);
  }, [loadData]);

  // Handle files added to queue
  const handleAddFiles = (files: FileList | File[]) => {
    const newItems: QueueItem[] = Array.from(files).map((file) => {
      const mediaType = detectMediaType(file);
      const cleanName = file.name.replace(/\.[^.]+$/, "");
      return {
        id: `q_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        file,
        previewUrl: URL.createObjectURL(file),
        mediaType,
        name: cleanName,
        duration: mediaType === "video" ? 10 : 10,
        campaign: "CMP-LOCAL",
        status: "pending",
      };
    });
    setUploadQueue((prev) => [...prev, ...newItems]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleAddFiles(e.dataTransfer.files);
    }
  };

  const handleUploadAll = async () => {
    const pending = uploadQueue.filter((q) => q.status === "pending" || q.status === "error");
    if (pending.length === 0) return;

    setIsUploadingAll(true);
    for (const item of pending) {
      setUploadQueue((prev) =>
        prev.map((q) => (q.id === item.id ? { ...q, status: "uploading", errorMsg: undefined } : q))
      );

      try {
        const formData = new FormData();
        formData.append("file", item.file);
        formData.append("name", item.name || item.file.name);
        formData.append("duration_seconds", String(item.duration));
        formData.append("campaign_id", item.campaign || "CMP-LOCAL");

        const res = await fetch(`${apiUrl}/api/v1/creatives/upload`, {
          method: "POST",
          body: formData,
        });

        if (res.ok) {
          setUploadQueue((prev) =>
            prev.map((q) => (q.id === item.id ? { ...q, status: "done" } : q))
          );
        } else {
          const err = await res.json().catch(() => ({}));
          setUploadQueue((prev) =>
            prev.map((q) =>
              q.id === item.id
                ? { ...q, status: "error", errorMsg: err.detail || "Lỗi upload" }
                : q
            )
          );
        }
      } catch (err) {
        setUploadQueue((prev) =>
          prev.map((q) =>
            q.id === item.id ? { ...q, status: "error", errorMsg: String(err) } : q
          )
        );
      }
    }
    setIsUploadingAll(false);
    loadData();
  };

  const removeQueueItem = (id: string) => {
    setUploadQueue((prev) => {
      const item = prev.find((q) => q.id === id);
      if (item) URL.revokeObjectURL(item.previewUrl);
      return prev.filter((q) => q.id !== id);
    });
  };

  const clearDoneQueue = () => {
    setUploadQueue((prev) => {
      prev.filter((q) => q.status === "done").forEach((q) => URL.revokeObjectURL(q.previewUrl));
      return prev.filter((q) => q.status !== "done");
    });
  };

  const handleToggle = async (id: string, current: boolean) => {
    try {
      await fetch(`${apiUrl}/api/v1/creatives/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_active: !current }),
      });
      setCreatives((prev) =>
        prev.map((c) => (c.id === id ? { ...c, is_active: !current } : c))
      );
    } catch (err) {
      console.error("Failed to toggle creative:", err);
    }
  };

  const handleUpdateDuration = async (id: string, duration: number) => {
    if (duration < 1) return;
    try {
      await fetch(`${apiUrl}/api/v1/creatives/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration_seconds: duration }),
      });
      setCreatives((prev) =>
        prev.map((c) => (c.id === id ? { ...c, duration_seconds: duration } : c))
      );
    } catch (err) {
      console.error("Failed to update duration:", err);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Bạn có chắc chắn muốn xóa nội dung "${name}" khỏi Playlist?`)) return;
    try {
      await fetch(`${apiUrl}/api/v1/creatives/${id}`, { method: "DELETE" });
      setCreatives((prev) => prev.filter((c) => c.id !== id));
      loadData();
    } catch (err) {
      console.error("Failed to delete creative:", err);
    }
  };

  // Reorder Item
  const handleMove = async (index: number, direction: "up" | "down") => {
    const targetIndex = direction === "up" ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= creatives.length) return;

    const newCreatives = [...creatives];
    const [moved] = newCreatives.splice(index, 1);
    newCreatives.splice(targetIndex, 0, moved);
    setCreatives(newCreatives);

    const ordered_ids = newCreatives.map((c) => c.id);
    try {
      await fetch(`${apiUrl}/api/v1/creatives/reorder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ordered_ids }),
      });
    } catch (err) {
      console.error("Failed to save reordered playlist:", err);
      loadData();
    }
  };

  // Total active rotation stats
  const activeCreatives = useMemo(() => creatives.filter((c) => c.is_active), [creatives]);
  const estimatedLoopDuration = useMemo(() => {
    return activeCreatives.reduce((acc, c) => acc + (c.duration_seconds || 10), 0);
  }, [activeCreatives]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <Link href="/" className="brand">
          <Activity size={20} />
          <span>Tracking AI</span>
        </Link>
        <nav className="nav-links">
          <Link href="/" className="nav-link">
            <Tv size={16} /> Màn hình Standee
          </Link>
          <Link href="/admin" className="nav-link active">
            <LayoutDashboard size={16} /> Quản trị Admin
          </Link>
        </nav>
      </header>

      <div className="admin-container">
        <div className="admin-header">
          <div>
            <span className="eyebrow">Content Management System</span>
            <h1>Quản trị Playlist & Thống kê Tracking</h1>
          </div>
          <div className="toolbar">
            <button className="button secondary" onClick={loadData} disabled={loading}>
              <RefreshCw size={16} className={loading ? "spin" : ""} /> Làm mới
            </button>
            <Link href="/" className="button primary">
              <MonitorPlay size={16} /> Mở Màn hình Standee
            </Link>
          </div>
        </div>

        {/* Overview Stats Cards */}
        {overview && (
          <div className="card-grid">
            <div className="stat-card">
              <span className="label">Lượt Tiếp Cận (Impressions)</span>
              <span className="value">{overview.impressions}</span>
              <span className="sub">Từ {overview.sessions} phiên phát hiện</span>
            </div>
            <div className="stat-card">
              <span className="label">Người Xem Thật (Viewers)</span>
              <span className="value" style={{ color: "var(--green)" }}>{overview.viewers}</span>
              <span className="sub">Nhìn trực diện vào màn hình</span>
            </div>
            <div className="stat-card">
              <span className="label">Xem Chăm Chú (Engaged)</span>
              <span className="value" style={{ color: "var(--blue)" }}>{overview.engaged_viewers}</span>
              <span className="sub">Duy trì chú ý liên tục</span>
            </div>
            <div className="stat-card">
              <span className="label">Tỉ Lệ Chú Ý Trung Bình</span>
              <span className="value" style={{ color: "var(--accent)" }}>
                {Math.round(overview.attention_rate * 100)}%
              </span>
              <span className="sub">Thời gian nhìn / Thời gian đứng</span>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="tab-bar">
          <button
            className={`tab-btn ${activeTab === "playlist" ? "active" : ""}`}
            onClick={() => setActiveTab("playlist")}
          >
            <Layers size={17} /> Quản lý Playlist ({creatives.length} nội dung)
          </button>
          <button
            className={`tab-btn ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => setActiveTab("analytics")}
          >
            <BarChart3 size={17} /> Báo cáo Hiệu quả từng Creative ({reports.length})
          </button>
        </div>

        {/* Tab 1: Playlist & Upload */}
        {activeTab === "playlist" && (
          <div>
            {/* Playlist Meta Banner */}
            <div className="playlist-meta-banner">
              <div className="playlist-meta-stats">
                <div className="playlist-meta-stat">
                  <span>Nội dung đang phát</span>
                  <strong>{activeCreatives.length} / {creatives.length} Active</strong>
                </div>
                <div className="playlist-meta-stat">
                  <span>Ước lượng 1 vòng lặp</span>
                  <strong>~{Math.round(estimatedLoopDuration)} giây</strong>
                </div>
              </div>
              <div style={{ display: "flex", gap: "10px" }}>
                <button
                  className="button secondary"
                  style={{ background: "rgba(255,255,255,0.15)", color: "#fff", borderColor: "rgba(255,255,255,0.2)" }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Plus size={16} /> Thêm Video / Ảnh mới
                </button>
              </div>
            </div>

            {/* Drag & Drop Upload Zone */}
            <div
              className={`upload-dropzone ${isDragging ? "dragging" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={32} />
              <p>Kéo thả nhiều Video (.mp4, .webm) hoặc Ảnh (.jpg, .png) vào đây</p>
              <span>hoặc nhấp để duyệt file từ máy tính</span>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="video/mp4,video/webm,image/*,.mp4,.webm,.mov,.jpg,.jpeg,.png"
                style={{ display: "none" }}
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleAddFiles(e.target.files);
                    e.target.value = "";
                  }
                }}
              />
            </div>

            {/* Upload Queue List */}
            {uploadQueue.length > 0 && (
              <div className="table-card" style={{ padding: "16px", marginTop: "16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                  <h3 style={{ margin: 0, fontSize: "15px", fontWeight: 700 }}>
                    Danh sách chờ tải lên ({uploadQueue.length} file)
                  </h3>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <button
                      className="button secondary"
                      style={{ minHeight: "32px", fontSize: "12px" }}
                      onClick={clearDoneQueue}
                      disabled={!uploadQueue.some((q) => q.status === "done")}
                    >
                      Dọn file đã xong
                    </button>
                    <button
                      className="button primary"
                      style={{ minHeight: "32px", fontSize: "12px" }}
                      onClick={handleUploadAll}
                      disabled={isUploadingAll || !uploadQueue.some((q) => q.status === "pending" || q.status === "error")}
                    >
                      {isUploadingAll ? <RefreshCw size={14} className="spin" /> : <Upload size={14} />}
                      {isUploadingAll ? "Đang tải lên..." : "Tải lên tất cả vào Playlist"}
                    </button>
                  </div>
                </div>

                <div className="upload-queue">
                  {uploadQueue.map((item) => (
                    <div key={item.id} className="queue-item">
                      {item.mediaType === "video" ? (
                        <video src={item.previewUrl} className="queue-thumb" muted playsInline />
                      ) : (
                        <img src={item.previewUrl} alt="preview" className="queue-thumb" />
                      )}
                      <div className="queue-info">
                        <div className="queue-fields">
                          <div>
                            <input
                              className="queue-input"
                              placeholder="Tên quảng cáo"
                              value={item.name}
                              disabled={item.status === "uploading" || item.status === "done"}
                              onChange={(e) =>
                                setUploadQueue((prev) =>
                                  prev.map((q) => (q.id === item.id ? { ...q, name: e.target.value } : q))
                                )
                              }
                            />
                          </div>
                          <div>
                            <input
                              className="queue-input"
                              type="number"
                              min="1"
                              max="300"
                              placeholder="Thời lượng (s)"
                              value={item.duration}
                              disabled={item.status === "uploading" || item.status === "done"}
                              onChange={(e) =>
                                setUploadQueue((prev) =>
                                  prev.map((q) =>
                                    q.id === item.id ? { ...q, duration: Number(e.target.value) } : q
                                  )
                                )
                              }
                            />
                          </div>
                          <div>
                            <input
                              className="queue-input"
                              placeholder="Campaign ID"
                              value={item.campaign}
                              disabled={item.status === "uploading" || item.status === "done"}
                              onChange={(e) =>
                                setUploadQueue((prev) =>
                                  prev.map((q) => (q.id === item.id ? { ...q, campaign: e.target.value } : q))
                                )
                              }
                            />
                          </div>
                        </div>
                        {item.errorMsg && (
                          <span style={{ fontSize: "11px", color: "var(--red)" }}>❌ {item.errorMsg}</span>
                        )}
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        {item.status === "done" && <CheckCircle2 size={18} color="var(--green)" />}
                        {item.status === "uploading" && <RefreshCw size={18} className="spin" color="var(--blue)" />}
                        <button
                          className="reorder-btn"
                          style={{ padding: "4px" }}
                          onClick={() => removeQueueItem(item.id)}
                          disabled={item.status === "uploading"}
                          title="Xóa khỏi hàng đợi"
                        >
                          <X size={16} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Playlist Table */}
            <div className="table-card" style={{ marginTop: "24px" }}>
              <div className="table-header">
                <h2>Thứ tự Phát trên Màn hình Standee</h2>
                <span style={{ fontSize: "12px", color: "var(--muted)" }}>
                  Standee sẽ phát tuần tự từ trên xuống dưới và lặp lại
                </span>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: "40px" }}>Thứ tự</th>
                    <th style={{ width: "70px" }}>Xem trước</th>
                    <th style={{ width: "60px" }}>Loại</th>
                    <th>Tên Quảng Cáo</th>
                    <th style={{ width: "130px" }}>Thời lượng (s)</th>
                    <th>Chiến dịch</th>
                    <th style={{ width: "90px" }}>Phát</th>
                    <th style={{ width: "80px", textAlign: "right" }}>Xóa</th>
                  </tr>
                </thead>
                <tbody>
                  {creatives.length === 0 && (
                    <tr>
                      <td colSpan={8} style={{ textAlign: "center", padding: "40px 20px", color: "var(--muted)" }}>
                        Chưa có nội dung nào trong playlist. Hãy kéo thả hoặc tải lên video/ảnh ở trên!
                      </td>
                    </tr>
                  )}
                  {creatives.map((c, index) => {
                    const fullMediaUrl = c.file_url.startsWith("http") ? c.file_url : `${apiUrl}${c.file_url}`;
                    return (
                      <tr key={c.id} style={{ opacity: c.is_active ? 1 : 0.6 }}>
                        <td>
                          <div className="reorder-btns">
                            <button
                              className="reorder-btn"
                              disabled={index === 0}
                              onClick={() => handleMove(index, "up")}
                              title="Di chuyển lên"
                            >
                              <ArrowUp size={12} />
                            </button>
                            <button
                              className="reorder-btn"
                              disabled={index === creatives.length - 1}
                              onClick={() => handleMove(index, "down")}
                              title="Di chuyển xuống"
                            >
                              <ArrowDown size={12} />
                            </button>
                          </div>
                        </td>
                        <td>
                          <div
                            style={{ cursor: "pointer", position: "relative" }}
                            onClick={() =>
                              setPreviewMedia({
                                url: fullMediaUrl,
                                type: c.media_type,
                                name: c.name,
                              })
                            }
                            title="Bấm để xem thử"
                          >
                            {c.media_type === "video" ? (
                              <video src={fullMediaUrl} className="table-thumb" muted preload="metadata" />
                            ) : (
                              <img src={fullMediaUrl} alt={c.name} className="table-thumb" />
                            )}
                          </div>
                        </td>
                        <td>
                          <span className={`badge ${c.media_type}`}>
                            {c.media_type === "video" ? <Film size={12} /> : <FileImage size={12} />}
                            {c.media_type.toUpperCase()}
                          </span>
                        </td>
                        <td>
                          <strong>{c.name}</strong>
                          <div style={{ fontSize: "11px", color: "var(--muted)" }}>{c.file_name}</div>
                        </td>
                        <td>
                          <div className="duration-input-wrap">
                            <input
                              type="number"
                              min="1"
                              max="300"
                              className="duration-input"
                              defaultValue={c.duration_seconds}
                              onBlur={(e) => handleUpdateDuration(c.id, Number(e.target.value))}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                  handleUpdateDuration(c.id, Number((e.target as HTMLInputElement).value));
                                }
                              }}
                            />
                            <span style={{ fontSize: "12px", color: "var(--muted)" }}>
                              {c.media_type === "video" ? "s (max)" : "giây"}
                            </span>
                          </div>
                        </td>
                        <td>
                          <code>{c.campaign_id}</code>
                        </td>
                        <td>
                          <label className="switch" title={c.is_active ? "Đang bật phát" : "Tạm tắt"}>
                            <input
                              type="checkbox"
                              checked={c.is_active}
                              onChange={() => handleToggle(c.id, c.is_active)}
                            />
                            <span className="slider" />
                          </label>
                        </td>
                        <td style={{ textAlign: "right" }}>
                          <button
                            className="icon-button button outline-danger"
                            onClick={() => handleDelete(c.id, c.name)}
                            title="Xóa khỏi playlist"
                          >
                            <Trash2 size={15} />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 2: Analytics by Creative */}
        {activeTab === "analytics" && (
          <div className="table-card">
            <div className="table-header">
              <h2>Báo cáo Thống kê Hiệu quả Từng Nội dung Quảng cáo</h2>
              <span style={{ fontSize: "12px", color: "var(--muted)" }}>Dữ liệu AI đo lường trực tiếp từ Standee</span>
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Nội dung (Creative)</th>
                  <th>Loại</th>
                  <th style={{ textAlign: "right" }}>Tiếp cận (Impressions)</th>
                  <th style={{ textAlign: "right" }}>Người xem (Viewers)</th>
                  <th style={{ textAlign: "right" }}>Chăm chú (Engaged)</th>
                  <th style={{ textAlign: "right" }}>Tổng time nhìn</th>
                  <th style={{ textAlign: "right" }}>Time nhìn TB</th>
                  <th style={{ textAlign: "right" }}>Tỉ lệ chú ý</th>
                  <th style={{ textAlign: "right" }}>Quay mặt đi</th>
                </tr>
              </thead>
              <tbody>
                {reports.length === 0 && (
                  <tr>
                    <td colSpan={9} style={{ textAlign: "center", padding: "32px", color: "var(--muted)" }}>
                      Chưa có dữ liệu tracking nào được ghi nhận. Hãy mở màn hình Standee và bật camera!
                    </td>
                  </tr>
                )}
                {reports.map((r) => (
                  <tr key={r.creative_id}>
                    <td>
                      <strong>{r.name}</strong>
                      <div style={{ fontSize: "11px", color: "var(--muted)" }}>ID: {r.creative_id}</div>
                    </td>
                    <td>
                      <span className={`badge ${r.media_type}`}>
                        {r.media_type === "video" ? <Film size={12} /> : <FileImage size={12} />}
                        {r.media_type}
                      </span>
                    </td>
                    <td style={{ textAlign: "right" }}><strong>{r.impressions}</strong></td>
                    <td style={{ textAlign: "right", color: "var(--green)", fontWeight: "700" }}>{r.viewers}</td>
                    <td style={{ textAlign: "right", color: "var(--blue)" }}>{r.engaged_viewers}</td>
                    <td style={{ textAlign: "right" }}>{r.total_attention_seconds.toFixed(1)}s</td>
                    <td style={{ textAlign: "right" }}>{r.average_attention_seconds.toFixed(1)}s</td>
                    <td style={{ textAlign: "right" }}>
                      <span
                        style={{
                          padding: "2px 8px",
                          borderRadius: "10px",
                          background: r.attention_rate > 0.6 ? "var(--green-soft)" : "var(--canvas)",
                          color: r.attention_rate > 0.6 ? "var(--green)" : "inherit",
                          fontWeight: "700",
                        }}
                      >
                        {Math.round(r.attention_rate * 100)}%
                      </span>
                    </td>
                    <td style={{ textAlign: "right", color: "var(--muted)" }}>{r.look_away_count} lần</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Preview Modal */}
      {previewMedia && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.85)",
            zIndex: 100,
            display: "grid",
            placeItems: "center",
            padding: "20px",
          }}
          onClick={() => setPreviewMedia(null)}
        >
          <div
            style={{
              background: "#1c2127",
              borderRadius: "10px",
              padding: "16px",
              maxWidth: "800px",
              width: "100%",
              maxHeight: "90vh",
              display: "flex",
              flexDirection: "column",
              gap: "12px",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", color: "#fff" }}>
              <h3 style={{ margin: 0, fontSize: "16px" }}>{previewMedia.name}</h3>
              <button
                className="icon-button"
                style={{ background: "transparent", color: "#fff", borderColor: "#444" }}
                onClick={() => setPreviewMedia(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div style={{ width: "100%", maxHeight: "70vh", overflow: "hidden", display: "grid", placeItems: "center" }}>
              {previewMedia.type === "video" ? (
                <video src={previewMedia.url} controls autoPlay style={{ maxWidth: "100%", maxHeight: "65vh", borderRadius: "6px" }} />
              ) : (
                <img src={previewMedia.url} alt={previewMedia.name} style={{ maxWidth: "100%", maxHeight: "65vh", objectFit: "contain", borderRadius: "6px" }} />
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
