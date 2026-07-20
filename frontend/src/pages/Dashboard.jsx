import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Sliders,
  AlertTriangle,
  CheckCircle,
  Download,
  RefreshCw,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useAuth } from "../context/AuthContext";
import api from "../api/client";
import toast from "react-hot-toast";

const FEATURE_LABELS = {
  followerRatio: "Follower Ratio",
  profileCompleteness: "Profile Completeness",
  engagementRate: "Engagement Rate",
  bioLength: "Bio Length",
  usernameAnomalyScore: "Username Anomaly",
  postFrequency: "Post Frequency",
  hasPicture: "Has Picture",
};

function ConfidenceGauge({ value }) {
  const pct = Math.round(value * 100);
  const r = 52,
    circ = 2 * Math.PI * r;
  const dash = circ * (1 - value);
  const color =
    value >= 0.7
      ? "var(--success)"
      : value >= 0.5
        ? "var(--warning)"
        : "var(--danger)";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "0.5rem",
      }}
    >
      <svg width={130} height={130} viewBox="0 0 130 130">
        <circle
          cx={65}
          cy={65}
          r={r}
          fill="none"
          stroke="var(--bg-elevated)"
          strokeWidth={12}
        />
        <circle
          cx={65}
          cy={65}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={12}
          strokeDasharray={circ}
          strokeDashoffset={dash}
          strokeLinecap="round"
          transform="rotate(-90 65 65)"
          style={{
            transition: "stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)",
          }}
        />
        <text
          x={65}
          y={60}
          textAnchor="middle"
          fill={color}
          fontSize={22}
          fontWeight={800}
        >
          {pct}%
        </text>
        <text
          x={65}
          y={78}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize={11}
        >
          Confidence
        </text>
      </svg>
    </div>
  );
}

export default function Dashboard() {
  const { user } = useAuth();
  const [showChart, setShowChart] = useState(true);
  const [tab, setTab] = useState("live");
  const [username, setUsername] = useState("");
  const [manual, setManual] = useState({
    username: "test_user",
    followers: 100,
    following: 200,
    posts: 10,
    has_pic: true,
    bio_length: 50,
    has_url: false,
    full_name: "",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const analyzeLive = async () => {
    if (!username.trim()) {
      setError("Enter a username.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const { data } = await api.post("/analysis/profile", {
        username: username.trim().replace("@", ""),
      });
      setResult(data);
      toast.success("Analysis complete!");
    } catch (e) {
      setError(e.response?.data?.detail || "Analysis failed.");
      toast.error("Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const analyzeManual = async () => {
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const { data } = await api.post("/analysis/manual", {
        ...manual,
        followers: Number(manual.followers),
        following: Number(manual.following),
        posts: Number(manual.posts),
        bio_length: Number(manual.bio_length),
      });
      setResult(data);
      toast.success("Analysis complete!");
    } catch (e) {
      setError(e.response?.data?.detail || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = async () => {
    if (!result) return;
    try {
      const res = await api.get(`/analysis/report/${result.history_id}`, {
        responseType: "blob",
      });
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report_${result.profile.username}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      toast.error("Failed to download report.");
    }
  };

  const shapData = result
    ? Object.entries(result.prediction.shap_values)
        .map(([k, v]) => ({
          name: FEATURE_LABELS[k] || k,
          value: parseFloat(v.toFixed(4)),
        }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    : [];

  return (
    <div>
      <div className="page-header">
        <h1>Profile Analysis</h1>
        <p>Analyze an Instagram profile for authenticity using AI</p>
      </div>

      {/* Tabs */}
      <div className="tabs" style={{ marginBottom: "1.5rem", maxWidth: 400 }}>
        <button
          className={`tab ${tab === "live" ? "active" : ""}`}
          onClick={() => setTab("live")}
        >
          <Search size={15} style={{ display: "inline", marginRight: 6 }} />
          Live Fetch
        </button>
        <button
          className={`tab ${tab === "manual" ? "active" : ""}`}
          onClick={() => setTab("manual")}
        >
          <Sliders size={15} style={{ display: "inline", marginRight: 6 }} />
          Manual Entry
        </button>
      </div>

      {/* Live Tab */}
      {tab === "live" && (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <p
            style={{
              fontSize: "0.875rem",
              color: "var(--text-secondary)",
              marginBottom: "1rem",
            }}
          >
            Enter any public Instagram username to fetch and analyze in
            real-time.
          </p>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <input
              id="live-username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && analyzeLive()}
              className="form-input"
              placeholder="@username or username"
              style={{ flex: 1, minWidth: 200 }}
            />
            <button
              id="live-analyze-btn"
              className="btn btn-primary"
              onClick={analyzeLive}
              disabled={loading}
            >
              {loading ? (
                <span className="spinner" />
              ) : (
                <>
                  <Search size={16} />
                  Analyze
                </>
              )}
            </button>
          </div>
          {error && (
            <div className="alert alert-error" style={{ marginTop: "0.75rem" }}>
              {error}
            </div>
          )}
        </div>
      )}

      {/* Manual Tab */}
      {tab === "manual" && (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <p
            style={{
              fontSize: "0.875rem",
              color: "var(--text-secondary)",
              marginBottom: "1rem",
            }}
          >
            Manually enter profile stats for analysis (useful for private
            profiles).
          </p>
          <div className="grid-3" style={{ marginBottom: "1rem" }}>
            {[
              { label: "Username", key: "username", type: "text" },
              { label: "Full Name", key: "full_name", type: "text" },
              { label: "Followers", key: "followers", type: "number" },
              { label: "Following", key: "following", type: "number" },
              { label: "Posts", key: "posts", type: "number" },
              {
                label: "Bio Length (chars)",
                key: "bio_length",
                type: "number",
              },
            ].map(({ label, key, type }) => (
              <div className="form-group" key={key}>
                <label className="form-label">{label}</label>
                <input
                  className="form-input"
                  type={type}
                  value={manual[key]}
                  onChange={(e) =>
                    setManual((m) => ({ ...m, [key]: e.target.value }))
                  }
                />
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: "1.5rem", marginBottom: "1rem" }}>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                cursor: "pointer",
                fontSize: "0.9rem",
              }}
            >
              <input
                type="checkbox"
                checked={manual.has_pic}
                onChange={(e) =>
                  setManual((m) => ({ ...m, has_pic: e.target.checked }))
                }
              />
              Has Profile Picture
            </label>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                cursor: "pointer",
                fontSize: "0.9rem",
              }}
            >
              <input
                type="checkbox"
                checked={manual.has_url}
                onChange={(e) =>
                  setManual((m) => ({ ...m, has_url: e.target.checked }))
                }
              />
              Has External URL
            </label>
          </div>
          <button
            id="manual-analyze-btn"
            className="btn btn-primary"
            onClick={analyzeManual}
            disabled={loading}
          >
            {loading ? (
              <span className="spinner" />
            ) : (
              <>
                <Search size={16} />
                Analyze Profile
              </>
            )}
          </button>
          {error && (
            <div className="alert alert-error" style={{ marginTop: "0.75rem" }}>
              {error}
            </div>
          )}
        </div>
      )}

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Profile Snapshot */}
            <div className="card" style={{ marginBottom: "1.25rem" }}>
              <h2
                style={{
                  fontSize: "1.1rem",
                  fontWeight: 700,
                  marginBottom: "1rem",
                }}
              >
                👤 Profile Snapshot — @{result.profile.username}
              </h2>
              <div className="stat-grid">
                <div className="stat-card">
                  <div className="stat-label">Followers</div>
                  <div className="stat-value" style={{ fontSize: "1.5rem" }}>
                    {result.profile.followersCount.toLocaleString()}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Following</div>
                  <div className="stat-value" style={{ fontSize: "1.5rem" }}>
                    {result.profile.followingCount.toLocaleString()}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Posts</div>
                  <div className="stat-value" style={{ fontSize: "1.5rem" }}>
                    {result.profile.postsCount.toLocaleString()}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Profile Pic</div>
                  <div className="stat-value" style={{ fontSize: "1.5rem" }}>
                    {result.profile.hasProfilePicture ? "✅" : "❌"}
                  </div>
                </div>
              </div>
              {result.profile.biography && (
                <p
                  style={{
                    marginTop: "0.75rem",
                    fontSize: "0.875rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  Bio: {result.profile.biography.slice(0, 120)}
                  {result.profile.biography.length > 120 ? "..." : ""}
                </p>
              )}
            </div>
            {/* Prediction */}
            <div className="card" style={{ marginBottom: "1.25rem" }}>
              <h2
                style={{
                  fontSize: "1.1rem",
                  fontWeight: 700,
                  marginBottom: "1rem",
                }}
              >
                🎯 Prediction Result
              </h2>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "2rem",
                  flexWrap: "wrap",
                }}
              >
                <div
                  style={{
                    padding: "1.25rem 2.5rem",
                    borderRadius: "var(--radius-lg)",
                    background:
                      result.prediction.label === "fake"
                        ? "var(--danger-bg)"
                        : "var(--success-bg)",
                    border: `2px solid ${result.prediction.label === "fake" ? "var(--danger)" : "var(--success)"}`,
                    textAlign: "center",
                  }}
                >
                  {result.prediction.label === "fake" ? (
                    <AlertTriangle size={32} color="var(--danger)" />
                  ) : (
                    <CheckCircle size={32} color="var(--success)" />
                  )}
                  <div
                    style={{
                      fontSize: "1.6rem",
                      fontWeight: 900,
                      color:
                        result.prediction.label === "fake"
                          ? "var(--danger)"
                          : "var(--success)",
                      marginTop: "0.25rem",
                    }}
                  >
                    {result.prediction.label.toUpperCase()}
                  </div>
                </div>
                <ConfidenceGauge value={result.prediction.confidence} />
                {result.prediction.low_confidence && (
                  <div className="alert alert-warning" style={{ flex: 1 }}>
                    <AlertTriangle size={18} /> Low confidence warning —
                    prediction may not be reliable.
                  </div>
                )}
              </div>
            </div>
            {/* Risk Profile */}
            <div className="card" style={{ marginBottom: "1.25rem" }}>
              <h2
                style={{
                  fontSize: "1.1rem",
                  fontWeight: 700,
                  marginBottom: "1rem",
                }}
              >
                🛡️ Risk Profile
              </h2>
              <div className="risk-grid">
                {Object.entries(result.prediction.risk_flags).map(
                  ([feat, info]) => (
                    <div
                      key={feat}
                      className={`risk-card ${info.flag === "⚠️" ? "warn" : "safe"}`}
                    >
                      <span style={{ fontSize: "1.1rem" }}>{info.flag}</span>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: "0.85rem" }}>
                          {FEATURE_LABELS[feat] || feat}
                        </div>
                        <div style={{ fontSize: "0.8rem", opacity: 0.8 }}>
                          {info.note}
                        </div>
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
            {/* SHAP Chart */}
            <div className="card" style={{ marginBottom: "1.25rem" }}>
              <h2
                style={{
                  fontSize: "1.1rem",
                  fontWeight: 700,
                  marginBottom: "1rem",
                }}
              >
                📊 Feature Importance (SHAP)
              </h2>
              <button
                className="btn btn-sm"
                style={{ marginBottom: "1rem" }}
                onClick={() => setShowChart(!showChart)}
              >
                {showChart ? "Hide" : "Show"} Chart
              </button>
              {showChart && (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={shapData}
                    layout="vertical"
                    margin={{ left: 20, right: 20 }}
                  >
                    <XAxis
                      type="number"
                      tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                      width={140}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--bg-card)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Bar dataKey="value" radius={4} barSize={20}>
                      {shapData.map((e, i) => (
                        <Cell
                          key={i}
                          fill={
                            e.value > 0 ? "var(--danger)" : "var(--success)"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
              <p
                style={{
                  fontSize: "0.78rem",
                  color: "var(--text-muted)",
                  marginTop: "0.5rem",
                }}
              >
                Green = pushes toward GENUINE · Red = pushes toward FAKE
              </p>
            </div>
            {/* Download */}
            <button className="btn btn-secondary" onClick={downloadPDF}>
              <Download size={16} /> Download PDF Report
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
