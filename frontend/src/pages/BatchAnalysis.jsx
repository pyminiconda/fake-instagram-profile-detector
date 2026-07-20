import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Layers,
  AlertTriangle,
  CheckCircle,
  Download,
  FileText,
  X,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import api from "../api/client";
import toast from "react-hot-toast";

export default function BatchAnalysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const inputRef = useRef();

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    if (!f.name.endsWith(".csv") && !f.name.endsWith(".txt")) {
      toast.error("Only CSV files are supported.");
      return;
    }
    setFile(f);
    setResult(null);
    setError("");
  };

  const analyze = async () => {
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }
    setError("");
    setLoading(true);
    setProgress(10);
    const form = new FormData();
    form.append("file", file);
    try {
      setProgress(40);
      const { data } = await api.post("/analysis/batch", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setProgress(100);
      setResult(data);
      toast.success(`Analyzed ${data.total} profiles!`);
    } catch (e) {
      setError(e.response?.data?.detail || "Batch analysis failed.");
      toast.error("Batch analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const downloadBatchPDF = async () => {
    try {
      const res = await api.get("/history/export/pdf", {
        responseType: "blob",
      });
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = "batch_report.pdf";
      a.click();
    } catch {
      toast.error("Export failed.");
    }
  };

  const pieData = result
    ? [
        { name: "Fake", value: result.fake_count, color: "var(--danger)" },
        {
          name: "Genuine",
          value: result.genuine_count,
          color: "var(--success)",
        },
      ]
    : [];

  return (
    <div>
      <div className="page-header">
        <h1>Batch Analysis</h1>
        <p>Analyze up to 50 Instagram profiles at once via CSV upload</p>
      </div>

      {/* Upload Area */}
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem" }}>
          Upload CSV File
        </h2>
        <div
          style={{
            fontSize: "0.875rem",
            color: "var(--text-secondary)",
            marginBottom: "1rem",
          }}
        >
          CSV format: one username per line (or with a header row "username").
          Max 50 profiles.
        </div>

        <div
          onClick={() => inputRef.current?.click()}
          style={{
            border: "2px dashed var(--border)",
            borderRadius: "var(--radius-lg)",
            padding: "2.5rem",
            textAlign: "center",
            cursor: "pointer",
            transition: "all var(--transition)",
            marginBottom: "1rem",
            background: file ? "rgba(108,99,255,0.05)" : "transparent",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.borderColor = "var(--brand-primary)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.borderColor = "var(--border)")
          }
        >
          {file ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.75rem",
              }}
            >
              <FileText size={24} color="var(--brand-secondary)" />
              <div>
                <div style={{ fontWeight: 600 }}>{file.name}</div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                  {(file.size / 1024).toFixed(1)} KB
                </div>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                }}
              >
                <X size={14} />
              </button>
            </div>
          ) : (
            <>
              <Upload
                size={32}
                style={{ margin: "0 auto 0.75rem", color: "var(--text-muted)" }}
              />
              <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
                Drop CSV here or click to browse
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Supports .csv files
              </div>
            </>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.txt"
          style={{ display: "none" }}
          onChange={handleFile}
        />

        {loading && (
          <div style={{ marginBottom: "1rem" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.8rem",
                color: "var(--text-muted)",
                marginBottom: "0.4rem",
              }}
            >
              <span>Analyzing profiles...</span>
              <span>{progress}%</span>
            </div>
            <div className="progress-track">
              <motion.div
                className="progress-fill"
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="alert alert-error" style={{ marginBottom: "1rem" }}>
            {error}
          </div>
        )}

        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button
            className="btn btn-primary"
            onClick={analyze}
            disabled={loading || !file}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <Layers size={16} />
                Analyze Batch
              </>
            )}
          </button>
          <a
            className="btn btn-ghost btn-sm"
            href="data:text/csv;charset=utf-8,username%0Aneymar%0Acristiano%0Atechcrunch"
            download="sample_batch.csv"
          >
            Download Sample CSV
          </a>
        </div>
      </div>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Summary Cards */}
            <div className="stat-grid" style={{ marginBottom: "1.25rem" }}>
              {[
                {
                  label: "Total Analyzed",
                  value: result.total,
                  color: "var(--brand-primary)",
                },
                {
                  label: "Fake Profiles",
                  value: result.fake_count,
                  color: "var(--danger)",
                },
                {
                  label: "Genuine Profiles",
                  value: result.genuine_count,
                  color: "var(--success)",
                },
                {
                  label: "Fake %",
                  value: `${result.fake_percentage}%`,
                  color: "var(--warning)",
                },
              ].map(({ label, value, color }) => (
                <div key={label} className="stat-card">
                  <div className="stat-label">{label}</div>
                  <div className="stat-value" style={{ color }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            {/* Pie chart */}
            <div className="card" style={{ marginBottom: "1.25rem" }}>
              <h2
                style={{
                  fontSize: "1rem",
                  fontWeight: 700,
                  marginBottom: "1rem",
                }}
              >
                Results Distribution
              </h2>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {pieData.map((e, i) => (
                      <Cell key={i} fill={e.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "var(--bg-card)",
                      border: "1px solid var(--border)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Detailed table */}
            <div className="card" style={{ marginBottom: "1.25rem" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "1rem",
                }}
              >
                <h2 style={{ fontSize: "1rem", fontWeight: 700 }}>
                  Detailed Results
                </h2>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={downloadBatchPDF}
                >
                  <Download size={14} /> Export PDF
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Username</th>
                      <th>Result</th>
                      <th>Confidence</th>
                      <th>Risk Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.results.map((r, i) => (
                      <tr key={i}>
                        <td
                          style={{
                            color: "var(--text-muted)",
                            fontSize: "0.8rem",
                          }}
                        >
                          {i + 1}
                        </td>
                        <td style={{ fontWeight: 600 }}>@{r.username}</td>
                        <td>
                          {r.error ? (
                            <span className="badge badge-warning">
                              ⚠️ Error
                            </span>
                          ) : (
                            <span className={`badge badge-${r.label}`}>
                              {r.label === "fake" ? "🚫" : "✅"}{" "}
                              {r.label?.toUpperCase()}
                            </span>
                          )}
                        </td>
                        <td style={{ fontSize: "0.875rem" }}>
                          {r.error ? "—" : `${Math.round(r.confidence * 100)}%`}
                        </td>
                        <td>
                          <span
                            style={{
                              fontSize: "0.8rem",
                              padding: "0.15rem 0.5rem",
                              borderRadius: "var(--radius-full)",
                              background:
                                r.risk_level === "High"
                                  ? "var(--danger-bg)"
                                  : r.risk_level === "Medium"
                                    ? "var(--warning-bg)"
                                    : "var(--success-bg)",
                              color:
                                r.risk_level === "High"
                                  ? "var(--danger)"
                                  : r.risk_level === "Medium"
                                    ? "var(--warning)"
                                    : "var(--success)",
                            }}
                          >
                            {r.risk_level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
