import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ShieldCheck,
  Search,
  BarChart2,
  Users,
  Brain,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Zap,
  Lock,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import styles from "./Landing.module.css";

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 30 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.6, delay, ease: [0.4, 0, 0.2, 1] },
});

const features = [
  {
    icon: Brain,
    title: "AI-Powered Detection",
    desc: "XGBoost & Random Forest models trained on real Instagram data with 80%+ accuracy.",
  },
  {
    icon: Zap,
    title: "Instant Analysis",
    desc: "Get results in seconds with live profile fetching via Apify & RapidAPI.",
  },
  {
    icon: BarChart2,
    title: "SHAP Explainability",
    desc: "Understand exactly why a profile is flagged — feature-level AI transparency.",
  },
  {
    icon: Search,
    title: "Batch Processing",
    desc: "Analyze up to 50 profiles at once via CSV upload for large-scale investigations.",
  },
  {
    icon: Lock,
    title: "Secure & Private",
    desc: "All data stored locally in SQLite. No third-party data sharing, ever.",
  },
  {
    icon: Users,
    title: "Team Admin Controls",
    desc: "Full admin panel to manage users, retrain models, and view system analytics.",
  },
];

const steps = [
  {
    step: "01",
    title: "Enter a Username",
    desc: "Type any public Instagram username into the analyzer.",
  },
  {
    step: "02",
    title: "AI Analyzes Profile",
    desc: "Our model inspects 7 key behavioral features in milliseconds.",
  },
  {
    step: "03",
    title: "Get the Verdict",
    desc: "See FAKE or GENUINE with confidence score, SHAP chart & risk flags.",
  },
];

const stats = [
  { value: "95%+", label: "Model Accuracy" },
  { value: "7", label: "AI Features" },
  { value: "50", label: "Batch Capacity" },
  { value: "<5s", label: "Avg. Response" },
];

const team = [
  {
    name: "Muhammad Moiz Nasim",
    role: "ML Engineer & Backend Developer",
    initials: "MN",
    desc: "Designed the AI pipeline, feature engineering, and FastAPI backend architecture.",
    color: "#6c63ff",
  },
  {
    name: "Muhammad Awais",
    role: "Frontend Developer & UI Designer",
    initials: "MA",
    desc: "Built the React frontend, design system, and user experience flows.",
    color: "#a78bfa",
  },
];

export default function Landing() {
  const { user } = useAuth();

  return (
    <div className={styles.page}>
      {/* ── Navbar ── */}
      <nav className={styles.navbar}>
        <div className={styles.navBrand}>
          <div className={styles.navLogo}>
            <ShieldCheck size={20} strokeWidth={2.5} />
          </div>
          <span className={styles.navName}>InstaGuard</span>
        </div>
        <div className={styles.navLinks}>
          <a href="#features">Features</a>
          <a href="#how-it-works">How It Works</a>
          {/* <a href="#team">Team</a> */}
        </div>
        <div className={styles.navActions}>
          {user ? (
            <Link to="/dashboard" className="btn btn-primary btn-sm">
              Go to Dashboard
            </Link>
          ) : (
            <>
              <Link to="/login" className="btn btn-ghost btn-sm">
                Login
              </Link>
              <Link to="/signup" className="btn btn-primary btn-sm">
                Get Started
              </Link>
            </>
          )}
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className={styles.hero}>
        <div className={styles.heroBg} />
        <div className={styles.heroOrb1} />
        <div className={styles.heroOrb2} />

        <motion.div
          className={styles.heroContent}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
        >
          <h1 className={styles.heroTitle}>
            Shield Your Feed from
            <br />
            <span className="gradient-text">Fake Profiles</span>
          </h1>

          <p className={styles.heroSub}>
            InstaGuard uses advanced machine learning to detect fake Instagram
            accounts with 95%+ accuracy — in under a second.
          </p>

          <div className={styles.heroCtas}>
            {user ? (
              <Link to="/dashboard" className="btn btn-primary btn-lg">
                Go to Dashboard <ArrowRight size={18} />
              </Link>
            ) : (
              <>
                <Link to="/signup" className="btn btn-primary btn-lg">
                  Start Detecting Free <ArrowRight size={18} />
                </Link>
                <Link to="/login" className="btn btn-secondary btn-lg">
                  Sign In
                </Link>
              </>
            )}
          </div>
        </motion.div>

        {/* Floating result card preview */}
        <motion.div
          className={styles.heroCard}
          initial={{ opacity: 0, x: 60, rotate: 3 }}
          animate={{ opacity: 1, x: 0, rotate: 0 }}
          transition={{ duration: 1, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
        >
          <div className={styles.heroCardHeader}>
            <div className={styles.hcDot} style={{ background: "#ef4444" }} />
            <div className={styles.hcDot} style={{ background: "#f59e0b" }} />
            <div className={styles.hcDot} style={{ background: "#22c55e" }} />
            <span
              style={{
                marginLeft: "auto",
                fontSize: "0.75rem",
                color: "var(--text-muted)",
              }}
            >
              InstaGuard Analysis
            </span>
          </div>
          <div className={styles.heroCardBody}>
            <div className={styles.hcUsername}>@suspicious_acc123</div>
            <div className={styles.hcResult}>
              <AlertTriangle size={18} color="#ef4444" />
              <span style={{ color: "#ef4444", fontWeight: 700 }}>
                FAKE PROFILE
              </span>
            </div>
            <div className={styles.hcConf}>
              Confidence: <b>94.2%</b>
            </div>
            <div className={styles.hcFlags}>
              <span
                className={styles.hcFlag}
                style={{
                  background: "var(--danger-bg)",
                  color: "var(--danger)",
                }}
              >
                ⚠️ No profile pic
              </span>
              <span
                className={styles.hcFlag}
                style={{
                  background: "var(--danger-bg)",
                  color: "var(--danger)",
                }}
              >
                ⚠️ Bot-like username
              </span>
              <span
                className={styles.hcFlag}
                style={{
                  background: "var(--success-bg)",
                  color: "var(--success)",
                }}
              >
                ✅ Has bio
              </span>
            </div>
          </div>
        </motion.div>
      </section>

      {/* ── Stats ── */}
      <section className={styles.statsSection}>
        <div className={styles.statsGrid}>
          {stats.map(({ value, label }, i) => (
            <motion.div
              key={label}
              className={styles.statItem}
              {...fadeUp(i * 0.1)}
            >
              <div className={styles.statValue}>{value}</div>
              <div className={styles.statLabel}>{label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className={styles.section}>
        <motion.div className={styles.sectionHeader} {...fadeUp()}>
          <div className={styles.sectionBadge}>Features</div>
          <h2>
            Everything You Need to
            <br />
            <span className="gradient-text">Detect Fake Accounts</span>
          </h2>
          <p>
            A complete suite of AI tools for Instagram authenticity
            verification.
          </p>
        </motion.div>

        <div className={styles.featuresGrid}>
          {features.map(({ icon: Icon, title, desc }, i) => (
            <motion.div
              key={title}
              className={styles.featureCard}
              {...fadeUp(i * 0.08)}
            >
              <div className={styles.featureIcon}>
                <Icon size={22} />
              </div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── How It Works ── */}
      <section id="how-it-works" className={styles.section}>
        <motion.div className={styles.sectionHeader} {...fadeUp()}>
          <div className={styles.sectionBadge}>How It Works</div>
          <h2>
            Three Steps to <span className="gradient-text">Truth</span>
          </h2>
          <p>Identifying fake profiles has never been easier or faster.</p>
        </motion.div>

        <div className={styles.stepsGrid}>
          {steps.map(({ step, title, desc }, i) => (
            <motion.div
              key={step}
              className={styles.stepCard}
              {...fadeUp(i * 0.15)}
            >
              <div className={styles.stepNum}>{step}</div>
              <h3>{title}</h3>
              <p>{desc}</p>
              {i < steps.length - 1 && (
                <div className={styles.stepArrow}>
                  <ArrowRight size={18} />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className={styles.cta}>
        <motion.div {...fadeUp()}>
          <h2>Ready to Detect Fake Profiles?</h2>
          <p>
            Join InstaGuard and start verifying Instagram authenticity with AI
            today.
          </p>
          {user ? (
            <Link to="/dashboard" className="btn btn-primary btn-lg">
              Go to Dashboard <ArrowRight size={18} />
            </Link>
          ) : (
            <Link to="/signup" className="btn btn-primary btn-lg">
              Create Free Account <ArrowRight size={18} />
            </Link>
          )}
        </motion.div>
      </section>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <div className={styles.footerBrand}>
          <ShieldCheck size={18} />
          <span>InstaGuard</span>
        </div>
        <p>
          Contact us:{" "}
          <a
            href="mailto:instagaurdofficial@gmail.com"
            style={{ color: "inherit", textDecoration: "underline" }}
          >
            instagaurdofficial@gmail.com
          </a>
        </p>
        <p>© 2026 InstaGuard · AI-Powered Instagram Fake Profile Detection</p>
      </footer>
    </div>
  );
}
