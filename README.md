# InstaGuard — Fake Instagram Profile Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![React](https://img.shields.io/badge/React-19-blue?logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow)

InstaGuard is a professional full-stack web application designed to detect fake or bot Instagram profiles. It leverages machine learning (Random Forest / XGBoost) trained on real Instagram data and provides real-time analysis through a sleek, modern React frontend.

## 👥 Team
- **Muhammad Moiz Nasim** — ML Engineer & Backend Developer
- **Muhammad Awais** — Frontend Developer & UI Designer

---

## 🏗️ Architecture
- **Frontend**: React 19, Vite, Framer Motion, Vanilla CSS (Design System)
- **Backend**: FastAPI (Python), PyJWT for Authentication, Pydantic for validation
- **Core ML Engine**: Scikit-Learn, XGBoost, SHAP (Explainability)
- **Database**: SQLite (`app.db`)

---

## 🚀 How to Run the Project

You will need to run both the backend API server and the frontend development server simultaneously.

### 1. Prerequisites
- **Python 3.9+**
- **Node.js 18+** & **npm**

### 2. Environment Setup

Copy the example environment file and fill in your own credentials:

```bash
cp .env.example .env
```

Open `.env` and fill in the required values (Instagram credentials, API keys, SMTP settings, etc.).  
See [`.env.example`](.env.example) for a description of every variable.

> **⚠️ Never commit your real `.env` file.** It is already listed in `.gitignore`.

### 3. Backend Setup (FastAPI)

Open a terminal in the root directory (`fake_instagram_detector`) and run:

```bash
# 1. Install all required Python dependencies
py -m pip install -r backend/requirements.txt

# 2. Start the FastAPI backend server
py -m uvicorn backend.main:app --reload --port 8000
```
*The backend API will be running at [http://localhost:8000](http://localhost:8000).*  
*Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).*

### 4. Frontend Setup (React + Vite)

Open a **new** terminal window, navigate to the `frontend` directory, and run:

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install all npm dependencies
npm install

# 3. Start the Vite development server
npm run dev
```
*The React web app will be running at [http://localhost:5173](http://localhost:5173).*

---

## 🔑 Default Admin Account
Sign up through the web interface. To promote your account to admin, update the `is_admin` column in `app.db` using any SQLite browser (e.g., [DB Browser for SQLite](https://sqlitebrowser.org/)).  
*(Note: `app.db` is generated automatically on first run.)*

---

## 📁 Project Structure
```text
fake_instagram_detector/
├── .env.example              # Template — copy to .env and fill in values
├── backend/                  # FastAPI Application
│   ├── main.py               # API entry point
│   ├── config.py             # App configurations & JWT secrets
│   ├── requirements.txt      # Python dependencies
│   ├── routers/              # API Endpoints (auth, analysis, users, admin)
│   ├── middleware/           # JWT & Auth logic
│   ├── core/                 # Core services (DB, ML, scraping, email)
│   └── schemas/              # Pydantic validation models
├── frontend/                 # React Web Application
│   ├── src/
│   │   ├── components/       # Layouts, Sidebar
│   │   ├── context/          # Auth & Theme Providers
│   │   ├── pages/            # Dashboard, Landing, Settings, Admin Pages
│   │   ├── api/              # Axios client with JWT interceptors
│   │   └── index.css         # Global Design System
│   └── vite.config.js
└── ml_pipeline/              # ML Training Pipeline
    ├── models/               # Serialized ML Models (.pkl)
    ├── data/                 # Datasets for training
    └── train_single.py       # Model training script
```

---

## 📄 License
This project is released under the [MIT License](LICENSE).
