# 📖 InstaGuard — User Manual

**Version:** 1.0.0 | **Authors:** Muhammad Moiz Nasim & Muhammad Awais

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Required Software Installation](#2-required-software-installation)
   - 2.1 [Install Python 3.9+](#21-install-python-39)
   - 2.2 [Install Node.js 18+](#22-install-nodejs-18)
   - 2.3 [Install Git (Optional)](#23-install-git-optional)
3. [Getting the Project Files](#3-getting-the-project-files)
   - 3.1 [Download via Git (Recommended)](#31-download-via-git-recommended)
   - 3.2 [Download as ZIP](#32-download-as-zip)
   - 3.3 [Recommended Folder Location](#33-recommended-folder-location)
4. [Project Structure Overview](#4-project-structure-overview)
5. [Environment Configuration (.env)](#5-environment-configuration-env)
   - 5.1 [Create the .env File](#51-create-the-env-file)
   - 5.2 [Environment Variable Reference](#52-environment-variable-reference)
   - 5.3 [Demo Mode vs. Live Mode](#53-demo-mode-vs-live-mode)
6. [Installing Dependencies](#6-installing-dependencies)
   - 6.1 [Python Dependencies](#61-python-dependencies)
   - 6.2 [Frontend (Node) Dependencies](#62-frontend-node-dependencies)
7. [Running the Application](#7-running-the-application)
   - 7.1 [Start the Backend (FastAPI)](#71-start-the-backend-fastapi)
   - 7.2 [Start the Frontend (React + Vite)](#72-start-the-frontend-react--vite)
   - 7.3 [Verifying Everything Works](#73-verifying-everything-works)
8. [Using the Application](#8-using-the-application)
   - 8.1 [Register & Login](#81-register--login)
   - 8.2 [Analyse a Profile](#82-analyse-a-profile)
   - 8.3 [View History](#83-view-history)
   - 8.4 [Admin Panel](#84-admin-panel)
9. [Stopping the Application](#9-stopping-the-application)
10. [Troubleshooting](#10-troubleshooting)
11. [Complete Quick-Start Checklist](#11-complete-quick-start-checklist)

---

## 1. System Requirements

| Component | Minimum |
|-----------|---------|
| **Operating System** | Windows 10 / macOS 12 / Ubuntu 20.04 |
| **RAM** | 4 GB (8 GB recommended) |
| **Disk Space** | ~2 GB free (for models, packages, node_modules) |
| **Python** | 3.9 or newer |
| **Node.js** | 18 or newer |
| **npm** | 9 or newer (bundled with Node.js) |
| **Internet** | Required for installing packages and for live scraping |

> **Note:** The application has been tested on Windows 10/11. All terminal commands in this manual use the Windows `py` launcher. On macOS/Linux, replace `py` with `python3`.

---

## 2. Required Software Installation

### 2.1 Install Python 3.9+

1. Open your browser and go to **https://www.python.org/downloads/**
2. Click **"Download Python 3.x.x"** (choose 3.9 or any newer version).
3. Run the downloaded installer.
4. **Important:** On the first screen, check the box **"Add Python to PATH"** before clicking Install.
5. Click **"Install Now"** and wait for it to finish.
6. Verify the installation by opening a terminal (PowerShell on Windows) and typing:

```powershell
py --version
```

You should see something like `Python 3.11.9`.

---

### 2.2 Install Node.js 18+

1. Open your browser and go to **https://nodejs.org/en/download/**
2. Download the **LTS (Long Term Support)** version for your operating system.
3. Run the installer and follow the default steps (keep all defaults).
4. Verify the installation by opening a terminal and typing:

```powershell
node --version
npm --version
```

You should see `v18.x.x` (or higher) for Node and `9.x.x` (or higher) for npm.

---

### 2.3 Install Git (Optional)

Git is only required if you want to clone the repository directly. If you downloaded the project as a ZIP file, you can skip this step.

1. Go to **https://git-scm.com/downloads**
2. Download and install Git for your operating system using the default settings.
3. Verify:

```powershell
git --version
```

---

## 3. Getting the Project Files

### 3.1 Download via Git (Recommended)

Open a terminal (PowerShell) and run:

```powershell
git clone https://github.com/pyminiconda/fake-instagram-profile-detector.git
```

This will create a folder called `fake-instagram-profile-detector` in your current directory.

### 3.2 Download as ZIP

1. Go to the project's GitHub page.
2. Click the green **"Code"** button then **"Download ZIP"**.
3. Extract the ZIP file to a location of your choice.

### 3.3 Recommended Folder Location

Place the project folder somewhere simple and without spaces in the path.

**Good locations:**
```
C:\Projects\fake_instagram_detector\
C:\Users\YourName\Documents\fake_instagram_detector\
```

**Avoid:**
```
C:\Users\Your Name\Desktop\My Projects\fake instagram detector\   (spaces cause issues)
```

After extracting, your folder should look like this:

```
fake_instagram_detector/        <- this is the ROOT folder
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── USER_MANUAL.md
├── backend/
├── frontend/
└── ml_pipeline/
```

> **All terminal commands in this manual must be run from inside this ROOT folder** unless stated otherwise.

---

## 4. Project Structure Overview

```
fake_instagram_detector/
│
├── .env.example              <- Template for your secret keys (copy to .env)
├── .env                      <- YOUR config file (never share this!)
│
├── backend/                  <- Python FastAPI server
│   ├── main.py               <- API entry point (run this)
│   ├── config.py             <- Loads settings from .env
│   ├── requirements.txt      <- All Python packages needed
│   ├── core/                 <- Database, ML engine, scraping, email
│   ├── routers/              <- API routes: auth, analysis, history, admin
│   ├── middleware/           <- JWT authentication logic
│   └── schemas/              <- Request/response data models
│
├── frontend/                 <- React web application (Vite)
│   ├── package.json          <- Node.js packages list
│   ├── vite.config.js        <- Dev server config (port 5173)
│   └── src/
│       ├── pages/            <- All UI pages (Dashboard, Landing, Admin...)
│       ├── components/       <- Reusable UI components
│       ├── context/          <- Auth and Theme state
│       ├── api/              <- Axios HTTP client
│       └── index.css         <- Global design system and styles
│
└── ml_pipeline/
    ├── models/
    │   ├── best_model.pkl    <- Pre-trained ML model (Random Forest/XGBoost)
    │   └── scaler.pkl        <- Feature scaler
    ├── data/                 <- Training datasets
    └── train_single.py       <- Script to re-train the model (optional)
```

---

## 5. Environment Configuration (.env)

The application uses a `.env` file to store secret credentials (API keys, passwords, etc.). This file is **never** included in the download — you must create it yourself.

### 5.1 Create the .env File

**Option A — Copy the template (recommended):**

In your terminal, from the ROOT folder, run:

```powershell
# Windows PowerShell
copy .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

**Option B — Manual creation:**

1. Open the ROOT folder in File Explorer.
2. Create a new file called exactly `.env` (with a dot at the start, no extension).
3. Open it with Notepad or any text editor.
4. Paste the contents from `.env.example` and fill in your values.

---

### 5.2 Environment Variable Reference

Open your `.env` file and fill in each value:

```ini
# Instagram Credentials
# Used by Instaloader to scrape live Instagram profiles.
# Leave EMPTY to run in Demo Mode (no real scraping).
INSTA_USERNAME=your_instagram_username
INSTA_PASSWORD=your_instagram_password

# Apify Token
# Used as a fallback scraper for more reliable data.
# Get your free token at: https://console.apify.com/account/integrations
APIFY_TOKEN=your_apify_api_token_here

# RapidAPI Key
# Used for the instagram120 API on RapidAPI.
# Sign up at: https://rapidapi.com/hub
RAPIDAPI_KEY=your_rapidapi_key_here

# JWT Secret Key
# Used to sign authentication tokens. MUST be long and random.
# Generate one with: py -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=paste_your_generated_secret_here
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS Origin
# The address of the frontend. Leave as-is for local development.
FRONTEND_ORIGIN=http://localhost:5173

# SMTP Email Settings
# Used to send OTP / password-reset emails.
# Use a Gmail App Password (NOT your real Google password).
# How to create an App Password: https://support.google.com/accounts/answer/185833
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_gmail_address@gmail.com
SMTP_PASSWORD=your_16_char_app_password
SMTP_SENDER=InstaGuard Security <your_gmail_address@gmail.com>
```

> **Minimum required for basic use:**
> You only *need* to set `JWT_SECRET_KEY`. All other fields are optional — without Instagram/Apify/RapidAPI keys the app runs in **Demo Mode**.

---

### 5.3 Demo Mode vs. Live Mode

| Mode | When it activates | What it does |
|------|------------------|--------------|
| **Demo Mode** | `INSTA_USERNAME` / `INSTA_PASSWORD` are empty | You enter profile metrics manually in the UI. No live scraping. |
| **Live Mode** | Valid Instagram credentials provided | The app automatically fetches profile data by username. |

Demo Mode is perfectly suitable for testing and grading the project.

---

## 6. Installing Dependencies

### 6.1 Python Dependencies

Open a terminal (PowerShell) and navigate to the ROOT folder:

```powershell
cd C:\Projects\fake_instagram_detector
```

Install all required Python packages with a single command:

```powershell
py -m pip install -r backend/requirements.txt
```

This will install FastAPI, uvicorn, scikit-learn, XGBoost, SHAP, instaloader, and all other backend packages. This may take **3–5 minutes** depending on your internet speed.

**Verify the installation:**

```powershell
py -c "import fastapi, sklearn, xgboost; print('All packages OK')"
```

You should see `All packages OK`.

---

### 6.2 Frontend (Node) Dependencies

Open a **second terminal** window and navigate to the `frontend` subfolder:

```powershell
cd C:\Projects\fake_instagram_detector\frontend
```

Install all Node.js packages:

```powershell
npm install
```

This downloads React, Vite, Axios, Framer Motion, Recharts, and all other frontend dependencies into `frontend/node_modules/`. This may take **1–3 minutes**.

---

## 7. Running the Application

The app has two servers that must run simultaneously in **two separate terminal windows**.

---

### 7.1 Start the Backend (FastAPI)

In **Terminal 1**, from the **ROOT folder**:

```powershell
py -m uvicorn backend.main:app --reload --port 8000
```

You should see output similar to:

```
=======================================================
  InstaGuard API v1.0.0 — Starting up...
=======================================================
  [DB]     SQLite connected.
  [ML]     Model ready: True
  [Fetch]  Demo mode: True
  [Docs]   http://localhost:8000/docs
=======================================================

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...]
```

> The backend is ready when you see **"Uvicorn running on http://0.0.0.0:8000"**

**Useful backend URLs:**

| URL | Purpose |
|-----|---------|
| `http://localhost:8000/api/health` | Health check — should return `{"status": "ok"}` |
| `http://localhost:8000/docs` | Interactive API documentation (Swagger UI) |
| `http://localhost:8000/redoc` | Alternative API docs (ReDoc) |

---

### 7.2 Start the Frontend (React + Vite)

In **Terminal 2**, from the **`frontend` subfolder**:

```powershell
cd C:\Projects\fake_instagram_detector\frontend
npm run dev
```

You should see:

```
  VITE v8.x.x  ready in 300 ms

  Local:   http://localhost:5173/
  Network: use --host to expose
```

> The frontend is ready when you see **"Local: http://localhost:5173/"**

---

### 7.3 Verifying Everything Works

1. Open your browser and go to **http://localhost:5173**
2. You should see the InstaGuard landing page.
3. In a new tab, go to **http://localhost:8000/api/health**
4. You should see: `{"status":"ok","app":"InstaGuard","version":"1.0.0"}`

If both pages load, the application is running correctly.

---

## 8. Using the Application

### 8.1 Register & Login

1. Navigate to **http://localhost:5173**
2. Click **"Get Started"** or **"Sign Up"**.
3. Fill in your name, email, and a password (minimum 8 characters).
4. Click **"Create Account"** — you will be logged in automatically.
5. To log in again later, use your email and password on the **Login** page.

> **Note:** User data is stored locally in `app.db` (SQLite). This file is created automatically the first time the backend starts.

---

### 8.2 Analyse a Profile

**In Live Mode** (Instagram credentials set in `.env`):
1. Go to the **Dashboard**.
2. Enter an Instagram username (e.g. `cristiano`) in the search bar.
3. Click **"Analyse"**.
4. The app fetches the profile, runs it through the ML model, and displays:
   - Fake / Real verdict with confidence score
   - SHAP explainability chart showing which features influenced the decision
   - Full profile metrics table

**In Demo Mode** (no credentials):
1. Go to the **Dashboard**.
2. You will see a form with manual feature inputs (follower count, post count, bio length, etc.).
3. Fill in the values and click **"Analyse"**.
4. The verdict and explanation are shown the same way.

---

### 8.3 View History

- Click **"History"** in the sidebar to see all past analyses.
- You can filter, sort, and export results to **PDF** or **Excel**.

---

### 8.4 Admin Panel

The Admin Panel provides access to:
- **User management** — view all registered users, toggle admin status
- **Model management** — view model metrics, re-trigger training
- **System statistics** — total analyses, fake/real ratio, etc.

**To access the Admin Panel:**

1. Download [DB Browser for SQLite](https://sqlitebrowser.org/) (free).
2. Open `app.db` (located in the ROOT folder) with it.
3. Go to the **"Browse Data"** tab and select the `users` table.
4. Find your user row and set `is_admin` to `1`.
5. Click **"Write Changes"** and close the file.
6. Log out of InstaGuard and log back in — the **Admin** link will appear in the sidebar.

---

## 9. Stopping the Application

To stop either server, switch to its terminal window and press:

```
Ctrl + C
```

Repeat for both Terminal 1 (backend) and Terminal 2 (frontend).

---

## 10. Troubleshooting

### `py` command not found
- Make sure Python was installed with **"Add to PATH"** checked.
- Restart your terminal after installing Python.
- On macOS/Linux, use `python3` instead of `py`.

### `npm` command not found
- Make sure Node.js was installed correctly.
- Restart your terminal after installing Node.js.

### Backend fails to start — `ModuleNotFoundError`
- Make sure you ran `py -m pip install -r backend/requirements.txt` from the ROOT folder.
- If the error mentions a specific package, install it manually:
  ```powershell
  py -m pip install <package_name>
  ```

### `app.db` permission error
- Close any SQLite browser tools that might have the file open.
- Make sure you are running the terminal from the ROOT folder, not a subdirectory.

### Frontend shows blank page or "Failed to fetch"
- Make sure the **backend is running** in Terminal 1 on port 8000.
- Check that no firewall is blocking port 8000.
- Visit `http://localhost:8000/api/health` — if it fails, the backend crashed. Check Terminal 1 for the error message.

### CORS errors in the browser console
- Confirm that `FRONTEND_ORIGIN=http://localhost:5173` is set in your `.env` file.
- Restart the backend after editing `.env`.

### Email / OTP not working
- Set `SMTP_USERNAME` and `SMTP_PASSWORD` in `.env` (use a Gmail App Password).
- Without SMTP settings, the email feature is disabled but the rest of the app still works normally.

### Port already in use
If ports 8000 or 5173 are occupied, find and kill the process:

```powershell
# Windows — find what is using port 8000
netstat -ano | findstr :8000

# Kill the process (replace XXXX with the PID shown)
taskkill /PID XXXX /F
```

---

## 11. Complete Quick-Start Checklist

Use this checklist for a fresh installation from scratch:

- [ ] **Install Python 3.9+** — https://www.python.org/downloads/ *(check "Add to PATH")*
- [ ] **Install Node.js 18+** — https://nodejs.org/en/download/
- [ ] **Download the project** — clone via Git or extract the ZIP
- [ ] **Open a terminal** and navigate to the ROOT project folder
- [ ] **Create the `.env` file** — `copy .env.example .env`
- [ ] **Edit `.env`** — set at minimum `JWT_SECRET_KEY` (generate with the command shown in the file)
- [ ] **Install Python packages** — `py -m pip install -r backend/requirements.txt`
- [ ] **Install Node packages** — open a second terminal, `cd frontend`, then `npm install`
- [ ] **Start backend** *(Terminal 1, ROOT folder)* — `py -m uvicorn backend.main:app --reload --port 8000`
- [ ] **Start frontend** *(Terminal 2, frontend folder)* — `npm run dev`
- [ ] **Open browser** — navigate to **http://localhost:5173**
- [ ] **Register an account** and start analysing profiles!

---

*For technical questions or bug reports, refer to the project's GitHub Issues page.*
