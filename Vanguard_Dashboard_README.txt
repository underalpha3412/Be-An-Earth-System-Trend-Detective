# Vanguard Earth Trends Dashboard
## Setup and launch guide (macOS)

This project has two parts:

- `nasa_dashboard.html`: the dashboard interface.
- `ai_backend.py`: the local Python API that sends graph data to NVIDIA Nemotron and returns the AI insights.

**Important:** The AI backend does not start automatically when you open the HTML file. You must start the backend manually in Terminal, then open the dashboard in your browser.

---

## 1. Files you should have

Keep these files together in the same project folder:

```text
Vanguard/
├── nasa_dashboard.html
├── ai_backend.py
├── requirements.txt
└── .env
```

The `.env` file contains your NVIDIA API key. Do not share it or upload it publicly.

## 2. Add your NVIDIA API key

1. Open the project folder in VS Code.
2. Create a new file named `.env` in the same folder as `ai_backend.py`.
3. Add this line, replacing the placeholder with your own NVIDIA API key:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

The backend uses the NVIDIA API endpoint and the model configured in `ai_backend.py`. You can create/manage an NVIDIA API key through the NVIDIA developer platform: https://build.nvidia.com/

Do not put quotation marks around the key unless NVIDIA specifically requires them. Do not add spaces around `=`.

## 3. Create a Python virtual environment

Open the project folder in VS Code. Choose **Terminal → New Terminal**.

Run these commands from the project folder:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal prompt should show `(.venv)`.

If `python3` is not found, install Python 3 from https://www.python.org/downloads/macos/ and reopen the terminal.

## 4. Install the required Python libraries

With `(.venv)` active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the `requirements.txt` file is missing or the command cannot find it, first make sure the terminal is in the folder containing that file. You can check the current folder with:

```bash
pwd
ls
```

## 5. Start the AI backend

In the same terminal, with `(.venv)` active, run:

```bash
uvicorn ai_backend:app --reload
```

Wait until the terminal displays a message similar to:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Keep this terminal window open while using the dashboard. The backend stops when you press `Ctrl + C` or close that terminal session.

You can check that the API is running by opening this address in your browser:

http://127.0.0.1:8000/docs

The FastAPI documentation page should appear.

## 6. Open the dashboard

Use a **second terminal** in VS Code. Keep the backend terminal running.

If you have the VS Code **Live Server** extension:

1. Open `nasa_dashboard.html` in VS Code.
2. Right-click inside the file.
3. Select **Open with Live Server**.

Alternatively, use another local static web server. Avoid relying on opening the HTML as a `file://` page if the browser blocks requests to the local API.

The dashboard should now be able to send graph data to the backend at:

`http://127.0.0.1:8000/analyze-graph`

The backend allows the default VS Code Live Server origins on ports `5500` and
`5501`. If your local server uses a different port, set
`DASHBOARD_ALLOWED_ORIGINS` in `.env` to a comma-separated list of the exact
origins, for example:

```env
DASHBOARD_ALLOWED_ORIGINS=http://127.0.0.1:5502,http://localhost:5502
```

## 7. Normal startup routine

Every time you want to use the dashboard:

**Terminal 1: start the backend**

```bash
cd /path/to/your/Vanguard
source .venv/bin/activate
uvicorn ai_backend:app --reload
```

**Then open `nasa_dashboard.html` with VS Code Live Server.**

Replace `/path/to/your/Vanguard` with the actual path to your project folder. If the terminal is already opened in the project folder, you do not need the `cd` command.

When finished, stop the backend with `Ctrl + C`.

## 8. Troubleshooting

### `pip: command not found`

Use Python to run pip:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If the virtual environment is active, use `python -m pip` instead.

### `uvicorn: command not found`

Activate the virtual environment and install the requirements again:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run:

```bash
python -m uvicorn ai_backend:app --reload
```

### The dashboard says it cannot connect to the AI backend

- Confirm the backend terminal is still running.
- Confirm `http://127.0.0.1:8000/docs` opens.
- Confirm the dashboard is being served locally, for example with VS Code Live Server.
- Check that the backend and dashboard are using the expected local address and port.

### The backend says `NVIDIA_API_KEY is missing`

- Confirm `.env` is in the same folder as `ai_backend.py`.
- Confirm the variable is written exactly as `NVIDIA_API_KEY=...`.
- Stop and restart Uvicorn after changing `.env`.

### NVIDIA returns an authentication or model error

Check that the API key is valid, that the NVIDIA API access is enabled for the account, and that the model configured in `ai_backend.py` is currently available to your account.

### Port 8000 is already in use

Stop the other process using port 8000, or configure a different port. If you change the port, update the dashboard's backend URL to match it.

---

## 9. About automatic startup

Automatic startup has **not** been implemented in this project.

A normal HTML page cannot start or stop a local Python process by itself because browsers restrict webpages from launching programs on your computer.

For now, use the two-step manual startup described above:

1. Start the Python backend in Terminal.
2. Open the dashboard in your browser.

A later improvement would be to create a small desktop launcher or package the dashboard and backend as a desktop application. That would make starting the project easier, but it is not part of the current setup.

## Security reminder

- Never commit `.env` to a public GitHub repository.
- Never paste your API key into `nasa_dashboard.html`.
- Do not share screenshots that expose your API key.
- Keep the API key only in the backend environment file.
