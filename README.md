# 🤖 OmniML — Autonomous HITL Auto-ML Pipeline

**OmniML** is a "Judge-Ready," production-stable autonomous machine learning pipeline. It leverages a multi-agent **LangGraph** workflow to automate the entire ML lifecycle: from architectural selection and real-world dataset sourcing (via Kaggle) to hyperparameter optimization (Optuna) and professional PDF evaluation.

Powered by **Groq (gpt-oss-120b)** for lightning-fast reasoning and **Chainlit** for a premium, ChatGPT-like user experience.

---

## ✨ Key Features

- **🧠 s:** A densely orchestrated LangGraph system featuring specialized agents:
  - **Architect:** Suggests neural network structures dynamically based on your problem description.
  - **Kaggle & HuggingFace Sourcer:** Searches and analyzes real-world datasets for a 1:1 match.
  - **EDA Analyzer:** Generates comprehensive real-time data profiling, including distributions, correlations, outliers, and Groq-powered AI insights via an interactive dashboard.
  - **Engineer:** Generates robust, production-grade PyTorch training scripts.
  - **Self-Healing Debugger:** Automatically intercepts PyTorch crashes and corrects syntactical or dimensional errors in the script mid-execution.
  - **Execution Sandbox:** Runs training in an isolated subprocess with full dependency management.
  - **ArXiv Comparator:** Fetches scholarly benchmarks from ArXiv and performs a quantitative Gap Analysis between your executed model and State-of-the-Art research.
  - **Evaluator:** Produces "Vast" multi-page professional Markdown and PDF reports.
- **📦 Real-World Data Sourcing:** Integrated APIs to find, download, and preprocess actual CSV datasets on-the-fly.
- **🧪 Hyperparameter Optimization:** Uses **Optuna** for automated tuning, ensuring "Judge-Ready" performance in minutes.
- **📊 Professional Analytics:** Automatically generates:
  - **Classification Confusion Matrices** (for binary/multi-class tasks).
  - **Loss Convergence Curves & Optuna Tuning Analytics.**
  - **Feature Correlation Heatmaps.**
- **🤚 Deep Human-in-the-Loop (HITL):** 
  - **Visual Flow Editor:** Drag, drop, and wire neural network architectures interactively via React Flow.
  - **Training Config Dashboard:** Surgically define epochs, learning rates, data splits, and optimizer strategies without touching code.
  - **Compute Choice:** Switch seamlessly between Local Subprocess and Cloud Execution environments.

---

## 🛠 Tech Stack

- **Orchestration:** LangGraph (Multi-Agent State Machine)
- **Inference Engine:** Groq (OpenAI-compatible gpt-oss-120b)
- **Frontend:** Chainlit 2.x (with custom persistence and rich UI elements)
- **ML Frameworks:** PyTorch, XGBoost, Scikit-Learn
- **Hyperparameter Tuning:** Optuna
- **Visualization:** Matplotlib, Seaborn
- **Data Source:** Kaggle Open Datasets

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/bytes06runner/Research.git
cd Research
```

### 2. Environment Setup
OmniML is optimized for Python 3.12.
```bash
python3 -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt  # Or pip install .
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
CHAINLIT_AUTH_SECRET=your_random_secret
```

---

## 🚀 Usage

Launch the OmniML Mission Control dashboard:
```bash
python start.py
```
The dashboard will be available at [http://localhost:8001](http://localhost:8001).

### Example Prompts
- *"I need an AI model to diagnose whether a patient has breast cancer based on static biopsy records and tabular cell measurements. Find me a dataset and build the optimal architecture."*
- *"Build a model to predict housing prices based on demographic data and historical sale records."*
- *"Detect credit card fraud patterns using anonymous transaction telemetry."*

---

## 📁 Project Structure

- `app.py`: The main Chainlit entry point and UI logic.
- `graph.py`: The LangGraph state machine definition (The "Brain" of OmniML).
- `start.py`: Production-safe launcher with AnyIO patches.
- `anomallm/`: The core OmniML SDK.
  - `trainer.py`: Industrial Auto-ML training engine.
  - `detector.py`: Automated feature scaling and discovery.
  - `persistence.py`: Custom SQLite persistence for chat history.
- `public/`: Custom CSS and branding assets.

---

## 📅 Roadmap
- [x] Integrate ArXiv Literature RAG for SOTA benchmarking.
- [x] Launch Interactive EDA & Training Configuration Dashboards.
- [x] Support seamless custom Hyperparameter tuning decoupled from Optuna runs.
- [ ] End-to-end Kaggle Cloud GPU execution polling.
- [ ] Support for Unstructured Text and Image datasets.

---
*OmniML: Autonomous, Transparent, and Professional Machine Learning.*

---

## 📆 Repository Info

- **Created:** April 3, 2026 (UTC) — [Initial commit](https://github.com/bytes06runner/OmniML/commit/40042c8d020cf55594b5f2af902e9a0535ec3034)
