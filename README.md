# praktor
Introducing Praktor: a Grok-4 based agent framework for Colab execution. | Plan → Run → Analyze → Knowledge → Train → Orchestrate → Report

Agentic Notebook Framework mentioned herein consists of EDA → Baselines → Report.

A lightweight **agentic orchestration framework** implemented entirely inside a single Jupyter/Colab notebook. It gives a chat model **controlled tools**—package installation and “write + run a Python script”—and uses them to build a reproducible pipeline that can discover data, normalize it, run EDA, train small baselines, and compile a final report. You can modify any part of this code as needed to suit your objectives, this is just one small use-case for this framework.

> **What this is:** a pragmatic pattern for effective tool-use in notebooks—not a monorepo or heavy orchestration library.  
> **What this is not:** an AutoGPT clone. The agent cannot do everything; it acts only through narrowly-scoped tools. However, you can easily adjust the parameters to become more or less agentic depending on your use case (feel free to swap in any use case you'd like over this repo by simply changing the messaging you give the architect, subagents, and earlier tools). This paritcular use-case usually takes approximately 20-30 minutes to run.

## ✨ Features

- **Tools** exposed to the model:
  - `install_packages_tool` → install required packages (validated list).
  - `execute_script_tool` → save an LLM-authored Python script and execute it in-kernel.
- **Agent + Orchestrator**
  - `Architect` runs a guarded plan–act loop with tool calls.
  - `MultiAgent` spawns focused sub-agents with a role/task and bounded rounds.
- **Modular sub-agent prompts** (no hard wiring): swap prompts and objectives to change behavior.
- **End-to-end artifacts**
  - Per-dataset folders under `/content/datasets/<slug>/` (raw, clean, figures, model).
  - Cross-run catalogs under `/content/collection/`.
  - Final `final_report.pdf` in `/content/` (if the Reporter persona runs).
- **Reproducible outputs**: scripts and figures are saved to disk; key paths are printed for auditability.

## 🧱 Architecture

### Tools (notebook-local)
- **`install_packages_tool(req: dict)`**  
  Uses `pip` to install packages requested by the agent. Captures and returns console output. Validated via a Pydantic schema in the notebook.
- **`execute_script_tool(req: dict)`**  
  Writes a Python file to disk (defaults to `generated_<id>.py`) and executes it in the same kernel using IPython. Ensures imports are at top level, creates directories before saving, and uses `matplotlib.pyplot.show()` for figures.

> These are thin wrappers around two Python functions defined in the notebook: `install_packages(packages)` and `save_and_execute_script(code, script_name=None)`.

### Core classes
- **`Agent`**
  - Wraps the chat loop (model messages ↔ tool calls ↔ tool results).
  - Enforces guardrails via a strict system prompt: create directories before saving, keep artifacts under `/content/...`, imports only at top-level, use `plt.show()`, and repair on failure.
  - `run(messages, max_iters=8)` executes until no more tool calls or the iteration cap is reached.
- **`MultiAgent`**
  - Thin orchestrator that can **`spawn(child_name, task, rounds=4)`**: launches a focused sub-agent with its own role/task and a bounded number of tool-use rounds.
  - Great for composing steps like: **Discovery → Normalize → EDA → Modeling → Reporting**.

### Model client
- Uses `xai_sdk.Client` (the code references model name `"grok-4"`). Any tools-capable chat model compatible with your SDK can be substituted.

## 📦 Outputs & Filesystem Contract

All artifacts live under `/content` (Colab-friendly; adjust if you’re not in Colab):

```
/content/
  datasets/
    <slug>/
      raw/...
      clean.csv
      manifest.json
      fig_missingness.png
      fig_corr.png
      fig_<col>_top10.png
      model/
        metrics.json
        learning_curve.png
        confusion_matrix.png      # classification only
        model.pkl (or params.json)
      index.html
  collection/
    catalog.json
    catalog.csv
    model_catalog.csv
  final_report.pdf
```

## 🧪 Default Pipeline (sub-agents you can swap)

> Implemented in the notebook as *instruction blocks* given to spawned sub-agents; they call the two tools to do work.

1. **Discovery / Curation**
   - Locate and download **public synthetic tabular datasets** with permissive licenses (Hugging Face/GitHub/raw CSVs). Kaggle optional if `KAGGLE_USERNAME`/`KAGGLE_KEY` exist.
   - Save raw files to `/content/datasets/<slug>/raw/`.

2. **Schema Normalization**
   - Read, infer dtypes, standardize column names to `lower_snake_case`, normalize missing values, parse dates.
   - Save cleaned CSV to `/content/datasets/<slug>/clean.csv`.
   - Write `/content/datasets/<slug>/manifest.json` with license, shapes, column summaries, and task/target guesses.
   - Aggregate a run-wide catalog under `/content/collection/`.

3. **EDA**
   - Plot missingness per column, correlation heatmap (for ≥2 numeric columns), and top-10 bar charts for up to 6 categorical columns.
   - Generate `/content/datasets/<slug>/index.html` linking to figures and artifacts.

4. **Auto-Modeling (Baselines)**
   - Heuristics choose a target: first `target/label/class/y/outcome`, else low-cardinality → classification, else numeric → regression.
   - **Classification:** `LogisticRegression` and `RandomForestClassifier` (macro-F1 or ROC AUC).  
   - **Regression:** `LinearRegression` and `RandomForestRegressor` (R²).  
   - Save metrics, learning curve, confusion matrix (if applicable), and serialized model to `/content/datasets/<slug>/model/`.

5. **Reporting**
   - Build catalogs under `/content/collection/`.
   - Compile `/content/final_report.pdf` (via `reportlab`) with executive summary, dataset cards, metrics, and an appendix (file inventory).


## 🧩 Modularity & Extensibility

- **Swap the model:** change the model name used by `xai_sdk.Client`.
- **Add/replace tools:** keep them **narrow and validated**. Ideas: HTTP fetch (allowlisted), SQL reader (read-only), HTML→Markdown, cloud uploader, notebook cell runner.
- **Rewrite sub-agents and the architect:** each step (Discovery, Normalization, EDA, Modeling, Reporting) is just a prompt. Replace with your own (e.g., documentation mining, embeddings benchmarking, summarization-only).

### Dial the “agenticness”
- Create different prompts for the architect and sub agents that are spawned to get extra creative with your use case. Simply just tell them what to do, and how much freedom they have, and they will follow your lead.

## 🚀 Quickstart

1. **Install dependencies**
   ```bash
   pip install xai-sdk pydantic ipython
   ```

2. **Set your API key securely** If you have a HF API key or Kaggle API key, be sure to add those to your Colab secrets and enable access. (They are not necessary but recommended).

   ```python
   import os
   os.environ["XAI_API_KEY"] = "xai-..."  # use your real key; do NOT hard-code in code
   ```

3. **Open the notebook and run cells**  
   - The first cell installs `xai-sdk` (if needed).  
   - Subsequent cells define tools, the `Agent`, the `MultiAgent`, and then run the orchestrated steps.

> ⚠️ The uploaded notebook included a placeholder API key in code. **Do not commit secrets.** Use environment variables as shown above.

## 🔐 Guardrails (built-in and recommended)

**Built-in via system prompt**
- Create directories before saving files.
- Keep **all artifacts under `/content/...`**.
- Imports at top level only; figures must call `plt.show()`.
- On failure, **repair and retry** within bounded iterations.

## 🧭 Use Cases

- **Data cataloging & profiling** of public datasets (with manifests + HTML indexes).
- **Auto-EDA** reports for incoming CSVs.
- **Baseline modeling** for quick feasibility checks.
- **Benchmark harness** for embeddings or classic ML (swap the Modeling persona).
- **Scrape → clean → summarize** for documentation or product data.

## ❓FAQ

**Is this safe to run?**  
Yes. But as always, exercise caution. Tools are narrow and prompts enforce guardrails. You should still review scripts and keep execution bounded.

**Do I need Colab?**  
No. But it is recommended. It uses `/content/...` paths by convention. Adjust paths if you’re in another environment.

**Can I change the model?**  
Yes—swap the model name passed to `xai_sdk.Client` and keep the tool API the same. xAI is the recommended SDK, however. Other SDKs likely will not work with this code structure and will need modification.

**Where are outputs saved?**  
Under `/content/datasets/<slug>/...`, `/content/collection/...`, and `/content/final_report.pdf`.

## 📝 License
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

