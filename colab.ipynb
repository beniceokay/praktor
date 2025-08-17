# --- Colab bootstrap: ensure dependencies and inline plotting ---
try:
    from xai_sdk import Client  
except ImportError:
    from IPython import get_ipython
    get_ipython().system("pip install -q xai-sdk pydantic ipython")

# --- Imports ---
import os
import json
import uuid
import traceback
import inspect
from typing import List, Optional
from pydantic import BaseModel, Field
from IPython import get_ipython

from xai_sdk import Client
from xai_sdk.chat import user, system, tool, tool_result

# --- API key (NO hard-coded secrets) ---
API_KEY = "YOUR_API_KEY"
if not API_KEY:
    raise RuntimeError(
        "Missing XAI_API_KEY. Set it with: import os; os.environ['XAI_API_KEY']='xai-...' and rerun."
    )

# === Client ===
client = Client(api_key=API_KEY, timeout=3600)

# === Tool Schemas ===
class InstallRequest(BaseModel):
    packages: List[str] = Field(..., description="List of pip packages to install")

class ExecuteRequest(BaseModel):
    code: str = Field(..., description="Python code to save and execute")
    script_name: Optional[str] = Field(None, description="Optional filename for script (must end with .py)")

# === Tool Implementations (execute IN the notebook kernel) ===
def install_packages(packages):
    ip = get_ipython()
    outputs = []
    for pkg in packages:
        try:
            print(f"📦 Installing: {pkg}")
            ip.system(f"pip install -q {pkg}")
            outputs.append(f"✅ Installed {pkg}")
        except Exception as e:
            msg = f"❌ Failed to install {pkg}: {e}"
            print(msg)
            outputs.append(msg)
    return "\n".join(outputs)

def save_and_execute_script(code, script_name=None):
    ip = get_ipython()
    try:
        if not script_name:
            script_name = f"generated_{uuid.uuid4().hex[:8]}.py"
        if not script_name.endswith(".py"):
            script_name += ".py"

        # Ensure imports are not indented (force top-level)
        sanitized = []
        for line in code.splitlines():
            if line.lstrip().startswith("import ") or line.lstrip().startswith("from "):
                sanitized.append(line.lstrip())
            else:
                sanitized.append(line)
        code = "\n".join(sanitized)

        # Friendly inline plotting defaults
        preamble = (
            "import matplotlib\n"
            "try:\n"
            "    import matplotlib.pyplot as plt\n"
            "except Exception:\n"
            "    pass\n"
        )
        file_path = f"/content/{script_name}"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(preamble + "\n" + code)

        print(f"🚀 Executing: {file_path}")
        ip.run_line_magic("run", file_path)
        return f"✅ Executed script: {file_path}"
    except Exception as e:
        err = f"❌ Script execution failed: {e}\n{traceback.format_exc()}"
        print(err)
        return err

# === Tool Bindings ===
def install_packages_tool(req: dict):
    data = InstallRequest(**req)
    return {"result": install_packages(data.packages)}

def execute_script_tool(req: dict):
    data = ExecuteRequest(**req)
    return {"result": save_and_execute_script(data.code, data.script_name)}

# === Tool Definitions ===
tool_definitions = [
    tool(
        name="install_packages_tool",
        description="Install pip packages into the environment.",
        parameters=InstallRequest.model_json_schema(),
    ),
    tool(
        name="execute_script_tool",
        description="Save and run a Python script (plots display inline).",
        parameters=ExecuteRequest.model_json_schema(),
    ),
]

TOOLS_MAP = {
    "install_packages_tool": install_packages_tool,
    "execute_script_tool": execute_script_tool,
}

# === Robust tool_result appender ===
def append_tool_result(chat_obj, call, result_obj):
    result_str = json.dumps(result_obj)

    # Try modern signature: tool_call_id + result
    try:
        chat_obj.append(tool_result(tool_call_id=call.id, result=result_str))
        return
    except TypeError:
        pass

    # Try legacy: tool_call_id + content
    try:
        chat_obj.append(tool_result(tool_call_id=call.id, content=result_str))
        return
    except TypeError:
        pass

    # Last resort: inspect function signature
    sig = inspect.signature(tool_result)
    kwargs = {}
    if 'tool_call_id' in sig.parameters:
        kwargs['tool_call_id'] = getattr(call, 'id', None)
    if 'result' in sig.parameters:
        kwargs['result'] = result_str
    elif 'content' in sig.parameters:
        kwargs['content'] = result_str
    if 'name' in sig.parameters:
        kwargs['name'] = getattr(getattr(call, 'function', None), 'name', None)
    chat_obj.append(tool_result(**kwargs))

# === Base Agent ===
class Agent:
    def __init__(self, name: str):
        self.name = name

    def _new_chat(self):
        return client.chat.create(
            model="grok-4",
            tools=tool_definitions,
            tool_choice="auto",
        )

    def run(self, messages, max_iters: int = 8):
        chat_obj = self._new_chat()
        chat_obj.append(system(
            "You are Grok, an intelligent Python agent inside Colab. "
            "You MUST use the provided tools to install packages and execute Python scripts. "
            "Never describe what you'd do—always call the tools. Explain your thought process as you call the tools. "
            "Scripts must:\n"
            "- All imports at TOP-LEVEL (no imports inside if/else/loops) to avoid indentation bugs\n"
            "- No indented imports\n"
            "- Save CSVs/plots under /content/ and print filenames\n"
            "- ALWAYS os.makedirs(f\"{base_dir}/model\", exist_ok=True) BEFORE any savefig/model save\n"
            "- Use matplotlib/seaborn with plt.show(); do NOT pass random_state to learning_curve\n"
            "If a script fails, fix and rerun."
        ))
        for m in messages:
            chat_obj.append(m)

        for _ in range(max_iters):
            response = chat_obj.sample()
            content = response.content or ""
            if content.strip():
                print(f"\n[{self.name}]:\n{content}\n")

            calls = getattr(response, "tool_calls", None) or []
            if calls:
                for call in calls:
                    fn_name = call.function.name
                    args = json.loads(call.function.arguments or "{}")
                    print(f"[TOOL CALL] {fn_name} {args}")
                    handler = TOOLS_MAP.get(fn_name)
                    if not handler:
                        append_tool_result(chat_obj, call, {"error": f"Unknown tool: {fn_name}"})
                        continue
                    result = handler(args)
                    append_tool_result(chat_obj, call, result)
                continue
            break

# === Multi-Agent Orchestrator ===
class MultiAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.children = []

    def spawn(self, child_name: str, task: str, rounds: int = 4):
        print(f"\n🧬 {self.name} spawning {child_name} for: {task}")
        child = Agent(child_name)
        self.children.append(child)
        messages = [
            user(
                task
                + "\n\nConstraints:\n"
                  "- Always call install_packages_tool before imports.\n"
                  "- All imports must be at TOP-LEVEL (no imports inside conditionals/loops).\n"
                  "- Save CSVs/plots to /content and print filenames.\n"
                  "- ALWAYS create /content/datasets/<slug>/model with os.makedirs(..., exist_ok=True) before saving figures/models.\n"
                  "- Use matplotlib/seaborn with plt.show(); do NOT pass random_state to learning_curve.\n"
                  "- If errors, fix and rerun."
            )
        ]
        child.run(messages, max_iters=rounds)

# ---------------- REPLACE FROM HERE ----------------

# (Optional) topics/keywords to bias the search (edit freely)
TARGET_TOPICS = ["synthetic", "tabular", "open-license", "classification", "regression"]

# --- Run orchestrator ---
architect = MultiAgent("Architect")

root_messages = [
    user(
        "SPAWN MULTIPLE SUB AGENTS WITH TOOLS TO MEET OBJECTIVES. RUN THE SCRIPTS.\n"
        "Goal: find, download, and curate **public synthetic TABULAR datasets** (CSV/Parquet) and then **train/evaluate** simple models.\n"
        "\nRules:\n"
        "- Use install_packages_tool and execute_script_tool for all work.\n"
        "- Prefer public sources with permissive/clear licenses (Hugging Face Hub via API, GitHub raw CSV/Parquet, academic mirrors). "
        "Kaggle only if KAGGLE_USERNAME/KAGGLE_KEY are set; otherwise skip.\n"
        "- Save ALL artifacts under BOTH /content/datasets/ and /content/datasets/<slug>/... and collection assets under /content/collection/.\n"
        "- For each dataset: validate schema, infer dtypes, detect NA, and create a manifest with license + basic stats.\n"
        "- Normalize each dataset to tidy CSV at /content/datasets/<slug>/clean.csv (lower_snake_case, NA normalized, numerics parsed, dates parsed).\n"
        "- Perform basic EDA (plots + stats). THEN: **Auto-train** a small model:\n"
        "   * Heuristic target detection (columns named: ['target','label','class','y','outcome'] or low-cardinality candidate for classification; else numeric w/ >20 uniq for regression).\n"
        "   * If classification: use Pipeline(ColumnTransformer(one-hot for categoricals, pass-through numerics) -> LogisticRegression + RandomForest). "
        "     Use class_weight='balanced' where applicable. Pick best by cross-val ROC AUC (binary) or macro F1 (multiclass).\n"
        "   * If regression: use Pipeline(... -> LinearRegression + RandomForestRegressor). Pick best by cross-val R^2.\n"
        "   * Train/val split with stratify (classification) or regular (regression). Create learning curve and, if classification, a confusion matrix.\n"
        "- Write per-dataset model artifacts to /content/datasets/<slug>/model/*\n"
        "- Collect a run-wide summary table at /content/collection/model_catalog.csv.\n"
        "- If anything fails, fix and rerun. Print all final file paths.\n"
        "- IMPORTANT GUARDS:\n"
        "   * All imports at top-level. Do NOT import inside if/else; only construct models in the blocks.\n"
        "   * Ensure os.makedirs(f\"/content/datasets/<slug>/model\", exist_ok=True) BEFORE any savefig or file save to /model.\n"
        "   * Do not pass random_state to sklearn.model_selection.learning_curve."
    )
]

# Kick off the plan
architect.run(root_messages, max_iters=4)

# 1) Finder — discover candidate datasets and resolve direct download links
architect.spawn(
    "DatasetFinder",
    f"""
Use install_packages_tool to install: requests, pandas, pyyaml, huggingface_hub, tqdm, python-dateutil.
Steps:
1) Search for **synthetic tabular datasets** related to topics {TARGET_TOPICS} from:
   - Hugging Face Datasets (use huggingface_hub API to list/search and fetch files)
   - GitHub (CSV/Parquet raw links only)
   - Kaggle ONLY if credentials are present (KAGGLE_USERNAME/KAGGLE_KEY), otherwise skip
2) For each candidate, resolve a direct download URL to CSV or Parquet. Prefer smaller samples if >1GB.
3) For each workable candidate:
   - Create slug from name (lowercase, hyphen/underscore only)
   - Write a candidate row to /content/collection/catalog_candidates.csv with:
     slug, source, url, filename, license, est_rows, est_cols, notes
Print a JSON summary of chosen candidates (slug -> url, license).

STRICT RULES:
- Keep all imports at top-level.
- Before any figure/model save under a dataset, ensure /content/datasets/<slug>/model exists.
- Do not pass random_state to learning_curve.
"""
)

# 2) Fetcher — download, integrity checks, and hashes
architect.spawn(
    "DatasetFetcher",
    """
Ensure: requests, pandas, tqdm, pyarrow installed.
Read /content/collection/catalog_candidates.csv and for each row:
- Download to /content/datasets/<slug>/raw/<filename>
- If parquet, keep as parquet and export a raw CSV snapshot when feasible (cap ~1e6 rows or 1GB; otherwise sample 100k)
- Verify file size > 1KB and readable by pandas
- Compute sha256 and store /content/datasets/<slug>/download_meta.json
- Append a record to /content/collection/catalog_downloaded.csv with slug, local paths, size, sha256, license
Print a compact table of slugs and their local paths.

STRICT RULES:
- Keep all imports at top-level.
- Ensure /content/datasets/<slug>/model exists before any saves to that dir.
"""
)

# 3) Normalizer — tidy CSV + manifest
architect.spawn(
    "SchemaNormalizer",
    """
Ensure: pandas, numpy, pyarrow, python-dateutil, unidecode.
For each /content/datasets/<slug>/raw file:
- Load using pandas (convert_dtypes())
- Standardize column names to lower_snake_case; strip/unidecode text
- Normalize NAs, trim string cells, attempt datetime parsing
- Write clean CSV -> /content/datasets/<slug>/clean.csv
- Create /content/datasets/<slug>/manifest.json with:
  {
    "slug": "...",
    "license": "...",
    "n_rows": ...,
    "n_cols": ...,
    "columns": [{"name":"...","dtype":"...","na_pct":...}],
    "target_guess": null,
    "task_guess": null,
    "notes": ""
  }
Aggregate manifests to /content/collection/catalog.json and /content/collection/catalog.csv.

STRICT RULES:
- Keep all imports at top-level.
- Ensure /content/datasets/<slug>/model exists before any saves to that dir.
"""
)

# 4) EDA — quick visuals & stats (no modeling yet)
architect.spawn(
    "EDAMaker",
    """
Ensure: pandas, matplotlib, seaborn, numpy.
For each /content/datasets/<slug>/clean.csv:
- Produce /content/datasets/<slug>/eda_overview.json (n_rows, n_cols, missing_by_col, numeric_summary, top categories)
- Generate:
  * /content/datasets/<slug>/fig_missingness.png (bar of NA counts per column)
  * /content/datasets/<slug>/fig_corr.png (numeric correlation heatmap if >=2 numeric columns)
  * For up to 6 categorical columns: top-10 value bars -> /content/datasets/<slug>/fig_<col>_top10.png
- Build /content/datasets/<slug>/index.html linking to clean.csv, manifest.json, and figures
Also create a collection landing page at /content/collection/index.html linking all datasets.

STRICT RULES:
- Keep all imports at top-level.
- ALWAYS os.makedirs(f"/content/datasets/<slug>/model", exist_ok=True) before any savefig to that folder.
- Use plt.show() after each savefig; optionally plt.close('all') to free memory.
"""
)

# 5) Trainer — AUTOMATIC MODELING (classification OR regression)
architect.spawn(
    "AutoTrainer",
    """
Ensure: scikit-learn, pandas, numpy, matplotlib, seaborn, joblib installed.
For each /content/datasets/<slug>/clean.csv:
- Heuristically pick a target column:
  1) Prefer a column named in ['target','label','class','y','outcome'] if present.
  2) Else: pick the first non-ID column with low cardinality (2..25 unique) -> classification.
  3) Else: pick the first numeric column with >20 unique values -> regression.
  If no valid target is found, skip dataset (record reason in /content/datasets/<slug>/model/skip.json).
- Drop known identifiers (columns matching r'^(id|uuid|index)$' or single-unique).
- Split 80/20. If classification and binary/multiclass, use stratify. Limit training rows to 2e6 for safety.
- Build preprocessing with ColumnTransformer:
    OneHotEncoder(handle_unknown='ignore', min_frequency=0.01) for categorical;
    'passthrough' for numeric.
- Train TWO small baselines:
    Classification: LogisticRegression(max_iter=200, class_weight='balanced') AND RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1).
    Regression: LinearRegression() AND RandomForestRegressor(n_estimators=300, n_jobs=-1).
- Cross-validate with 5-fold:
    Classification scoring: ROC AUC if binary else macro F1.
    Regression scoring: R^2.
  Pick best by mean CV score.
- Fit best on train split; evaluate on holdout:
    Classification: accuracy, macro_f1, ROC AUC (binary only), confusion_matrix image -> /content/datasets/<slug>/model/confusion_matrix.png
    Regression: MAE, RMSE, R^2
- Plot learning curve -> /content/datasets/<slug>/model/learning_curve.png
- Permutation importance (n_repeats=5) on a 10k-row sample if tabular size is big; save CSV -> /content/datasets/<slug>/model/perm_importance.csv
- Save artifacts:
    /content/datasets/<slug>/model/best_model.joblib
    /content/datasets/<slug>/model/metrics.json
    /content/datasets/<slug>/model/report.txt (short summary)
Append a row per dataset to /content/collection/model_catalog.csv with: slug, task, target, model_name, cv_score, holdout_metrics_path, model_path.
PRINT a table of slugs, chosen target, task type, best model, and key scores.

STRICT RULES:
- Keep all imports at top-level; do NOT import inside if/else blocks. Only construct models in the blocks.
- Before ANY savefig/model/JSON save to /model, call: os.makedirs(f"/content/datasets/<slug>/model", exist_ok=True).
- Use plt.show() after saves; do not pass random_state to learning_curve.
"""
)

# 6) Curator — optional union preview (unchanged, helpful for context)
architect.spawn(
    "Curator",
    """
Read /content/collection/catalog.csv and manifests.
Create a preview union of up to 3 compatible datasets (same subset of columns; inner-join on common names) and save to
  /content/collection/preview_union.csv (limit 200k rows).
Write /content/collection/compatibility_report.json summarizing shared schemas/conflicts.
Print both file paths.

STRICT RULES:
- Imports at top-level. Create output dirs before saving.
"""
)

# 7) Reporter — compact results summary (NO chain-of-thought)
architect.spawn(
    "Reporter",
    """
Install reportlab.
Generate /content/final_report.pdf with:
- Executive Summary: count of datasets downloaded, modeled, total rows/cols.
- Catalog table: slug, license, n_rows, n_cols, target, task, best model, CV score, holdout highlights.
- Per-dataset mini-cards (thumbnail corr fig if present; link to dataset index.html; link to model metrics).
- Appendix: file inventory (paths + sizes) under /content/datasets and /content/collection.

Also include a concise 'Reasoning Summary (high-level)'.
Content to include:
1) Title page and overview of objectives. Be EXTREMELY THOROUGH AND DETAILED ABOUT EVERYTHING.
2) Data section: read /content/synthetic_data.csv if present; show shape, head(5), and key stats.
3) Methods section: brief description of data generation, modeling, and evaluation steps (no chain-of-thought).
4) Results section: embed figures if present — /content/correlation_heatmap.png, /content/pairplot.png, /content/confusion_matrix.png, /content/learning_curve.png, any /content/training_curves_*.png. Scale images to fit the page.
5) Metrics: if /content/train_report.json exists, summarize key metrics in a table.
6) Artifacts appendix: list all created files under /content relevant to this run with sizes.
7) Clear conclusion and next steps.

Save the PDF to /content/final_report.pdf and print the exact path at the end.
STRICT RULES:
- Imports at top-level. Create output dirs before saving any images or files.
"""
)
