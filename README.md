# CS229 LLM Routing Project

Code and data for the final report (`cs229-report.tex`). The project studies
*supervised LLM routing*: given a question and a set of candidate models that
vary in price and capability, train a small classifier to pick the best model
per example, and then read off when the resulting policy is Pareto-improving
on the cost--accuracy plane.

The headline benchmark is a 787-example mixed-task CSV across five task
families (math, science, commonsense, humanities, social science), graded
deterministically against ten OpenRouter models. The report does the analysis
on the six *viable* models (the four that scored badly under the prompt format
are excluded).

## Repository layout

```text
cs229-report.tex                           Final CS229 report (self-contained)
README.md                                  This file

scripts/
  prepare_mixed_tasks.py                   Build the mixed-task CSV (StrategyQA + GSM8K + SciQ + MMLU)
  run_openrouter_multimodel.py             Call N OpenRouter models on the mixed-task CSV
  build_router_dataset.py                  Turn raw model outputs into a router-ready CSV
  run_class_experiments.py                 Main pair sweep: 10 router families x 30 seeds
  run_weak_pair_experiments.py             Weak-pair / negative-result experiments
  run_architecture_analysis.py             Bias-variance, feature importance, per-task routing
  run_hyperparam_sweep.py                  Per-family hyperparameter ablation on the headline pair
  analyze_decidable_routing.py             Decidable-lift metric on the disagreement subset
  analyze_pair_costs.py                    Cost summaries / pgfplots data for Table 1 + Figure 1

routing/
  prompts.py                               Prompt templates
  graders.py                               Numeric / yes-no / multiple-choice graders
  features.py                              Prompt-only and cheap-probe feature extraction
  router_models.py                         scikit-learn router definitions

baselines/router_baselines.py              Routing evaluation harness (fixed/random/oracle/learned)

data/
  openrouter_mixed_10models_787_router_dataset.csv   The benchmark used in the report
  raw/                                     Raw OpenRouter outputs
  tasks/                                   Source task CSVs

results/                                   class_*.csv files, one per analysis
archive/                                   Outdated experiments (see archive/README.md)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then put your key in .env:
# OPENROUTER_API_KEY=sk-or-v1-...
```

The `.env` file is gitignored.

## Reproducing the report numbers

The dataset CSV is committed, so the analysis scripts run end-to-end without
any OpenRouter calls:

```bash
# Main pair sweep (15 viable pairs, 10 router families, 30 seeds)
python scripts/run_class_experiments.py

# Architecture deep-dive: feature importance, per-task routing, bias-variance,
# and the 6-viable-models pair sweep
python scripts/run_architecture_analysis.py

# Hyperparameter ablation on the headline pair (Table 2 in the report)
python scripts/run_hyperparam_sweep.py

# Decidable lift on the strong pair and a few weaker contrasts
python scripts/analyze_decidable_routing.py

# Weak-pair negative result (predictability vs volume of disagreement)
python scripts/run_weak_pair_experiments.py

# Cost summaries used by Table 1 and the cost-accuracy figure
python scripts/analyze_pair_costs.py
```

Each script writes one or more CSVs to `results/` (all prefixed `class_`).

## Rebuilding the dataset (optional, costs OpenRouter credits)

```bash
# 1. Build the mixed-task CSV
python scripts/prepare_mixed_tasks.py \
  --output data/tasks/mixed_router_800.csv \
  --per-task-limit 160 \
  --shuffle

# 2. Run all 10 candidate models on every example
python scripts/run_openrouter_multimodel.py \
  --tasks data/tasks/mixed_router_800.csv \
  --output data/raw/openrouter_mixed_10models.jsonl \
  --route gemini_flash_lite=google/gemini-2.0-flash-lite-001:reasoning \
  --route gemini_flash=google/gemini-2.5-flash:reasoning \
  --route deepseek_v3=deepseek/deepseek-chat-v3:reasoning \
  --route llama_70b=meta-llama/llama-3.3-70b-instruct:reasoning \
  --route qwen_30b=qwen/qwen3-30b-a3b:reasoning \
  --route granite_8b=ibm-granite/granite-3.1-8b-instruct:reasoning \
  --route mistral_nemo=mistralai/mistral-nemo:reasoning \
  --route minimax_m3=minimax/minimax-m1:reasoning \
  --route ring_1t=inclusionai/ring-1t:reasoning \
  --route step_flash=stepfun-ai/step-3:reasoning \
  --concurrency 4 \
  --request-delay-seconds 0.25 \
  --max-retries 8 \
  --resume

# 3. Flatten into a router-ready CSV
python scripts/build_router_dataset.py \
  --input data/raw/openrouter_mixed_10models.jsonl \
  --output data/openrouter_mixed_10models_787_router_dataset.csv \
  --strategies gemini_flash_lite gemini_flash deepseek_v3 llama_70b qwen_30b granite_8b mistral_nemo minimax_m3 ring_1t step_flash \
  --cheap-strategy granite_8b \
  --cost-mode actual
```

Total OpenRouter spend on the committed dataset was about $0.79.

## What's archived

`archive/` holds older iterations of the project that aren't used by the final
report: the StrategyQA-only direct/cascade pipeline, 5-model and snapshot
versions of the dataset, the per-model correctness-predictor approach, the
abandoned local-Llama experiments, the granite-probe variant, the milestone
draft, and 44 stale result CSVs. See `archive/README.md` for details.
