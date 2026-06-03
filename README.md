# CS229 Routing Project

This project studies cost-aware routing for LLM inference. The main question is:

> Can a learned router decide when to accept a cheap model answer and when to escalate to a stronger model?

The current OpenRouter setup is:

- `cheap_direct`: `mistralai/mistral-nemo`
- `strong_reasoning`: `qwen/qwen3-235b-a22b-2507`
- `escalate_strong`: the same strong response, but with cascade cost equal to the cheap call(s) plus the strong call

The main experiment compares metadata-only routing against cascade routing that also uses cheap-model answer features such as confidence, uncertainty phrases, and sample agreement.

## Repository Layout

```text
scripts/prepare_strategyqa.py        Download/convert StrategyQA into a task CSV
scripts/prepare_mixed_tasks.py       Download/convert StrategyQA, GSM8K, and SciQ
scripts/run_openrouter_strategies.py Run cheap and strong OpenRouter models
scripts/run_openrouter_multimodel.py Run arbitrary named OpenRouter model routes
scripts/build_router_dataset.py      Convert raw model outputs into router-training CSVs
scripts/evaluate_model_groups.py     Test accuracy routing within fixed model groups
baselines/router_baselines.py        Run routing evaluations and save result tables
routing/prompts.py                   Prompt templates for cheap and strong model calls
routing/graders.py                   Parse/grade numeric, yes/no, and multiple-choice answers
routing/features.py                  Extract prompt and cheap-answer features for routers
routing/router_models.py             Learned router definitions: logistic, forest, MLP, TF-IDF
data/tasks/                          Input task CSVs, before model calls
data/raw/                            Raw model outputs, before router training
data/*.csv                           Flattened router datasets
results/                             Baseline and router evaluation CSVs
cs229-milestone.tex                  Milestone writeup draft/template
```

Fixed policies are evaluated in `baselines/router_baselines.py`. Learned router
model definitions live in `routing/router_models.py`.

## How The Files Interact

The project has four stages:

```text
1. Task data
   data/tasks/strategyqa_50.csv
   columns: example_id, task_type, answer_type, prompt, answer

2. Model outputs
   data/raw/openrouter_strategyqa_50_outputs.jsonl
   produced by scripts/run_openrouter_strategies.py

3. Router dataset
   data/openrouter_strategyqa_direct_router_dataset.csv
   data/openrouter_strategyqa_cascade_router_dataset.csv
   produced by scripts/build_router_dataset.py

4. Router results
   results/openrouter_direct_metadata.csv
   results/openrouter_cascade_features.csv
   produced by baselines/router_baselines.py
```

The important distinction is that model-running code and router-learning code
are separate. `scripts/run_openrouter_strategies.py` spends API calls and records
what each model answered. `baselines/router_baselines.py` does not call any
models; it only learns routing rules from saved CSVs.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenRouter key in a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
OPENROUTER_API_KEY=sk-or-v1-your-real-key
```

The real `.env` file is ignored by Git.

## Prepare StrategyQA

Create a StrategyQA task file:

```bash
python scripts/prepare_strategyqa.py \
  --output data/tasks/strategyqa_50.csv \
  --limit 50 \
  --shuffle
```

This uses Hugging Face `datasets`:

```python
load_dataset("ChilleD/StrategyQA", split="train")
```

This writes rows with:

```text
example_id,task_type,answer_type,prompt,answer
```

For a quick smoke test, use `--limit 10`. For a larger run, increase the limit.

## Run A Small StrategyQA OpenRouter Batch

Generate raw model outputs:

```bash
python scripts/run_openrouter_strategies.py \
  --tasks data/tasks/strategyqa_50.csv \
  --output data/raw/openrouter_strategyqa_50_outputs.jsonl \
  --limit 10 \
  --cheap-samples 1 \
  --cheap-prompt-mode brief_reasoning \
  --concurrency 4 \
  --request-delay-seconds 0.25 \
  --max-retries 8
```

Build a direct-routing dataset:

```bash
python scripts/build_router_dataset.py \
  --input data/raw/openrouter_strategyqa_50_outputs.jsonl \
  --output data/openrouter_strategyqa_direct_router_dataset.csv \
  --strategies cheap_direct strong_reasoning \
  --cost-mode normalized
```

Build a cascade-routing dataset:

```bash
python scripts/build_router_dataset.py \
  --input data/raw/openrouter_strategyqa_50_outputs.jsonl \
  --output data/openrouter_strategyqa_cascade_router_dataset.csv \
  --strategies cheap_direct escalate_strong \
  --cost-mode normalized
```

Use `--cost-mode actual` to train/evaluate with OpenRouter's reported
`usage.cost` values. Actual costs are usually very small, so use larger
`--cost-weights` values when optimizing dollar utility.

## Evaluate Similar-Cost Model Groups

For the accuracy-routing question, keep the candidate set in a similar price
range and set `cost_weight=0`. The helper below repeats the train/test split
across several seeds so single-split luck is less misleading:

```bash
python scripts/evaluate_model_groups.py \
  --data data/openrouter_mixed_5models_500_router_dataset.csv \
  --models mistral_nemo qwen_235b gemini_flash llama_70b deepseek_v3 \
  --group gemini_llama=gemini_flash,llama_70b \
  --group llama_deepseek=llama_70b,deepseek_v3 \
  --group gemini_llama_deepseek=gemini_flash,llama_70b,deepseek_v3 \
  --features task_type prompt_chars prompt_words question_start num_numbers has_math_symbols has_code_like_text question_mark_count capitalized_word_count contains_parenthesis contains_quote has_comparison_word has_negation_word has_quantity_word cheap_answer_chars cheap_answer_words cheap_parsed_answer cheap_contains_uncertainty cheap_self_confidence cheap_sample_agreement \
  --output results/model_group_accuracy_promising_with_mlp_summary.csv \
  --details-output results/model_group_accuracy_promising_with_mlp_details.csv
```

If `--group` is omitted, all model pairs and triples are evaluated. Use
`--skip-mlp` for a faster scan.

## Run Baselines

Metadata-only direct routing:

```bash
python baselines/router_baselines.py \
  --data data/openrouter_strategyqa_direct_router_dataset.csv \
  --models cheap_direct strong_reasoning \
  --features task_type prompt_chars prompt_words question_start num_numbers has_math_symbols has_code_like_text question_mark_count capitalized_word_count contains_parenthesis contains_quote has_comparison_word has_negation_word has_quantity_word cheap_answer_chars cheap_answer_words cheap_parsed_answer cheap_contains_uncertainty cheap_self_confidence cheap_sample_agreement \
  --cost-weights 0 0.02 0.05 0.1 \
  --text-feature prompt \
  --results-output results/openrouter_direct_metadata.csv
```

Cascade routing with cheap-response features:

```bash
python baselines/router_baselines.py \
  --data data/openrouter_strategyqa_cascade_router_dataset.csv \
  --models cheap_direct escalate_strong \
  --features task_type prompt_chars prompt_words question_start num_numbers has_math_symbols has_code_like_text question_mark_count capitalized_word_count contains_parenthesis contains_quote has_comparison_word has_negation_word has_quantity_word cheap_answer_chars cheap_answer_words cheap_parsed_answer cheap_contains_uncertainty cheap_self_confidence cheap_sample_agreement \
  --cost-weights 0 0.02 0.05 0.1 \
  --threshold-feature cheap_self_confidence \
  --thresholds 0.5 0.7 0.9 \
  --text-feature prompt \
  --results-output results/openrouter_cascade_features.csv
```

## Routers And Metrics

Implemented policies and routers:

- Always choose each candidate model
- Random routing
- Oracle routing
- Confidence-threshold routing
- Logistic regression
- Random forest
- MLP router
- TF-IDF + logistic regression
- TF-IDF + MLP

Metrics:

- `accuracy`: routed-answer correctness
- `avg_cost`: average normalized route cost
- `utility`: `correct - cost_weight * cost`
- `regret`: utility gap from oracle routing
- `route_acc`: agreement with oracle route labels

## Current Project Plan

The strongest project direction is mixed-task specialist routing, not only
StrategyQA escalation. Start with a small balanced dataset:

```text
150 StrategyQA or commonsense examples
150 GSM8K-style math examples
150 SciQ or ARC science examples
optional: 150 code examples
```

Compare a cheap generalist against one or more cheap specialist models. Good
early candidates:

```text
cheap generalist: mistralai/mistral-nemo
reasoning/math:  qwen/qwen3-235b-a22b-2507
science/general: google/gemini-2.0-flash-001
code optional:   qwen coder or codestral-style model
```

Then scale toward roughly 200 examples per task type if OpenRouter latency and
cost are manageable.

The final analysis should compare:

- Random split vs held-out-task split
- Metadata-only features vs cheap-response features
- Fixed policies vs learned routers
- Accuracy-cost tradeoff across several `cost_weight` values
- Normalized route costs vs actual OpenRouter `usage.cost`

## Mixed Specialist Routing

Prepare a balanced mixed-task CSV:

```bash
python scripts/prepare_mixed_tasks.py \
  --output data/tasks/mixed_specialist_600.csv \
  --per-task-limit 150 \
  --shuffle
```

Run cheap specialist candidates on the same examples:

```bash
python scripts/run_openrouter_multimodel.py \
  --tasks data/tasks/mixed_specialist_600.csv \
  --output data/raw/openrouter_mixed_specialists.jsonl \
  --route cheap_general=mistralai/mistral-nemo:full_reasoning \
  --route math_specialist=qwen/qwen3-235b-a22b-2507:reasoning \
  --route science_specialist=google/gemini-2.5-flash:reasoning \
  --route humanities_specialist=qwen/qwen3-235b-a22b-2507:reasoning \
  --limit 150 \
  --concurrency 4 \
  --request-delay-seconds 0.25 \
  --max-retries 8 \
  --resume
```

Build a router dataset with actual OpenRouter costs:

```bash
python scripts/build_router_dataset.py \
  --input data/raw/openrouter_mixed_specialists.jsonl \
  --output data/openrouter_mixed_specialists_router_dataset.csv \
  --strategies cheap_general math_specialist science_specialist humanities_specialist \
  --cheap-strategy cheap_general \
  --cost-mode actual
```

Train/evaluate fixed policies, manual task routing, and learned routers:

```bash
python baselines/router_baselines.py \
  --data data/openrouter_mixed_specialists_router_dataset.csv \
  --models cheap_general math_specialist science_specialist humanities_specialist \
  --features task_type prompt_chars prompt_words question_start num_numbers has_math_symbols has_code_like_text question_mark_count capitalized_word_count contains_parenthesis contains_quote has_comparison_word has_negation_word has_quantity_word cheap_answer_chars cheap_answer_words cheap_parsed_answer cheap_contains_uncertainty cheap_self_confidence cheap_sample_agreement \
  --cost-weights 0 1000 5000 10000 20000 \
  --task-router commonsense=cheap_general math=math_specialist science=science_specialist humanities=humanities_specialist \
  --text-feature prompt \
  --results-output results/openrouter_mixed_specialists.csv
```

For normalized conceptual costs instead of dollar costs, build with
`--cost-mode normalized` and use smaller weights like `0 0.02 0.05 0.1`.

Held-out-task evaluation only makes sense after creating a mixed dataset with
multiple task types, such as StrategyQA, math, and MMLU. For StrategyQA-only
experiments, use the default random split.
