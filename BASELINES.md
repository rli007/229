# Cost-Aware Escalation Baselines

This code evaluates baselines for an instance-level router whose main project framing is:

> Given an input and the cheap strategy's first response, predict whether a stronger
> strategy is worth its additional cost.

Recommended OpenRouter setup:

- `cheap_direct`: `meta-llama/llama-3.2-3b-instruct:free`
- `strong_reasoning`: `meta-llama/llama-3.3-70b-instruct:free`
- `escalate_strong`: same 70B output, but with cascade cost equal to cheap feature call(s) plus 70B.

The cleanest project story is binary escalation: accept `cheap_direct` or escalate
to `strong_reasoning`. The code can still evaluate 3-way routing when
`medium_fewshot` is included.

## Generate OpenRouter Outputs

Set your key outside the repo:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Run a small end-to-end sample:

```bash
python scripts/run_openrouter_strategies.py \
  --tasks data/tasks/sample_tasks.csv \
  --output data/raw/openrouter_sample_outputs.jsonl \
  --limit 20 \
  --cheap-samples 3
```

For full experiments, replace `data/tasks/sample_tasks.csv` with the mixed benchmark task CSV.

## Build Router Datasets

Direct-routing dataset, where the router chooses 3B or 70B before observing the cheap answer:

```bash
python scripts/build_router_dataset.py \
  --input data/raw/openrouter_sample_outputs.jsonl \
  --output data/openrouter_direct_router_dataset.csv \
  --strategies cheap_direct strong_reasoning
```

Cascade-routing dataset, where the router has already paid for cheap features and chooses accept vs escalate:

```bash
python scripts/build_router_dataset.py \
  --input data/raw/openrouter_sample_outputs.jsonl \
  --output data/openrouter_cascade_router_dataset.csv \
  --strategies cheap_direct escalate_strong
```

## Run Baselines

Metadata-only direct routing:

```bash
python baselines/router_baselines.py \
  --data data/openrouter_direct_router_dataset.csv \
  --models cheap_direct strong_reasoning \
  --features task_type prompt_chars prompt_words num_numbers has_math_symbols has_code_like_text question_mark_count \
  --cost-weight 0.05 \
  --text-feature prompt \
  --results-output results/openrouter_direct_metadata.csv
```

Cascade routing with cheap-response features:

```bash
python baselines/router_baselines.py \
  --data data/openrouter_cascade_router_dataset.csv \
  --models cheap_direct escalate_strong \
  --features task_type prompt_chars prompt_words num_numbers has_math_symbols has_code_like_text question_mark_count cheap_answer_chars cheap_answer_words cheap_contains_uncertainty cheap_self_confidence cheap_sample_agreement \
  --cost-weight 0.05 \
  --threshold-feature cheap_self_confidence \
  --thresholds 0.5 0.7 0.9 \
  --text-feature prompt \
  --results-output results/openrouter_cascade_features.csv
```

Held-out task evaluation example:

```bash
python baselines/router_baselines.py \
  --data data/openrouter_cascade_router_dataset.csv \
  --models cheap_direct escalate_strong \
  --features task_type prompt_chars prompt_words num_numbers has_math_symbols has_code_like_text question_mark_count cheap_answer_chars cheap_answer_words cheap_contains_uncertainty cheap_self_confidence cheap_sample_agreement \
  --cost-weight 0.05 \
  --threshold-feature cheap_self_confidence \
  --text-feature prompt \
  --heldout-task-types knowledge \
  --results-output results/openrouter_cascade_heldout_knowledge.csv
```

## Legacy Local Llama Path

If your Hugging Face cache has the models, use model IDs or snapshot paths:

```bash
python scripts/run_local_llama_strategies.py \
  --backend hf \
  --tasks data/tasks/sample_tasks.csv \
  --output data/raw/local_llama_outputs.jsonl \
  --cheap-model meta-llama/Llama-3.2-1B \
  --strong-model meta-llama/Meta-Llama-3.1-8B \
  --cheap-tokenizer meta-llama/Meta-Llama-3.1-8B \
  --cheap-samples 3
```

On this machine, the 8B tokenizer resolves offline. The 1B cache may need
`--cheap-tokenizer meta-llama/Meta-Llama-3.1-8B`, which is included above.
If a model ID does not resolve, pass the full local snapshot path.

The local `Qwen/Qwen2.5-0.5B-Instruct` cache currently appears to contain tokenizer
files but not model weights, so it is not used by default.

## Data Format

Use one CSV row per example. Include any pre-routing features you want the router to use, plus two columns for each candidate model:

- `<model>_correct`: `1` if that model solved the example, otherwise `0`.
- `<model>_cost`: inference cost, latency, token cost, or a normalized proxy.

Example final CSV columns:

```text
example_id,task_type,prompt_chars,prompt_words,num_numbers,cheap_answer_words,cheap_self_confidence,cheap_direct_correct,cheap_direct_cost,strong_reasoning_correct,strong_reasoning_cost
```

## Policies Included

- `always_<model>` for every candidate model.
- `random`.
- `oracle`, which picks the best model using ground-truth outcomes.
- optional confidence-threshold cascade: accept the first model when confidence is high,
  otherwise escalate to the last model.
- `logistic_router`, a first supervised baseline.
- `random_forest_router`, a nonlinear supervised baseline.
- `mlp_router`, a small neural-network router over tabular features.
- `tfidf_logistic_router`, a prompt text + tabular logistic router.
- `tfidf_mlp_router`, a prompt text + tabular neural router.

## Metrics

- `accuracy`: correctness of the routed system.
- `avg_cost`: average selected model cost.
- `utility`: average `correct - cost_weight * cost`.
- `regret`: average gap from the oracle utility.
- `route_acc`: how often the router selected the same model as the oracle label.

## Project Choices To Decide

1. **Candidate model set.**
   - Direct: `cheap_direct` vs `strong_reasoning`.
   - Cascade: `cheap_direct` vs `escalate_strong`.

2. **Routing target.**
   - Recommended: cost-aware utility, `correct - lambda * cost`.
   - Report accuracy and average cost separately so the tradeoff is interpretable.

3. **Features available to router.**
   - First baseline: metadata only.
   - Main project angle: metadata plus cheap-response confidence/consistency.
   - Stretch: embeddings or held-out-task generalization.

4. **Evaluation split.**
   - Random example split: easiest first milestone choice.
   - Held-out task split: stronger evidence that routing generalizes.
   - Time-based split: best if data arrives sequentially.
