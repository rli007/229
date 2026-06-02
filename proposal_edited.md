# Edited Project Proposal: Learning When to Escalate

## Working Title

**Learning When to Escalate: Cost-Aware Cascaded Routing for Mixed Reasoning Tasks**

## Motivation

Large language model systems often face a practical tradeoff between quality and inference cost. A cheap direct-answer strategy may solve easy examples at low cost, while a stronger reasoning strategy may be necessary for harder examples but is more expensive. Always using the strongest strategy wastes computation on easy inputs, while always using the cheapest strategy loses accuracy on difficult inputs.

This project studies **cost-aware escalation**: given an input and a cheap model's first response, can we predict whether it is worth escalating to a stronger strategy? This differs from Mixture-of-Experts routing, where routing happens inside one jointly trained neural network, usually per token or per layer. Our setting is external, per-example routing among complete inference strategies with explicit accuracy-cost tradeoffs.

Prior work has studied LLM routing and cascades broadly, so our novelty claim is deliberately narrower: we will empirically test which low-cost signals actually predict the marginal value of escalation, especially cheap-model confidence and consistency features, and whether these signals improve over simple metadata-only baselines.

## Problem Setup

For each task example \(x_i\), the system can choose between local, no-API inference strategies:

1. **cheap_direct**: Llama 3.2 1B with a short direct-answer prompt.
2. **strong_reasoning**: Llama 3.1 8B with a reasoning prompt.
3. **medium_fewshot**: optional extension using Llama 3.2 1B with few-shot prompting or another local intermediate model.

The main decision can be framed either as binary escalation:

\[
\text{accept cheap answer} \quad \text{vs.} \quad \text{escalate}
\]

or as multi-way routing among all strategies. We will emphasize the binary escalation interpretation in the report because it is easier to explain, requires less data, and matches the deployment question: "Should we pay for the stronger local inference path?"

For each strategy \(m\), we observe correctness \(c_{i,m} \in \{0,1\}\) and normalized cost \(k_m\). The cost-aware utility is:

\[
U(i,m) = c_{i,m} - \lambda k_m.
\]

The oracle route for training/evaluation is:

\[
y_i = \arg\max_m U(i,m).
\]

We will report accuracy and average cost separately, plus utility and regret relative to the oracle.

## Data and Tasks

The preferred dataset design is a mixed automatically gradable benchmark with spread-out task types:

- **Math reasoning:** GSM8K or SVAMP, graded by normalized numeric answer.
- **Commonsense/yes-no reasoning:** StrategyQA, graded by yes/no exact match.
- **Knowledge/multiple choice:** a small MMLU subset, graded by answer letter.

For feasibility, the milestone can use a smaller sample, for example 300-1000 total examples across 2-3 task families. We should avoid code-generation tasks for the first version because execution-based grading adds complexity.

## Features

We will compare feature groups:

1. **Metadata features**
   - task type
   - prompt length
   - word count
   - number count
   - presence of math symbols
   - presence of code-like text

2. **Cheap-response features**
   - cheap answer length
   - cheap self-reported confidence, if prompted
   - whether the cheap response contains uncertainty phrases
   - cheap sampled-answer agreement, if we sample multiple cheap responses

3. **Optional semantic features**
   - prompt embeddings or TF-IDF features, if time permits.

The key ablation will compare metadata-only routing against metadata plus cheap-response features. That is where the project can be meaningfully different from a generic LLM router.

## Baselines and Models

Baselines:

- always use cheap_direct
- always use strong_reasoning
- always use medium_fewshot, if included
- random routing
- oracle routing
- threshold cascade based on cheap confidence, if confidence is available

Learned routers:

- logistic regression
- random forest
- optional gradient boosting

The first milestone should include the fixed-policy baselines plus logistic regression and random forest. The final report should add ablations and possibly held-out-task evaluation.

## Evaluation

Main metrics:

- routed accuracy
- average normalized cost
- utility \(c - \lambda k\)
- regret against oracle routing
- route accuracy against oracle labels

Recommended evaluation splits:

1. Random example split for fast baseline development.
2. Held-out task or held-out dataset split for the final report, e.g., train on GSM8K and StrategyQA, test on SVAMP or MMLU subset.

## Risks and Mitigations

The main risk is that the strong strategy may dominate or the cheap strategy may already solve most examples. We mitigate this by using mixed task types and reporting oracle label balance. If almost all oracle labels choose one strategy, we will adjust cost weights, sample harder examples, or simplify to binary escalation.

Another risk is noisy grading for open-ended answers. We mitigate this by choosing automatically gradable tasks first: multiple choice, yes/no, and numeric final-answer tasks.

## Immediate Implementation Plan

1. Use the included synthetic raw-output file to validate the data pipeline.
2. Replace synthetic rows with real task examples and per-strategy outputs.
3. Run `build_router_dataset.py` to produce a router CSV.
4. Run `router_baselines.py` on metadata features.
5. Add cheap-response features and rerun the ablation.
6. Decide whether to include embeddings or held-out-task evaluation after seeing first real results.
