# CS229 Project Plan

## Locked Choices

1. **Experiment shape:** hybrid.
   - Direct routing: choose 3B vs 70B using only question metadata.
   - Cascade routing: run 3B first, then decide whether to accept or escalate to 70B.
   - Main story: cascade routing, because it tests whether the cheap model's behavior helps.

2. **Models via OpenRouter:**
   - Cheap model: `meta-llama/llama-3.2-3b-instruct:free`
   - Strong model: `meta-llama/llama-3.3-70b-instruct:free`

3. **Cost definition:**
   - Use normalized model-size/compute proxy.
   - Cheap 3B call cost: `1`
   - Strong 70B call cost: `8`
   - Cascade escalation cost: cheap feature call(s) plus strong call.

4. **Features:**
   - Metadata: task type, prompt length, word count, number count, math-symbol flag, code-like flag, question-mark count.
   - Cheap-response features: self-confidence, answer length, uncertainty phrases, sample agreement.
   - Text features: TF-IDF over the prompt.

5. **Datasets:**
   - Math: GSM8K or SVAMP.
   - Commonsense: StrategyQA.
   - Knowledge: MMLU subset.

6. **Evaluation splits:**
   - Random train/test split.
   - Held-out task split.

7. **Routers/baselines:**
   - Always cheap.
   - Always strong.
   - Random.
   - Confidence threshold.
   - Logistic regression.
   - Random forest.
   - MLP router.
   - TF-IDF + logistic regression.
   - TF-IDF + MLP.
   - Oracle.

## Main Research Question

Can a learned router predict when to escalate from a free cheap Llama 3.2 3B response to a free strong Llama 3.3 70B response, and do cheap-response confidence/consistency features improve over metadata-only routing?

## First Real Run

Start small:

```text
50 math
50 StrategyQA
50 MMLU
```

Then scale to:

```text
200 examples per task
```

if rate limits and runtime are manageable.

