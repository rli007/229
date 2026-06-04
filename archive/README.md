# Archive

Older artifacts from earlier iterations of this project. Nothing in here is
required to reproduce the final report (`cs229-report.tex`); kept around only
as a record of what was tried and discarded.

## What's in here

- `scripts/` --- five superseded scripts:
  - `prepare_strategyqa.py`, `run_openrouter_strategies.py` --- the original
    StrategyQA-only pipeline (cheap-direct vs strong-reasoning cascade) before
    the mixed-task setup.
  - `run_local_llama_strategies.py` --- abandoned local-Llama experiments;
    the project ended up using OpenRouter end-to-end.
  - `evaluate_correctness_predictor_router.py` --- a Shnitzer-style per-model
    binary correctness predictor; superseded by the unified multiclass
    formulation in `scripts/run_class_experiments.py`.
  - `evaluate_model_groups.py` --- earlier model-group sweep, replaced by the
    pair sweep in `run_class_experiments.py` and the viable-only sweep in
    `run_architecture_analysis.py`.

- `data/` --- 13 stale router datasets from earlier experiments:
  - StrategyQA-only direct/cascade datasets.
  - 5-model mixed datasets (`openrouter_mixed_5models_*`), pre-10-model.
  - 10-model snapshot datasets, pre-787-example final.
  - The granite-probe variant of the 787 dataset (alternate cheap probe,
    not used in the final report).
  - Generic / smoke / example CSVs.

- `results/` --- 44 result CSVs that correspond to the archived scripts and
  datasets above (cascade baselines, correctness-predictor sweeps, model-group
  accuracy tables, 5-model results, StrategyQA results, etc.).

- `milestone/` --- the course milestone deliverable from May:
  `cs229-milestone.tex`, `cs229.sty`, and `reference.bib`. The final report
  is self-contained (inline bibliography, standard `article` class), so the
  `cs229.sty` and `reference.bib` files are no longer needed in the repo
  root.
