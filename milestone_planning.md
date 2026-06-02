# CS229 Milestone Planning Notes

## Main Issues To Fix From Proposal Feedback

1. **Define the routing problem precisely.**
   - State the input to the router: e.g., a task instance, prompt, query, example, or feature vector.
   - State the output of the router: one model from a fixed candidate set.
   - State the objective: maximize correctness, minimize cost subject to quality, or optimize a weighted utility.

2. **Name the models being routed between.**
   - Replace vague phrases like "different models" with a concrete set.
   - Example: `{cheap baseline model, medium model, strong model}` or `{logistic regression, random forest, gradient boosted trees}` depending on your actual project.

3. **Name the tasks being evaluated.**
   - Specify the task family and dataset source.
   - Example: classification tasks, math word problems, code generation problems, retrieval questions, tabular prediction tasks, etc.

4. **Describe the supervision.**
   - Explain how each training example is labeled.
   - For a model router, a natural label is the model with the best validation outcome on that instance, possibly after applying a cost penalty.
   - Include estimated data size: number of examples, number of tasks/datasets, train/validation/test split.

5. **Include at least one baseline.**
   - Strong baselines for a router project:
     - Always choose the cheapest model.
     - Always choose the strongest model.
     - Random routing.
     - Simple supervised router such as logistic regression on hand-engineered features.

6. **Show preliminary results, even if early.**
   - Include a small table with accuracy, cost, utility, or regret.
   - If results are not final, mark them as preliminary and explain what they show.

7. **Add error analysis.**
   - Identify where the baseline/router fails.
   - Examples: ambiguous instances, out-of-distribution tasks, examples where cheap model confidence is misleading, class imbalance in oracle labels.

## Questions I Need From You

1. What is the full project title?
2. What are the full names of all team members?
3. What exactly are you routing between? List the candidate models/systems.
4. What tasks or datasets are you using?
5. How many supervised examples do you have or expect to have?
6. What is the label for each example: best model, correct/incorrect per model, cost-adjusted utility, or something else?
7. What baseline result do you already have, if any?
8. What preliminary metric should the milestone emphasize: accuracy, cost, latency, calibration, regret, F1, or another metric?
9. Are there any teammate contribution details you want included?

## Submission Checklist

- [ ] Full project title appears at top.
- [ ] Full names of all team members appear at top.
- [ ] Motivation is concrete and avoids generic ML exposition.
- [ ] Method states model set, task set, inputs, labels, and objective.
- [ ] Preliminary experiments include at least one baseline.
- [ ] Error analysis describes observed failure modes.
- [ ] Next steps follow from preliminary results.
- [ ] Contributions section names each teammate's work.
- [ ] Draft is at most 3 pages excluding references.

