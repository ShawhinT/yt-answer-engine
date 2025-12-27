# Experiment Registry

| ID | Description | Query Set | Best Run | Recall@3 (Hybrid) | MRR (Hybrid) | Notes |
|----|-------------|-----------|----------|-------------------|--------------|-------|

<!-- run.py will add rows here -->

---

## Adding New Experiments

To create a new experiment:

1. Copy `exp_001/` structure:
   ```bash
   cp -r experiments/exp_001 experiments/exp_XXX
   ```

2. Create `experiment.yaml` with exp_id, name, description, pipeline

3. Modify `src/main.py` to implement new method (must export `run_experiment(run_id, max_queries)`)

4. Run experiment:
   ```bash
   # Run single experiment
   python -m utils.experiments --exp exp_XXX --run-id r001

   # Run all experiments
   python -m utils.experiments --all --max-queries 5
   ```

5. Compare results in this registry
