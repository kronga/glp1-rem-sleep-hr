# Heart rate rises most during REM sleep after semaglutide initiation

Code for the paper *Heart rate rises most during REM sleep after semaglutide initiation:
a matched longitudinal cohort study* (citation to be added).

Participants of the Human Phenotype Project (HPP) 10K cohort who started semaglutide
were matched 1:1 to non-initiators on propensity score and visit timing. Each outcome was
estimated as a difference-in-differences. Sleep-stage heart rate from home sleep tests is
the primary sleep outcome.

## Repository structure

```
steps/   analysis pipeline (cohort, matching, estimation, sensitivity analyses)
src/     shared helpers (medication cohort, confounders, propensity matching, FDR)
paper/   figure and number generation
```

## Requirements

Python 3.11 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data availability

HPP data are available to researchers through the HPP data-access process. They are not
included in this repository.

## Reproducing the results

```bash
make all
```

This runs the pipeline in order:

| Target | Output |
|---|---|
| `transitions`, `upstream` | Medication exposure tables and intermediate cohorts |
| `cohort` | Matched cohorts and primary estimates for every outcome |
| `secondary` | Secondary sleep analyses, alternative matching designs, robustness analyses |
| `figs` | Main and supplementary figures (`paper/figures/final/`) |
| `numbers` | Every number reported in the text (`paper/NUMBERS.md`, `paper/tables/`) |

## License

MIT. See `LICENSE`.
