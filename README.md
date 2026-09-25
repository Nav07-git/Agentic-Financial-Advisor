# Buy or Wait? — HackerRank Orchestrate

This project produces a deterministic financial recommendation for every row
in `dataset/requests.csv`. It uses only the supplied dataset and Python's
standard library; no API keys, packages, or network access are required.

## Run

Use Python 3.11+ from the repository root:

```bash
python code/main.py
```

The command reads `dataset/` and writes `output.csv` at the repository root.
It processes all requests, validates each final row before writing it, and
returns the output path on success.

## What the pipeline does

1. Resolves cash-relevant records and supplied exchange rates conservatively.
   Missing image amounts are never guessed or treated as zero.
2. Forecasts the next 90 days and calculates the Stage 1 safe amount and
   earliest full-payment date.
3. Generates, validates, and ranks Stage 2 full, partial, installment, and
   wait plans, including only explicitly permitted flexible spending changes.
4. Validates the exact submission schema, dates, payment plans, enums, and
   explanations before writing the final CSV.

## Submission contents

- `output.csv` — generated predictions (250 rows plus header)
- `code.zip` — runnable source, this README, and `evaluation/usage_report.md`
- `log.txt` — development transcript

The token/cost report for the final deterministic run is at
`evaluation/usage_report.md`.
