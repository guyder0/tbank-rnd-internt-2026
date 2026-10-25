# Repository Guidelines

## Project Structure & Module Organization

This workspace currently contains the KION movie recommendation dataset, rather than an application or packaged library.

- `KION_DATASET/README.md` documents the dataset, fields, benchmark context, and research citation.
- `KION_DATASET/interactions.csv.zip` contains interaction data.
- `KION_DATASET/data_original.zip` and `data_en.zip` provide Russian and English metadata.
- `KION_DATASET/data_original/` and `data_en/` hold extracted data; `sample_submission.csv` provides the recommendation submission format.
- `task.md` is currently empty; record task requirements there when defined.

There are no established source, test, or asset directories. If adding analysis code, keep reusable modules in `src/`, tests in `tests/`, and exploratory notebooks in `notebooks/`. Treat these as proposed conventions, not existing infrastructure.

## Development & Validation Commands

No build system, dependency manifest, or test runner is configured. Run these commands from the repository root when preparing data:

- `unzip -l KION_DATASET/data_en.zip` lists archive contents before extraction.
- `unzip -t KION_DATASET/interactions.csv.zip` checks archive integrity.
- `mkdir -p data/raw && unzip KION_DATASET/data_en.zip -d data/raw` extracts English metadata into a separate working directory.

Document dependency installation and execution commands alongside any new tooling. Exclude extracted data and generated outputs from version control before committing.

## Coding Style & Naming Conventions

No formatter or linter is configured. Use descriptive `snake_case` names and four-space indentation for new Python code. Keep Markdown headings descriptive and commands reproducible from the repository root. Preserve upstream archive names and dataset column names.

## Testing Guidelines

No testing framework or coverage threshold is established. For new Python modules, prefer `pytest` with `tests/test_<module>.py` files and document `python -m pytest` once configured. Use small fixtures to check parsing, identifier relationships, missing-value handling, and chronological splits without requiring full dataset extraction.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace, so no existing commit convention can be verified. Use concise, imperative subjects such as `Add interaction validation script`. Keep changes focused. Pull requests should describe the purpose, affected files, validation performed, and any data or dependency requirements. For recommendation experiments, record the split strategy, random seed, and evaluation metric.

## Dataset Handling

Preserve supplied archives unchanged. Keep derived tables, model artifacts, and local configuration separate from source files. Follow the citation guidance in `KION_DATASET/README.md` when publishing research using this dataset.
