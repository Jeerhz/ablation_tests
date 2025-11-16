# Ablation Tests for MuLSD

This folder contains Python scripts to run ablation tests for the MuLSD project.
Make sure this folder is located inside the main **MuLSD** directory.

---

## Requirements

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) as the Python package manager.
You can also use `pip` or `poetry` since this project is small.

To install `uv`, follow the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

Create python virtual environment in root folder with:

```bash
uv venv .venv
```

And install dependencies with

```bash
uv sync
```

---

## Scripts Overview

1. Create benchmark environement used to call muLSDwith: **`build_benchmark.py`**

   - Downloads the York Urban Dataset.
   - Converts it into a structured benchmark folder.
   - Creates one benchmark folder per selected options, putting a `Makefile` inside.

2. Call muLSD on all images with: **`main.py`**

   - Lists all possible combinations of ablation options.
   - For each combination:
     - Creates a specialized benchmark folder.
     - Runs the benchmark using `make`.

3. Compute the metrics and run tests on results with: **`eval.py`**
   - Run it inside `ablation_tests` folder after creating `benchmark_*` folders.
   - Extract scores from benchmark
   - Compute averages on combinations or on images
   - Run Wilcoxon tests for every pair of datasets (with option/without options)

---

## How to Run **`main.py`**

### Step-by-step:

1. **Go to the `ablation_tests` folder**

   ```bash
   cd ablation_tests
   ```

2. **(To do once) Set up the benchmark environment**

   This will download and prepare the dataset.

   ```bash
   uv run python main.py --setup
   ```

3. **Run all benchmark tests**

   This creates and executes all option combinations.

   ```bash
   uv run python main.py --run
   ```

4. **Clean up all generated benchmark folders**

   This removes all folders like `benchmark_*`.

   ```bash
   uv run python main.py --clean
   ```

5. **(Optional) Run all steps at once**

   If no arguments are provided, all three actions will run:

   ```bash
   uv run python main.py
   ```

---

## Available Options

These are the option flags used to generate benchmark variants:

```
"n", "p", "f", "c", "o", "e", "w"
```

Refer to ReadMe in MuLSD folder to learn more about them.
