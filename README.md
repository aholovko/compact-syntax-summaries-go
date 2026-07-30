# Compact Syntax Summaries for Low-Resource Go Style Diagnosis and Repair — Replication Package

## Study purpose

This package supports the accompanying paper's study of compact syntax summaries for low-resource Go style diagnosis and
repair. It contains source-free run records and study rows, plus the code needed to regenerate the reported analysis.
Source-free means these files carry no Stack V2 source code; the package also contains no model files or checkpoints.

## Reproduce the results

After [installing `uv`](https://docs.astral.sh/uv/getting-started/installation/), run from the repository root:

```bash
uv run analysis/rerun_analysis.py
```

On first use, `uv` may install CPython 3.12 or 3.13 and the locked packages; the analysis itself never uses the network.
After that first run (or an explicit `uv sync --locked`), the same analysis can be verified offline:

```bash
uv run --offline --no-sync --locked analysis/rerun_analysis.py
```

## What the command does

The command validates 14 evaluation runs: 12 fine-tuned and 2 zero-shot. It also validates 2,206 source-free study rows.
It then recomputes the result inventory (`results.json`) and the paper's Tables 8.1–8.9 into `reproduced/`,
compares the 10 generated files with `expected/`, and leaves all inputs unchanged.

`expected/` is not an input to the calculations: every output is recomputed from the released records before any
comparison.

Full-precision numbers in `results.json` are compared with an absolute tolerance of `1e-7` because numerical
optimization can vary slightly across platforms. The CSV tables must match byte-for-byte.

The analysis itself does not retrain models, run inference, download data or weights, reconstruct Stack V2 source data,
use a GPU, invoke Go, or access the network.

## Experimental conditions

- C0 fine-tunes on the main tasks with raw Go code.
- C1 adds a compact syntax-summary sidecar to each main-task input.
- C2 uses the C1 inputs and replaces 20% of the training examples with syntax-summary generation.
- C2-control replaces that 20% with selected duplicated main-task examples under the same step count. It differs from
  C2 in content, length-selection rule, and realized token exposure.
- zero-shot-raw and zero-shot-syntax evaluate the base model without fine-tuning, using raw and summary-bearing
  prompts.

## Paperflow

During the study, an internal tool called Paperflow scheduled the fixed condition-and-seed combinations and collected
the run records. The released code handles training, evaluation, and analysis without Paperflow.

## Historical compute

The 12 fine-tuned runs used Lightning AI managed Studio on GCP (instance type `a2-ultragpu-1g`) with one NVIDIA
A100-SXM4-80GB GPU per run. Together they took 25.6 training hours and 35.4 end-to-end hours. Monetary cost is not
reported because no billing record was retained.

## Optional retraining

Retraining is not required to reproduce the reported results. The training and evaluation source is included for
inspection, and this section explains how to start a new run.

`paper4` was the project's internal identifier for this study. It remains in the Python namespace, variable names, and
prepared-data paths but does not denote an experimental condition or release version.

The prepared data is available from the immutable
[Zenodo record](https://doi.org/10.5281/zenodo.21698768) as the versioned asset
`compact-syntax-summaries-go-prepared-data-v1.0.0.tar.gz`. Place the archive next to this repository and extract it to
`../prepared-paper4`:

```bash
test -f ../compact-syntax-summaries-go-prepared-data-v1.0.0.tar.gz && \
  test ! -e ../prepared-paper4 && \
  tar -xzf ../compact-syntax-summaries-go-prepared-data-v1.0.0.tar.gz -C ..
```

Raw Stack V2 data alone is not enough: training also requires the generated targets, summaries, adjudications, and fixed
selection artifacts in this asset.

The base model is gated. Request access and accept its license on the
[official Hugging Face model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Once access is granted,
download the full snapshot outside this repository using the official
[Hugging Face `hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```bash
uvx hf auth login
uvx hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ../Llama-3.2-1B-Instruct
export PAPER4_MODEL_DIR=../Llama-3.2-1B-Instruct
```

`PAPER4_MODEL_DIR` must point to the downloaded model directory containing `original/tokenizer.model` and either a
single `model.safetensors` or `model.safetensors.index.json` plus all referenced shards.

Training also requires a CUDA GPU and CPython 3.12 or 3.13. Go `>=1.26.4,<1.27` from the
[official Go downloads page](https://go.dev/dl/) and its accompanying `gofmt` must be on `PATH`.

Then install go-critic v0.14.4 and put it on `PATH`:

```bash
go install github.com/go-critic/go-critic/cmd/go-critic@v0.14.4
export PATH="$(go env GOPATH)/bin:$PATH"
```

The archived runs did not record their exact Go and go-critic builds, so these versions define the supported environment
rather than the verified historical one.

Confirm that the model directory is set and the output path is absent, then start a C0 training run with seed 42:

```bash
test -n "${PAPER4_MODEL_DIR:-}" && \
  test ! -e ../paper4-c0-seed42 && \
  uv run --locked --group training python -m go_ast_assistant.paper4.run_experiment \
    --condition C0 --seed 42 --study-data-dir ../prepared-paper4 --model-dir "$PAPER4_MODEL_DIR" \
    --output-dir ../paper4-c0-seed42 --device cuda
```

Other fine-tuned runs use condition `C0`, `C1`, `C2`, or `C2-control` and seed `42`, `43`, or `44`. Each run needs its
own absent output directory. This command does not run the 2 zero-shot baselines.

The output directory retains only `records.jsonl`, `results.yaml`, `selection_trace.json`, and `manifest.yaml`;
temporary checkpoints and model weights are discarded after evaluation.

PyTorch may print warnings that some CUDA operations remain nondeterministic; those warnings do not stop the run.
Compare the new run's metrics with the released historical records; exact numerical agreement is not expected.

## Repository contents

- `analysis/`: deterministic, source-free regeneration of the results.
- `config/`: fixed experiment definitions and the manuscript-result inventory.
- `data/runs/`: normalized source-free records and metrics for 12 fine-tuned and 2 zero-shot runs.
- `data/study/`: 2,206 compact source-free study rows and global metadata.
- `expected/`: canonical results and Tables 8.1–8.9.
- `src/go_ast_assistant/paper4/`: training and evaluation source for inspection.
- `serializer/`: standalone Go syntax-summary command and its tests.
- `tests/`: analysis, package-contract, and optional training checks.

The serializer reads JSONL input, one `{ "id": "...", "code": "..." }` object per line. The default analysis command
does not run it. To test and run it locally:

```bash
cd serializer
GOTOOLCHAIN=local go test ./...
GOTOOLCHAIN=local go run ./cmd/go-syntax summarize \
  --in input.jsonl --out summaries.jsonl
```

## License and citation

The canonical repository is <https://github.com/aholovko/compact-syntax-summaries-go>.
Version 1.0.0 is identified by tag `v1.0.0`:
<https://github.com/aholovko/compact-syntax-summaries-go/releases/tag/v1.0.0>.

External data and model references are:

- Stack V2-derived data: DOI `10.57967/hf/5304`, revision
  `7b951fd57d19286153b46ba219aa2cb87fcc4d2b`.
- Base model: `meta-llama/Llama-3.2-1B-Instruct`.

Use `CITATION.cff` to cite this package; a full citation for the accompanying paper will be added on publication. The
root `LICENSE` applies MIT to project-authored software and documentation, and CC BY 4.0 to project-generated analysis
data, annotations, metadata, and numerical outputs. Neither grant relicenses third-party model material, reference
material, or code.
