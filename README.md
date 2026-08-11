# exa-models-benchmark

Benchmark pipeline for the paper *ExaModels.jl: an Algebraic Modeling System for Nonlinear
Programming on GPUs* ([exanauts/exa-models-paper](https://github.com/exanauts/exa-models-paper)).
The raw measurements live on this repository's **`results` branch**, one self-contained bundle per run;
the paper's tracked tables and figures are regenerated from them by the pipeline here and deployed into
the paper checkout with `make deploy REPO=/path/to/exa-models-paper`.

This repository holds the callback-level benchmark for the ExaModels.jl paper. It times each NLPModels.jl
derivative callback — `obj`, `cons!`, `grad!`, `jac_coord!`, `hess_coord!`, the matrix-vector products, and model
creation — for **ExaModels.jl** against **JuMP** and **AMPL**, on CPU and five GPU backends, across three problem
suites (Lukšan–Vlček, COPS, PGLIB-OPF).

> **Reproducing the paper?** You do not need this file. Every table and figure in the paper is regenerated
> from the results already archived on the `results` branch, on any machine, in minutes, without a GPU. The
> exact command sequence is in the [root README](../README.md). This file is about *taking new
> measurements* on new hardware.

This README is the handoff guide for **running the benchmark on a new cluster / newer GPUs**. For the
internal data-integrity rules and audit log, see `CLAUDE.md`.

---

## 1. Prerequisites

- **Julia 1.12.0** (the paper's pinned version; `Project.toml` requires ≥ 1.10). Recommended: install via
  [`juliaup`](https://github.com/JuliaLang/juliaup), then `juliaup add 1.12.0`.
- **GitHub network access** for `Pkg.instantiate` — five packages are pulled straight from git rather than
  from the registry, each pinned to `main` in `Project.toml [sources]`: `ExaModels`, `ExaModelsAMPL`,
  `ExaModelsPower`, `LuksanVlcekBenchmark` and `COPSBenchmark`. The exact commits actually used are the
  ones recorded in the committed `Manifest.toml`, which is what `Pkg.instantiate()` resolves against.
- **GPU drivers/toolkits**, only for the backend(s) you run:
  | Backend | Precision | Requires |
  |---------|-----------|----------|
  | `cuda`   | fp64 | NVIDIA driver + CUDA toolkit (CUDA.jl) |
  | `amdgpu` | fp64 | ROCm (AMDGPU.jl) |
  | `oneapi` | fp64 | Intel Level Zero (oneAPI.jl) |
  | `metal`  | fp32 | Apple Silicon + macOS (Metal.jl) |
  | `cpu` / `reference` | fp64 | none (CPU only) |
- **AMPL** is only needed for the `reference` run (JuMP + AMPL baselines, CPU). The GPU comparisons do not need it.

> **Precision caveat:** all GPU backends run fp64 except `metal`, which is **fp32** (Metal.jl constraint) — never
> compare fp32 timings against fp64 results as if equivalent.

## 2. Copy-paste recipes

Every recipe below is self-contained: paste it into a shell on the benchmark machine. The common preamble
(once per machine) is:

```bash
git clone git@github.com:exanauts/exa-models-paper.git
cd exa-models-paper/benchmark
make setup                                # installs Julia via juliaup if missing
export PATH="$HOME/.juliaup/bin:$PATH"    # if juliaup was just installed
```

`make setup` is part of the standard workflow: every recipe starts with it. It is idempotent —
after the first run it is a fast no-op, so including it costs seconds and protects against a
stale or missing environment (e.g. after a `git pull` that changes the Manifest). The FIRST run
per clone does the full precompilation, which is parallel across cores — on a cluster, run it in
a multi-core compute allocation (e.g. `salloc -c 16`), not a core-limited login shell — **and with
the target accelerator visible** (e.g. `--gres=gpu:1`): GPU-runtime JLLs bake their choice in at
precompile time, so a CPU-only first setup silently breaks every later GPU run from that depot.

> **If a depot is already poisoned** (GPU runs fail with "CUDA.jl could not find an appropriate
> CUDA runtime … JLLs were precompiled without an NVIDIA driver"), re-running setup with a GPU
> does NOT repair it — `Pkg.precompile` skips caches it considers valid. Force the recompile,
> on a GPU-visible node:
>
> ```julia
> pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
> Base.compilecache(pkg)
> ```
>
> (The error text blames login nodes and containers; the actual cause is usually a CPU-only
> batch allocation.)

`make setup` uses the committed `Manifest.toml` — **do not run `Pkg.update()`**, which would break version
pinning and reproducibility.

### Smoke test first (any machine, ~2 min)

```bash
make nvidia-opf SECONDS=0.2 QUICK=1     # substitute the vendor for this machine
```

### NVIDIA GPU machine

```bash
make setup && make nvidia && make save
```

### AMD GPU machine

```bash
make setup && make amd && make save
```

### Intel GPU machine

```bash
make setup && make intel && make save
```

### Apple silicon (Metal)

```bash
make setup && make apple && make save    # fp32
```

### Main CPU machine (all baselines)

```bash
make setup
make reference && make save              # JuMP + AMPL baselines (all suites)
make cpu cpu-mt && make save             # ExaModels CPU, single-thread + multi-thread
```

The reference baselines are the most time-consuming runs (hours; large OPF JuMP builds dominate), so
`make reference` shards them across pinned cores by default — each shard is its own single-threaded
process on its own core (`taskset`), so per-callback timings keep the single-core discipline.
`NPROC` defaults to min(8, cores); override with `NPROC=k`, or `NPROC=1` for a single process.

### Thread-scaling curve

```bash
make cpu-scaling MT_THREADS_LIST='1 2 4 8 16 32' && make save
```

### Multi-GPU node: split instances across all GPUs

This is the default: plain `make nvidia` (likewise `amd` / `intel`) counts the node's devices and
shards across all of them automatically, so on an 8-GPU node it already finishes roughly 8x faster.
Each device runs a disjoint subset of the instances (`SHARD=i/N`); the CSVs carry
`_dev<i>_shard<i>of<N>` tags and merge cleanly at collection time.

```bash
make nvidia && make save              # auto-detects and uses ALL GPUs (validated on a 2x GV100 node)
make nvidia NGPUS=4 && make save      # override: force the device count
make nvidia DEVICE=1 && make save     # override: pin to one device, no sharding
```

Manual variant (e.g. two GPUs, two shells):

```bash
make nvidia DEVICE=0 SHARD=1/2    # shell 1
make nvidia DEVICE=1 SHARD=2/2    # shell 2, concurrently; then make save once
```

`SHARD` also works without GPUs, e.g. splitting the CPU reference across two nodes
(`make reference SHARD=1/2` on one, `SHARD=2/2` on the other).

### Slurm cluster

First run `make setup` separately in an interactive allocation (precompilation is multi-core):

```bash
salloc -c 16 --time=00:30:00
make setup
exit
```

Then interactive benchmarking: `salloc --gres=gpu:1 ...`, then any recipe above.

Batch: per-leg scripts for the campaign live in [`slurm/`](slurm/) — one file per run
prefixed by cluster (`orcd-*` for the MIT ORCD legs, `jlse-*` for JLSE, `local-apple` for the Mac). **Every leg
runs its own `make setup`**: the overhead is seconds when the environment is current, and
it makes each job self-contained and robust — no ordering between legs, no stale-env
failures after a `git pull`. Submit any subset in any order:

```bash
sbatch slurm/orcd-reference.sbatch
sbatch slurm/orcd-nvidia-h100.sbatch
```

Every leg script runs `make setup` itself and it is idempotent, so the first leg you submit
pays the one-time environment build and the rest find nothing to do. There is no separate
setup job to submit. Adjust partition/`-w` node pins to live queue conditions.

**Do not front-load that precompilation onto a cheap CPU-only allocation.** CUDA.jl's JLLs
bake the runtime choice in at precompile time, so precompiling with no NVIDIA driver visible
writes "no CUDA runtime" into the *shared* depot and every later GPU leg then dies within
seconds of starting. Letting each leg run its own `make setup` avoids this, because a GPU leg
always has its device visible via `--gres`. If it does happen, a GPU-visible `make setup` will
*not* repair it -- `Pkg.precompile` skips packages it considers already built, so the poisoned
cache must be invalidated explicitly:

```julia
pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
Base.compilecache(pkg)
```

If compute nodes have no outbound network or git credentials, run `make save` from the login node after the
job finishes — the results persist in the clone's `data/results/`.

### AD-framework comparison (paper section 8.4)

```bash
make compare-setup                        # once: python venv (jax/torch/casadi), pins frozen
make compare-ad                           # fair single-core CPU: ExaModels (1t + MT), ADNLPModels,
                                          #   JAX, PyTorch, CasADi (sx / mx / map modes)
make compare-ad-gpu                       # CUDA: ExaModels, JAX, PyTorch
make compare-cold                         # cold-start compile timing: one fresh process per
                                          #   framework and size (first call vs warm reuse)
make save                                 # compare_*.csv are archived with everything else
```

The CPU rows run under `taskset -c 0` with all thread pools capped at 1; the recorded
`cpu_wall_ratio` column audits the single-core discipline. Python package pins live in
`compare/requirements.lock`.

### Monitoring a long run

Every run tees its full output to `data/results/logs/<host>_<backend>[_dev..][_shard..]_<utc>.log` with
progress counters (`[OPF ExaModels 37/120] ... (elapsed 512s)`), and maintains a live partial CSV
(`results/partial_<host>_p<pid>.csv`), so `tail -f` the log or watch the partial CSV row count.

## 3. Make targets and options

One target per hardware vendor, each with per-suite granularity so nothing forces a full run:

| Target | Backend | Precision |
|---|---|---|
| `make nvidia` | CUDA | fp64 |
| `make amd` | AMDGPU (ROCm) | fp64 |
| `make intel` | oneAPI | fp64 |
| `make apple` | Metal | fp32 |
| `make cpu` | ExaModels CPU, 1 thread | fp64 |
| `make cpu-mt` | ExaModels CPU, `MT_THREADS` threads | fp64 |
| `make cpu-scaling` | `cpu-mt` sweep over `MT_THREADS_LIST` | fp64 |
| `make reference` | JuMP + AMPL baselines | fp64 |
| `make jump-opf` | JuMP OPF (= `reference` restricted to OPF) | fp64 |

Per-suite variants: `make <vendor>-lv`, `make <vendor>-cops`, `make <vendor>-opf` (e.g. `make amd-opf`).

**Options** (append `VAR=value`): `JULIA=/path/to/julia`, `SECONDS` (per-callback timing budget; CPU rows
report the **minimum** over per-call timings, GPU rows the **sync-bracketed batch mean**, best of three
batches, with batch size `bt_n` and spread `bt_spread` recorded per row), `MT_THREADS`, `MT_THREADS_LIST`,
`DEVICE=<i>` (pin to one GPU), `SHARD=i/N` (run the i-th of N disjoint instance subsets), `NGPUS`,
`QUICK=1` (small instance sizes), `SUITES=LV,COPS` (explicit suite list).

**Resource etiquette:** check `nvidia-smi` / `nvtop` / `htop` before a full run; don't saturate a shared machine.
A *flat* GPU time that doesn't grow with problem size signals GPU scalar indexing — investigate before trusting it.

## 4. Outputs

Each run writes to `benchmark/data/results/` — a **temporary staging directory**, fully gitignored on
`main`; the only durable copy of raw results is the `results` branch (below):
- `<hostname>_<tag>.csv` — one row per (problem, size, modeling system) with the timing columns.
- `<hostname>_<tag>_hw.toml` (or `_hw.txt`) — hardware/software metadata for that run.

CSV schema (must stay stable for ingestion):
```
suite,problem,size,ams,nvar,ncon,nnzj,nnzh,tobj,tcon,tgrad,tjac,thess,thprod,tjprod,tjtprod,tcreate,bt_n,bt_spread
```

## 5. Returning results: `make save` and the `results` branch

`make save` (script: `save_results.sh`) archives everything from `data/results/` to the dedicated **`results`
branch** under a unique directory `runs/<UTC-timestamp>-<hostname>-<uuid>/`, so concurrent runs from different
machines never conflict. Each run directory is self-contained for the paper and for auditing:

- `results/` — the raw CSVs, `*_hw.toml` hardware info, and `logs/` with the full benchmark output;
- `Manifest.toml` — the exact Julia package versions used;
- `run.toml` — run metadata: host, OS, Julia version, code commit, file inventory.

Anyone can inspect the raw data behind the paper's tables at
`https://github.com/exanauts/exa-models-paper/tree/results/runs/`.

On the paper machine, select the runs to include (by UUID substring; empty = all), generate, review, deploy:

```bash
make results RUNS='66899914 2c126f43'   # fetch selected runs, regenerate tables+figures into
                                        #   data/build/, compile data/preview/standalone_results.pdf
make deploy                             # copy build/ into results/ (the tracked paper inputs)
make pdf                                # rebuild the paper
```

Runs are applied oldest-to-newest, so a rerun from the same machine supersedes its earlier files.

Platform labels are assigned by `hardware_table.jl`. The label's letter names the accelerator class and
the number counts within that class: `C<n>` for CPU-only platforms, `N<n>` NVIDIA, `A<n>` AMD, `I<n>`
Intel, `M<n>` Apple. A platform with no pin gets the next free number in its class, *positionally*, in
reading order over whichever `*_hw.toml` files happen to be present, and a warning naming its run ids is
printed. Positional assignment means adding or losing a single run bundle silently renumbers platforms in
every table, figure and sentence, so every platform whose label appears in the prose is pinned in
`data/labels.toml`, keyed by run-UUID substring or by bare hostname. Pin a new machine before its label
reaches the text.

To regenerate the paper tables/figures step by step. All of these run in the `benchmark/data` project and
write into the `benchmark/data/build/` staging directory, which `make deploy` then copies into the paper's
tracked `results/`:

```bash
cd benchmark/data
julia --project=. collect.jl        # merge all results/*.csv → results/combined.csv
julia --project=. hardware_table.jl # → build/tables/hardware.tex
julia --project=. table.jl          # → build/tables/*.tex   (SGM summary + per-suite results)
julia --project=. gpu_table.jl      # → build/tables/gpu_summary*.tex
julia --project=. compare_table.jl  # → build/tables/compare_ad.tex, compare_ad_opf.tex
julia --project=. plot_opf_pgf.jl   # → build/figures/*.tex  (pgfplots axes)
julia --project=. breakdown_out.jl  # → build/tables/breakdown*.tex, build/figures/breakdown*.tex
```

Or from `benchmark/`: `make tables` (collect + hardware_table + table + gpu_table + compare_table),
`make plots` (collect + plot_opf_pgf) and `make solve-tables` (breakdown_out). Note that
`make solve-tables` is **not** a prerequisite of `make tables` or `make pipeline`; the Section 8.5
artifacts are only regenerated when it is run explicitly.

**Data-integrity rule:** every number in the paper must come from this pipeline. No value is ever typed or
estimated by hand — see `CLAUDE.md`.

## 6. Suites and sizes

- **LV** (Lukšan–Vlček): 18 scalable sparse equality-constrained NLPs, parameterized by `N`.
- **COPS**: 18 scalable NLPs from optimal control / PDE / parameter estimation, parameterized by a discretization.
- **PGLIB-OPF**: AC optimal power flow on real grid topologies, two formulations (polar, rectangular). Instance
  data lives under `data/pglib_opf_*`.

Exact size sets are defined in `cases.jl` (and `cases_quick.jl` / `cases_minimal.jl` for fast subsets).

## 7. Notes for the cluster operator

- Run each backend on hardware that actually has that accelerator; the script auto-detects the device name.
- For `oneapi` in fp64, set `IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1` (emulated, slow — fp32 is the
  paper-reported oneAPI precision).
- If the git-sourced packages fail to resolve, confirm outbound GitHub access from the compute node.

## Reproducing every table and figure from the stored data

This is the reproducibility claim of the repository: **given the archived raw results, anyone can
regenerate every table and figure in the paper, on any machine, in minutes.** No GPU and no benchmark
run are involved.

### Prerequisites

- `git`, and network access to this repository (the first step fetches the `results` branch from `origin`).
- Julia. `Project.toml` requires 1.10 or newer; the campaign used 1.12, which is the channel
  `make setup` installs via juliaup.
- A TeX distribution, for the final PDF (see above).

### Instantiate the generator environment

`make setup` instantiates only the top-level this repository project, which is the environment that *runs*
benchmarks. It does **not** instantiate `data/`, and all of the generators run in that
project. On a fresh clone, `make collect` therefore fails with a missing-package error until you do
this once:

```
julia --project=benchmark/data -e 'using Pkg; Pkg.instantiate()'
```

That is the only environment regeneration needs. `solve/` does not have to be instantiated:
the Section 8.5 table generator (`breakdown_out.jl`) runs in the `benchmark/data` project and only reads
CSVs. `make solve-setup` is needed only to *run* the Section 8.5 experiment, and requires a GPU.

### The sequence

```
make fetch-results          # pull the archived run bundles from the results branch
                            #   into data/results/

cp data/_hw_rescue/*.toml data/results/
                            # the Section 8.4 / 8.5 host descriptions are not part of
                            # any run bundle; see the note below

make tables                 # runs collect + hardware first, then table.jl, gpu_table.jl,
                            #   compare_table.jl  ->  data/build/tables/*.tex
make plots                  # plot_opf_pgf.jl                 ->  data/build/figures/*.tex
make solve-tables           # breakdown_out.jl (Section 8.5)  ->  build/tables/ and build/figures/

make deploy REPO=/path/to/exa-models-paper
                            # copy build/ tables and figures into the paper checkout

latexmk -C /path/to/exa-models-paper && latexmk -synctex=1 -pdf -cd /path/to/exa-models-paper/main.tex
```

What each step does:

- `make fetch-results` fetches `origin/results`, checks it out into a temporary worktree, moves any CSVs
  and `*_hw.toml` already in `data/results/` aside into a timestamped stash, and copies every
  run bundle's `results/` into `data/results/` (and any `solve-results/` into
  `solve/results/`). Bundles are applied oldest to newest, so a rerun from the same machine
  supersedes its earlier files. `RUNS='<uuid-substring> ...'` restricts it to selected runs; empty means
  all.
- `make collect` (a prerequisite of `tables` and `plots`) merges every CSV in
  `data/results/` into `data/results/combined.csv`, which is what every generator
  reads.
- `make hardware` (a prerequisite of `tables`) reads the `*_hw.toml` files and writes
  `build/tables/hardware.tex`.
- `make deploy` copies the staged `build/` artifacts into the paper checkout given by `REPO`.

### Expected outcome

The paper tracks its `results/` directory. After the sequence above, `git diff -- results/` in the paper checkout

should report no changes: the artifacts regenerated from the archived data are identical to the ones
committed alongside the paper. If it does report changes, the deployed artifacts and the stored data
disagree, and the difference is the thing to investigate.

### The one-shot pipeline, and what it leaves out

`make pipeline` chains `fetch-results -> collect -> hardware -> tables -> plots -> deploy -> pdf`, plus a
standalone preview PDF. It is convenient, but note two gaps:

- **`make solve-tables` is not a prerequisite of `make tables` or of `make pipeline`.** The Section 8.5
  breakdown table and figures are regenerated only by running `make solve-tables` explicitly, before
  `make deploy`. Running `make pipeline` alone leaves the Section 8.5 artifacts at whatever is already in
  `results/`. Since `main.tex` `\input`s `results/tables/breakdown_facts` for numbers quoted in the
  Section 8.5 prose, this matters.
- **`make pipeline` does not copy `data/_hw_rescue/*.toml`.** Those two files
  (`node4513_compare_hw.toml`, `node5000_compare_hw.toml`) describe the Section 8.4 comparison host and
  the Section 8.5 GPU host. They are not present in any archived run bundle, and no script copies them,
  so without the explicit `cp` above the hardware table loses those two platforms.

## What needs the original hardware, and what does not

**Regenerating the tables and figures needs nothing but this repository.** Any machine, minutes, no
accelerator, per the section above.

**Regenerating the measurements is a different matter.** They are wall-clock timings of derivative
callbacks on specific processors and specific GPUs. Rerunning the harness elsewhere will produce a valid
benchmark, but it will not reproduce the paper's numbers, and it will not reproduce them on the same
hardware either beyond run-to-run noise. What follows is what each result actually required, as recorded
in the archived `*_hw.toml` files and in `slurm/`.

| Result | Target | What it needs |
|---|---|---|
| CPU baselines, Section 8.2 (`make cpu`, `make cpu-mt`, `make cpu-scaling`) | ExaModels on CPU, single-threaded and at 4 / 8 / 16 threads | A many-core x86 node, run exclusively. The campaign used AMD EPYC 9474F nodes. The single-thread baseline is the denominator of every speedup in the paper, so it must not share a machine. |
| JuMP and AMPL reference, Section 8.2 (`make reference`) | Reference modeling-system timings | A CPU node. Slowest leg by far: the campaign allocated 24 hours and shards it across pinned cores (`NPROC`, default `min(8, cores)`), each shard a single-threaded process under `taskset`. The AMPL path writes `.nl` files with `ExaModelsAMPL` and reads them back through `AmplNLReader` / `ASL_jll`; the harness does not invoke a separate AMPL installation. |
| NVIDIA GPUs, Section 8.3 (`make nvidia`) | CUDA, Float64 | The campaign collected six NVIDIA platforms: A100-SXM4-80GB, H100 80GB HBM3, H200, L40S, RTX PRO 6000 Blackwell Server Edition, and B200. Each needs that specific device. `make nvidia` shards across every visible GPU by default. |
| AMD GPU, Section 8.3 (`make amd`) | AMDGPU / ROCm, Float64 | One AMD GPU node with ROCm. One AMD platform was collected, sharded across eight devices. |
| Apple GPU, Section 8.3 (`make apple`) | Metal, **Float32** | Apple silicon running macOS. Collected on an M2 Pro. Metal.jl constrains this leg to Float32, so its timings are not directly comparable with the Float64 columns; the generated tables and figures mark it. |
| Intel GPU (`make intel`) | oneAPI, Float64 | The target exists and the harness supports oneAPI, but **no Intel GPU results were collected in this campaign.** There are no oneAPI CSVs on the `results` branch and no Intel platform in the paper. `slurm/jlse-intel.sbatch` exists but produced no archived results. |
| AD-framework comparison, Section 8.4 (`make compare-setup`, `make compare-ad`, `make compare-ad-opf`, `make compare-ad-gpu`, `make compare-ad-opf-gpu`) | ExaModels vs JAX, PyTorch, CasADi | Two machines: a CPU node for the single-core legs (all thread pools capped at 1, `taskset -c 0`, audited by the recorded `cpu_wall_ratio` column) and an NVIDIA GPU node for the CUDA legs. Also a Python 3.9 or newer interpreter **with development headers**, since `torch.compile` shells out to a C compiler; `make compare-setup` builds the venv and refuses to overwrite `compare/requirements.lock` from a CPU-only environment. CasADi is absent from the GPU rows because it has no GPU AD. |
| Solve-time breakdown, Section 8.5 (`make solve-setup`, `make solve-breakdown`) | MadNLP with LiftedKKT, four device configurations | A separate Julia project (`solve/`) needing MadNLP, MadNLPGPU and CUDSS. The CPU-only configuration ran on a CPU node; the three configurations that touch a device ran on an H200. `make solve-setup` must be run with a GPU visible, because the GPU runtime JLLs bake their choice in at precompile time. Its CSVs land in `solve/results/`, not `data/results/`, and travel in the run bundle under `solve-results/`. |

For instructions on running any of these, including the multi-GPU sharding, the Slurm scripts and the
depot-poisoning failure mode that a CPU-only precompile causes, see the running instructions above.

## Citing

If you use these benchmarks, cite the paper:

> Sungho Shin, Michel Schanen, Francois Pacaud, Alexis Montoison, and Mihai Anitescu.
> *ExaModels.jl: an Algebraic Modeling System for Nonlinear Programming on GPUs.*
> Submitted to Mathematical Programming Computation.

The paper sources live at [exanauts/exa-models-paper](https://github.com/exanauts/exa-models-paper);
ExaModels.jl itself lives at [exanauts/ExaModels.jl](https://github.com/exanauts/ExaModels.jl).
