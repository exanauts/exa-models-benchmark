# ============================================================================
# ExaModels Paper — Benchmark Pipeline
# ============================================================================
#
# Workflow on ANY benchmark machine (H200 node, JLSE AMD/Intel, Mac, ...):
#
#   git clone git@github.com:exanauts/exa-models-paper.git
#   cd exa-models-paper/benchmark
#   make setup          # instantiate the Julia environment (once)
#   make nvidia         # or: amd | intel | apple | cpu | cpu-mt | reference
#   make save           # git commit + push the new result CSVs
#
# Granular runs (per suite), e.g.:
#   make nvidia-opf     # OPF only
#   make amd-lv         # Luksan-Vlcek only
#   make intel-cops     # COPS only
#
# Options (append VAR=value):
#   JULIA=/path/to/julia   SECONDS=2.0   MT_THREADS=8   QUICK=1 (small instances)
#
# Post-processing (on the paper machine, after pulling results):
#   make pipeline       # collect -> hardware -> tables -> plots -> pdf
#
# ============================================================================

# --- Configuration -----------------------------------------------------------

SHELL := /bin/bash

JULIA      ?= julia
SECONDS    ?= 2.0
SUITES     ?=
MT_THREADS ?= 8
MT_THREADS_LIST ?= 1 2 4 8 16
DEVICE     ?=
SHARD      ?=
NGPUS      ?=
QUICK      ?=

BENCH := $(abspath .)
DATA  := $(BENCH)/data
REPO  := $(abspath ..)

QUICK_FLAG = $(if $(QUICK),quick,)

# $(call run,<backend-arg>,<precision>,<suites>,<threads>,<env-prefix>)
#   backend-arg: reference | nothing | CPU | CUDA | AMDGPU | oneAPI | Metal
#   precision:   fp64 | fp32   (fp32 appends the flag benchmark.jl expects)
define run
	mkdir -p $(DATA)/results/logs && cd $(DATA) && set -o pipefail && \
	$(if $(SHARD),EXA_SHARD=$(SHARD),) $(5) $(JULIA) --project=$(BENCH) -t$(4) $(BENCH)/benchmark.jl \
		$(1) $(SECONDS) $(3) $(if $(filter fp32,$(2)),fp32,) $(QUICK_FLAG) \
		2>&1 | tee $(DATA)/results/logs/$$(hostname -s)_$(1)$(if $(DEVICE),_dev$(DEVICE),)$(if $(SHARD),_shard$(subst /,of,$(SHARD)),)_$$(date -u +%Y%m%dT%H%M%SZ).log
endef

# GPU device selection (multi-GPU machines): make nvidia DEVICE=1 pins the run
# to that device and records it in the CSV tag (_dev1), so per-device runs of
# the same suite never collide and can execute in parallel.
CUDA_DEV_ENV  = $(if $(DEVICE),CUDA_VISIBLE_DEVICES=$(DEVICE) EXA_DEVICE=$(DEVICE),)
ROCM_DEV_ENV  = $(if $(DEVICE),ROCR_VISIBLE_DEVICES=$(DEVICE) EXA_DEVICE=$(DEVICE),)
ONEAPI_DEV_ENV= $(if $(DEVICE),ZE_AFFINITY_MASK=$(DEVICE) EXA_DEVICE=$(DEVICE),)

# --- Setup -------------------------------------------------------------------

.PHONY: setup
setup:                              ## install julia via juliaup if missing, then instantiate the env
	@command -v $(JULIA) >/dev/null 2>&1 || command -v $$HOME/.juliaup/bin/julia >/dev/null 2>&1 || { \
		echo "julia not found; installing via juliaup..."; \
		curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12; \
	}
	@PATH="$$HOME/.juliaup/bin:$$PATH"; export PATH; \
	$(JULIA) --project=$(BENCH) -e 'using Pkg; Pkg.instantiate(); Pkg.status()'

# --- Vendor targets (full suite: LV + COPS + OPF) ----------------------------

.PHONY: nvidia-single amd-single intel-single apple cpu cpu-mt jump-opf

# Device-count autodetection per vendor (used when neither DEVICE nor NGPUS is set)
COUNT_nvidia = $$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)
COUNT_amd    = $$(rocm-smi --showid 2>/dev/null | grep -o "^GPU\[[0-9]*\]" | sort -u | wc -l || true)
COUNT_intel  = $$(sycl-ls 2>/dev/null | grep -ci "level_zero.*gpu" || true)

# make <vendor>: shards across ALL detected GPUs by default.
# Overrides: DEVICE=<i> pins one GPU; NGPUS=<k> forces the device count;
# <vendor>-single is the plain one-process body.
define GPU_DISPATCH
.PHONY: $(1)
$(1):
	@if [ -n "$(DEVICE)" ]; then \
		$$(MAKE) $(1)-single DEVICE=$(DEVICE); \
	else \
		n="$(NGPUS)"; \
		if [ -z "$$$$n" ]; then n=$$(COUNT_$(1)); fi; \
		case "$$$$n" in (""|*[!0-9]*) n=1;; esac; \
		[ "$$$$n" -ge 1 ] || n=1; \
		if [ "$$$$n" -le 1 ]; then \
			$$(MAKE) $(1)-single; \
		else \
			echo "Detected $$$$n devices; sharding across all (override: DEVICE=i or NGPUS=k)"; \
			$$(MAKE) $(1)-parallel NGPUS=$$$$n; \
		fi; \
	fi
endef
$(foreach v,nvidia amd intel,$(eval $(call GPU_DISPATCH,$(v))))

nvidia-single:                      ## NVIDIA GPU (CUDA), Float64, one device
	$(call run,CUDA,fp64,$(SUITES),1,$(CUDA_DEV_ENV))

amd-single:                         ## AMD GPU (AMDGPU/ROCm), Float64, one device
	$(call run,AMDGPU,fp64,$(SUITES),1,$(ROCM_DEV_ENV))

intel-single:                       ## Intel GPU (oneAPI), Float64, one device
	$(call run,oneAPI,fp64,$(SUITES),1,$(ONEAPI_DEV_ENV))

apple:                              ## Apple GPU (Metal), Float32
	$(call run,Metal,fp32,$(SUITES),1,)

cpu:                                ## ExaModels CPU, single thread, Float64
	$(call run,nothing,fp64,$(SUITES),1,)

cpu-mt:                             ## ExaModels CPU, KernelAbstractions threads
	$(call run,CPU,fp64,$(SUITES),$(MT_THREADS),)

reference-single:                   ## JuMP + AMPL reference timings, one process
	$(call run,reference,fp64,$(SUITES),1,)

# make reference: shards the baselines across NPROC single-threaded processes by
# default, each pinned to its own core (taskset), preserving the single-core
# timing discipline.  NPROC defaults to min(8, cores); NPROC=1 disables
# sharding; an explicit SHARD=i/N runs that one shard (e.g. across nodes).
NPROC ?= $(shell n=$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8); if [ $$n -gt 8 ]; then echo 8; else echo $$n; fi)

.PHONY: reference reference-single
reference:                          ## JuMP + AMPL reference timings (sharded across NPROC cores)
	@if [ -n "$(SHARD)" ] || [ "$(NPROC)" -le 1 ]; then \
		$(MAKE) reference-single SHARD=$(SHARD) SUITES="$(SUITES)" SECONDS=$(SECONDS) QUICK="$(QUICK)"; \
	else \
		echo "Sharding reference across $(NPROC) pinned cores (override: NPROC=k; NPROC=1 for one process)"; \
		pids=""; rc=0; \
		for i in $$(seq 0 $$(( $(NPROC) - 1 ))); do \
			pin=""; command -v taskset >/dev/null 2>&1 && pin="taskset -c $$i"; \
			$$pin $(MAKE) reference-single SHARD=$$(( i + 1 ))/$(NPROC) SUITES="$(SUITES)" SECONDS=$(SECONDS) QUICK="$(QUICK)" & \
			pids="$$pids $$!"; \
		done; \
		for p in $$pids; do wait $$p || rc=1; done; \
		exit $$rc; \
	fi

jump-opf:                           ## JuMP OPF run (alias: reference restricted to OPF)
	$(MAKE) reference SUITES=OPF

.PHONY: cpu-scaling
cpu-scaling:                        ## cpu-mt at each thread count in MT_THREADS_LIST
	for t in $(MT_THREADS_LIST); do \
		$(MAKE) cpu-mt MT_THREADS=$$t SUITES=$(SUITES) SECONDS=$(SECONDS) QUICK=$(QUICK) || exit 1; \
	done

# --- Multi-GPU parallel: split the instance list across NGPUS devices --------
# make nvidia-parallel NGPUS=8  ->  8 concurrent runs, DEVICE=0..7, disjoint shards.

define PAR_RULES
.PHONY: $(1)-parallel
$(1)-parallel:
	@pids=""; rc=0; \
	for i in $$$$(seq 0 $$$$(( $(NGPUS) - 1 ))); do \
		$$(MAKE) $(1)-single DEVICE=$$$$i SHARD=$$$$(( i + 1 ))/$(NGPUS) SUITES="$(SUITES)" SECONDS=$(SECONDS) QUICK="$(QUICK)" & \
		pids="$$$$pids $$$$!"; \
	done; \
	for p in $$$$pids; do wait $$$$p || rc=1; done; \
	exit $$$$rc
endef
$(foreach v,nvidia amd intel,$(eval $(call PAR_RULES,$(v))))

# --- Per-suite granular targets: <vendor>-{lv,cops,opf} ----------------------

VENDORS := nvidia amd intel apple cpu cpu-mt reference
define SUITE_RULES
.PHONY: $(1)-lv $(1)-cops $(1)-opf
$(1)-lv:    ; $$(MAKE) $(1) SUITES=LV
$(1)-cops:  ; $$(MAKE) $(1) SUITES=COPS
$(1)-opf:   ; $$(MAKE) $(1) SUITES=OPF
endef
$(foreach v,$(VENDORS),$(eval $(call SUITE_RULES,$(v))))

# --- Section 8.4: AD-framework comparison ------------------------------------

COMPARE_N       ?= 200000
COMPARE_COLD_N  ?= 20,200,2000,20000,200000
COMPARE_VENV    := $(DATA)/compare-venv
COMPARE_PY      := $(COMPARE_VENV)/bin/python
TASKSET         := $(shell command -v taskset >/dev/null 2>&1 && echo "taskset -c 0" || echo "")
COMPARE_ENV     := OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false

.PHONY: compare-setup compare-ad compare-ad-gpu compare-cold

compare-setup:                      ## python venv for jax/torch/casadi (cpu wheels); freezes requirements.lock
	python3 -m venv $(COMPARE_VENV)
	$(COMPARE_PY) -m pip install -q --upgrade pip
	$(COMPARE_PY) -m pip install -q -r compare/requirements.in
	$(COMPARE_PY) -m pip freeze > compare/requirements.lock
	@echo "Python env ready; pins written to compare/requirements.lock"

compare-ad:                         ## fair single-core CPU comparison: examodels, adnlpmodels, jax, torch, casadi
	cd $(DATA) && $(TASKSET) $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl examodels cpu $(COMPARE_N) $(SECONDS)
	cd $(DATA) && $(TASKSET) $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl adnlpmodels cpu $(COMPARE_N) $(SECONDS)
	cd $(DATA) && $(JULIA) --project=$(BENCH) -t$(MT_THREADS) $(BENCH)/compare/compare_ad.jl examodels cpu-mt $(COMPARE_N) $(SECONDS)
	cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework jax --device cpu --n $(COMPARE_N) --seconds $(SECONDS)
	cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework torch --device cpu --n $(COMPARE_N) --seconds $(SECONDS)
	cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework casadi --mode sx --device cpu --n $(COMPARE_N) --seconds $(SECONDS)
	cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework casadi --mode mx --device cpu --n $(COMPARE_N) --seconds $(SECONDS)
	cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework casadi --mode map --device cpu --n $(COMPARE_N) --seconds $(SECONDS)

compare-ad-gpu:                     ## GPU comparison (CUDA): examodels, jax, torch
	cd $(DATA) && $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl examodels cuda $(COMPARE_N) $(SECONDS)
	cd $(DATA) && $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework jax --device cuda --n $(COMPARE_N) --seconds $(SECONDS)
	cd $(DATA) && $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework torch --device cuda --n $(COMPARE_N) --seconds $(SECONDS)

compare-cold:                       ## cold-start compile timing: fresh process per framework and size
	cd $(DATA) && COMPARE_N=$(COMPARE_COLD_N) $(TASKSET) $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_cold.jl
	cd $(DATA) && COMPARE_N=$(COMPARE_COLD_N) $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_cold.py

# --- Save results ------------------------------------------------------------

.PHONY: save
save:                               ## archive results to the 'results' branch (UUID run dir)
	$(BENCH)/save_results.sh

# --- Post-processing (on the paper machine) ----------------------------------

.PHONY: fetch-results pipeline results deploy collect hardware tables plots pdf

RUNS ?=

fetch-results:                      ## pull archived runs into data/results/ (RUNS='id1 id2' selects by uuid/substring; empty = all)
	git -C $(REPO) fetch origin results
	@tmp=$$(mktemp -d); \
	git -C $(REPO) worktree add --detach $$tmp origin/results >/dev/null 2>&1; \
	rm -f $(DATA)/results/*.csv $(DATA)/results/*_hw.toml; rm -rf $(DATA)/results/logs; \
	sel="$(RUNS)"; n=0; \
	for d in $$tmp/runs/*/; do \
		name=$$(basename $$d); \
		if [ -n "$$sel" ]; then keep=0; for s in $$sel; do case "$$name" in *"$$s"*) keep=1;; esac; done; [ $$keep -eq 1 ] || continue; fi; \
		[ -d "$$d/results" ] || continue; \
		cp -R $$d/results/* $(DATA)/results/ 2>/dev/null || true; \
		for h in $$d/results/*_hw.toml; do \
			[ -f "$$h" ] || continue; hb=$$(basename $$h); \
			grep -q '^run_id' $(DATA)/results/$$hb 2>/dev/null || \
				{ echo "run_id = \"$$name\""; cat $$h; } > $(DATA)/results/$$hb; \
		done; \
		echo "  + $$name"; n=$$((n+1)); \
	done; \
	git -C $(REPO) worktree remove --force $$tmp; \
	echo "Merged $$n run(s), $$(ls $(DATA)/results/*.csv 2>/dev/null | wc -l | tr -d ' ') CSVs (oldest to newest; later runs win filename collisions)."

pipeline: results deploy pdf
	@echo "Pipeline complete."

results: fetch-results collect hardware tables plots  ## fetch (RUNS-selectable) + generate + preview PDF
	@mkdir -p $(DATA)/preview/results && \
		ln -sfn ../../build/tables $(DATA)/preview/results/tables && \
		ln -sfn ../../build/figures $(DATA)/preview/results/figures && \
		ln -sfn ../build $(DATA)/preview/build
	cd $(DATA)/preview && latexmk -pdf -interaction=nonstopmode ../standalone_results.tex >/dev/null 2>&1 || true
	@[ -f $(DATA)/preview/standalone_results.pdf ] && echo "Preview: $(DATA)/preview/standalone_results.pdf" || echo "(preview PDF failed; build/ artifacts still generated)"

deploy:                             ## copy build/ tables+figures into the paper's tracked inputs (results/)
	mkdir -p $(REPO)/results/tables $(REPO)/results/figures
	cp $(DATA)/build/tables/*.tex $(REPO)/results/tables/ 2>/dev/null || true
	cp $(DATA)/build/figures/*.tex $(REPO)/results/figures/ 2>/dev/null || true
	@echo "Deployed build/ -> results/ (tracked paper inputs)."

collect:
	cd $(DATA) && $(JULIA) --project=. collect.jl

hardware:
	cd $(DATA) && $(JULIA) --project=. hardware_table.jl

tables: collect hardware
	cd $(DATA) && $(JULIA) --project=. table.jl && $(JULIA) --project=. gpu_table.jl

plots: collect
	cd $(DATA) && $(JULIA) --project=. plot_opf_pgf.jl

pdf:
	cd $(REPO) && latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex

# --- Help --------------------------------------------------------------------

.PHONY: help
help:
	@echo "ExaModels Paper Benchmark Pipeline"
	@echo "==================================="
	@echo ""
	@echo "On a benchmark machine (clone -> make -> commit -> push):"
	@echo "  make setup                 instantiate Julia environment (once)"
	@echo "  make nvidia|amd|intel|apple|cpu|cpu-mt|reference"
	@echo "                             full suite (LV + COPS + OPF) on that backend"
	@echo "  make <vendor>-lv|-cops|-opf   one suite only, e.g. make amd-opf"
	@echo "  (GPU targets auto-shard across all devices; reference auto-shards across NPROC cores)"
	@echo "  make compare-setup|compare-ad|compare-ad-gpu|compare-cold   section 8.4 AD comparison"
	@echo "  make save                  archive results to the results branch (runs/<uuid>)"
	@echo ""
	@echo "  make cpu-scaling           cpu-mt sweep over MT_THREADS_LIST (default: 1 2 4 8 16)"
	@echo "  GPU targets auto-shard across ALL detected devices by default;"
	@echo "  DEVICE=i pins one GPU, NGPUS=k forces the count, <vendor>-single = plain body"
	@echo ""
	@echo "Options: JULIA=/path/julia  SECONDS=2.0  MT_THREADS=8  MT_THREADS_LIST='1 8 16'  QUICK=1  DEVICE=0"
	@echo ""
	@echo "Post-processing (paper machine):"
	@echo "  make fetch-results         pull archived runs (results branch) into data/results"
	@echo "  make results [RUNS='uuid1 uuid2']   fetch selected runs + tables/figures + preview"
	@echo "  make deploy                copy generated artifacts into the paper inputs"
	@echo "  make pipeline              fetch nothing; results -> deploy -> pdf"
	@echo "  make collect|hardware|tables|plots|pdf"
	@echo ""
	@echo "Backends -> precision: CUDA/AMDGPU/oneAPI/CPU fp64; Metal fp32."
