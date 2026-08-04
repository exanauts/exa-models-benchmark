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

COMPARE_N       ?= 20,200,2000,20000,200000
# elec sizes are electron COUNTS, so 3*np variables and a DENSE Hessian of
# 3np(3np+1)/2 entries -- 180300 at np=200. The suite uses 5/50/200; 5 is too
# small to time meaningfully here, so the ladder starts where the structure
# starts to cost something.
COMPARE_ELEC_N  ?= 50,100,200
COMPARE_COLD_N  ?= 20,200,2000,20000,200000
COMPARE_VENV    := $(DATA)/compare-venv
COMPARE_PY      := $(COMPARE_VENV)/bin/python
TASKSET         := $(shell command -v taskset >/dev/null 2>&1 && echo "taskset -c 0" || echo "")
COMPARE_ENV     := OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false

.PHONY: pglib compare-setup compare-ad compare-ad-elec compare-ad-opf compare-ad-opf-gpu compare-ad-gpu compare-cold

compare-setup:                      ## python venv for jax/torch/casadi (cpu wheels); freezes requirements.lock
	@# `python3` is 3.6 on Rocky 8; jax/torch need >= 3.9 and the old resolver
	@# silently installs a jax whose jaxlib wheel does not exist, so the leg dies
	@# at `import jaxlib` after the venv looks like it built fine.
	@#
	@# The same reasoning applies to requirements.in: adding jax[cuda12] does
	@# nothing if an existing venv is reused, so the CPU-only jaxlib stays and
	@# jax keeps silently falling back to CPU while the CSV says device=cuda.
	@# The venv records a checksum of requirements.in and is rebuilt when it
	@# changes.
	@#
	@# A venv left by an earlier, wrong interpreter must be rebuilt, not reused.
	@# `python3 -m venv` on an existing directory leaves it alone, so after the
	@# 3.9-or-newer selection was added the 3.6 venv from the failed run was
	@# still sitting there and would have reproduced the identical ModuleNotFound
	@# error with the fix correctly in place -- a fix that looks like it did not
	@# work. Compare the interpreter version and rebuild on mismatch.
	@py=$${PYTHON:-}; \
	 if [ -z "$$py" ]; then \
	   for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do \
	     command -v $$c >/dev/null 2>&1 || continue; \
	     v=$$($$c -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])'); \
	     [ "$$v" -ge 309 ] && { py=$$c; break; }; \
	   done; \
	 fi; \
	 [ -n "$$py" ] || { echo "compare-setup: no python >= 3.9 found; set PYTHON=/path/to/python3" >&2; exit 1; }; \
	 echo "compare-setup: using $$py ($$($$py --version 2>&1))"; \
	 want=$$($$py -c 'import sys;print("%d.%d"%sys.version_info[:2])'); \
	 req=$$(cksum compare/requirements.in | awk '{print $$1}'); \
	 if [ -x $(COMPARE_PY) ]; then \
	   have=$$($(COMPARE_PY) -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none); \
	   hadreq=$$(cat $(COMPARE_VENV)/.requirements.cksum 2>/dev/null || echo none); \
	   if [ "$$have" != "$$want" ]; then \
	     echo "compare-setup: existing venv is python $$have, want $$want -- rebuilding"; \
	     rm -rf $(COMPARE_VENV); \
	   elif [ "$$hadreq" != "$$req" ]; then \
	     echo "compare-setup: requirements.in changed since this venv was built -- rebuilding"; \
	     rm -rf $(COMPARE_VENV); \
	   fi; \
	 fi; \
	 [ -x $(COMPARE_PY) ] || $$py -m venv $(COMPARE_VENV)
	$(COMPARE_PY) -m pip install -q --upgrade pip
	$(COMPARE_PY) -m pip install -q -r compare/requirements.in
	@cksum compare/requirements.in | awk '{print $$1}' > $(COMPARE_VENV)/.requirements.cksum
	$(COMPARE_PY) -m pip freeze > compare/requirements.lock
	@echo "Python env ready; pins written to compare/requirements.lock"

compare-ad:                         ## fair single-core CPU comparison, swept over problem size
	@# Swept over COMPARE_N rather than run at one size. A single point cannot
	@# show where a framework's fixed overhead stops dominating, and section 8.4
	@# was quoting speedups measured only at N=200000 -- exactly the regime that
	@# flatters whichever backend amortises setup best. Same ladder as
	@# compare-cold, so the two halves of the section describe the same sizes.
	@#
	@# ADNLPModels is dropped from the comparison: at N=200000 it measured
	@# 45623 ms per gradient and 376238 ms per Hessian, roughly 80000x and
	@# 350000x ExaModels. A ratio that large says the two are not doing
	@# comparable work rather than that one is faster, and it dominated the
	@# leg's wall time. Its runner has been removed from compare_ad.jl.
	@#
	@# CasADi SX is dropped too: at N=200000 it builds a scalar expression graph
	@# of millions of nodes, which dominated wall time and memory (17 GB RSS)
	@# without saying anything MX does not. --mode mx and --mode map remain.
	@# Record the node once, so these rows can carry a platform label. Once per
	@# leg, not per invocation: record_hw includes benchmark.jl, which pulls in
	@# JuMP and PowerModels.
	cd $(DATA) && $(JULIA) --project=$(BENCH) $(BENCH)/compare/record_hw.jl CPU
	@set -e; for n in $$(echo "$(COMPARE_N)" | tr ',' ' '); do \
	  echo "=== compare-ad N=$$n ==="; \
	  (cd $(DATA) && $(TASKSET) $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl examodels cpu $$n $(SECONDS)); \
	  (cd $(DATA) && $(JULIA) --project=$(BENCH) -t$(MT_THREADS) $(BENCH)/compare/compare_ad.jl examodels cpu-mt $$n $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework jax --device cpu --n $$n --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework torch --device cpu --n $$n --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework casadi --mode mx --device cpu --n $$n --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework casadi --mode map --device cpu --n $$n --seconds $(SECONDS)); \
	done

compare-ad-elec:                    ## COPS elec sweep -- NOT part of the reported study
	@# Kept working, deliberately not run. A fair comparison on elec turned out
	@# to be very hard: the objective is a sum over pairs, so each framework's
	@# best implementation is a different algorithm rather than a different
	@# idiom, and the measured gap moved by 25x on a formulation change of mine
	@# alone. That makes the number a statement about the harness. The reported
	@# comparison is Luksan-Vlcek and AC-OPF.
	@# Opposite structure to Luksan-Vlcek: COPS elec has a DENSE Hessian, so
	@# colouring is worthless and every framework must form the whole lower
	@# triangle. ExaModels runs FIRST at each size because it exports the start
	@# point the Python frameworks read -- elec's start comes from Julia's RNG
	@# and numpy cannot reproduce it. CasADi map mode is skipped: elec's
	@# objective is over pairs, which is not what an element template tests.
	@set -e; for n in $$(echo "$(COMPARE_ELEC_N)" | tr ',' ' '); do \
	  echo "=== compare-ad elec np=$$n ==="; \
	  (cd $(DATA) && COMPARE_PROBLEM=elec $(TASKSET) $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl examodels cpu $$n $(SECONDS)); \
	  (cd $(DATA) && COMPARE_PROBLEM=elec $(JULIA) --project=$(BENCH) -t$(MT_THREADS) $(BENCH)/compare/compare_ad.jl examodels cpu-mt $$n $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem elec --framework jax --device cpu --n $$n --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem elec --framework torch --device cpu --n $$n --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem elec --framework casadi --mode mx --device cpu --n $$n --seconds $(SECONDS)); \
	done

# AC-OPF polar comparison. The cases span three decades of variable count; the
# exporter runs first for each because it writes the network AND ExaModels'
# reference values, which is what every Python framework is checked against.
# Four decades of variable count, ending at the largest case in the library.
# case78484 is ~8.5x case9241 in buses, so its exported payload -- which carries
# ExaModels' reference jac and hess values, the thing every framework is checked
# against -- goes from 39 MB to a few hundred. That is affordable on the legs'
# current memory (256G on GPU, whole-node on CPU) and was NOT affordable on the
# 8 GB default that OOM-killed the first GPU attempt.
COMPARE_OPF_CASES ?= pglib_opf_case118_ieee pglib_opf_case1354_pegase pglib_opf_case9241_pegase pglib_opf_case78484_epigrids
PGLIB_DIR         ?= $(DATA)/pglib-opf

pglib:                              ## fetch the pglib-opf cases if absent
	@# The suite legs clone this lazily from inside benchmark.jl, so a leg that
	@# runs only the comparison never triggers it and the OPF targets find no
	@# case files. Same pin as benchmark.jl: v23.07.
	@[ -d $(PGLIB_DIR) ] || git clone -q --depth 1 --branch v23.07 \
	    https://github.com/power-grid-lib/pglib-opf.git $(PGLIB_DIR)
	@ls $(PGLIB_DIR)/pglib_opf_case118_ieee.m >/dev/null

compare-ad-opf: pglib               ## AC-OPF polar comparison, CPU
	@# Record the node, or these rows carry no platform label. compare-ad and
	@# compare-ad-gpu already do this; the OPF targets were written later and
	@# did not, so the first OPF bundles show "--" for hardware.
	cd $(DATA) && $(JULIA) --project=$(BENCH) $(BENCH)/compare/record_hw.jl CPU
	@set -e; for c in $(COMPARE_OPF_CASES); do \
	  echo "=== compare-ad-opf $$c ==="; \
	  m=$(PGLIB_DIR)/$$c.m; \
	  j=$(DATA)/results/opf_$$c.json; \
	  (cd $(DATA) && $(JULIA) --project=$(BENCH) $(BENCH)/compare/export_opf.jl $$m results/opf_$$c.json); \
	  (cd $(DATA) && COMPARE_PROBLEM=opf COMPARE_OPF_CASE=$$m $(TASKSET) $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl examodels cpu 0 $(SECONDS)); \
	  (cd $(DATA) && COMPARE_PROBLEM=opf COMPARE_OPF_CASE=$$m $(JULIA) --project=$(BENCH) -t$(MT_THREADS) $(BENCH)/compare/compare_ad.jl examodels cpu-mt 0 $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem opf --opf-json $$j --framework jax --device cpu --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem opf --opf-json $$j --framework torch --device cpu --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem opf --opf-json $$j --framework casadi --mode mx --device cpu --seconds $(SECONDS)); \
	done

compare-ad-opf-gpu: pglib           ## AC-OPF polar comparison, CUDA
	@# CasADi has no GPU AD, so it is absent here by capability rather than by choice.
	cd $(DATA) && $(JULIA) --project=$(BENCH) $(BENCH)/compare/record_hw.jl CUDA
	@set -e; for c in $(COMPARE_OPF_CASES); do \
	  echo "=== compare-ad-opf-gpu $$c ==="; \
	  m=$(PGLIB_DIR)/$$c.m; \
	  j=$(DATA)/results/opf_$$c.json; \
	  (cd $(DATA) && $(JULIA) --project=$(BENCH) $(BENCH)/compare/export_opf.jl $$m results/opf_$$c.json); \
	  (cd $(DATA) && COMPARE_PROBLEM=opf COMPARE_OPF_CASE=$$m $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl examodels cuda 0 $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem opf --opf-json $$j --framework jax --device cuda --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --problem opf --opf-json $$j --framework torch --device cuda --seconds $(SECONDS)); \
	done

compare-ad-gpu:                     ## GPU comparison (CUDA), swept over problem size
	cd $(DATA) && $(JULIA) --project=$(BENCH) $(BENCH)/compare/record_hw.jl CUDA
	@set -e; for n in $$(echo "$(COMPARE_N)" | tr ',' ' '); do \
	  echo "=== compare-ad-gpu N=$$n ==="; \
	  (cd $(DATA) && $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_ad.jl examodels cuda $$n $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework jax --device cuda --n $$n --seconds $(SECONDS)); \
	  (cd $(DATA) && $(COMPARE_PY) $(BENCH)/compare/compare_ad.py --framework torch --device cuda --n $$n --seconds $(SECONDS)); \
	done

compare-cold:                       ## cold-start compile timing: fresh process per framework and size
	cd $(DATA) && COMPARE_N=$(COMPARE_COLD_N) $(TASKSET) $(JULIA) --project=$(BENCH) -t1 $(BENCH)/compare/compare_cold.jl
	cd $(DATA) && COMPARE_N=$(COMPARE_COLD_N) $(COMPARE_ENV) $(TASKSET) $(COMPARE_PY) $(BENCH)/compare/compare_cold.py

# --- Save results ------------------------------------------------------------

.PHONY: save save-partial
save-partial:                       ## archive an unfinished run's partial CSVs (marked -partial)
	SAVE_PARTIAL=1 $(BENCH)/save_results.sh

save:                               ## archive results to the 'results' branch (UUID run dir)
	$(BENCH)/save_results.sh

# --- Post-processing (on the paper machine) ----------------------------------

.PHONY: fetch-results pipeline results deploy review collect hardware tables plots pdf

RUNS ?=

fetch-results:                      ## pull archived runs into data/results/ (RUNS='id1 id2' selects by uuid/substring; empty = all)
	git -C $(REPO) fetch origin results
	@tmp=$$(mktemp -d); \
	git -C $(REPO) worktree add --detach $$tmp origin/results >/dev/null 2>&1; \
	rm -f $(DATA)/results/*.csv $(DATA)/results/*_hw.toml; rm -rf $(DATA)/results/logs; \
	rm -f $(SOLVE)/results/*.csv; \
	sel="$(RUNS)"; n=0; \
	for d in $$tmp/runs/*/; do \
		name=$$(basename $$d); \
		if [ -n "$$sel" ]; then keep=0; for s in $$sel; do case "$$name" in *"$$s"*) keep=1;; esac; done; [ $$keep -eq 1 ] || continue; fi; \
		if [ -d "$$d/solve-results" ]; then \
			mkdir -p $(SOLVE)/results; \
			cp $$d/solve-results/*.csv $(SOLVE)/results/ 2>/dev/null || true; \
		fi; \
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

review: tables plots                ## one PDF with EVERY generated table and figure, for eyeballing before deploy
	@# Depends on BOTH generators. It used to depend on neither, so `make tables
	@# review` rebuilt the tables and rendered them beside whatever figures were
	@# last built -- the preview looked coherent and was not. The Apple bundle
	@# reached the tables and the hardware list while the figures still showed
	@# ten series without it.
	@# Globbed, not hand-listed: a renamed or new artifact must appear without
	@# editing preview.tex. Hand-listing is how the old preview silently lost its
	@# OPF figure. gpu_summary and *_all_callbacks are skipped: they are wrappers
	@# over results/, i.e. deployed paths that do not exist before deploy.
	@# Explicit reading order, then anything new appended so a fresh artifact
	@# still appears without editing this list.
	@{ printf '%s\n' '\section{Tables}'; \
	   emit() { n=$$1; [ -f "$(DATA)/build/tables/$$n.tex" ] || return 0; \
	     case " $$SEEN " in *" $$n "*) return 0;; esac; SEEN="$$SEEN $$n"; \
	     e=$$(printf '%s' "$$n" | sed 's/_/\\_/g'); \
	     printf '\\artifacttable{%s}{%s}\n' "$$n" "$$e"; }; \
	   SEEN=""; \
	   for n in hardware compare_ad compare_ad_opf gpu_summary_lv gpu_summary_cops \
	            gpu_summary_opf_polar gpu_summary_opf_rect \
	            results_LV results_COPS results_OPF-polar results_OPF-rect; do emit $$n; done; \
	   for f in $(DATA)/build/tables/*.tex; do n=$$(basename $$f .tex); \
	     case "$$n" in gpu_summary|sgm_summary) continue;; esac; emit $$n; done; \
	   printf '%s\n' '\section{Figures}'; \
	   femit() { n=$$1; [ -f "$(DATA)/build/figures/$$n.tex" ] || return 0; \
	     case " $$FSEEN " in *" $$n "*) return 0;; esac; FSEEN="$$FSEEN $$n"; \
	     e=$$(printf '%s' "$$n" | sed 's/_/\\_/g'); \
	     printf '\\artifactfigure{%s}{%s}\n' "$$n" "$$e"; }; \
	   FSEEN=""; \
	   for s in LV COPS OPF OPF_rect; do femit "$${s}_all_callbacks"; done; \
	   for f in $(DATA)/build/figures/*_all_callbacks.tex; do femit $$(basename $$f .tex); done; \
	 } > $(DATA)/build/_artifacts.tex
	cd $(DATA) && latexmk -pdf -synctex=1 -interaction=nonstopmode preview.tex >/dev/null 2>&1 || true
	@[ -f $(DATA)/preview.pdf ] && echo "Review PDF: $(DATA)/preview.pdf" || { echo "build failed; see $(DATA)/preview.log" >&2; exit 1; }

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
	cd $(DATA) && $(JULIA) --project=. table.jl && $(JULIA) --project=. gpu_table.jl && $(JULIA) --project=. compare_table.jl

plots: collect
	cd $(DATA) && $(JULIA) --project=. plot_opf_pgf.jl

pdf:
	cd $(REPO) && latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex

# --- Solve-time breakdown (GPU-resident callbacks) ---------------------------
#
# A separate Julia project (benchmark/solve) because this leg needs MadNLP,
# MadNLPGPU and CUDSS on top of the modelling stack, and the callback benchmark
# should not have to resolve a solver stack it never uses.
#
# Its CSVs land in benchmark/solve/results/, NOT data/results/, so save_results.sh
# (which bundles data/results) cannot sweep them into a campaign archive.

SOLVE := $(BENCH)/solve

.PHONY: solve-setup solve-breakdown solve-tables

solve-setup:                        ## instantiate the solve-breakdown env (run with a GPU VISIBLE)
	@PATH="$$HOME/.juliaup/bin:$$PATH"; export PATH; \
	$(JULIA) --project=$(SOLVE) -e 'using Pkg; Pkg.instantiate(); Pkg.status()'

# -t1: the CPU configurations are the single-threaded baseline of Section 8.2.
solve-breakdown:                    ## 4-config LiftedKKT breakdown (SUITE=all|rosenbrock|opf, REPS=3)
	mkdir -p $(SOLVE)/results/logs && cd $(SOLVE) && set -o pipefail && \
	$(JULIA) --project=$(SOLVE) -t1 $(SOLVE)/solve_breakdown.jl $(or $(SUITE),all) $(or $(REPS),3) \
		2>&1 | tee $(SOLVE)/results/logs/$$(hostname -s)_breakdown_$$(date -u +%Y%m%dT%H%M%SZ).log

solve-tables:                       ## regenerate the breakdown table + figure from solve/results/*.csv
	cd $(DATA) && $(JULIA) --project=. breakdown_out.jl

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
	@echo "Solve-time breakdown (Section 8.5, needs one GPU):"
	@echo "  make solve-setup           instantiate benchmark/solve (with a GPU visible)"
	@echo "  make solve-breakdown       run the 4 configurations (SUITE=, REPS=)"
	@echo "  make solve-tables          regenerate its table + figure"
	@echo ""
	@echo "Backends -> precision: CUDA/AMDGPU/oneAPI/CPU fp64; Metal fp32."
