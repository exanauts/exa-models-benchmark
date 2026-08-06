#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-compare-gpu.sbatch
# Partitions: mit_preemptable starts in ~2 min (2-day cap); switch to the
# ou_cheme partitions when they are free.  Queue-time predictions from
# --test-only are unreliable; probe with a short job if in doubt.
#SBATCH --job-name=exa-compare-opf-gpu
#SBATCH --output=exa-compare-opf-gpu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# A BLOCK of identically-configured nodes, the same trick the CPU legs use, and
# for the same reason: `gpu:h100:1` alone constrains the card but not the host,
# and bare `gpu:1` constrains neither, so two runs of this leg can land on
# different silicon and be reduced together as if they had not.
#
# These eight are homogeneous on every axis sinfo exposes -- h100:4, 128 CPUs,
# 2055056 MB -- and all eight are in mit_preemptable. node1702 is one of them,
# which is the point: it is where the saved compare-opf-gpu bundle was measured
# (Xeon Platinum 8462Y+ / H100 80GB HBM3), so a re-run stays comparable with the
# rows already on the results branch instead of starting a second population.
#
# Eight nodes rather than one is also the scheduling win -- the constraint is
# what buys consistency, and the block is what buys queue time.
#
# gres stays TYPED even though -C already implies the model. That is not
# redundancy: `--gres=gpu:1` together with this -C is rejected outright with
# "Requested node configuration is not available", while `--gres=gpu:h100:1`
# with the identical -C schedules fine (verified with sbatch --test-only, which
# placed it on node2803). Slurm appears to resolve an untyped gres against the
# partition before the constraint narrows it. Do not "simplify" this back.
# The ou_cheme_gpu RTX PRO 6000 BLOCK (Sungho, 2026-08-05). Three nodes,
# homogeneous: rtx_pro_6000:8, 128 CPUs, 2055020 MB.
#
# Moved off the H200 pair because that pair is saturated -- node5000 and node5001
# each report AllocTRES gres/gpu:h200=8, every device committed -- and Slurm
# estimated our start at 2026-08-07T16:27, some 43 hours out. The PRO 6000 nodes
# are in the same partition we own and are not contended that way.
#
# This is the campaign's THIRD GPU model, after H100 and H200. All three GPU legs
# move together so the rows stay internally consistent, but they SUPERSEDE the
# archived H100/H200 GPU numbers rather than extending them, exactly as the CPU
# legs supersede the EPYC ones.
#
# gres stays TYPED alongside -C: an untyped --gres=gpu:1 with a node constraint is
# rejected at submit with "Requested node configuration is not available".
#SBATCH -C "node5003|node5004|node5005"
#SBATCH --gres=gpu:rtx_pro_6000:1
# 1 hour, not 4. Sized from measured wall time: compare-gpu 6m57s (19703562),
# compare-opf-gpu 11m42s (19703564), solve-gpu 7m12s (19703566). Slurm backfills
# on REQUESTED time, so a four-hour ask to do seven minutes of work forfeits every
# gap shorter than four hours -- which is how these legs came to sit PENDING while
# nodes in their own -C block were idle.
#SBATCH --time=01:00:00
#SBATCH --partition=ou_cheme_gpu
# --mem is REQUIRED even on a GPU leg. Without it Slurm applies DefMemPerCPU,
# which on ORCD gave this job 8000M for 8 CPUs -- and the OPF comparison was
# OOM-killed at MaxRSS 7.8 GB, exactly at the ceiling, after ExaModels had
# already produced its CUDA numbers. The CPU legs were fixed after the same
# failure months earlier; the GPU legs were never covered, and two of them
# (h100, pro6000) have been running on 8 GB and simply not hitting it.
#
# --mem=0 is "all the memory on this node", the same request the CPU legs make.
# A fixed 128G would be another ceiling to discover the hard way on the next
# larger case, and these legs already hold a GPU on the node.
# 64G, sized from sacct rather than from headroom-by-guess. Measured MaxRSS on
# the three GPU legs that have now run the ladder out to case78484_epigrids:
# compare-gpu 6.4 GB, compare-opf-gpu 12.7 GB (this leg), solve-gpu 7.5 GB.
#
# 256G was 20x the largest of those, and over-requesting is not free: Slurm
# schedules on REQUESTED memory, so the ask itself excluded nodes and is a large
# part of why these legs waited. 64G keeps ~5x over the worst case -- enough for
# an instance several times larger than any in the study -- and fits far more
# nodes.
# Our PROPORTIONAL SHARE of the node, not a flat figure (Sungho, 2026-08-05:
# "we divide it by the existing all cpu by the one we requested, or for the gpu
# node divide it by the number of devices and request accordingly").
#
# A flat number is a guess that has to be revisited every time a case grows -- it
# went 8G, 256G, 64G, 16G, 32G, 64G in one evening. A share is a rule: take the
# fraction of the node's memory matching the fraction of the node we occupy, so
# exactly as many of these legs fit per node as there are slots, and no leg
# either starves a neighbour or is starved by one.
#
#   1 of 4 GPUs on a 2055056 MB node = 513764 MB
#
# Rounded DOWN to a clean boundary so the request can never exceed what the node
# advertises; the block nodes differ by a few MB of kernel overhead and an exact
# figure would be unschedulable on the smallest of them.
# 4x the MEASURED peak RSS of 12.1 GB, from this leg's own run history in sacct
# (`sacct -o JobName,MaxRSS` over every run back to 2026-07-25), rounded up to a
# tidy size, floored at 32G.
#
# Measured rather than chosen, because every figure this file has carried was a
# guess: 8G inherited from DefMemPerCPU (OOM-killed at 9.9 GB), then 256G, 64G,
# 16G, 32G, 64G, then a proportional share of the node. The share rule reads well
# and is wrong here -- memory need does not track core count for these legs.
#
# Note the POPULATION. Sizing off a single run undersamples badly: this leg's own
# runs span a wide range depending on which framework variants compile, so the
# figure is the maximum over runs, not the last one.
#
# Pre-fix runs peaked at 177 GB and are EXCLUDED, same dense _accumulate.
#SBATCH --mem=64G

set -euo pipefail
# Under Slurm this is the submit dir; run directly with bash from anywhere:
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# SLURM_SUBMIT_DIR is wherever sbatch was TYPED, which is not necessarily where
# this script needs to run. Submitting from the repo root -- `sbatch
# benchmark/slurm/<leg>.sbatch` -- lands here at the root, where there is no
# Makefile, and the leg dies instantly with "No rule to make target 'setup'".
# That cost three scheduling slots on 2026-08-05 after an hour of queueing.
#
# The $0 fallback above cannot save it: Slurm copies the script to
# /var/spool/slurmd/job<N>/slurm_script, so dirname $0 is the spool directory.
# So locate the tree by what it CONTAINS, and fail loudly rather than run half a
# leg from the wrong directory.
[ -f Makefile ] || { [ -f benchmark/Makefile ] && cd benchmark; }
[ -f Makefile ] || { echo "cannot find the benchmark Makefile from $PWD (submit from benchmark/, or from the repo root)" >&2; exit 1; }

# Per-leg Julia depot. Legs sharing one depot serialise on Pkg's precompile
# lock, and a leg killed mid-precompile can leave that lock behind: compare-cpu
# then sat for 2h25 on a 32-core allocation having burned ten seconds of CPU,
# with no output and no timeout, because `make setup` was waiting on a lock left
# by a cancelled sibling.
#
# First entry is writable and per-leg, so locks and new precompile caches are
# private. The shared depot follows read-only, so packages already installed
# there are reused and this costs no extra download or compile in the common
# case. JULIA_DEPOT_PATH is a search path: writes go to the first entry only.
export JULIA_DEPOT_PATH="$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}:$HOME/.julia"
mkdir -p "$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}"

# ORCD ships Julia 1.12.0 at ~/julia-1.12.0; juliaup here defaults to 1.11.7,
# while the depot and Manifest are built for 1.12.x per the README pin.
#
# Setting PATH is NOT enough: the Makefile setup recipe does its own
#   PATH="$HOME/.juliaup/bin:$PATH"; export PATH; $(JULIA) ...
# so juliaup wins over anything this script puts on PATH. JULIA ?= julia in
# the Makefile does defer to an exported environment variable, so that is the
# lever. Guarded, so sites without that path keep the juliaup default.
export PATH="$HOME/.juliaup/bin:$PATH"
[ -x "$HOME/julia-1.12.0/bin/julia" ] && export JULIA="$HOME/julia-1.12.0/bin/julia"

# Bound the setup: a wedged depot lock should fail loudly in minutes, not
# consume the whole wall-clock limit in silence.
#
# `timeout` is GNU coreutils and does not exist on macOS, where it is gtimeout
# if installed at all -- so the unconditional form broke the Apple leg outright:
# "timeout: command not found", make setup never ran, exit 1. Degrade to running
# without a bound rather than failing on the site that has no timeout binary.
if command -v timeout >/dev/null 2>&1; then _TIMEOUT="timeout 45m"
elif command -v gtimeout >/dev/null 2>&1; then _TIMEOUT="gtimeout 45m"
else _TIMEOUT=""; echo "note: no timeout(1) available; running make setup unbounded"; fi
$_TIMEOUT make setup || { echo "make setup failed or timed out" >&2; exit 1; }
# Section 8.4 AD comparison, CUDA side (ExaModels, JAX, PyTorch).
# torch.compile shells out to a C compiler and #includes Python.h. ORCD's
# /usr/bin/python3.12 ships no headers, so inductor fails with a
# CalledProcessError and PyTorch silently drops to CUDA graphs only -- on the
# OPF Jacobian that is 2.7x slower (0.118 ms compiled against 0.325 ms graphed)
# and gets reported as PyTorch's cost. miniforge's 3.12 does ship headers, and a
# venv built on it inherits them, so no CPATH is needed: compare-setup picks a
# header-bearing interpreter and this only has to put one on PATH.
# `module` is a shell FUNCTION defined by the login profile, and an sbatch script
# runs a non-interactive shell that never sources it -- so `module load` here is
# "command not found", the || branch fires, and the leg proceeds with
# /usr/bin/python3.12 (3.12.1, no Python.h) instead of miniforge's 3.12.12.
# That is how 19678269 and 19678270 produced 25 and 9 "torch.compile
# unavailable" and a PyTorch row that is eager and unreportable.
#
# Sourcing must NOT run under `set -u`: the init references unset variables and
# the shell exits silently mid-script, which is how this looked like it had
# worked once before.
set +u
for _mi in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
  [ -f "$_mi" ] && . "$_mi" 2>/dev/null
done
set -u
module load gcc/14.3.0 miniforge/25.11.0-0 2>/dev/null || \
  echo "note: module load failed; torch.compile will be unavailable" >&2
make compare-setup
make compare-ad-opf-gpu
# --- what this leg produced, BY CASE NAME (not row counts) ---
# A row count is a weak check: a leg can emit the right number of rows while
# substituting, duplicating or dropping cases and still look correct. This
# names every (suite, problem, size) that cases.jl expects and is absent.
# Must be the SAME julia the run used. Bare `julia` takes the juliaup
# default (1.11.7 on ORCD) and dies loading CSV.jl against a 1.12
# manifest -- which then reports CASE COVERAGE INCOMPLETE for a leg
# whose data is fine, and cost the cpu-mt leg its save.
"${JULIA:-julia}" --project=. check_cases.jl || echo "CASE COVERAGE INCOMPLETE - see MISSING lines above"
# Per-instance failures only warn, so a COMPLETED job that dropped a whole
# suite is otherwise indistinguishable from a good one.
# `|| true`: grep exits 1 when it finds nothing, and these scripts run under
# `set -euo pipefail`, so a leg with ZERO per-instance failures killed itself
# here -- before `make save` -- and reported FAILED. Exactly inverted: a clean
# run failed, a run with failures continued. It cost a100 and compare-gpu
# their bundles and sent me hunting an OOM that never happened.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20 || true

# save_results.sh refuses to bundle when no non-partial CSVs exist, so a leg
# that produced nothing cannot push an empty run directory.

# Which (case, framework) pairs this leg owed and did not produce.
# A framework failure no longer ends the leg, so absence has to be
# reported here or a leg that dropped one still looks successful.
PY_BIN="$PWD/data/compare-venv/bin/python"
"$PY_BIN" compare/check_compare.py data/results \
  --problem opf --frameworks examodels,jax,torch --device cuda \
  || echo "COMPARE COVERAGE INCOMPLETE"

make save
