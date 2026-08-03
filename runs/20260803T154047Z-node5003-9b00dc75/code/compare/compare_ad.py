"""Section 8.4 AD-framework comparison — Python side (JAX / PyTorch / CasADi).

Usage: python compare_ad.py --framework {jax,torch,casadi} [--device {cpu,cuda}]
                            [--n N] [--seconds S] [--out-dir results]

Fair protocol (see the paper's benchmark protocol):
- CPU runs are intended to be launched under `taskset -c <core>` with all
  thread pools capped at 1 (the make target does this); the measured
  process-CPU/wall ratio is recorded so single-core discipline is auditable.
- Hessian timing includes extraction of solver-ready sparse values
  (lower-triangle COO) from the color-based HVPs, using precomputed index
  maps (the analogue of model-build-time structure analysis).
- Reported statistic: minimum over repetitions after warmup.
- Correctness of the Hessian values is asserted against analytic entries.
"""
import argparse, csv, os, socket, time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--framework", required=True, choices=["jax", "torch", "casadi"])
p.add_argument("--mode", default="mx", choices=["sx", "mx", "map"],
               help="casadi expression mode (sx: scalar; mx: matrix; map: SX element template)")
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
p.add_argument("--n", type=int, default=200_000)
p.add_argument("--seconds", type=float, default=2.0)
p.add_argument("--out-dir", default="results")
args = p.parse_args()
N = args.n


def bench(fn, seconds, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter(); fn(); dt = max(time.perf_counter() - t0, 1e-9)
    reps = max(3, min(10_000, int(seconds / dt)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def cpu_ratio(fn, reps=30):
    t0w, t0c = time.perf_counter(), time.process_time()
    for _ in range(reps):
        fn()
    return (time.process_time() - t0c) / (time.perf_counter() - t0w)


def x_init():
    return np.array([1.0 if i % 2 == 1 else -1.2 for i in range(N)])


def check_hess(vals, x):
    assert abs(vals[0] - (1200 * x[0] ** 2 - 400 * x[1] + 2)) < 1e-6
    assert abs(vals[N] - (-400 * x[0])) < 1e-6


def run_jax():
    import jax, jax.numpy as jnp
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    x0 = jnp.array(x_init())

    def rosenbrock(x):
        return jnp.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    jit_grad = jax.jit(jax.grad(rosenbrock))
    idx = np.arange(N)
    color = idx % 3
    color_d, rows_d = jnp.array(color), jnp.array(idx)
    color_s, rows_s = jnp.array(color[:-1]), jnp.array(idx[:-1] + 1)

    def hess_vals(x):
        hvps = []
        for c in range(3):
            v = jnp.zeros(N).at[c::3].set(1.0)
            _, hvp = jax.jvp(jax.grad(rosenbrock), (x,), (v,))
            hvps.append(hvp)
        H = jnp.stack(hvps)
        return jnp.concatenate([H[color_d, rows_d], H[color_s, rows_s]])

    jit_hess = jax.jit(hess_vals)
    jit_grad(x0).block_until_ready(); jit_hess(x0).block_until_ready()
    tg = bench(lambda: jit_grad(x0).block_until_ready(), args.seconds)
    th = bench(lambda: jit_hess(x0).block_until_ready(), args.seconds)
    r = cpu_ratio(lambda: jit_hess(x0).block_until_ready())
    check_hess(np.array(jit_hess(x0)), np.array(x0))
    return tg, th, r


def run_torch():
    import torch
    torch.set_num_threads(1)
    dev = torch.device("cuda" if args.device == "cuda" else "cpu")
    x0 = torch.tensor(x_init(), dtype=torch.float64, device=dev)

    def rosenbrock(x):
        return torch.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    color = torch.tensor(np.arange(N) % 3, device=dev)
    rows_d = torch.tensor(np.arange(N), device=dev)
    color_s, rows_s = color[:-1], torch.tensor(np.arange(N - 1) + 1, device=dev)

    def eval_grad():
        x = x0.clone().requires_grad_(True)
        g, = torch.autograd.grad(rosenbrock(x), x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return g

    def eval_hess():
        x = x0.clone().requires_grad_(True)
        g, = torch.autograd.grad(rosenbrock(x), x, create_graph=True)
        hvps = []
        for c in range(3):
            v = torch.zeros(N, dtype=torch.float64, device=dev)
            v[c::3] = 1.0
            hvp, = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=True)
            hvps.append(hvp)
        H = torch.stack(hvps)
        out = torch.cat([H[color, rows_d], H[color_s, rows_s]])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return out

    tg = bench(eval_grad, args.seconds)
    th = bench(eval_hess, args.seconds)
    r = cpu_ratio(eval_hess)
    check_hess(eval_hess().cpu().numpy(), x0.cpu().numpy())
    return tg, th, r


def run_casadi():
    if args.device != "cpu":
        raise SystemExit("casadi: cpu only")
    import casadi as ca
    if args.mode == "sx":
        x = ca.SX.sym("x", N)
        f = ca.sum1(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
    elif args.mode == "mx":
        x = ca.MX.sym("x", N)
        f = ca.sum1(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
    else:  # map: SX element template applied via Function.map (closest to ExaModels)
        xi, xip = ca.SX.sym("xi"), ca.SX.sym("xip")
        fi = ca.Function("fi", [xi, xip], [100 * (xi ** 2 - xip) ** 2 + (xi - 1) ** 2])
        x = ca.MX.sym("x", N)
        f = ca.sum2(fi.map(N - 1)(x[:-1].T, x[1:].T))
    grad_f = ca.gradient(f, x)
    hess_f, _ = ca.hessian(f, x)
    opts = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O3"]}, "jit_cleanup": True}
    fn_grad = ca.Function("grad_%s_jit" % args.mode, [x], [grad_f], opts)
    fn_hess = ca.Function("hess_%s_jit" % args.mode, [x], [hess_f], opts)
    x0 = x_init()
    fn_grad(x0); fn_hess(x0)
    tg = bench(lambda: fn_grad(x0), args.seconds)
    th = bench(lambda: fn_hess(x0), args.seconds)
    r = cpu_ratio(lambda: fn_hess(x0))
    H = np.array(fn_hess(x0).full()) if N <= 5000 else None  # dense check only when small
    if H is not None:
        x0n = x0
        assert abs(H[0, 0] - (1200 * x0n[0] ** 2 - 400 * x0n[1] + 2)) < 1e-6
    return tg, th, r


tg, th, r = {"jax": run_jax, "torch": run_torch, "casadi": run_casadi}[args.framework]()
host = socket.gethostname()
os.makedirs(args.out_dir, exist_ok=True)
fwtag = args.framework + ("-" + args.mode if args.framework == "casadi" else "")
out = os.path.join(args.out_dir, f"compare_{host}_{fwtag}_{args.device}.csv")
new = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["framework", "device", "n", "threads", "tgrad_ms", "thess_ms", "cpu_wall_ratio", "hostname"])
    w.writerow([fwtag, args.device, N, 1, tg * 1e3, th * 1e3, round(r, 2), host])
print(f"{args.framework}/{args.device} N={N}  grad {tg*1e3:.3f} ms  hess {th*1e3:.3f} ms  cpu/wall {r:.2f} -> {out}")
