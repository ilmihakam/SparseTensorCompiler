## Ubuntu `arm64` VM Setup on Apple M4 for Benchmark Hardware Counters

This path uses a standard Ubuntu guest in UTM. It does **not** use Asahi Linux.

The goal is to answer one question first: can the VM expose usable `perf` hardware
events on this M4 machine? If the answer is no, stop and report the failure rather
than treating the counters as rigorous.

### 1. Create the VM

Use UTM with Apple virtualization:

```text
UTM -> Create a New Virtual Machine -> Virtualize -> Linux
- ISO: Ubuntu Server 24.04 arm64
- Architecture: arm64
- Memory: 8 GB to 12 GB
- CPU cores: 4
- Disk: 30 GB+
- Network: Shared Network (NAT)
- Enable SSH during install
```

Do not use `x86_64` guests or Rosetta for this workflow.

### 2. Install guest tools

Inside the Ubuntu guest:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  git \
  python3 \
  linux-tools-common \
  linux-tools-generic \
  util-linux
```

If `perf` reports a kernel-version mismatch, install the matching package:

```bash
sudo apt install -y "linux-tools-$(uname -r)"
```

### 3. Hard feasibility gate

Run these checks before integrating the VM into the benchmark workflow:

```bash
uname -m
perf --version
perf list | head -n 40
printf 'int main(void) { return 0; }\n' | cc -x c -O2 - -o /tmp/hwc_smoke
taskset -c 0 perf stat -e cycles,instructions /tmp/hwc_smoke
```

Success means all of the following are true:

- `uname -m` reports `aarch64`
- `perf stat` succeeds for `cycles` and `instructions`
- `perf list` shows usable hardware events for the guest
- repeated pinned runs produce plausible counts

Do not assume Intel event names such as `L2_rqsts.*` or `llc-load-misses` on this setup.
On `arm64`, the usable event names must come from the guest's `perf list`.

### 4. Failure conditions

Treat the VM as **not feasible for rigorous counters** if any of these happen:

- `perf stat` reports hardware events as unsupported
- only software events are available
- counts are obviously unstable across identical pinned runs
- the available event set cannot be mapped cleanly into benchmark CSV fields

### 5. Failure report template

If the gate fails, record:

```text
Host:
- Apple M4
- macOS version
- UTM version

Guest:
- Ubuntu version
- kernel version
- perf version

Evidence:
- output of `perf list`
- exact failing `perf stat` command
- exact stderr/stdout from the failure

Conclusion:
- VM PMU access is not sufficient for rigorous benchmarking on this machine.
```

### 6. Running the benchmark pipeline after the gate passes

Once the gate passes, use the benchmark runners with explicit event lists:

```bash
cd benchmarks/scripts/run
python3 benchmark_spmv.py \
  --max-matrices 1 \
  --formats csr \
  --configs baseline \
  --trials 3 \
  --hwc-mode perf \
  --hwc-events cycles,instructions \
  --no-taco
```

If you want the run to fail whenever one requested event is unavailable:

```bash
python3 benchmark_spmv.py \
  --max-matrices 1 \
  --formats csr \
  --configs baseline \
  --trials 3 \
  --hwc-mode perf \
  --hwc-events cycles,instructions \
  --hwc-strict \
  --no-taco
```

The CSV output will include:

- `hwc_status`
- `hwc_tool`
- `hwc_events_requested`
- `hwc_events_recorded`
- `hwc_event_values_json`
- normalized columns such as `hwc_cycles` and `hwc_instructions` when present
