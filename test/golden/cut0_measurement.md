# Cut-0 measurement: GPU-active versus CPU host time

- Device: CUDA device 0
- Fixture: `example_studies/Kneel_1/1024/2806.tif`
- Cost: `DIRECT_DILATION`; backface OFF; Canny `3/0/150`; dilation `6`
- Warmups: 3
- Measured evaluations: 20

| Metric | Median (µs) | p95 (µs) |
|---|---:|---:|
| CPU host wall time | 98.364 | 105.189 |
| GPU CUDA-event elapsed time | 98.62399846 | 105.0240025 |

- GPU/CPU median ratio: 1.002643228
- `gpu_active_per_eval_ge_cpu_host_per_eval`: true
- `gpu_active_over_1ms`: false
- `u12_band_reachable_by_premise`: true

Interpretation: U12 band-reachability premise is satisfied.
