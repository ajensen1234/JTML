# Cut-0 measurement: GPU-active versus CPU host time

- Device: CUDA device 0
- Fixture: `example_studies/Kneel_1/1024/2806.tif`
- Cost: `DIRECT_DILATION`; backface OFF; Canny `3/0/150`; dilation `6`
- Warmups: 3
- Measured evaluations: 20

| Metric | Median (µs) | p95 (µs) |
|---|---:|---:|
| CPU host wall time | 116.964 | 225.788 |
| GPU CUDA-event elapsed time | 117.4079999 | 226.879999 |

- GPU/CPU median ratio: 1.003796039
- `gpu_active_per_eval_ge_cpu_host_per_eval`: true
- `gpu_active_over_1ms`: false
- `u12_band_reachable_by_premise`: true

Interpretation: U12 band-reachability premise is satisfied.
