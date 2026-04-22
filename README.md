# GEqDiff
Graph Equivariant Diffusion

# LEGO Sampling Sweep

- Dataset: `/scratch/angiod/GEqDiff/lego/dataset_shell_small_test.npz`
- Device: `cuda:0`
- Indices: `0 1 2 3 4 5 6 7`

| Rank | Model | Config | Valid-like | Mean validity | Mean overlap | Mean components | Mean shift | Mean energy delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `production__best_model` | `heun100_clash_cohesion_refine` | 0/8 | 0.328 | 0.773 | 1.875 | 2.480 | 1.901 |
| 2 | `production__last_model` | `heun100_clash_refine` | 0/8 | 0.237 | 0.731 | 2.750 | 2.476 | 1.575 |
| 3 | `production__last_model` | `heun100_clash` | 0/8 | 0.235 | 0.731 | 2.750 | 2.476 | 1.586 |
| 4 | `production__last_model` | `heun100_clash_cohesion_refine` | 0/8 | 0.221 | 0.739 | 2.625 | 2.472 | 1.571 |
| 5 | `production__last_model` | `heun100_clash_cohesion` | 0/8 | 0.220 | 0.738 | 2.625 | 2.472 | 1.582 |
| 6 | `production__last_model` | `heun100_noguide` | 0/8 | 0.211 | 0.739 | 2.750 | 2.476 | 1.588 |
| 7 | `production__best_model` | `heun100_clash_refine` | 0/8 | 0.189 | 0.767 | 1.875 | 2.483 | 2.425 |
| 8 | `production__best_model` | `heun100_clash` | 0/8 | 0.171 | 0.768 | 1.875 | 2.483 | 1.903 |
| 9 | `production__best_model` | `heun100_clash_cohesion` | 0/8 | 0.168 | 0.774 | 1.875 | 2.480 | 1.901 |
| 10 | `production__best_model` | `heun100_noguide` | 0/8 | 0.166 | 0.771 | 1.875 | 2.482 | 2.426 |
| 11 | `production_noclash__last_model` | `heun100_clash_cohesion_refine` | 0/8 | 0.069 | 0.967 | 2.750 | 2.401 | 1.486 |
| 12 | `production_noclash__last_model` | `heun100_clash_cohesion` | 0/8 | 0.069 | 0.968 | 2.750 | 2.401 | 1.490 |
| 13 | `production_noclash__last_model` | `heun100_clash_refine` | 0/8 | 0.062 | 0.965 | 2.750 | 2.401 | 1.487 |
| 14 | `production_noclash__last_model` | `heun100_clash` | 0/8 | 0.061 | 0.966 | 2.750 | 2.401 | 1.491 |
| 15 | `production_noclash__last_model` | `heun100_noguide` | 0/8 | 0.060 | 0.969 | 2.750 | 2.401 | 1.493 |
| 16 | `production_noclash__best_model` | `heun100_clash_refine` | 0/8 | 0.026 | 0.878 | 3.000 | 2.482 | 2.151 |
| 17 | `production_noclash__best_model` | `heun100_clash_cohesion_refine` | 0/8 | 0.026 | 0.894 | 2.875 | 2.480 | 2.151 |
| 18 | `production_noclash__best_model` | `heun100_clash` | 0/8 | 0.026 | 0.893 | 2.875 | 2.482 | 2.152 |
| 19 | `production_noclash__best_model` | `heun100_clash_cohesion` | 0/8 | 0.025 | 0.910 | 2.750 | 2.480 | 2.152 |
| 20 | `production_noclash__best_model` | `heun100_noguide` | 0/8 | 0.024 | 0.901 | 3.000 | 2.481 | 2.419 |

# Comments
Best config found (validity-maximizing in this sweep):

sampler=heun
late_refine_from_step=3, late_refine_factor=2
clash_guidance=true
clash_guidance_strength=1.5
clash_guidance_max_norm=2.0
clash_guidance_weight_schedule=late_linear
clash_guidance_auto_scale=false
output: s4_heun_refine_gx09.npz
metrics: s4_heun_refine_gx09_metrics.json
Improvement vs best no-guidance baseline (s1_heun_refine_noguide):

mean validity: 28.33 -> 29.31
mean effective overlap volume: 0.1110 -> 0.0832
mean clashing brick pairs: 2.3 -> 1.9
mean micro-overlapping pairs: 1.1 -> 0.8
valid-like geometries remained 0/10 in all tested configs.