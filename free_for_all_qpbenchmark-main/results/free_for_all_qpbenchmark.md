# Free-for-all test set

| Number of problems | 1 |
|:-------------------|:--------------------|
| Benchmark version  | 2.2.1 |
| Date               | 2024-07-20 08:44:29.346564+00:00 |
| CPU                | [AMD Ryzen 7 2700X Eight-Core Processor](#cpu-info) |
| Run by             | [@basti](https://github.com/basti/) |

Benchmark reports are copious as we aim to document comparison factors as much as possible. You can also [jump to results](#results-by-settings) directly.

## Contents

* [Description](#description)
* [Solvers](#solvers)
* [Settings](#settings)
* [CPU info](#cpu-info)
* [Known limitations](#known-limitations)
* [Results by settings](#results-by-settings)
    * [Default](#default)
    * [High accuracy](#high-accuracy)
    * [Low accuracy](#low-accuracy)
    * [Mid accuracy](#mid-accuracy)
* [Results by metric](#results-by-metric)
    * [Success rate](#success-rate)
    * [Computation time](#computation-time)
    * [Optimality conditions](#optimality-conditions)
        * [Primal residual](#primal-residual)
        * [Dual residual](#dual-residual)
        * [Duality gap](#duality-gap)

## Description

Community-built test set to benchmark QP solvers.

## Solvers

| solver   | version     |
|:---------|:------------|
| clarabel | 0.9.0       |
| cvxopt   | 0.0.0       |
| daqp     | 0.5.1       |
| ecos     | 2.0.14      |
| highs    | 1.7.2       |
| osqp     | 0.6.7.post0 |
| piqp     | 0.4.1       |
| proxqp   | 0.6.6       |
| qpalm    | 1.2.3       |
| qpoases  | 3.2.1       |
| quadprog | 0.1.12      |
| scs      | 3.2.6       |

All solvers were called via [qpsolvers](https://github.com/qpsolvers/qpsolvers) v4.3.2.

## CPU info

| Property | Value |
|----------|-------|
| `arch` | X86_64 |
| `arch_string_raw` | x86_64 |
| `bits` | 64 |
| `brand_raw` | AMD Ryzen 7 2700X Eight-Core Processor |
| `count` | 16 |
| `cpuinfo_version_string` | 9.0.0 |
| `family` | 23 |
| `flags` | `3dnowprefetch`, `abm`, `adx`, `aes`, `apic`, `arat`, `avx`, `avx2`, `bmi1`, `bmi2`, `clflush`, `clflushopt`, `clzero`, `cmov`, `cmp_legacy`, `constant_tsc`, `cpuid`, `cr8_legacy`, `cx16`, `cx8`, `de`, `extd_apicid`, `f16c`, `fma`, `fpu`, `fsgsbase`, `fxsr`, `fxsr_opt`, `ht`, `hypervisor`, `ibpb`, `lahf_lm`, `lm`, `mca`, `mce`, `misalignsse`, `mmx`, `mmxext`, `movbe`, `msr`, `mtrr`, `nonstop_tsc`, `nopl`, `nx`, `osvw`, `osxsave`, `pae`, `pat`, `pclmulqdq`, `pdpe1gb`, `pge`, `pni`, `popcnt`, `pse`, `pse36`, `rdrand`, `rdrnd`, `rdseed`, `rdtscp`, `rep_good`, `sep`, `sha`, `sha_ni`, `smap`, `smep`, `ssbd`, `sse`, `sse2`, `sse4_1`, `sse4_2`, `sse4a`, `ssse3`, `syscall`, `topoext`, `tsc`, `tsc_reliable`, `virt_ssbd`, `vme`, `vmmcall`, `xgetbv1`, `xsave`, `xsavec`, `xsaveerptr`, `xsaveopt` |
| `hz_actual_friendly` | 3.9926 GHz |
| `hz_advertised_friendly` | 3.9926 GHz |
| `l1_data_cache_size` | 262144 |
| `l1_instruction_cache_size` | 524288 |
| `l2_cache_associativity` | 6 |
| `l2_cache_line_size` | 512 |
| `l2_cache_size` | 4194304 |
| `l3_cache_size` | 524288 |
| `model` | 8 |
| `python_version` | 3.12.4.final.0 (64 bit) |
| `stepping` | 2 |
| `vendor_id_raw` | AuthenticAMD |

## Settings

There are 4 settings: *default*, *high_accuracy*, *low_accuracy* and *mid_accuracy*. They validate solutions using the following tolerances:

| tolerance   |   default |   high_accuracy |   low_accuracy |   mid_accuracy |
|:------------|----------:|----------------:|---------------:|---------------:|
| ``dual``    |         1 |           1e-09 |          0.001 |          1e-06 |
| ``gap``     |         1 |           1e-09 |          0.001 |          1e-06 |
| ``primal``  |         1 |           1e-09 |          0.001 |          1e-06 |
| ``runtime`` |        10 |          10     |         10     |         10     |

Solvers for each settings are configured as follows:

| solver   | parameter                        | default   | high_accuracy   | low_accuracy   | mid_accuracy   |
|:---------|:---------------------------------|:----------|:----------------|:---------------|:---------------|
| clarabel | ``tol_feas``                     | -         | 1e-09           | 0.001          | 1e-06          |
| clarabel | ``tol_gap_abs``                  | -         | 1e-09           | 0.001          | 1e-06          |
| clarabel | ``tol_gap_rel``                  | -         | 0.0             | 0.0            | 0.0            |
| cvxopt   | ``feastol``                      | -         | 1e-09           | 0.001          | 1e-06          |
| daqp     | ``dual_tol``                     | -         | 1e-09           | 0.001          | 1e-06          |
| daqp     | ``primal_tol``                   | -         | 1e-09           | 0.001          | 1e-06          |
| ecos     | ``feastol``                      | -         | 1e-09           | 0.001          | 1e-06          |
| highs    | ``dual_feasibility_tolerance``   | -         | 1e-09           | 0.001          | 1e-06          |
| highs    | ``primal_feasibility_tolerance`` | -         | 1e-09           | 0.001          | 1e-06          |
| highs    | ``time_limit``                   | 10.0      | 10.0            | 10.0           | 10.0           |
| osqp     | ``eps_abs``                      | -         | 1e-09           | 0.001          | 1e-06          |
| osqp     | ``eps_rel``                      | -         | 0.0             | 0.0            | 0.0            |
| osqp     | ``time_limit``                   | 10.0      | 10.0            | 10.0           | 10.0           |
| piqp     | ``check_duality_gap``            | -         | True            | True           | True           |
| piqp     | ``eps_abs``                      | -         | 1e-09           | 0.001          | 1e-06          |
| piqp     | ``eps_duality_gap_abs``          | -         | 1e-09           | 0.001          | 1e-06          |
| piqp     | ``eps_duality_gap_rel``          | -         | 0.0             | 0.0            | 0.0            |
| piqp     | ``eps_rel``                      | -         | 0.0             | 0.0            | 0.0            |
| proxqp   | ``check_duality_gap``            | -         | True            | True           | True           |
| proxqp   | ``eps_abs``                      | -         | 1e-09           | 0.001          | 1e-06          |
| proxqp   | ``eps_duality_gap_abs``          | -         | 1e-09           | 0.001          | 1e-06          |
| proxqp   | ``eps_duality_gap_rel``          | -         | 0.0             | 0.0            | 0.0            |
| proxqp   | ``eps_rel``                      | -         | 0.0             | 0.0            | 0.0            |
| qpalm    | ``eps_abs``                      | -         | 1e-09           | 0.001          | 1e-06          |
| qpalm    | ``eps_rel``                      | -         | 0.0             | 0.0            | 0.0            |
| qpalm    | ``time_limit``                   | 10.0      | 10.0            | 10.0           | 10.0           |
| qpoases  | ``predefined_options``           | default   | reliable        | fast           | -              |
| qpoases  | ``time_limit``                   | 10.0      | 10.0            | 10.0           | 10.0           |
| scs      | ``eps_abs``                      | -         | 1e-09           | 0.001          | 1e-06          |
| scs      | ``eps_rel``                      | -         | 0.0             | 0.0            | 0.0            |
| scs      | ``time_limit_secs``              | 10.0      | 10.0            | 10.0           | 10.0           |

## Known limitations

The following [issues](https://github.com/qpsolvers/qpbenchmark/issues) have been identified as impacting the fairness of this benchmark. Keep them in mind when drawing conclusions from the results.

- [#60](https://github.com/qpsolvers/qpbenchmark/issues/60): Conversion to SOCP limits performance of ECOS
- [#88](https://github.com/qpsolvers/qpbenchmark/issues/88): CPU thermal throttling

## Results by settings

### Default

Solvers are compared over the whole test set by [shifted geometric mean](https://github.com/qpsolvers/qpbenchmark#shifted-geometric-mean) (shm). Lower is better, 1.0 is the best.

|        |   [Success rate](#success-rate) (%) |   [Runtime](#computation-time) (shm) |   [Primal residual](#primal-residual) (shm) |   [Dual residual](#dual-residual) (shm) |   [Duality gap](#duality-gap) (shm) |
|:-------|------------------------------------:|-------------------------------------:|--------------------------------------------:|----------------------------------------:|------------------------------------:|
| proxqp |                               100.0 |                                  1.0 |                                         1.0 |                                     1.0 |                                 1.0 |

