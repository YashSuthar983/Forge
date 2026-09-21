# Netlib LP — GPU first-order + crossover

```
date            : 2026-09-21 11:45:00 UTC
host            : shreyas-radeon
cpu             : AMD Ryzen 7 7700X 8-Core Processor
gpu             : AMD Radeon RX 9060 XT (RADV GFX1200), fp64, 16304 MiB
vulkan loader   : 1.3.275     glslangValidator: 15.1.0
compiler        : g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
commit          : 89a0f35
```

**Reference** `--engine simplex --time-limit 60` (reaches `ProvedOptimalFP`).
**GPU** `--engine hpr --backend vulkan --no-fo-certificates --no-presolve --time-limit 90`.

GPU wall time includes host<->device transfer. `--no-presolve` is required:
with presolve enabled `lift_reduced_candidate` loses the basis on the way
back from the reduced space and the same runs return `NoSolutionFound`
despite identical objectives. That reproduces on `--backend cpu`, so it is
an orchestration bug, not a device bug.

## Summary

| metric | value |
|---|---|
| instances | 93 |
| CPU reference optimal | 93 |
| **GPU objective agrees (rel < 1e-6)** | **93 / 93** |
| GPU objective disagrees | 0 |
| **GPU `ProvedOptimalFP`** | **92 / 93** |

The one miss is `fit2d`: `Interrupted / FeasibleWithGap`, objective still
correct to 1.899e-10, out of budget at 90 s. It is 25 rows x 10500 columns,
so the row-parallel SpMV gets 25 threads of work on a 32-CU device -- a
shape problem, not a size problem, and the case a cost model should route
to CPU.

## Per instance

| instance | reference objective | GPU objective | rel err | GPU status / proof | GPU ms |
|---|---|---|---|---|---|
| `25fv47` | 5.5018458883e+03 | 5.5018458883e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 7306.052 |
| `80bau3b` | 9.8722419241e+05 | 9.8722419241e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 7139.103 |
| `adlittle` | 2.2549496316e+05 | 2.2549496316e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 278.308 |
| `afiro` | -4.6475314286e+02 | -4.6475314286e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 224.087 |
| `agg2` | -2.0239252356e+07 | -2.0239252356e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6691.973 |
| `agg3` | 1.0312115935e+07 | 1.0312115935e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6838.740 |
| `agg` | -3.5991767287e+07 | -3.5991767287e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6161.556 |
| `bandm` | -1.5862801845e+02 | -1.5862801845e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1005.213 |
| `beaconfd` | 3.3592485807e+04 | 3.3592485807e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 10514.657 |
| `blend` | -3.0812149846e+01 | -3.0812149846e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 316.140 |
| `bnl1` | 1.9776295615e+03 | 1.9776295615e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6088.709 |
| `bnl2` | 1.8112365404e+03 | 1.8112365404e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6236.045 |
| `boeing1` | -3.3521356751e+02 | -3.3521356751e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 12349.046 |
| `boeing2` | -3.1501872802e+02 | -3.1501872802e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1312.720 |
| `bore3d` | 1.3730803942e+03 | 1.3730803942e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 5922.087 |
| `brandy` | 1.5185098965e+03 | 1.5185098965e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 781.127 |
| `capri` | 2.6900129138e+03 | 2.6900129138e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 5135.237 |
| `cycle` | -5.2263930249e+00 | -5.2263930249e+00 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6433.222 |
| `czprob` | 2.1851966989e+06 | 2.1851966989e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 8833.061 |
| `d2q06c` | 1.2278421081e+05 | 1.2278421081e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 9821.040 |
| `d6cube` | 3.1549166667e+02 | 3.1549166667e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 11544.803 |
| `degen2` | -1.4351780000e+03 | -1.4351780000e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 451.867 |
| `degen3` | -9.8729400000e+02 | -9.8729400000e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1817.288 |
| `dfl001` | 1.1266396047e+07 | 1.1266396047e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 16354.275 |
| `e226` | -1.1638929066e+01 | -1.1638929066e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1353.216 |
| `etamacro` | -7.5571523325e+02 | -7.5571523325e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 4560.827 |
| `fffff800` | 5.5567956482e+05 | 5.5567956482e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 11177.118 |
| `finnis` | 1.7279106560e+05 | 1.7279106560e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 4873.604 |
| `fit1d` | -9.1463780924e+03 | -9.1463780924e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 7389.950 |
| `fit1p` | 9.1463780924e+03 | 9.1463780924e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 3844.730 |
| `fit2d` | -6.8464293294e+04 | -6.8464293307e+04 | 1.899e-10 | `Interrupted/FeasibleWithGap` | 90448.526 |
| `fit2p` | 6.8464293294e+04 | 6.8464293294e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 44428.954 |
| `forplan` | -6.6421896127e+02 | -6.6421896127e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 14324.541 |
| `ganges` | -1.0958573613e+05 | -1.0958573613e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 4058.045 |
| `gfrd-pnc` | 6.9022359995e+06 | 6.9022359995e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1970.399 |
| `greenbea` | -7.2555248130e+07 | -7.2555248130e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 10393.386 |
| `greenbeb` | -4.3022602612e+06 | -4.3022602612e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 10626.928 |
| `grow15` | -1.0687094129e+08 | -1.0687094129e+08 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 788.763 |
| `grow22` | -1.6083433648e+08 | -1.6083433648e+08 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1137.327 |
| `grow7` | -4.7787811815e+07 | -4.7787811815e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1303.649 |
| `israel` | -8.9664482186e+05 | -8.9664482186e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1152.832 |
| `kb2` | -1.7499001299e+03 | -1.7499001299e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1638.195 |
| `lotfi` | -2.5264706062e+01 | -2.5264706062e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 7452.210 |
| `maros` | -5.8063743701e+04 | -5.8063743701e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 8361.204 |
| `maros-r7` | 1.4971851665e+06 | 1.4971851665e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 9690.960 |
| `modszk1` | 3.2061972906e+02 | 3.2061972907e+02 | 3.119e-11 | `Optimal/ProvedOptimalFP` | 4768.937 |
| `nesm` | 1.4076036488e+07 | 1.4076036488e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6018.651 |
| `perold` | -9.3807552782e+03 | -9.3807552782e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 5049.650 |
| `pilot4` | -2.5811392589e+03 | -2.5811392589e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6546.975 |
| `pilot87` | 3.0171034733e+02 | 3.0171034733e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 25615.153 |
| `pilot.ja` | -6.1131364656e+03 | -6.1131364656e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 11814.284 |
| `pilot` | -5.5748972927e+02 | -5.5748970616e+02 | 4.145e-08 | `Optimal/ProvedOptimalFP` | 18110.284 |
| `pilotnov` | -4.4972761882e+03 | -4.4972761882e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 10388.653 |
| `pilot.we` | -2.7201075328e+06 | -2.7201075328e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6330.547 |
| `recipe` | -2.6661600000e+02 | -2.6661600000e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 371.500 |
| `sc105` | -5.2202061212e+01 | -5.2202061212e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 407.886 |
| `sc205` | -5.2202061212e+01 | -5.2202061212e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 2471.930 |
| `sc50a` | -6.4575077059e+01 | -6.4575077059e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 284.277 |
| `sc50b` | -7.0000000000e+01 | -7.0000000000e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 309.429 |
| `scagr25` | -1.4753433061e+07 | -1.4753433061e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1316.655 |
| `scagr7` | -2.3313898243e+06 | -2.3313898243e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1848.382 |
| `scfxm1` | 1.8416759028e+04 | 1.8416759028e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 644.073 |
| `scfxm2` | 3.6660261565e+04 | 3.6660261565e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1410.088 |
| `scfxm3` | 5.4901254550e+04 | 5.4901254550e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1302.047 |
| `scorpion` | 1.8781248227e+03 | 1.8781248227e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 2224.036 |
| `scrs8` | 9.0429695380e+02 | 9.0429695380e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 2538.779 |
| `scsd1` | 8.6666666743e+00 | 8.6666666743e+00 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 256.585 |
| `scsd6` | 5.0500000078e+01 | 5.0500000078e+01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 371.537 |
| `scsd8` | 9.0499999993e+02 | 9.0499999993e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 713.187 |
| `sctap1` | 1.4122500000e+03 | 1.4122500000e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 273.364 |
| `sctap2` | 1.7248071429e+03 | 1.7248071429e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 380.658 |
| `sctap3` | 1.4240000000e+03 | 1.4240000000e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 221.946 |
| `seba` | 1.5711600000e+04 | 1.5711600000e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 956.311 |
| `share1b` | -7.6589318579e+04 | -7.6589318579e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 5385.954 |
| `share2b` | -4.1573224074e+02 | -4.1573224074e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1274.844 |
| `shell` | 1.2088253460e+09 | 1.2088253460e+09 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1820.310 |
| `ship04l` | 1.7933245380e+06 | 1.7933245380e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 854.896 |
| `ship04s` | 1.7987147004e+06 | 1.7987147004e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1316.583 |
| `ship08l` | 1.9090552114e+06 | 1.9090552114e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1555.209 |
| `ship08s` | 1.9200982105e+06 | 1.9200982105e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 683.555 |
| `ship12l` | 1.4701879193e+06 | 1.4701879193e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 2346.149 |
| `ship12s` | 1.4892361344e+06 | 1.4892361344e+06 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1063.469 |
| `sierra` | 1.5394362184e+07 | 1.5394362184e+07 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 740.724 |
| `stair` | -2.5126695119e+02 | -2.5126695119e+02 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 2684.880 |
| `standata` | 1.2576995000e+03 | 1.2576995000e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 967.230 |
| `standgub` | 1.2576995000e+03 | 1.2576995000e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 755.706 |
| `standmps` | 1.4060175000e+03 | 1.4060175000e+03 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 785.971 |
| `stocfor1` | -4.1131976219e+04 | -4.1131976219e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 510.678 |
| `stocfor2` | -3.9024408538e+04 | -3.9024408538e+04 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1097.331 |
| `tuff` | 2.9214776509e-01 | 2.9214776509e-01 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 9556.105 |
| `vtp.base` | 1.2983146246e+05 | 1.2983146246e+05 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 1563.795 |
| `wood1p` | 1.4429024116e+00 | 1.4429024116e+00 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 6306.731 |
| `woodw` | 1.3044763331e+00 | 1.3044763331e+00 | 0.000e+00 | `Optimal/ProvedOptimalFP` | 10256.507 |
