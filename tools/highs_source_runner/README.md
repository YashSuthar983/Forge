# Native source-HiGHS benchmark runner

This binary is the only accepted `highs-source` timing lane. It calls the
HiGHS C++ API, reads the model before the measured interval, and reports the
`Highs::getRunTime()` delta as versioned JSON. The ordinary HiGHS CLI's
two-decimal `HiGHS run time` text is intentionally never parsed.

Configure it against the exact clean, pinned source/build used for evidence:

```sh
cmake -S tools/highs_source_runner -B build-highs-runner \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/same/compiler/as/sor \
  -DHIGHS_SOURCE_DIR=/path/to/HiGHS \
  -DHIGHS_BUILD_DIR=/path/to/HiGHS/build \
  -DHIGHS_EXPECTED_COMMIT=<full-40-character-commit>
cmake --build build-highs-runner --parallel
build-highs-runner/sor_highs_source_runner --identity
```

Configuration refuses a dirty or unpinned source checkout, a mismatched build
tree/compiler, or a HiGHS build lacking Release, `-O3`, `NDEBUG`, and
`-march=native`. Runtime refuses unknown arguments and rejected HiGHS options.
The Python claim preflight independently checks the emitted identity against
SOR and the official wheel before any measurements are made.

