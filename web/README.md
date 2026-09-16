# FORGE Demo Console (web)

Thin browser UI over `forge_solve` / `forge_check`. **Not** a modelling product -  says CLI/API is enough; this is for demos and the video.

## Run

```bash
# from repo root (forge/)
cmake --build build -j"$(nproc)"   # need forge_solve, forge_check

./web/run.sh
# or: cd web && source .venv/bin/activate && python app.py
```

Open **http://127.0.0.1:8765**

Optional env:

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_BIN_DIR` | `../build` then `../build-native` | Directory with `forge_solve`, `forge_check`, `forge_gen` |
| `FORGE_EXAMPLES` | `../examples` | Preset MPS/QPS |
| `FORGE_WEB_TIMEOUT` | `90` | Max subprocess seconds |
| `FORGE_WEB_HOST` / `FORGE_WEB_PORT` | `127.0.0.1` / `8765` | Bind address |

## UX idea (simpler than commercial consoles)

Commercial UIs dump parameters, logs, and modelling chrome up front. FORGE's demo is three steps:

1. **Choose** a problem (or drop an MPS/QPS)  
2. **Solve** - one big button; engine/GPU live under Advanced  
3. **Answer** - plain-language verdict + objective; residuals auto-verified  

Same binaries as the CLI. Not a modelling environment (PS does not require a polished GUI).

## Video line

> "Choose a problem, press Solve, get a verified answer - same from-scratch engine as the terminal."
