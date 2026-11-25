# Local model server (optional)

Runbook for an [Ollama](https://ollama.com) container that lets `src/rag.py` answer with open models on your own machine. It is optional and separate from the devcontainer: nothing here is needed for the Gemini runs. It is set up for CPU only (the author's machine has no GPU).

The models are the four `provider = "ollama"` tables in `rag_models.toml`: `llama3.1:8b`, `qwen3:8b`, `gemma4:e4b` and `llama3.2:3b`. How the runner uses them (fixed settings, log fields, digest check) is described in [../docs/rag-runner.md](../docs/rag-runner.md). The image is `ollama/ollama:0.40.18`, pinned by tag and digest in `compose.yaml`.

## Prerequisites and context

- Host: Docker with Compose, a clone of the repository, and `python3` 3.11 or newer (`pull-models.sh` reads `rag_models.toml` with `tomllib` when called without arguments). Run steps 1 to 3 from the repository root on the host.
- Devcontainer: running (for step 3 and the runs in step 4). Run step 4 inside it.
- Disk about 20 GB for the models, and RAM: an 8B model at Q4 needs about 5 GB, and `gemma4:e4b` more. With the default 9 GB cap, stop other containers first.

## Steps

1. Start the server (host):
   ```
   docker compose -f local-llm/compose.yaml up -d
   ```
   Validation: `docker ps --filter name=ragfortextmining-ollama` shows the container as `healthy` after a short wait (the healthcheck runs `ollama list`).
2. Download the models (host; the container name is `ragfortextmining-ollama`). With no arguments the script pulls every model with `provider = "ollama"` in `rag_models.toml`; or pass tags:
   ```
   local-llm/pull-models.sh
   local-llm/pull-models.sh llama3.1:8b qwen3:8b
   ```
   Validation: the script ends with `ollama list` showing every tag you need.
3. Attach the running devcontainer to the server's network `ragfortextmining-llm`. Run this on the **host**, not inside the devcontainer:
   ```
   local-llm/connect-devcontainer.sh
   ```
   Validation: the script prints `connected ...` or `... is already connected ...`. The devcontainer configuration is not changed.
4. In the devcontainer, try one development question first, then the final run:
   ```
   python src/rag.py --dev Q006 --model llama3.1:8b
   python src/rag.py --model llama3.1:8b
   ```
   Validation: the summary line shows the answers written to `runs/`. More on runs and reruns: [../docs/runbooks/run-m3-models.md](../docs/runbooks/run-m3-models.md).

The runner reaches the server at `OLLAMA_BASE_URL`, default `http://ragfortextmining-ollama:11434` (the container name on the shared network). To use a server elsewhere, set the variable in `.env` (see `.env.example`) or in the environment.

## Stop conditions

- Step 3 reports `no running devcontainer found` or `more than one devcontainer matches`: start (or disambiguate) the devcontainer and repeat; do not run it inside the devcontainer.
- The runner stops with the command to run when the server is down or the tag is missing: run step 1 or 2.
- Memory runs out: the Ollama process is killed (not other containers). Stop other containers or lower the model size.
- Each call has a 900 s timeout and five errors in a row abort that model's run, so a hung server can cost up to 5 x 900 s before the abort. Finished answers stay in the log, and a rerun continues.
- The model digest is checked once, at the start of a run: a tag re-pulled in the middle of a run is not detected.
- If one model fails, the others in the same `rag.py` command still run and the exit code is 1.

## Expected result

The container is healthy, the tags are listed, and each run writes `runs/m3_runs.<model>.jsonl`. On CPU, answers take minutes: the committed final runs of the four models took about 5 to 17 minutes per 10 answers (sum of the logged call latencies, on one laptop).

## Recovery and shutdown

The connection is lost when the devcontainer is recreated or the network is removed: repeat step 3 (it is safe to repeat).

- `docker compose -f local-llm/compose.yaml stop`: stops the container and frees its RAM; the models stay.
- `local-llm/connect-devcontainer.sh --disconnect`, then `docker compose -f local-llm/compose.yaml down`: removes the container. Disconnect first, otherwise the network cannot be removed.
- `docker compose -f local-llm/compose.yaml down -v`: also deletes the volume `ragfortextmining-ollama-models` with the downloaded models (about 20 GB).

## Server settings

Defined in `compose.yaml`:
- `OLLAMA_MEM_LIMIT` (default `9g`) caps the container's memory; the swap limit is set to the same value, so there is no swap.
- One model is loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`) and an idle model is unloaded after 5 minutes.
- The port is published on `127.0.0.1:11434` only (for `curl` and debugging); it is not exposed to the network.
- The image is pinned by tag and digest, so every run talks to the same server version.

## References

[../docs/rag-runner.md](../docs/rag-runner.md), [../docs/runbooks/README.md](../docs/runbooks/README.md), `compose.yaml`, `connect-devcontainer.sh`, `pull-models.sh`.
