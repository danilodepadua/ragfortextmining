# Local model server (optional)

An [Ollama](https://ollama.com) container that lets `src/rag.py` answer with open models on your own machine. It is optional and separate from the devcontainer: nothing here is needed for the Gemini runs. It is set up for CPU only (the author's machine has no GPU).

The models are the four `provider = "ollama"` tables in `rag_models.toml`: `llama3.1:8b`, `qwen3:8b`, `gemma4:e4b` and `llama3.2:3b`. How the runner uses them (fixed settings, log fields, digest check) is described in the main [README](../README.md) (paragraph "Local models (Ollama)" under "M3: RAG answers"). The image is `ollama/ollama:0.40.18`, pinned by tag and digest in `compose.yaml`.

## Steps

1. Start the server (from the repository root, on the host):
   ```
   docker compose -f local-llm/compose.yaml up -d
   ```
2. Download the models (the container name is `ragfortextmining-ollama`). With no arguments the script reads `rag_models.toml` with the host's `python3` (3.11 or newer, for `tomllib`) and pulls every model with `provider = "ollama"`; or pass tags:
   ```
   local-llm/pull-models.sh
   local-llm/pull-models.sh llama3.1:8b qwen3:8b
   ```
3. Attach the running devcontainer to the server's network `ragfortextmining-llm`. Run this on the **host**, not inside the devcontainer:
   ```
   local-llm/connect-devcontainer.sh
   ```
   The connection is lost when the devcontainer is recreated or the network is removed: repeat the command then (it is safe to repeat). The devcontainer configuration is not changed.
4. In the devcontainer, try one development question first, then the final run:
   ```
   python src/rag.py --dev Q006 --model llama3.1:8b
   python src/rag.py --model llama3.1:8b
   ```
   If the server is down or the tag is missing, the runner stops with the command to run.

The runner reaches the server at `OLLAMA_BASE_URL`, default `http://ragfortextmining-ollama:11434` (the container name on the shared network). To use a server elsewhere, set the variable in `.env` (see `.env.example`) or in the environment.

## Stopping

- `docker compose -f local-llm/compose.yaml stop`: stops the container and frees its RAM; the models stay.
- `local-llm/connect-devcontainer.sh --disconnect`, then `docker compose -f local-llm/compose.yaml down`: removes the container. Disconnect first, otherwise the network cannot be removed.
- `docker compose -f local-llm/compose.yaml down -v`: also deletes the volume `ragfortextmining-ollama-models` with the downloaded models (about 20 GB).

## Settings

- `OLLAMA_MEM_LIMIT` (default `9g`) caps the container's memory; the swap limit is set to the same value, so there is no swap. If memory runs out, the Ollama process is killed instead of other containers.
- One model is loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`) and an idle model is unloaded after 5 minutes.
- The port is published on `127.0.0.1:11434` only (for `curl` and debugging); it is not exposed to the network.
- The image is pinned by tag and digest, so every run talks to the same server version.

## Warnings

- **RAM.** An 8B model at Q4 needs about 5 GB, and `gemma4:e4b` more. With the default 9 GB cap, stop other containers first.
- **Speed.** On CPU, answers take minutes: the committed final runs of the four models took about 5 to 17 minutes per 10 answers (sum of the logged call latencies, on one laptop).
- **Timeouts and errors.** Each call has a 900 s timeout and five errors in a row abort that model's run, so a hung server can cost up to 5 x 900 s before the abort. Finished answers stay in the log, and a rerun continues. The model digest is checked once, at the start of a run: a tag re-pulled in the middle of a run is not detected.
- **Several models.** If one model fails, the others in the same `rag.py` command still run and the exit code is 1.
