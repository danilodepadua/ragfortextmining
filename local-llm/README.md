# Local model server (optional)

An [Ollama](https://ollama.com) container that lets `src/rag.py` answer with open models on your own machine. It is optional and separate from the devcontainer: nothing here is needed for the Gemini runs. It is set up for CPU only (the author's machine has no GPU).

The models are the four `provider = "ollama"` tables in `rag_models.toml`: `llama3.1:8b`, `qwen3:8b`, `gemma4:e4b` and `llama3.2:3b`. How the runner uses them (fixed settings, log fields, digest check) is described in the main [README](../README.md) (paragraph "Local models (Ollama)" under "M3: RAG answers").

## Steps

1. Start the server (from the repository root, on the host):
   ```
   docker compose -f local-llm/compose.yaml up -d
   ```
2. Download the models. With no arguments it pulls every model with `provider = "ollama"` in `rag_models.toml`; or pass tags:
   ```
   local-llm/pull-models.sh
   local-llm/pull-models.sh llama3.1:8b qwen3:8b
   ```
3. Attach the running devcontainer to the server's network `ragfortextmining-llm`. Run this on the **host**, not inside the devcontainer:
   ```
   local-llm/connect-devcontainer.sh
   ```
   Repeat it after a devcontainer rebuild or after the network is recreated. The devcontainer configuration is not changed.
4. In the devcontainer, try one development question first, then the final run:
   ```
   python src/rag.py --dev Q006 --model llama3.1:8b
   python src/rag.py --model llama3.1:8b
   ```

The runner reaches the server at `OLLAMA_BASE_URL`, default `http://ragfortextmining-ollama:11434` (the container name on the shared network). To use a server elsewhere, set the variable in `.env` (see `.env.example`) or in the environment.

## Stopping

- `docker compose -f local-llm/compose.yaml stop`: stops the container and frees its RAM; the models stay.
- `local-llm/connect-devcontainer.sh --disconnect`, then `docker compose -f local-llm/compose.yaml down`: removes the container. Disconnect first, otherwise the network cannot be removed.
- `docker compose -f local-llm/compose.yaml down -v`: also deletes the downloaded models (about 20 GB).

## Settings

- `OLLAMA_MEM_LIMIT` (default `9g`) caps the container's memory; the swap limit is set to the same value, so there is no swap. If memory runs out, the Ollama process is killed instead of other containers.
- One model is loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`) and an idle model is unloaded after 5 minutes.
- The port is published on `127.0.0.1:11434` only (for `curl` and debugging); it is not exposed to the network.
- The image is pinned by tag and digest, so every run talks to the same server version.

## Warnings

- **RAM.** An 8B model at Q4 needs about 5 GB, and `gemma4:e4b` more. With the default 9 GB cap, stop other containers first.
- **Speed.** On CPU, expect minutes per answer (an estimate, not measured), so 10 answers per model take a long time.
- **Timeouts and errors.** Each call has a 900 s timeout. Five errors in a row abort that model's run; finished answers stay in the log, and a rerun continues.
