# Devcontainer and execution contexts

The devcontainer (`.devcontainer/devcontainer.json`) is an alternative to a local virtual environment: Python 3.12 with all dependencies, `pytest`, `ipykernel` and `nbconvert` preinstalled by `.devcontainer/post-create.sh`. It mounts the frozen `data/` read-only so that no command run inside it can change the corpus.

## Contexts

| Context | Use it for |
| --- | --- |
| Host (or a local virtual environment with a writable `data/`) | `src/curate.py` `collect`, `finalize`; `src/build_dataset.py` `build`, `manifest`; git commands that change files under `data/`; the Ollama server in `local-llm/` |
| Devcontainer | `python -m pytest`, `python src/build_dataset.py check`, `python src/retrieve.py`, `python src/rag.py` (they write only to `runs/`) |

## Behavior

- Setup: open the repository in VS Code and choose **Reopen in Container**, or run `devcontainer up --workspace-folder .`.
- Put `GEMINI_API_KEY` in `.env` (see [README.md](../README.md)). The key is not forwarded from the host environment, so an empty host variable can never shadow the `.env` value.
- `data/` is mounted **read-only**: the corpus is frozen (`data/MANIFEST.json` records the sha256 of every file), so `src/curate.py` and `src/build_dataset.py`, which write there, fail instead of changing it. To rebuild the corpus, run them on the host (see [runbooks/rebuild-data.md](runbooks/rebuild-data.md)) or remove the `data` entry from `mounts` in `.devcontainer/devcontainer.json`.
- Run git commands that change files under `data/` (`pull`, `checkout`, `merge`, `stash`) on the host: inside the container they fail on the read-only mount and can leave the working tree half-updated.
- Write run outputs (logs, answers) to `runs/`, which is writable and tracked in git.
- The optional Ollama server runs on the host, outside the devcontainer. Attaching the devcontainer to its network is described in [../local-llm/README.md](../local-llm/README.md).

## Related

- [data-and-run-outputs.md](data-and-run-outputs.md) for the frozen files.
