# Runbooks

Step-by-step procedures. Each runbook is self-contained: prerequisites and execution context first, numbered steps with validation checks, stop conditions, the expected result and recovery.

| Runbook | Context | Purpose |
| --- | --- | --- |
| [reproduce-results.md](reproduce-results.md) | Devcontainer | Reproduce the committed retrieval and RAG results from a fresh clone with 0 API calls |
| [run-models.md](run-models.md) | Devcontainer | Run or rerun the RAG runner for one or more models, development runs included |
| [rebuild-data.md](rebuild-data.md) | Host | Rebuild the frozen `data/` corpus and manifest (deliberate, rare) |
| [../../local-llm/README.md](../../local-llm/README.md) | Host and devcontainer | Start the Ollama server, pull models, connect the devcontainer, stop it |

Reference pages are indexed in the [README](../../README.md).
