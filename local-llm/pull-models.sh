#!/usr/bin/env bash
# Downloads models into the running Ollama container.
#   local-llm/pull-models.sh                      every model with provider = "ollama" in rag_models.toml
#   local-llm/pull-models.sh llama3.1:8b qwen3:8b  only these tags
set -euo pipefail

if [ "$#" -eq 0 ]; then
  root="$(cd "$(dirname "$0")/.." && pwd)"
  mapfile -t tags < <(python3 - "$root/rag_models.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    config = tomllib.load(f)
for name, table in config["models"].items():
    if table.get("provider") == "ollama":
        print(name)
PY
  )
  set -- "${tags[@]}"
fi

for tag in "$@"; do
  echo "== pulling $tag"
  docker exec ragfortextmining-ollama ollama pull "$tag"
done
docker exec ragfortextmining-ollama ollama list
