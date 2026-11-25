#!/usr/bin/env bash
# Attaches this project's running devcontainer to the model server's network, so the runner can reach
# http://ragfortextmining-ollama:11434 from inside it. The devcontainer configuration is not changed.
#   local-llm/connect-devcontainer.sh               connect (safe to repeat)
#   local-llm/connect-devcontainer.sh --disconnect  detach, needed before `docker compose down`
# A rebuilt devcontainer, or a server whose network was recreated, has to be connected again.
set -euo pipefail

network=ragfortextmining-llm
case "${1:-}" in
  "" | --disconnect) ;;
  *) echo "usage: $0 [--disconnect]" >&2; exit 2 ;;
esac

# The label holds the path the devcontainer was opened with: try the plain and the symlink-resolved path.
root="$(cd "$(dirname "$0")/.." && pwd)"
container=""
for candidate in "$root" "$(cd "$root" && pwd -P)"; do
  container="$(docker ps -q --filter "label=devcontainer.local_folder=$candidate")"
  [ -n "$container" ] && break
done
if [ -z "$container" ]; then
  echo "no running devcontainer found for $root" >&2
  exit 1
fi
if [ "$(echo "$container" | wc -l)" -ne 1 ]; then
  echo "more than one devcontainer matches $root: $container" >&2
  exit 1
fi

connected="$(docker inspect -f '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}} {{end}}' "$container")"
case "${1:-}" in
  --disconnect)
    if [[ " $connected " == *" $network "* ]]; then
      docker network disconnect "$network" "$container"
      echo "disconnected $container from $network"
    else
      echo "$container was not connected to $network"
    fi
    ;;
  "")
    if [[ " $connected " == *" $network "* ]]; then
      echo "$container is already connected to $network"
    else
      docker network connect "$network" "$container"
      echo "connected $container to $network"
    fi
    ;;
esac
