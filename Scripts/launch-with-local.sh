#!/usr/bin/env bash
# Branche pi (https://pi.dev) sur `qwen38 serve` (http://127.0.0.1:8848) et le lance
# dans ce dépôt. Usage :
#   Scripts/launch-with-local.sh            # interactif
#   Scripts/launch-with-local.sh T-1.0      # une fiche, non interactif (voir run-fiche.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${QWEN38_BASE:-http://127.0.0.1:8848}"
command -v pi >/dev/null || { echo "pi absent : npm install -g --ignore-scripts @earendil-works/pi-coding-agent"; exit 1; }
health=$(curl -sf --max-time 10 "$BASE/healthz") || { echo "serveur qwen38 injoignable sur $BASE (lancer : qwen38 serve …)"; exit 1; }
model=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('model',''))" "$health")
[ -n "$model" ] || { echo "healthz ne donne pas de modèle : $health"; exit 1; }
echo "modèle serveur : $model"
# Fusionne le fournisseur local dans ~/.pi/agent/models.json (créé si absent).
mkdir -p ~/.pi/agent
python3 - "$BASE" "$model" <<'PY'
import json, os, sys
base, model = sys.argv[1], sys.argv[2]
path = os.path.expanduser("~/.pi/agent/models.json")
data = json.load(open(path)) if os.path.exists(path) else {}
providers = data.setdefault("providers", {})
providers["qwen38-local"] = {
    "baseUrl": base + "/v1",
    "api": "openai-completions",
    "apiKey": "local",
    # qwen38 serve n'a pas de rôle `developer` (il le prendrait pour `user`) ;
    # il accepte `reasoning_effort`, ce qui active la réflexion (PLAN.md §0.4).
    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": True},
    "models": [{
        "id": model, "name": "Qwen3.8 Flash-Next (local)", "reasoning": True,
        "input": ["text"], "contextWindow": 32768, "maxTokens": 4096,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }],
}
json.dump(data, open(path, "w"), indent=2, ensure_ascii=False)
print("~/.pi/agent/models.json : fournisseur qwen38-local écrit")
PY
[ -d reference ] || Scripts/link-references.sh
export PI_MODEL="qwen38-local/$model"
if [ $# -ge 1 ]; then exec Scripts/run-fiche.sh "$1"; fi
echo "Message à envoyer : Exécute la fiche tasks/T-1.0.md en suivant AGENTS.md."
exec pi --approve --thinking low --model "$PI_MODEL"
