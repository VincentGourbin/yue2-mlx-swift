#!/usr/bin/env bash
# Reference performance measurement before Jalon 4 optimizations (fiche T-4.0, GATE G-5).
# Cools the GPU 120s, then runs `yue2 generate --profile` on song.json (short settings:
# 64/100/4) twice back to back (A, A) to see the intra-variant spread, and writes the 4
# phase durations + peak memory of the second run to .local-runs/bench/<date>[-tag].txt.
# Extra args ($@, e.g. `--quant qint8`) are passed through to `yue2 generate`, and tag the
# output file so a variant point (B in an A/B/B/A protocol) never overwrites the baseline.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${YUE2_MODELS_DIR:?YUE2_MODELS_DIR non defini}"
bin=.xcodebuild/Build/Products/Release/yue2
[ -x "$bin" ] || { echo "BENCH FAILED : binaire absent, lancer Scripts/check-build.sh"; exit 1; }

echo "cooling down 120s..." >&2
sleep 120

mkdir -p .local-runs/bench
date_tag=$(date +%Y-%m-%d)
tag=$(printf '%s' "$*" | tr -c 'A-Za-z0-9' '-' | sed 's/^-*//; s/-*$//')
out_file=".local-runs/bench/${date_tag}${tag:+-$tag}.txt"
# Captured once, at script scope: `run_once` sees its own `$@`/`$1`, not the script's.
extra_args=("$@")

run_once() {
    local out_dir="$1"
    # `${extra_args[@]+"${extra_args[@]}"}`: bash 3.2's `set -u` treats a zero-element array
    # as unbound even after `extra_args=()`, and the simpler `${extra_args[@]-}` fix doesn't
    # actually fix it either — it expands to one *empty-string* argument, which ArgumentParser
    # then rejects as "Unexpected argument ''". This nested form is the one that truly yields
    # zero words for an empty array while still splatting every element of a non-empty one.
    "$bin" generate --request reference/yue/examples/song.json --out "$out_dir" --profile \
        --abc-max-tokens 64 --semantic-max-tokens 100 --semantic-min-tokens 0 --ode-steps 4 \
        --models-dir "$YUE2_MODELS_DIR" "${extra_args[@]+"${extra_args[@]}"}" 2>&1
}

# Parses `generated <id>: <audio>s audio in <e2e>s (abc <a>s, semantic <s>s, nar <n>s, vae <v>s)`
# and `Peak Process: <mb> MB` from one run's combined output.
extract() {
    local log="$1"
    echo "$log" | grep -oE 'abc [0-9.]+ s, semantic [0-9.]+ s, nar [0-9.]+ s, vae [0-9.]+ s' \
        | grep -oE '[0-9.]+' | tr '\n' ' '
    echo "$log" | grep -oE 'Peak Process: [0-9.]+ MB' | head -1 | grep -oE '[0-9.]+'
}

log_a1=$(run_once .local-runs/bench/run-a1)
log_a2=$(run_once .local-runs/bench/run-a2)

read -r a1_abc a1_sem a1_nar a1_vae a1_peak <<< "$(extract "$log_a1")"
read -r a2_abc a2_sem a2_nar a2_vae a2_peak <<< "$(extract "$log_a2")"

e2e_a1=$(echo "$a1_abc + $a1_sem + $a1_nar + $a1_vae" | bc)
e2e_a2=$(echo "$a2_abc + $a2_sem + $a2_nar + $a2_vae" | bc)
dispersion=$(echo "scale=4; 100 * ($e2e_a2 - $e2e_a1) / (($e2e_a1 + $e2e_a2) / 2)" | bc | tr -d -- '-')

{
    echo "date: ${date_tag}"
    echo "commit: $(git rev-parse --short HEAD)"
    echo "run A1: abc=${a1_abc}s sem=${a1_sem}s nar=${a1_nar}s vae=${a1_vae}s peak=${a1_peak}MB e2e=${e2e_a1}s"
    echo "run A2: abc=${a2_abc}s sem=${a2_sem}s nar=${a2_nar}s vae=${a2_vae}s peak=${a2_peak}MB e2e=${e2e_a2}s"
    echo "dispersion (e2e, A1 vs A2): ${dispersion}%"
} | tee "$out_file" >&2

echo "BENCH ${date_tag} abc=${a2_abc} sem=${a2_sem} nar=${a2_nar} vae=${a2_vae} peak=${a2_peak}"
