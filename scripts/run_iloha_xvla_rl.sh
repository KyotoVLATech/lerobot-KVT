#!/usr/bin/env bash
# Iloha sushi X-VLA: ordinary EXPO-FT, with post-episode human rewards.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    cat <<'EOF'
Usage: bash scripts/run_iloha_xvla_rl.sh [--dry-run] [server|client|eval] [extra arguments...]

  server  Start the normal EXPO-FT learner (default; GPU terminal).
  client  Start the robot client (second terminal; Enter to start each episode).
  eval    Run the original X-VLA evaluation command, without RL updates.

Defaults: sushi checkpoint 060000, 20 seconds, unlimited RL episodes, Quality: High task.
Evaluation defaults to 5 episodes, as in the original evaluation command.
The RL client asks for a reward in [0, 1] after each episode ends.
The learner updates soft prompts, the edit policy and critics; RTC is not used.

Optional environment overrides:
  POLICY_PATH, DATASET_PATH, TASK, EPISODE_TIME_S, NUM_EPISODES,
  HOST, PORT, CONTROL_HZ, STATE_SOURCE, OUTPUT_DIR

Example (separate GPU and robot PCs):
  bash scripts/run_iloha_xvla_rl.sh server
  HOST=192.168.1.10 bash scripts/run_iloha_xvla_rl.sh client

Extra arguments go to the selected Python script, e.g.:
  bash scripts/run_iloha_xvla_rl.sh server --resume outputs/rl/previous/latest
EOF
}

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi
mode=${1:-server}
if (( $# > 0 )); then
    shift
fi

POLICY_PATH=${POLICY_PATH:-outputs/train/xvla_iloha-dataset-sushi/checkpoints/060000/pretrained_model}
DATASET_PATH=${DATASET_PATH:-datasets/iloha-dataset-sushi}
TASK=${TASK:-Grab the edge of the towel and fold it twice. Quality: High}
EPISODE_TIME_S=${EPISODE_TIME_S:-20}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8106}
CONTROL_HZ=${CONTROL_HZ:-30}
STATE_SOURCE=${STATE_SOURCE:-command}

case "$mode" in
    server)
        # Separate output per run; resumed checkpoints keep their own replay/state.
        OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl/xvla_sushi_expo_ft_$(date +%Y%m%d_%H%M%S)}
        command=(uv run --extra xvla iloha_xvla_rl.py
            --policy_path "$POLICY_PATH"
            --output_dir "$OUTPUT_DIR"
            --port "$PORT" --control_hz "$CONTROL_HZ"
            --episodes "${NUM_EPISODES:-0}")
        ;;
    client)
        command=(uv run --extra xvla iloha_rl.py
            --host "$HOST" --port "$PORT"
            --observation_format xvla --state_source "$STATE_SOURCE"
            --reward_mode terminal-score
            --episode_time_s "$EPISODE_TIME_S" --fps "$CONTROL_HZ"
            --task "$TASK")
        ;;
    eval)
        command=(uv run --extra xvla iloha_eval.py
            --policy_path "$POLICY_PATH" --dataset_path "$DATASET_PATH"
            --episode_time_s "$EPISODE_TIME_S" --num_episodes "${NUM_EPISODES:-5}"
            --task "$TASK")
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

command+=("$@")
printf 'Running: '
printf '%q ' "${command[@]}"
printf '\n'
if (( ! dry_run )); then
    exec "${command[@]}"
fi
