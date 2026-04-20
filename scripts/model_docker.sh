#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_EXTRA="${MODEL_EXTRA:-pi}"
IMAGE_NAME="${IMAGE_NAME:-lerobot-kvt-train}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache/model-docker}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
DOCKERFILE="${DOCKERFILE:-${REPO_ROOT}/docker/Dockerfile.train}"

usage() {
  cat <<EOF
Usage:
  $0 --model {sarm|pi|xvla} build
  $0 --model {sarm|pi|xvla} shell
  $0 --model {sarm|pi|xvla} login
  $0 --model {sarm|pi|xvla} run -- <command>

Environment variables:
  IMAGE_NAME       Docker image base name. Default: lerobot-kvt-train
  CONTAINER_NAME  Optional fixed container name. Default: let Docker choose one.
  CACHE_ROOT      Host cache root for Hugging Face, W&B, torch, uv.
  DATASET_DIR     Host dataset directory mounted to /workspace/lerobot/datasets.
  OUTPUT_DIR      Host output directory mounted to /workspace/lerobot/outputs.

Examples:
  $0 --model sarm build
  $0 --model sarm login
  $0 --model sarm shell
  $0 --model sarm run -- lerobot-info
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

require_model() {
  case "${MODEL_EXTRA}" in
    sarm|pi|xvla) ;;
    *) die "--model must be one of: sarm, pi, xvla" ;;
  esac
}

model_profile() {
  require_model
  case "${MODEL_EXTRA}" in
    sarm|pi) echo "pi-sarm" ;;
    xvla) echo "xvla" ;;
  esac
}

model_extras() {
  require_model
  case "${MODEL_EXTRA}" in
    sarm|pi) echo "pi sarm" ;;
    xvla) echo "xvla" ;;
  esac
}

image_tag() {
  echo "${IMAGE_NAME}:$(model_profile)"
}

container_name_args() {
  if [ -n "${CONTAINER_NAME}" ]; then
    echo "--name ${CONTAINER_NAME}"
  fi
}

prepare_dirs() {
  mkdir -p \
    "${CACHE_ROOT}/home" \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/wandb" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/triton" \
    "${CACHE_ROOT}/uv" \
    "${DATASET_DIR}" \
    "${OUTPUT_DIR}"
}

gpu_args() {
  if docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia"; then
    echo "--gpus all"
  fi
}

tty_args() {
  if [ -t 0 ]; then
    echo "-it"
  else
    echo "-i"
  fi
}

docker_run_base() {
  prepare_dirs
  docker run --rm \
    $(tty_args) \
    $(gpu_args) \
    --ipc=host \
    --network=host \
    --user "$(id -u):$(id -g)" \
    $(container_name_args) \
    -e HOME=/cache/home \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_CACHE=/cache/huggingface/hub \
    -e HF_LEROBOT_HOME=/cache/huggingface/lerobot \
    -e WANDB_DIR=/cache/wandb \
    -e WANDB_CACHE_DIR=/cache/wandb \
    -e USER="${USER:-lerobot}" \
    -e LOGNAME="${LOGNAME:-${USER:-lerobot}}" \
    -e TORCH_HOME=/cache/torch \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/torch/inductor \
    -e TRITON_CACHE_DIR=/cache/triton \
    -e UV_CACHE_DIR=/cache/uv \
    -e UV_PROJECT_ENVIRONMENT=/opt/lerobot/.venv \
    -e UV_LINK_MODE=copy \
    -e MODEL_EXTRAS="$(model_extras)" \
    -v "${REPO_ROOT}:/workspace/lerobot" \
    -v "${CACHE_ROOT}/home:/cache/home" \
    -v "${CACHE_ROOT}/huggingface:/cache/huggingface" \
    -v "${CACHE_ROOT}/wandb:/cache/wandb" \
    -v "${CACHE_ROOT}/torch:/cache/torch" \
    -v "${CACHE_ROOT}/triton:/cache/triton" \
    -v "${CACHE_ROOT}/uv:/cache/uv" \
    -v "${DATASET_DIR}:/workspace/lerobot/datasets" \
    -v "${OUTPUT_DIR}:/workspace/lerobot/outputs" \
    -w /workspace/lerobot \
    "$(image_tag)" \
    "$@"
}

build_image() {
  require_model
  prepare_dirs
  docker build \
    -f "${DOCKERFILE}" \
    --build-arg MODEL_EXTRAS="$(model_extras)" \
    -t "$(image_tag)" \
    "${REPO_ROOT}"
}

run_login() {
  require_model
  docker_run_base bash -lc '
    set -euo pipefail
    echo "Logging in to Hugging Face. Token is cached under ${HF_HOME}."
    if command -v hf >/dev/null 2>&1; then
      hf auth login
    else
      huggingface-cli login
    fi
    echo "Logging in to Weights & Biases. Token is cached under ${WANDB_CACHE_DIR}."
    wandb login
  '
}

run_shell() {
  require_model
  docker_run_base bash -lc '
    exec bash
  '
}

run_command() {
  require_model
  [ "$#" -gt 0 ] || die "run requires a command after --"
  docker_run_base bash -lc '
    exec "$@"
  ' bash "$@"
}

ACTION=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      [ "$#" -ge 2 ] || die "--model requires a value"
      MODEL_EXTRA="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    build|shell|login|run)
      ACTION="$1"
      shift
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "${ACTION}" ] || {
  usage
  exit 1
}

case "${ACTION}" in
  build) build_image ;;
  shell) run_shell "$@" ;;
  login) run_login ;;
  run) run_command "$@" ;;
  *) die "unknown action: ${ACTION}" ;;
esac
