#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

# Defaults for the bimanual UR5e + PGI dataset collected with
# configs/bimanual_vr_joint.yaml. DATASET_ROOT can still be overridden with an
# environment variable or by using the legacy <dataset_root> <episode> form.
DEFAULT_DATASET_ROOT="${PROJECT_ROOT}/data/ur5e-bimanual-vr-joint"
DATASET_ROOT=${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}
REPO_ID="FANYECHEN/ur5e-bimanual-vr-joint"

LEFT_ROBOT_IP="192.168.1.3"
LEFT_GRIPPER_IP="192.168.1.8"
LEFT_GRIPPER_PORT="8888"
RIGHT_ROBOT_IP="192.168.1.5"
RIGHT_GRIPPER_IP="192.168.1.7"
RIGHT_GRIPPER_PORT="8887"

usage() {
    echo "Usage: bash scripts/replay_data.sh <episode> [replay options...]" >&2
    echo "   or: bash scripts/replay_data.sh <dataset_root> <episode> [replay options...]" >&2
    exit 2
}

if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
    EPISODE=$1
    shift
elif [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
    DATASET_ROOT=$1
    EPISODE=$2
    shift 2
else
    usage
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN_DEFAULT="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN_DEFAULT=python
fi
PYTHON_BIN=${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_dataset.py" \
    "${DATASET_ROOT}" \
    "${EPISODE}" \
    --repo-id "${REPO_ID}" \
    --left-robot-ip "${LEFT_ROBOT_IP}" \
    --left-gripper-ip "${LEFT_GRIPPER_IP}" \
    --left-gripper-port "${LEFT_GRIPPER_PORT}" \
    --right-robot-ip "${RIGHT_ROBOT_IP}" \
    --right-gripper-ip "${RIGHT_GRIPPER_IP}" \
    --right-gripper-port "${RIGHT_GRIPPER_PORT}" \
    "$@"
