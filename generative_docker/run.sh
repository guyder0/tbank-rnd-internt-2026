#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo 'Usage: generative_docker/run.sh /absolute/data-directory [container arguments]' >&2
    exit 2
fi
data_dir="$(cd -- "$1" && pwd -P)"
shift
image="${SID_IMAGE:-kion-generative:latest}"
gpu_mode="${SID_GPU:-auto}"
gpu_args=()
device_args=()
case "$gpu_mode" in
    auto)
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
            && [[ "$(docker info --format '{{range $name, $runtime := .Runtimes}}{{if eq $name "nvidia"}}nvidia{{end}}{{end}}' 2>/dev/null)" == nvidia ]]; then
            gpu_args=(--gpus all)
            echo 'NVIDIA GPU/runtime detected; exposing GPU to container.'
        else
            echo 'Starting without GPU passthrough; application will use CPU if GPU is unavailable.'
        fi
        ;;
    cpu) device_args=(--device cpu) ;;
    gpu) gpu_args=(--gpus all); device_args=(--device gpu) ;;
    *) echo 'SID_GPU must be auto, cpu or gpu' >&2; exit 2 ;;
esac
exec docker run --rm --user "$(id -u):$(id -g)" \
    "${gpu_args[@]}" --volume "$data_dir:/data" "$image" "${device_args[@]}" "$@"
