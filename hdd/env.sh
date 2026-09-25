# Sourced by the launch scripts. On the server everything this project writes, caches included,
# must stay under ~/workspace; left alone, torch.hub, torch.compile, Triton, the CUDA JIT and pip
# write to ~/.cache, ~/.triton, ~/.nv and /tmp.
W="$HOME/workspace"
export TORCH_HOME="$W/.cache/torch" \
       TORCHINDUCTOR_CACHE_DIR="$W/.cache/torchinductor" \
       TRITON_CACHE_DIR="$W/.cache/triton" \
       CUDA_CACHE_PATH="$W/.cache/nv" \
       PIP_CACHE_DIR="$W/.cache/pip" \
       XDG_CACHE_HOME="$W/.cache" \
       MPLCONFIGDIR="$W/.cache/matplotlib" \
       TMPDIR="$W/.tmp"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TMPDIR"
