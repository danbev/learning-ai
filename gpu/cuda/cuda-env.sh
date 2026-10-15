export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    printf 'CUDA compiler not found at %s/bin/nvcc; set CUDA_HOME to your toolkit directory.\n' "$CUDA_HOME" >&2
    return 1 2>/dev/null || exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
