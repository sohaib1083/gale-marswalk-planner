#!/bin/bash
# Run a command once the GPU has enough free memory (other jobs on this laptop share the 4 GB GPU).
#   tools/when_gpu_free.sh 2600 python3 navcam3d/finetune_depth.py ...
need=$1; shift
ok=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$free" ] && [ "$free" -ge "$need" ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -ge 3 ] && break       # free for three checks in a row (~1 minute), not just a gap between jobs
  sleep 20
done
echo "GPU free (${free} MiB), starting: $*"
exec "$@"
