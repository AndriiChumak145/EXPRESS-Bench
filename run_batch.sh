#!/bin/bash
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate fine-eqa
export LD_LIBRARY_PATH=/home/dani/miniconda3/envs/fine-eqa/lib

export USE_LOCAL_QWEN=1
export USE_LOCAL_GEMMA=0
export QWEN_THINKING=0
export GEMMA_THINKING=0

MAX_RESTARTS=100
restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "================================================="
    echo "Starting benchmark run (Attempt $((restart_count+1)))"
    echo "================================================="
    
    python main.py -cf fine_eqa.yaml
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Benchmark completed successfully! All episodes processed."
        break
    else
        echo "Benchmark crashed with exit code $exit_code (Likely OOM). Restarting in 5 seconds to bypass memory leak..."
        restart_count=$((restart_count+1))
        sleep 5
    fi
done

if [ $restart_count -eq $MAX_RESTARTS ]; then
    echo "Reached maximum restarts ($MAX_RESTARTS). Exiting."
fi
