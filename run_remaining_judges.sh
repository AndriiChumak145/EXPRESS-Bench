#!/bin/bash
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate qwen36
export LD_LIBRARY_PATH=/home/dani/miniconda3/envs/qwen36/lib:$LD_LIBRARY_PATH

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Cleaning up background llama.cpp server (PID $SERVER_PID)..."
        kill -9 $SERVER_PID 2>/dev/null
        pkill -f llama_cpp.server 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

run_eval() {
    local output_file=$1
    local qwen_model=$2
    local gemma_model=$3
    echo "========================================================="
    echo "Starting evaluation loop for: $output_file"
    echo "========================================================="
    
    while true; do
        python3 eval_all_2044.py --output_file "outputs/all_2044_variance_results/$output_file" \
            --episodes_per_batch 250 \
            --num_evals_per_episode 10 \
            ${qwen_model:+--qwen_model "$qwen_model"} \
            ${gemma_model:+--gemma_model "$gemma_model"}
        exit_status=$?

        if [ $exit_status -eq 3 ]; then
            echo "Batch completed cleanly. Restarting process to free VRAM..."
            sleep 2
        elif [ $exit_status -eq 0 ]; then
            echo "Evaluation for $output_file completed perfectly! Moving to next model..."
            break
        else
            echo "Script crashed unexpectedly with code $exit_status. Stopping loop."
            break
        fi
    done
}

# 1. 16-bit Qwen Instruct (Native)
# export USE_LOCAL_QWEN=1
# export QWEN_QUANTIZATION_OVERRIDE=0
# export QWEN_THINKING=0
# export USE_LOCAL_GEMMA=0
# run_eval "all_2044_variance_results_qwen16bit.pkl" "Qwen/Qwen3-VL-8B-Instruct" ""

# 3a. Gemma-4-E4B-it
# export USE_LOCAL_QWEN=0
# export QWEN_QUANTIZATION_OVERRIDE=0
# export QWEN_THINKING=0
# export USE_LOCAL_GEMMA=1
# run_eval "all_2044_variance_results_gemma.pkl" "" "google/gemma-4-E4B-it"

# 4b. Gemma-4-31B-IT-NVFP4 (Local llama.cpp Server)
# echo "Starting llama.cpp NVFP4 server in the background..."
# bash /home/dani/concept-scenesplat/nvfp4/start_llama_server.sh > /home/dani/concept-scenesplat/EXPRESS-Bench/llama_server.log 2>&1 &
# SERVER_PID=$!
# echo "Waiting 30 seconds for server to load the 30GB model into VRAM..."
# sleep 30

# export USE_LOCAL_QWEN=0
# export USE_LOCAL_GEMMA=0
# export USE_LOCAL_VLLM=1
# export QWEN_QUANTIZATION_OVERRIDE=0
# export QWEN_THINKING=0
# run_eval "all_2044_variance_results_gemma_31b_nvfp4.pkl" "" ""

# echo "Killing NVFP4 server to free 30GB VRAM..."
# kill -9 $SERVER_PID
# pkill -f llama_cpp.server
# sleep 10

# 3b. Gemma-4-12B-it (Resuming with full VRAM!)
# export USE_LOCAL_QWEN=0
# export QWEN_QUANTIZATION_OVERRIDE=0
# export QWEN_THINKING=0
# export USE_LOCAL_GEMMA=1
# export USE_LOCAL_VLLM=0
# run_eval "all_2044_variance_results_gemma_12b.pkl" "" "google/gemma-4-12B-it"

# 5. Qwen 8B Thinking (Native 16-bit)
# export USE_LOCAL_QWEN=1
# export QWEN_QUANTIZATION_OVERRIDE=0
# export QWEN_THINKING=0
# export USE_LOCAL_GEMMA=0
# run_eval "all_2044_variance_results_qwen8b_thinking_16bit.pkl" "Qwen/Qwen3-VL-8B-Thinking" ""

# # 6. Qwen 8B Thinking (FP8)  - didn't finish due to very long thinking time 
# export USE_LOCAL_QWEN=1
# export QWEN_QUANTIZATION_OVERRIDE=0  # Native FP8 handles quantization internally without bitsandbytes
# export QWEN_THINKING=0
# export USE_LOCAL_GEMMA=0
# run_eval "all_2044_variance_results_qwen8b_thinking_fp8.pkl" "Qwen/Qwen3-VL-8B-Thinking-FP8" ""


# qwen3.6-27B-NVFP4 (Local llama.cpp Server)
echo "Starting llama.cpp NVFP4 server in the background..."
bash /home/dani/concept-scenesplat/nvfp4/start_llama_server.sh > /home/dani/concept-scenesplat/EXPRESS-Bench/llama_server.log 2>&1 &
SERVER_PID=$!
echo "Waiting 60 seconds for server to load the 26GB model into VRAM..."
sleep 60

export USE_LOCAL_QWEN=0
export USE_LOCAL_GEMMA=0
export USE_LOCAL_VLLM=1
export QWEN_QUANTIZATION_OVERRIDE=0
export QWEN_THINKING=0
run_eval "all_2044_variance_results_qwen3.6_27b_nvfp4_no_reasoning.pkl" "" ""

echo "Killing NVFP4 server to free 26GB VRAM..."
kill -9 $SERVER_PID
pkill -f llama_cpp.server
sleep 10



echo "All judge tests finished successfully!"
