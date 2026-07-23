import pickle
import numpy as np
import math
import re

def score(results):
    C, C_star, p_path, l_path, d_T = [], [], [], [], []
    for result in results:
        if result["path_len"] != float("inf"):
            EAC = result["EAC"].replace("Your mark:", "").strip()
            try:
                lines = [line.strip() for line in EAC.split("\n") if line.strip()]
                if not lines:
                    raise ValueError("Empty output")
                last_line = lines[-1]
                match = re.search(r"([0-9](?:\.[0-9]+)?)\s*[,/]\s*([0-9])", last_line)
                if match:
                    grd = float(match.group(1))
                    acc = int(match.group(2))
                else:
                    grd = float("nan")
                    acc = float("nan")
            except Exception as e:
                grd = float("nan")
                acc = float("nan")
            
            if not math.isnan(grd) and not math.isnan(acc):
                C.append(grd*acc)
                C_star.append(acc)
                p_path.append(result["path_len"])
                l_path.append(result["geodesic_distance"])
        if result["goal_dis"] != float("inf"):
            d_T.append(result["goal_dis"])

    if not C:
        return 0.0

    weight_path = np.array(l_path) / np.maximum(p_path, l_path)
    C_avg = np.mean(100.0 * (np.clip(C, 0, 5) / 5))
    return C_avg

def analyze(filename):
    with open("experiment_result/results.pkl", "rb") as f:
        original_results = pickle.load(f)
        
    try:
        with open(filename, "rb") as f:
            generated_data = pickle.load(f)
    except FileNotFoundError:
        print(f"{filename} not found, skipping.")
        return
        
    if not generated_data:
        print(f"{filename} is empty.")
        return
        
    max_runs = max(len(v) for v in generated_data.values())
    
    # Calculate how many episodes were fully completed
    episodes_completed = sum(1 for v in generated_data.values() if len(v) == max_runs)
    
    run_averages = []
    episode_maes = []
    
    for i in range(max_runs):
        run_results = []
        for r in original_results:
            q_ind = r["question_ind"]
            if q_ind in generated_data and len(generated_data[q_ind]) > i:
                pseudo_r = r.copy()
                pseudo_r["EAC"] = generated_data[q_ind][i]
                run_results.append(pseudo_r)
        
        c_avg = score(run_results)
        run_averages.append(c_avg)

    overall_mean = np.mean(run_averages)
    leaderboard_mae = np.mean(np.abs(np.array(run_averages) - overall_mean))

    print(f"\n==========================================")
    print(f"Results for {filename} (Runs: {max_runs}, Episodes: {episodes_completed}/2044)")
    print(f"==========================================")
    print(f"Final Benchmark Scores Across {max_runs} Runs: {[round(r, 2) for r in run_averages]}")
    print(f"Mean Final Score: {overall_mean:.2f}")
    print(f"Leaderboard Variance (MAE of Final Scores): {leaderboard_mae:.3f}")
    
    for r in original_results:
        q_ind = r["question_ind"]
        if q_ind in generated_data and len(generated_data[q_ind]) == max_runs:
            if r["path_len"] != float("inf"):
                episode_c_scores = []
                for i in range(max_runs):
                    pseudo_r = r.copy()
                    pseudo_r["EAC"] = generated_data[q_ind][i]
                    c = score([pseudo_r]) 
                    episode_c_scores.append(c)
                
                mean_c = np.mean(episode_c_scores)
                mae = np.mean(np.abs(np.array(episode_c_scores) - mean_c))
                episode_maes.append(mae)
                
    if episode_maes:
        avg_episode_mae = np.mean(episode_maes)
        print(f"Average Episode-Level MAE: {avg_episode_mae:.2f}")

files = [
    "outputs/all_2044_variance_results/all_2044_variance_results_qwen16bit.pkl",
    "outputs/all_2044_variance_results/all_2044_variance_results_gemma.pkl",
    "outputs/all_2044_variance_results/all_2044_variance_results_gemma_12b.pkl",
    "outputs/all_2044_variance_results/all_2044_variance_results_gemma_31b_nvfp4.pkl",
    "outputs/all_2044_variance_results/all_2044_variance_results_qwen8b_thinking_16bit.pkl",
    "outputs/all_2044_variance_results/all_2044_variance_results_qwen8b_thinking_fp8.pkl"
]
for f in files:
    analyze(f)
