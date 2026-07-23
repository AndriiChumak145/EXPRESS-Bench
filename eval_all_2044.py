import os
import sys
import pickle
import logging
import torch
import argparse
from tqdm import tqdm
import gpt


logging.basicConfig(level=logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Evaluate 2044 episodes")
    parser.add_argument("--output_file", type=str, default="all_2044_variance_results.pkl", help="Output file to save the benchmark results")
    parser.add_argument("--qwen_model", type=str, default=None, help="Override the Qwen model string in gpt.py")
    parser.add_argument("--gemma_model", type=str, default=None, help="Override the Gemma model string in gpt.py")
    parser.add_argument("--episodes_per_batch", type=int, default=250, help="Number of episodes to process before exiting to free VRAM")
    parser.add_argument("--num_evals_per_episode", type=int, default=2, help="Number of evaluations per episode")
    args = parser.parse_args()
    
    EPISODES_PER_BATCH = args.episodes_per_batch
    NUM_EVALS_PER_EPISODE = args.num_evals_per_episode
    
    if args.qwen_model:
        gpt.QWEN_MODEL = args.qwen_model
    if args.gemma_model:
        gpt.GEMMA_MODEL = args.gemma_model

    force_think = os.environ.get("QWEN_THINKING", "0") == "1"
    score_prompt = "./prompt/evaluation_thinking.txt" if force_think else "./prompt/evaluation.txt"
    
    with open("experiment_result/results.pkl", "rb") as f:
        results = pickle.load(f)
    
    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            all_scores = pickle.load(f)
        print(f"Resuming from existing progress. {len(all_scores)} episodes fully processed.")
    else:
        all_scores = {} 
        
    episodes_processed_this_run = 0

    for r in tqdm(results, total=len(results)):
        question_ind = r["question_ind"]
        
        if question_ind in all_scores and len(all_scores[question_ind]) >= NUM_EVALS_PER_EPISODE:
            continue
            
        cnt_step = r["cnt_step"]
        img_path = f"experiment_result/{question_ind}/{cnt_step}.png"
        
        ex_prompt = f"Question: {r['question']}\nAnswer: {r['answer']}\nResponse: {r['gen_answer']}\nYour mark: "
        
        if question_ind not in all_scores:
            all_scores[question_ind] = []
            
        runs_needed = NUM_EVALS_PER_EPISODE - len(all_scores[question_ind])
        for _ in range(runs_needed):
            EAC = gpt.ask_model(score_prompt, ex_prompt, img_path, force_think_tag=force_think)
            all_scores[question_ind].append(EAC)
            
        # Clean up memory
        torch.cuda.empty_cache()
            
        # Save incrementally 
        if question_ind % 10 == 0:
            with open(output_file, "wb") as f:
                pickle.dump(all_scores, f)

        episodes_processed_this_run += 1
        
        # Exit cleanly to release VRAM every EPISODES_PER_BATCH episodes
        if episodes_processed_this_run >= EPISODES_PER_BATCH:
            with open(output_file, "wb") as f:
                pickle.dump(all_scores, f)
            print("\nBatch limit reached. Exiting cleanly to clear VRAM!")
            sys.exit(3)

    # Final save
    with open(output_file, "wb") as f:
        pickle.dump(all_scores, f)
        
    print(f"All 2044 episodes evaluated N times successfully and saved to {output_file}!")
    sys.exit(0)

if __name__ == "__main__":
    main()
