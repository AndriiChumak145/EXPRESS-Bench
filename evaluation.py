import numpy as np
import re
import math


def score(results):
    C, C_star, p_path, l_path, d_T = [], [], [], [], []
    for result in results:
        if result["path_len"] != float("inf"):
            EAC = result["EAC"].replace("Your mark:", "").strip()
            try:
                # To handle models that hallucinate text and put the score at the end:
                # take the last non-empty line
                lines = [line.strip() for line in EAC.split('\n') if line.strip()]
                if not lines:
                    raise ValueError("Empty output")
                last_line = lines[-1]
                
                match = re.search(r'([0-9](?:\.[0-9]+)?)\s*[,/]\s*([0-9])', last_line)
                if match:
                    grd = float(match.group(1))
                    acc = int(match.group(2))
                else:
                    print(f"WARNING: Could not extract score via regex for episode. Output was:\n{last_line}\nAssigning NaN.")
                    grd = float('nan')
                    acc = float('nan')
            except Exception as e:
                print(f"CRITICAL: Failed to parse EAC format. Assigning NaN. Error: {e}")
                grd = float('nan')
                acc = float('nan')
            C.append(grd*acc)
            C_star.append(acc)
            p_path.append(result["path_len"])
            l_path.append(result["geodesic_distance"])
        if result["goal_dis"] != float("inf"):
            d_T.append(result["goal_dis"])

    weight_path = l_path / np.maximum(p_path, l_path)
    C_avg = np.mean(100.0 * (np.clip(C, 0, 5) / 5))
    C_star_avg = np.mean(100.0 * (np.clip(C_star, 0, 5) / 5))
    E_path = np.mean(100.0 * (np.clip(C, 0, 5) / 5) * weight_path)
    d_T_avg = np.mean(d_T)

    return C_avg, C_star_avg, E_path, d_T_avg

