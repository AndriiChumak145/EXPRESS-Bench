#!/usr/bin/env python3
"""
EXPRESS-Bench Video Generator
Generates side-by-side visualization videos and step frames for EXPRESS-Bench (Fine-EQA) episodes.

Layout:
  - Left Hero Panel: <step>_draw.png / <step>.png (Egocentric Camera View with A, B, C Prompt Points)
  - Right Main Panel: <step>_semantic.png (Semantic Region Map & Score Figure)
  - Right Reasoning Column (optional): Candidate Frontiers & Scores (LSV/GSV/SV) + Predicted Region
  - Bottom Banner: Question, Ground Truth Answer, Predicted Model Answer / Exploration Status
  - Saved Frames: Saved to EXPRESS-Bench/videos/<qid>/frames/step_XX.png
"""

import os
import sys
import glob
import re
import pickle
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw_obj: ImageDraw.ImageDraw) -> List[str]:
    """Wraps text into multiple lines given pixel max_width."""
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    
    for word in words:
        test_line = " ".join(current_line + [word])
        try:
            bbox = draw_obj.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
        except AttributeError:
            w, _ = draw_obj.textsize(test_line, font=font)
            
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def parse_fine_eqa_logs(episode_dir: str, results_root: Optional[str] = None) -> Tuple[Dict[int, Dict[str, str]], Dict[int, str]]:
    """
    Parses Fine-EQA / EXPRESS-Bench logs to extract per-step scores (LSV, GSV, SV),
    chosen frontier point letters (A, B, C), and predicted spatial regions.
    """
    scores_per_step: Dict[int, Dict[str, str]] = {}
    region_per_step: Dict[int, str] = {}
    
    search_dirs = [episode_dir]
    parent_dir = os.path.dirname(episode_dir)
    if parent_dir and os.path.exists(parent_dir):
        search_dirs.append(parent_dir)
    if results_root and os.path.exists(results_root):
        search_dirs.append(results_root)
        
    log_files: List[str] = []
    for d in search_dirs:
        log_files.extend(glob.glob(os.path.join(d, "*.log")))
        
    log_files = sorted(list(set(log_files)))

    for lfile in log_files:
        try:
            with open(lfile, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            lsv_matches = re.findall(r'Exp - LSV:\s*(\[.*?\]|[\d\.\s]+)\s*GSV:\s*([\d\.]+)\s*SV:\s*(\[.*?\]|[\d\.\s]+)', content)
            pt_matches = re.findall(r'Predicted point:\s*([A-D])', content)
            
            for step_idx, (lsv, gsv, sv) in enumerate(lsv_matches):
                chosen_pt = pt_matches[step_idx] if step_idx < len(pt_matches) else "N/A"
                
                # Format LSV / SV lists cleanly with A, B, C labels
                def format_score_str(raw_str: str) -> str:
                    cleaned = raw_str.replace('[', '').replace(']', '').strip()
                    parts = cleaned.split()
                    if len(parts) >= 3:
                        return f"A:{parts[0]}  B:{parts[1]}  C:{parts[2]}"
                    return cleaned

                scores_per_step[step_idx] = {
                    "lsv": format_score_str(str(lsv)),
                    "gsv": str(gsv).strip(),
                    "sv": format_score_str(str(sv)),
                    "chosen_pt": chosen_pt
                }
                
            reg_matches = re.findall(r'Predicted region:\s*(.*?),\s*Confidence:\s*([\d\.]+)', content)
            for s_idx, (reg, conf) in enumerate(reg_matches):
                try:
                    conf_val = float(conf)
                    region_per_step[s_idx] = f"{reg} (conf: {conf_val:.2f})"
                except ValueError:
                    region_per_step[s_idx] = f"{reg} (conf: {conf})"
        except OSError as err:
            logging.warning(f"Could not read log file {lfile}: {err}")

    return scores_per_step, region_per_step


def load_fine_eqa_metadata(episode_dir: str) -> Tuple[str, str, str, int]:
    """
    Loads question, ground_truth answer, predicted answer, and step count
    from the episode's result.pkl file.
    """
    question = "Unknown Question"
    expected_answer = "N/A"
    model_answer = "Model answer unavailable."
    cnt_step = 0
    
    pkl_path = os.path.join(episode_dir, "result.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    question = data.get("question", question)
                    expected_answer = data.get("answer", expected_answer)
                    model_answer = data.get("gen_answer", model_answer)
                    cnt_step = data.get("cnt_step", cnt_step)
        except (pickle.UnpicklingError, OSError, AttributeError) as err:
            logging.warning(f"Error reading result.pkl in {episode_dir}: {err}")
    else:
        logging.warning(f"result.pkl not found in {episode_dir}. Checking fallback JSON metadata.")
        json_files = glob.glob(os.path.join(episode_dir, "*_answer.json"))
        if json_files:
            try:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    import json
                    jdata = json.load(f)
                    question = jdata.get("question", question)
                    expected_answer = jdata.get("ground_truth", jdata.get("answer", expected_answer))
                    model_answer = jdata.get("prediction", jdata.get("pred", model_answer))
            except (json.JSONDecodeError, OSError) as err:
                logging.warning(f"Error reading answer JSON in {episode_dir}: {err}")

    return question, expected_answer, model_answer, cnt_step


def draw_fine_eqa_reasoning_panel(
    draw: ImageDraw.ImageDraw,
    step: int,
    scores_info: Dict[str, str],
    region_info: str,
    main_canvas_w: int,
    panel_w: int,
    canvas_h: int,
    font_title: ImageFont.ImageFont,
    font_text: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont
) -> None:
    """Renders the right-hand reasoning and scores column."""
    px = main_canvas_w
    draw.rectangle([px, 0, px + panel_w, canvas_h], fill=(15, 18, 24))
    draw.line([(px, 0), (px, canvas_h)], fill=(0, 255, 120), width=3)
    
    card_w = panel_w - 40
    card_x1 = px + 20
    
    # Header Banner
    draw.rectangle([card_x1, 15, card_x1 + card_w, 55], fill=(30, 35, 45), outline=(0, 255, 120), width=2)
    draw.text((card_x1 + 15, 23), f"STEP {step} REASONING & SCORES", fill=(0, 255, 120), font=font_title)
    
    # 1. Candidate Frontiers & Scores Card
    card1_y1 = 70
    card1_y2 = 330
    draw.rectangle([card_x1, card1_y1, card_x1 + card_w, card1_y2], fill=(22, 26, 35), outline=(60, 70, 90), width=2)
    draw.rectangle([card_x1, card1_y1, card_x1 + card_w, card1_y1 + 35], fill=(35, 42, 55))
    draw.text((card_x1 + 15, card1_y1 + 8), "★ Candidate Frontiers & Scores", fill=(255, 215, 0), font=font_title)
    
    cy = card1_y1 + 45
    if scores_info:
        draw.text((card_x1 + 15, cy), f"LSV (Local Scores):  {scores_info.get('lsv', 'N/A')}", fill=(200, 220, 255), font=font_small)
        cy += 30
        draw.text((card_x1 + 15, cy), f"GSV (Global Score): {scores_info.get('gsv', 'N/A')}", fill=(200, 220, 255), font=font_small)
        cy += 30
        draw.text((card_x1 + 15, cy), f"SV (Combined Score): {scores_info.get('sv', 'N/A')}", fill=(255, 215, 0), font=font_small)
        cy += 35
        chosen = scores_info.get('chosen_pt', 'N/A')
        draw.text((card_x1 + 15, cy), f"★ Chosen Action: Frontier {chosen}", fill=(0, 255, 120), font=font_title)
        cy += 35
        draw.text((card_x1 + 15, cy), "Frontier Options A, B, C match array index order 0, 1, 2.", fill=(160, 170, 185), font=font_small)
    else:
        draw.text((card_x1 + 15, cy), "Frontier evaluation: Evaluating visual prompt points A, B, C...", fill=(160, 170, 185), font=font_small)
        
    # 2. Predicted Spatial Region Card
    card2_y1 = 340
    card2_y2 = 550
    draw.rectangle([card_x1, card2_y1, card_x1 + card_w, card2_y2], fill=(22, 26, 35), outline=(60, 70, 90), width=2)
    draw.rectangle([card_x1, card2_y1, card_x1 + card_w, card2_y1 + 35], fill=(35, 42, 55))
    draw.text((card_x1 + 15, card2_y1 + 8), "📍 Predicted Spatial Region", fill=(0, 200, 255), font=font_title)
    
    ry = card2_y1 + 50
    if region_info:
        draw.text((card_x1 + 15, ry), f"Region: {region_info}", fill=(255, 255, 255), font=font_text)
    else:
        draw.text((card_x1 + 15, ry), "Region prediction: Integrating TSDF region semantics...", fill=(160, 170, 185), font=font_small)

    # 3. Method Info Card
    card3_y1 = 570
    card3_y2 = 880
    draw.rectangle([card_x1, card3_y1, card_x1 + card_w, card3_y2], fill=(22, 26, 35), outline=(60, 70, 90), width=2)
    draw.rectangle([card_x1, card3_y1, card_x1 + card_w, card3_y1 + 35], fill=(35, 42, 55))
    draw.text((card_x1 + 15, card3_y1 + 8), "⚙ Fine-EQA Method Context", fill=(200, 255, 220), font=font_title)
    
    my = card3_y1 + 50
    draw.text((card_x1 + 15, my), "• Evaluates candidate prompt points (A, B, C) directly on image", fill=(180, 190, 210), font=font_small)
    my += 35
    draw.text((card_x1 + 15, my), "• Combines Local (LSV) + Global (GSV) direction scores", fill=(180, 190, 210), font=font_small)
    my += 35
    draw.text((card_x1 + 15, my), "• Integrates spatial region probabilities into 2D TSDF Map", fill=(180, 190, 210), font=font_small)


def create_express_bench_video(
    episode_dir: str,
    output_video_path: Optional[str] = None,
    fps: float = 2.0,
    show_reasoning: bool = True
) -> bool:
    """
    Creates a step-by-step MP4 video and saves step frame PNGs for an EXPRESS-Bench episode.
    """
    if not os.path.exists(episode_dir):
        logging.error(f"Episode directory {episode_dir} does not exist.")
        return False
        
    png_files = glob.glob(os.path.join(episode_dir, "[0-9]*.png"))
    if not png_files:
        logging.error(f"No step PNG files found in {episode_dir}")
        return False
        
    steps: set = set()
    for f in png_files:
        base = os.path.basename(f)
        prefix = base.split('_')[0].split('.')[0]
        if prefix.isdigit():
            steps.add(int(prefix))
            
    if not steps:
        logging.error(f"Could not extract step numbers from {episode_dir}")
        return False
        
    sorted_steps = sorted(list(steps))
    num_steps = len(sorted_steps)
    
    question, expected_answer, model_answer, cnt_step = load_fine_eqa_metadata(episode_dir)
    scores_per_step, region_per_step = parse_fine_eqa_logs(episode_dir)
    
    banner_h = 180
    main_canvas_w = 1920
    panel_w = 640 if show_reasoning else 0
    canvas_w = main_canvas_w + panel_w
    canvas_h = 1080
    img_area_h = canvas_h - banner_h # 900px
    
    hero_w = 900
    hero_h = 880
    
    grid_w = 900
    grid_h = 880
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_path):
        font_title = ImageFont.truetype(font_path, 22)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_small = ImageFont.truetype(font_path, 16)
    else:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if output_video_path is None:
        output_video_path = os.path.join(episode_dir, "video.mp4")
    
    output_dir = os.path.dirname(output_video_path)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (canvas_w, canvas_h))
    
    if not video_writer.isOpened():
        logging.error(f"Failed to open VideoWriter for path {output_video_path}")
        return False

    logging.info(f"Generating EXPRESS-Bench Video for {episode_dir} ({num_steps} steps) -> {output_video_path}...")

    try:
        for idx, step_num in enumerate(sorted_steps):
            canvas = Image.new("RGB", (canvas_w, canvas_h), color=(25, 28, 35))
            draw = ImageDraw.Draw(canvas)
            
            is_final_step = (idx == num_steps - 1)
            
            # 1. Left Hero Panel: Egocentric Camera View (<step>_draw.png / <step>.png)
            draw_path = os.path.join(episode_dir, f"{step_num}_draw.png")
            if not os.path.exists(draw_path):
                draw_path = os.path.join(episode_dir, f"{step_num}.png")
            if not os.path.exists(draw_path):
                draw_path = os.path.join(episode_dir, f"step_{step_num:03d}.png")
                
            has_exact_hero = os.path.exists(draw_path)
            used_hero_step = step_num
            
            if not has_exact_hero:
                # Fallback to latest available camera view image in episode_dir
                for prev_s in reversed(sorted_steps):
                    for cand_name in [f"{prev_s}_draw.png", f"{prev_s}.png", f"step_{prev_s:03d}.png"]:
                        cand_path = os.path.join(episode_dir, cand_name)
                        if os.path.exists(cand_path):
                            draw_path = cand_path
                            used_hero_step = prev_s
                            break
                    if os.path.exists(draw_path):
                        break
                        
            if os.path.exists(draw_path):
                with Image.open(draw_path) as d_img:
                    hero_img = d_img.convert("RGB")
            else:
                hero_img = Image.new("RGB", (hero_w, hero_h), color=(40, 44, 52))
                hdraw = ImageDraw.Draw(hero_img)
                hdraw.text((200, 400), f"Step {step_num} Visual View", fill=(180, 180, 180), font=font_title)

            hero_copy = hero_img.copy()
            hero_copy.thumbnail((hero_w, hero_h), Image.Resampling.LANCZOS)
            
            hx = (920 - hero_copy.width) // 2
            hy = (img_area_h - hero_copy.height) // 2
            canvas.paste(hero_copy, (hx, hy))
            
            if is_final_step:
                if has_exact_hero:
                    hero_title_text = f"FINAL ANSWER STEP (Step {step_num})"
                else:
                    hero_title_text = f"FINAL ANSWER STEP (View from Step {used_hero_step})"
                hero_color = (200, 150, 0)
                hero_outline = (255, 215, 0)
            else:
                hero_title_text = f"STEP {step_num}: Visual View & Frontiers (A, B, C)"
                hero_color = (0, 160, 75)
                hero_outline = (0, 255, 120)
            
            draw.rectangle([hx, hy, hx + hero_copy.width, hy + hero_copy.height], outline=hero_outline, width=4)
            draw.rectangle([hx, hy, hx + 440, hy + 28], fill=hero_color)
            draw.text((hx + 10, hy + 5), hero_title_text, fill=(255, 255, 255), font=font_small)

            # 2. Right Main Panel: Semantic Region Map (<step>_semantic.png)
            sem_path = os.path.join(episode_dir, f"{step_num}_semantic.png")
            has_exact_sem = os.path.exists(sem_path)
            
            if not has_exact_sem:
                # Find latest available semantic png or map png
                all_sems = sorted(glob.glob(os.path.join(episode_dir, "*_semantic.png")))
                if not all_sems:
                    all_sems = sorted(glob.glob(os.path.join(episode_dir, "*_map.png")))
                if all_sems:
                    sem_path = all_sems[-1]
                    
            if os.path.exists(sem_path):
                with Image.open(sem_path) as s_img:
                    sem_img = s_img.convert("RGB")
            else:
                sem_img = Image.new("RGB", (grid_w, grid_h), color=(40, 44, 52))
                sdraw = ImageDraw.Draw(sem_img)
                sdraw.text((200, 400), f"Step {step_num} Semantic Figure", fill=(180, 180, 180), font=font_title)

            sem_copy = sem_img.copy()
            sem_copy.thumbnail((grid_w, grid_h), Image.Resampling.LANCZOS)
            
            mx = 940 + (grid_w - sem_copy.width) // 2
            my = (img_area_h - sem_copy.height) // 2
            
            cell_bg = Image.new("RGB", (sem_copy.width, sem_copy.height), color=(255, 255, 255))
            cell_bg.paste(sem_copy, (0, 0))
            canvas.paste(cell_bg, (mx, my))
            
            sem_outline = (0, 200, 255) if has_exact_sem else (255, 165, 0)
            if not has_exact_sem and is_final_step:
                slot3_title = f"★ FINAL STEP: Exploration Terminated (Last Map State)"
                banner_color = (200, 120, 0)
                banner_w = 480
            else:
                slot3_title = f"Semantic Map & Scores (Step {step_num})"
                banner_color = (0, 0, 0, 200)
                banner_w = 320
                
            draw.rectangle([mx, my, mx + sem_copy.width, my + sem_copy.height], outline=sem_outline, width=3)
            draw.rectangle([mx, my, mx + banner_w, my + 25], fill=banner_color)
            draw.text((mx + 8, my + 4), slot3_title, fill=(255, 255, 255), font=font_small)

            # 3. Bottom Overlay Banner
            draw.rectangle([0, img_area_h, main_canvas_w, canvas_h], fill=(18, 22, 28))
            draw.line([(0, img_area_h), (main_canvas_w, img_area_h)], fill=(0, 255, 120), width=3)

            draw.text((30, img_area_h + 20), "Question: ", fill=(255, 215, 0), font=font_title)
            draw.text((170, img_area_h + 22), question, fill=(255, 255, 255), font=font_text)

            draw.text((30, img_area_h + 70), "Expected GT: ", fill=(0, 230, 120), font=font_title)
            draw.text((200, img_area_h + 72), expected_answer, fill=(200, 255, 220), font=font_text)

            if is_final_step:
                # Detect if model was forced to answer at step limit vs voluntarily stopped
                is_forced_answer = (num_steps > 3)
                ans_label = "Model Response (FORCED AT STEP LIMIT): " if is_forced_answer else "Model Response (Voluntary Stop): "
                ans_color = (255, 80, 180) if is_forced_answer else (0, 200, 255)
                
                draw.text((30, img_area_h + 120), ans_label, fill=ans_color, font=font_title)
                
                # Dynamically measure exact text width to avoid any overlap
                try:
                    bbox = draw.textbbox((0, 0), ans_label, font=font_title)
                    label_w = bbox[2] - bbox[0]
                except AttributeError:
                    label_w, _ = draw.textsize(ans_label, font=font_title)
                    
                draw.text((30 + label_w + 10, img_area_h + 122), model_answer, fill=(255, 255, 255), font=font_text)
            else:
                draw.text((30, img_area_h + 120), "Status: ", fill=(255, 165, 0), font=font_title)
                draw.text((130, img_area_h + 122), f"Exploring... (Step {idx+1}/{num_steps})", fill=(200, 200, 200), font=font_text)

            # 4. Right-Hand Reasoning Panel (if enabled)
            if show_reasoning:
                scores_info = scores_per_step.get(idx, {})
                region_info = region_per_step.get(idx, "")
                draw_fine_eqa_reasoning_panel(
                    draw, step_num, scores_info, region_info,
                    main_canvas_w, panel_w, canvas_h,
                    font_title, font_text, font_small
                )

            # Save frame image under frames/step_XX.png
            frame_path = os.path.join(frames_dir, f"step_{idx:02d}.png")
            canvas.save(frame_path)

            cv_img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
            
            num_repeats = 2
            for _ in range(num_repeats):
                video_writer.write(cv_img)
                
            if is_final_step:
                for _ in range(8): # 4 seconds extra buffer at 2fps
                    video_writer.write(cv_img)
    finally:
        video_writer.release()
        
    logging.info(f"Successfully generated video: {output_video_path}")
    logging.info(f"Saved step frames to: {frames_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MP4 Video for Fine-EQA / EXPRESS-Bench Episodes")
    parser.add_argument("--episode-dir", type=str, default=None, help="Direct path to episode directory (e.g. EXPRESS-Bench/experiment_result/0)")
    parser.add_argument("--question-id", type=str, default=None, help="Question ID to search (e.g. 0, 100, 109)")
    parser.add_argument("--results-root", type=str, default="EXPRESS-Bench/experiment_result", help="Root directory to search when --question-id is specified")
    parser.add_argument("--output-video", type=str, default=None, help="Output MP4 video path")
    parser.add_argument("--fps", type=float, default=2.0, help="Video framerate (default: 2.0 FPS)")
    parser.add_argument("--no-reasoning", action="store_false", dest="show_reasoning", help="Disable right-hand reasoning panel")
    args = parser.parse_args()

    ep_dir = args.episode_dir

    if ep_dir is None and args.question_id:
        logging.info(f"Searching for question ID {args.question_id} under {args.results_root}...")
        matches = glob.glob(os.path.join(args.results_root, f"**/{args.question_id}"), recursive=True)
        if matches:
            ep_dir = matches[0]
            logging.info(f"Found episode directory: {ep_dir}")
        else:
            logging.error(f"Could not find directory for question ID {args.question_id} under {args.results_root}")
            sys.exit(1)

    if not ep_dir:
        logging.error("Must specify either --episode-dir or --question-id.")
        sys.exit(1)

    create_express_bench_video(
        episode_dir=ep_dir,
        output_video_path=args.output_video,
        fps=args.fps,
        show_reasoning=args.show_reasoning
    )
