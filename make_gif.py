import os
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont

def make_gif(episode_dir, output_path, fps=2):
    """
    Reads an episode directory and stitches X_draw.png, X_semantic.png, and X_map.png
    into a side-by-side grid, then saves as a GIF.
    """
    # Find all png files to determine max steps
    png_files = glob.glob(os.path.join(episode_dir, "[0-9]*.png"))
    if not png_files:
        print(f"No png files found in {episode_dir}")
        return
        
    # Extract step integers
    steps = set()
    for f in png_files:
        basename = os.path.basename(f)
        # Get prefix integer (e.g. '12' from '12_map.png')
        step_str = basename.split('_')[0].split('.')[0]
        if step_str.isdigit():
            steps.add(int(step_str))
            
    if not steps:
        print("Could not determine steps.")
        return
        
    max_steps = max(steps)
    print(f"Found {max_steps + 1} steps in {episode_dir}")
    
    frames = []
    
    # Pre-determine the target width for each column to keep layout perfectly fixed
    target_height = 512
    def get_target_width(pattern):
        files = glob.glob(os.path.join(episode_dir, pattern))
        if files:
            with Image.open(files[0]) as sample:
                aspect = sample.width / sample.height
                return int(target_height * aspect)
        return target_height # Fallback if none exist in entire episode

    width_rgb = get_target_width("*_draw.png") or get_target_width("*.png")
    width_sem = get_target_width("*_semantic.png")
    width_map = get_target_width("*_map.png")
    total_width = width_rgb + width_sem + width_map

    for i in range(max_steps + 1):
        rgb_path = os.path.join(episode_dir, f"{i}_draw.png")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(episode_dir, f"{i}_prompt_points.png")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(episode_dir, f"{i}.png")
            
        semantic_path = os.path.join(episode_dir, f"{i}_semantic.png")
        map_path = os.path.join(episode_dir, f"{i}_map.png")
        
        try:
            # Helper to load and resize, or return black box of exact correct size
            def load_and_resize(path, target_w):
                if os.path.exists(path):
                    img = Image.open(path).convert('RGB')
                    return img.resize((target_w, target_height), Image.Resampling.LANCZOS)
                return Image.new('RGB', (target_w, target_height), color='black')

            img_rgb = load_and_resize(rgb_path, width_rgb)
            img_sem = load_and_resize(semantic_path, width_sem)
            img_map = load_and_resize(map_path, width_map)
            
            # Stitch side-by-side into fixed layout
            combined = Image.new('RGB', (total_width, target_height))
            combined.paste(img_rgb, (0, 0))
            combined.paste(img_sem, (width_rgb, 0))
            combined.paste(img_map, (width_rgb + width_sem, 0))
            
            # Add a small step counter overlay
            draw = ImageDraw.Draw(combined)
            draw.text((10, 10), f"Step: {i}", fill="red")
            
            frames.append(combined)
            print(f"Processed step {i}")
        except Exception as e:
            print(f"Error processing step {i}: {e}")
            
    if frames:
        print(f"Saving GIF to {output_path}...")
        # Save as looping GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(args.duration * 1000), # Duration in ms per frame
            loop=0
        )
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GIF from EXPRESS-Bench episode results.")
    parser.add_argument("episode_dir", type=str, help="Path to the episode directory (e.g., experiment_result/0)")
    parser.add_argument("--output", type=str, default="episode_rollout.gif", help="Output GIF filename")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of each frame in seconds")
    args = parser.parse_args()
    
    make_gif(args.episode_dir, args.output, args.fps)
