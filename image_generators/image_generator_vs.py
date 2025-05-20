import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict, Any
from collections import Counter


SHAPE_TYPES = ['circle', 'square', 'triangle']
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'magenta', 'pink', "cyan", "gray", "lime"]

def generate_trial_image(
    trial_type: str,
    grid_size: int,
    cell_size: int
) -> Tuple[Image.Image, int, str, Dict[str, int]]:
    """
    Generates an image for basic visual attention trials.
    trial_type ∈ {color, shape, color_of_number, shape_of_number}

    Returns:
        img (PIL.Image): The generated image.
        target_number (int): The number in the grid cell that is the target or is queried.
        target_feature (str): The color or shape feature that is the target or is queried.
        color_counts (Dict[str, int]): Counts of each color in the grid.
    """
    total_cells = grid_size * grid_size
    target_idx_flat = random.randrange(total_cells)
    target_row, target_col = divmod(target_idx_flat, grid_size)
    target_number_on_grid = target_idx_flat + 1

    grid_shapes: List[List[str]] = [[random.choice(SHAPE_TYPES) for _ in range(grid_size)] for _ in range(grid_size)]
    grid_colors: List[List[str]] = [[random.choice(COLORS) for _ in range(grid_size)] for _ in range(grid_size)]
    
    actual_target_feature = ""

    img = Image.new('RGB', (grid_size * cell_size, grid_size * cell_size), 'white')
    draw = ImageDraw.Draw(img)

    if trial_type == "color": # Find number on target color
        actual_target_feature = random.choice(COLORS)
        grid_colors[target_row][target_col] = actual_target_feature
        other_colors = [c for c in COLORS if c != actual_target_feature]
        if not other_colors: other_colors = [actual_target_feature] # Fallback if only one color

        for r in range(grid_size):
            for c in range(grid_size):
                if r == target_row and c == target_col:
                    pass # Already set
                else:
                    grid_colors[r][c] = random.choice(other_colors)
        # Shapes are random for all cells
        grid_shapes = [[random.choice(SHAPE_TYPES) for _ in range(grid_size)] for _ in range(grid_size)]

    elif trial_type == "shape": # Find number on target shape
        actual_target_feature = random.choice(SHAPE_TYPES)
        grid_shapes[target_row][target_col] = actual_target_feature
        other_shapes = [s for s in SHAPE_TYPES if s != actual_target_feature]
        if not other_shapes: other_shapes = [actual_target_feature]

        for r in range(grid_size):
            for c in range(grid_size):
                if r == target_row and c == target_col:
                    pass
                else:
                    grid_shapes[r][c] = random.choice(other_shapes)
        # Colors are random for all cells
        grid_colors = [[random.choice(COLORS) for _ in range(grid_size)] for _ in range(grid_size)]
    # some of these are not used, but are maintained here for ease and laziness reasons
    elif trial_type == "color_of_number":
        actual_target_feature = grid_colors[target_row][target_col]

    elif trial_type == "shape_of_number": 
        actual_target_feature = grid_shapes[target_row][target_col]
    else:
        raise ValueError(f"Unknown trial_type: {trial_type}")

    for r_idx in range(grid_size):
        for c_idx in range(grid_size):
            x_center, y_center = c_idx * cell_size + cell_size // 2, r_idx * cell_size + cell_size // 2
            shape_to_draw = grid_shapes[r_idx][c_idx]
            color_to_draw = grid_colors[r_idx][c_idx]
            
            size_factor = cell_size * 0.3 # Radius or half-side
            
            if shape_to_draw == 'circle':
                draw.ellipse([x_center - size_factor, y_center - size_factor, 
                              x_center + size_factor, y_center + size_factor], fill=color_to_draw)
            elif shape_to_draw == 'square':
                draw.rectangle([x_center - size_factor, y_center - size_factor, 
                                x_center + size_factor, y_center + size_factor], fill=color_to_draw)
            elif shape_to_draw == 'triangle': 
                p1 = (x_center, y_center - size_factor)
                p2 = (x_center - size_factor * math.sqrt(3)/2, y_center + size_factor/2)
                p3 = (x_center + size_factor * math.sqrt(3)/2, y_center + size_factor/2)
                draw.polygon([p1, p2, p3], fill=color_to_draw)


    font = ImageFont.load_default(size=max(10,int(cell_size*0.2))) # Dynamic font size
    for r_idx in range(grid_size):
        for c_idx in range(grid_size):
            num_label = str(r_idx * grid_size + c_idx + 1)
            x_center, y_center = c_idx * cell_size + cell_size // 2, r_idx * cell_size + cell_size // 2
            
            text_bbox = draw.textbbox((0,0), num_label, font=font) # Pillow >= 10 needs (x,y) in bbox
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            draw.text((x_center - text_w / 2, y_center - text_h / 2), num_label, fill='black', font=font)

    flat_colors_list = [color for row_of_colors in grid_colors for color in row_of_colors]
    current_color_counts = Counter(flat_colors_list)
    
    return img, target_number_on_grid, actual_target_feature, current_color_counts


def generate_chain_image(
    grid_size: int,
    cell_size: int,
    chain_length: int
) -> Tuple[Image.Image, Tuple[str, str], str, List[Tuple[str,str]], Dict[str, int]]:
    """
    Generates a chain-following puzzle image.
    Each cell has a unique (shape, color) pair.
    Labels guide the chain.

    Returns:
        img (PIL.Image): The rendered grid image.
        start_pair (Tuple[str, str]): The (shape, color) of the starting cell.
        final_color (str): The color of the shape in the final cell of the chain.
        chain_path (List[Tuple[str,str]]): The sequence of (shape, color) pairs forming the chain.
        color_counts (Dict[str, int]): Counts of each color across the entire grid.
    """
    all_possible_pairs = [(s, c) for s in SHAPE_TYPES for c in COLORS]
    num_cells_needed = grid_size * grid_size
    if num_cells_needed > len(all_possible_pairs):
        raise ValueError(f"Grid size {grid_size}x{grid_size} needs {num_cells_needed} unique pairs, but only {len(all_possible_pairs)} are available with current SHAPE_TYPES and COLORS.")

    sampled_pairs = random.sample(all_possible_pairs, num_cells_needed)
    grid_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    random.shuffle(grid_positions) # randomize

    # Mappings: cell_pos -> (shape, color) and (shape, color) -> cell_pos
    cell_to_pair_map: Dict[Tuple[int,int], Tuple[str,str]] = {pos: pair for pos, pair in zip(grid_positions, sampled_pairs)}
    pair_to_cell_map: Dict[Tuple[str,str], Tuple[int,int]] = {pair: pos for pos, pair in cell_to_pair_map.items()}
    
    overall_color_counts = Counter(pair[1] for pair in cell_to_pair_map.values())

    img = Image.new('RGB', (grid_size * cell_size, grid_size * cell_size), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(10,int(cell_size*0.15))) # Smaller font for labels

    for (r, c), (shape, color) in cell_to_pair_map.items():
        x_center, y_center = c * cell_size + cell_size // 2, r * cell_size + cell_size // 2
        size_factor = cell_size * 0.35 # Radius or half-side for shapes
        if shape == 'circle':
            draw.ellipse([x_center - size_factor, y_center - size_factor, x_center + size_factor, y_center + size_factor], fill=color)
        elif shape == 'square':
            draw.rectangle([x_center - size_factor, y_center - size_factor, x_center + size_factor, y_center + size_factor], fill=color)
        elif shape == 'triangle':
            p1 = (x_center, y_center - size_factor)
            p2 = (x_center - size_factor * math.sqrt(3)/2, y_center + size_factor/2)
            p3 = (x_center + size_factor * math.sqrt(3)/2, y_center + size_factor/2)
            draw.polygon([p1, p2, p3], fill=color)
    if chain_length + 1 > len(sampled_pairs):
        raise ValueError(f"Chain length {chain_length} + 1 is too long for the number of unique pairs ({len(sampled_pairs)}) in the grid.")
    
    actual_chain_path = random.sample(sampled_pairs, chain_length + 1)
    start_chain_pair = actual_chain_path[0]
    final_chain_color = actual_chain_path[-1][1] 
    for i in range(chain_length): # Up to the second to last element of the chain
        current_cell_pair = actual_chain_path[i]
        next_cell_pair_in_chain = actual_chain_path[i+1] 
        
        (r_curr, c_curr) = pair_to_cell_map[current_cell_pair]
        label_text = f"{next_cell_pair_in_chain[1]} {next_cell_pair_in_chain[0]}" # "Color Shape"

        x_text_center, y_text_center = c_curr * cell_size + cell_size // 2, r_curr * cell_size + cell_size // 2
        text_bbox = draw.textbbox((0,0), label_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.text((x_text_center - text_w / 2, y_text_center - text_h / 2), label_text, fill='black', font=font)

    cells_with_guiding_labels_pairs = actual_chain_path[:chain_length]

    for (r_cell, c_cell), pair_in_cell in cell_to_pair_map.items():
        if pair_in_cell in cells_with_guiding_labels_pairs:
            continue
        dummy_target_shape, dummy_target_color = random.choice(all_possible_pairs)
        distractor_label_text = f"{dummy_target_color} {dummy_target_shape}"
        
        x_text_center, y_text_center = c_cell * cell_size + cell_size // 2, r_cell * cell_size + cell_size // 2
        text_bbox = draw.textbbox((0,0), distractor_label_text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.text((x_text_center - text_w / 2, y_text_center - text_h / 2), distractor_label_text, fill='black', font=font)

    return img, start_chain_pair, final_chain_color, actual_chain_path, overall_color_counts

def generate_chain_description_string(num_steps: int) -> str:
    """Generates an example descriptive string for a chain task instance."""
    if not 1 <= num_steps <= 5: 
        num_steps = 3 
    example_items = [
        ("blue", "triangle"), ("red", "square"), ("blue", "circle"),
        ("magenta", "triangle"), ("green", "circle"), ("purple", "square")
    ]
    
    path_parts = [f"you might start at a {example_items[0][0]} {example_items[0][1]}"]
    for i in range(1, num_steps + 1):
        if i < len(example_items):
            path_parts.append(f"then go to a {example_items[i][0]} {example_items[i][1]}")
        else: # Should not happen if num_steps is within bounds of example_items
            path_parts.append("then to another item") 
            
    path_description = ", ".join(path_parts)
    steps_str = f"{num_steps} step{'s' if num_steps != 1 else ''}"
    
    final_color_example = example_items[num_steps][0] if num_steps < len(example_items) else "unknown"
    return f"{steps_str}, {path_description}. The answer would be {{{final_color_example}}}."


def generate_two_shot_examples(
    grid_size: int,
    cell_size: int,
    chain_length: int,
    odir: str = "fs_examples_visual_attention" # Specific dir name
) -> List[Tuple[Image.Image, str, str]]:
    """
    Builds two few-shot examples specifically for the 'chain' trial.
    """
    os.makedirs(odir, exist_ok=True)
    examples = []
    for i in range(2): 
        img, start_pair, final_color_answer, full_chain_path, _ = generate_chain_image(
            grid_size, cell_size, chain_length
        )
        

        prompt_text = (
            f"Starting at the {start_pair[1]} {start_pair[0]} (this is cell 0 of the chain), "
            f"follow the labels for {len(full_chain_path)-1} steps. "
        )

        path_description_parts = []
        for step_idx in range(len(full_chain_path) - 1):
            next_shape_in_chain = full_chain_path[step_idx+1][0]
            next_color_in_chain = full_chain_path[step_idx+1][1]
            if step_idx == 0:
                 path_description_parts.append(f"The label in the start cell points to the {next_color_in_chain} {next_shape_in_chain} (cell 1).")
            else:
                 path_description_parts.append(f"From there, the label points to the {next_color_in_chain} {next_shape_in_chain} (cell {step_idx+1}).")
        
        prompt_text += " ".join(path_description_parts)
        prompt_text += (
            f" After these steps, what is the color of the shape in the final cell (cell {len(full_chain_path)-1})? "
            f"Answer with only the color name in curly braces, e.g. {{red}}."
        )
        
        # The answer is just the final color
        answer_text = f"{{{final_color_answer}}}"
        
        img.save(os.path.join(odir, f"two_shot_chain_example_{i}.png"))
        examples.append((img, prompt_text, answer_text))
        
    return examples