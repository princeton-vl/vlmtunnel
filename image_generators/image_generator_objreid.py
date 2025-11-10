import os
import random
import math
import copy
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw
import numpy as np 

CANVAS_SIZE_DEFAULT = 512 
COLORS = ['red', 'green', 'blue', 'orange', 'purple', 'teal', 'magenta', 'gold']
SHAPES = ['circle', 'oval', 'square', 'rectangle', 'triangle', 'polygon', 'line']

GLOBAL_ROT_RANGE         = (-math.pi, math.pi)
GLOBAL_SCALE_RANGE       = (0.3, 2.0)
RESCALE_MIN              = 0.1
RESCALE_MAX              = 0.33
NO_MIN_TRANSLATE_PX      = 15
NO_MAX_TRANSLATE_PX      = 30
NO_MIN_ROT               = -math.pi / 5
NO_MAX_ROT               = math.pi / 5
JITTER_MIN_ROT           = math.radians(6)

PROB_TRANSLATE           = 0.15
PROB_ROTATE              = 0.15
PROB_SCALE               = 0.15
PROB_RESHAPE             = 0.05
NO_MIN_TRANSFORMS        = 1
NO_MAX_TRANSFORMS        = 2

def rotate_point(x: float, y: float, cx: float, cy: float, ang: float) -> Tuple[float, float]:
    dx, dy = x - cx, y - cy
    ca, sa = math.cos(ang), math.sin(ang)
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca

def compute_shape_radius(info: Dict[str, Any]) -> float:
    t = info['type']
    if t == 'circle': return info['size'] / 2.0
    if t in ('square', 'rectangle'):
        return math.hypot(info['width'] / 2.0, info['height'] / 2.0)
    if t == 'oval':
        # For ovals, account for potential rotation expansion
        w, h = info['width'] / 2.0, info['height'] / 2.0
        # When rotated with expand=True, diagonal becomes the bounding box
        # Plus the 2.5x scaling factor used in drawing (lines 60-62)
        base_radius = math.hypot(w, h)
        return base_radius * 1.5  # Conservative estimate for 2.5x temp image + rotation
    if t == 'triangle': return info['size'] * math.sqrt(3) / 3.0 
    if t == 'polygon': return info['size'] 
    if t == 'line': return info['size'] / 2.0
    return 0.0

def draw_composite(shapes: List[Dict[str, Any]], offs: List[Tuple[float, float]],
                   canvas_size: int, center: Tuple[int, int]) -> Image.Image:
    cx, cy = center
    img = Image.new('RGB', (canvas_size, canvas_size), 'white')
    d = ImageDraw.Draw(img)
    for info, (dx, dy) in sorted(zip(shapes, offs), key=lambda x_item: x_item[0]['z']):
        x0, y0 = cx + dx, cy + dy
        typ, col = info['type'], info['color']
        ang = info.get('angle', 0.0)

        if typ == 'circle':
            r = info['size'] / 2.0
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=col)
        elif typ == 'oval':
            w, h = info['width'] / 2.0, info['height'] / 2.0
            temp_img = Image.new("RGBA", (int(w*2.5), int(h*2.5)), (0,0,0,0))
            temp_draw = ImageDraw.Draw(temp_img)
            temp_draw.ellipse((w*0.25, h*0.25, w*2.25, h*2.25), fill=col)
            if ang != 0: temp_img = temp_img.rotate(-math.degrees(ang), expand=True, resample=Image.BICUBIC) # Negative for PIL rotate
            img.paste(temp_img, (int(x0 - temp_img.width/2), int(y0 - temp_img.height/2)), temp_img)
            
        elif typ in ('square', 'rectangle'):
            w, h = info['width'] / 2.0, info['height'] / 2.0
            pts = [(x0 - w, y0 - h), (x0 + w, y0 - h),
                   (x0 + w, y0 + h), (x0 - w, y0 + h)]
            if ang != 0: pts = [rotate_point(px, py, x0, y0, ang) for px, py in pts]
            d.polygon(pts, fill=col)
        elif typ == 'triangle': 
            s = info['size']
            ht = s * math.sqrt(3) / 2.0
            verts = [(x0, y0 - 2 / 3 * ht),
                     (x0 - s / 2, y0 + 1 / 3 * ht),
                     (x0 + s / 2, y0 + 1 / 3 * ht)]
            if ang != 0: verts = [rotate_point(px, py, x0, y0, ang) for px, py in verts]
            d.polygon(verts, fill=col)
        elif typ == 'polygon': 
            rad, n_sides = info['size'], info['n']
            current_angle = info.get('angle', 0.0) # Using the shape's own angle
            verts = [(x0 + rad * math.cos(current_angle + 2 * math.pi * i / n_sides), 
                      y0 + rad * math.sin(current_angle + 2 * math.pi * i / n_sides)) for i in range(n_sides)]
            d.polygon(verts, fill=col)
        elif typ == 'line':
            line_len = info['size'] / 2.0
            p1, p2 = (x0 - line_len, y0), (x0 + line_len, y0)
            if ang != 0:
                p1 = rotate_point(p1[0], p1[1], x0, y0, ang)
                p2 = rotate_point(p2[0], p2[1], x0, y0, ang)
            d.line([p1, p2], fill=col, width=max(1, int(info.get('size',10) / 10.0)))
    return img

def composite_bounds(shapes: List[Dict[str, Any]], offs: List[Tuple[float, float]]):
    all_x_coords, all_y_coords = [], []
    if not shapes: return 0.0,0.0,0.0,0.0
    for info, (dx, dy) in zip(shapes, offs):
        shape_extent = compute_shape_radius(info) 
        # This is an approximation. For rotated non-circular shapes, true bounds are complex; this ensures no intersections at the cost of some degrees of freedom
        all_x_coords.extend([dx - shape_extent, dx + shape_extent])
        all_y_coords.extend([dy - shape_extent, dy + shape_extent])
    if not all_x_coords : return 0.0,0.0,0.0,0.0
    return min(all_x_coords), max(all_x_coords), min(all_y_coords), max(all_y_coords)

def composite_fits_canvas(shapes: List[Dict[str, Any]], offs: List[Tuple[float, float]], canvas_size: int) -> bool:
    """Check if all shapes remain fully visible when drawn with center at (canvas_size//2, canvas_size//2)"""
    center_x = center_y = canvas_size // 2
    
    for shape, (dx, dy) in zip(shapes, offs):
        shape_radius = compute_shape_radius(shape)
        # Final drawing coordinates will be: (center_x + dx, center_y + dy)
        x0, y0 = center_x + dx, center_y + dy
        
        # Check if shape (with radius) stays within [0, canvas_size) bounds
        if (x0 - shape_radius < 0 or x0 + shape_radius >= canvas_size or
            y0 - shape_radius < 0 or y0 + shape_radius >= canvas_size):
            return False
    
    return True

def _is_connected_img(img: Image.Image) -> bool:
    np_img_lum = np.array(img.convert('L'))
    np_img_bool = np_img_lum < 250 
    if not np_img_bool.any(): return False
    rows, cols = np_img_bool.shape
    visited = np.zeros_like(np_img_bool, dtype=bool)
    
    # Find first foreground pixel using np.argwhere
    foreground_pixels = np.argwhere(np_img_bool)
    if foreground_pixels.size == 0: return False 
    start_node = tuple(foreground_pixels[0])

    q = [start_node] 
    visited[start_node] = True
    count_visited = 0
    while q:
        r, c = q.pop(0)
        count_visited += 1
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and \
               np_img_bool[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return count_visited == np_img_bool.sum()

def composite_connected(shapes: List[Dict[str, Any]], offs: List[Tuple[float, float]], canvas_size: int) -> bool:
    img = draw_composite(shapes, offs, canvas_size, (canvas_size // 2, canvas_size // 2))
    return _is_connected_img(img)

def _generate_touching_offsets(shapes: List[Dict[str, Any]], canvas_size: int) -> List[Tuple[float, float]]:
    offsets: List[Tuple[float, float]] = [(0.0, 0.0)]
    for i in range(1, len(shapes)):
        placed_current = False
        for _attempt_place in range(100): 
            ref_idx = random.randrange(i) # Connect to a random previous shape
            ref_offset_x, ref_offset_y = offsets[ref_idx]
            
            r_i = compute_shape_radius(shapes[i])
            r_ref = compute_shape_radius(shapes[ref_idx])
            
            dist_for_touch = r_i + r_ref - random.uniform(0, 0.1 * min(r_i, r_ref)) 
            angle = random.uniform(0, 2 * math.pi)
            
            new_dx = ref_offset_x + dist_for_touch * math.cos(angle)
            new_dy = ref_offset_y + dist_for_touch * math.sin(angle)

            is_too_overlapping = False
            for k, (prev_ox, prev_oy) in enumerate(offsets):
                if k == ref_idx: continue # Don't check against the shape we're attaching to here
                dist_to_prev = math.hypot(new_dx - prev_ox, new_dy - prev_oy)
                if dist_to_prev < 0.75 * (r_i + compute_shape_radius(shapes[k])): 
                    is_too_overlapping = True; break
            
            if not is_too_overlapping:
                offsets.append((new_dx, new_dy))
                placed_current = True
                break
        if not placed_current:
            offsets.append((random.uniform(-canvas_size*0.1, canvas_size*0.1), 
                            random.uniform(-canvas_size*0.1, canvas_size*0.1)))
    return offsets

def _random_shape_list(canvas_size: int) -> List[Dict[str, Any]]:
    num_shapes = random.randint(2, 5)
    shape_list: List[Dict[str, Any]] = []
    for z_order in range(num_shapes):
        shape_type = random.choice(SHAPES)
        color = random.choice(COLORS)
        shape_info: Dict[str, Any] = {'type': shape_type, 'color': color, 'angle': random.uniform(0, 2*math.pi) if shape_type != 'circle' else 0.0, 'z': z_order}

        if shape_type in ('square', 'rectangle', 'oval'):
            width = random.uniform(canvas_size * 0.07, canvas_size * 0.2)
            height = width if shape_type == 'square' else width * random.uniform(0.6, 1.4)
            shape_info.update(width=width, height=height)
        elif shape_type == 'polygon':
            radius = random.uniform(canvas_size * 0.07, canvas_size * 0.2)
            shape_info.update(size=radius, n=random.randint(5, 8))
        else: 
            size_param = random.uniform(canvas_size * 0.07, canvas_size * 0.2)
            shape_info.update(size=size_param)
        shape_list.append(shape_info)
    return shape_list

def gen_shapes_offsets(canvas_size: int, require_conn: bool) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]]]:
    for _attempt in range(500): # Increased attempts
        shapes = _random_shape_list(canvas_size)
        offsets = _generate_touching_offsets(shapes, canvas_size) 
        if composite_fits_canvas(shapes, offsets, canvas_size):
            if not require_conn or composite_connected(shapes, offsets, canvas_size):
                return shapes, offsets
    raise RuntimeError(f"Failed to generate shape composite (conn={require_conn}) after many attempts.")

def apply_affine(img: Image.Image, scale: float, angle_rad: float,
                 center_img_coords: Tuple[int, int], # Center of the transformation IN THE IMAGE COORDS
                 canvas_size: int) -> Image.Image:
    cx, cy = center_img_coords

    
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Transformation matrix elements for rotation then scaling centered at origin
    a_rs = scale * cos_a
    b_rs = scale * -sin_a 
    d_rs = scale * sin_a
    e_rs = scale * cos_a

    c_val = cx - a_rs * cx - b_rs * cy
    f_val = cy - d_rs * cx - e_rs * cy
    
    transform_matrix = (a_rs, b_rs, c_val, d_rs, e_rs, f_val)
    
    return img.transform((canvas_size, canvas_size), Image.AFFINE,
                         transform_matrix, resample=Image.BICUBIC, fillcolor="white")


def jitter_shapes(shapes: List[Dict[str, Any]], offs: List[Tuple[float, float]], canvas_size: int) \
        -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]], List[str], List[Dict[str,str]]]:
    new_s = [copy.deepcopy(s_item) for s_item in shapes]
    new_o = list(offs) 
    transforms_applied_names: List[str] = []
    jittered_shape_attributes: List[Dict[str,str]] = []

    num_to_jitter = NO_MIN_TRANSFORMS
    if len(new_s) > 0:
        num_to_jitter = random.randint(NO_MIN_TRANSFORMS, min(NO_MAX_TRANSFORMS, len(new_s)))
    else: return new_s, new_o, [], [] # No shapes to jitter
        
    indices_to_jitter = random.sample(range(len(new_s)), num_to_jitter)

    for idx in indices_to_jitter:
        current_shape_info = new_s[idx]
        jittered_shape_attributes.append({"type": current_shape_info['type'], "color": current_shape_info['color']})
        chosen_transforms_for_this_shape = set()

        possible_mandatory = ['translate', 'rotate', 'scale']
        if current_shape_info['type'] == 'circle': possible_mandatory.remove('rotate')
        if not possible_mandatory: # e.g. a single circle, only translate/scale left
            if random.random() < 0.5 : chosen_transforms_for_this_shape.add('translate')
            else: chosen_transforms_for_this_shape.add('scale')
        else: chosen_transforms_for_this_shape.add(random.choice(possible_mandatory))

        if 'translate' not in chosen_transforms_for_this_shape and random.random() < PROB_TRANSLATE: chosen_transforms_for_this_shape.add('translate')
        if current_shape_info['type'] != 'circle' and 'rotate' not in chosen_transforms_for_this_shape and random.random() < PROB_ROTATE: chosen_transforms_for_this_shape.add('rotate')
        if 'scale' not in chosen_transforms_for_this_shape and random.random() < PROB_SCALE: chosen_transforms_for_this_shape.add('scale')
        if 'reshape' not in chosen_transforms_for_this_shape and random.random() < PROB_RESHAPE:
            if any(s_type != current_shape_info['type'] for s_type in SHAPES): chosen_transforms_for_this_shape.add('reshape')
        
        transforms_applied_names.extend(list(chosen_transforms_for_this_shape))

        if 'translate' in chosen_transforms_for_this_shape:
            dist = random.uniform(NO_MIN_TRANSLATE_PX, NO_MAX_TRANSLATE_PX); theta = random.uniform(0, 2 * math.pi)
            dx_orig, dy_orig = new_o[idx]; new_o[idx] = (dx_orig + dist * math.cos(theta), dy_orig + dist * math.sin(theta))
        if 'rotate' in chosen_transforms_for_this_shape:
            rot_mag = random.uniform(JITTER_MIN_ROT, abs(NO_MAX_ROT))
            current_shape_info['angle'] = (current_shape_info.get('angle',0.0) + rot_mag * random.choice([-1,1]))
        if 'scale' in chosen_transforms_for_this_shape:
            delta_sc = random.uniform(RESCALE_MIN, RESCALE_MAX); factor = 1.0 + delta_sc if random.random() < 0.5 else 1.0 - delta_sc
            factor = max(0.1, factor)
            if current_shape_info['type'] in ('square', 'rectangle', 'oval'):
                current_shape_info['width'] = max(1.0, current_shape_info['width'] * factor)
                current_shape_info['height'] = max(1.0, current_shape_info['height'] * factor)
            else: current_shape_info['size'] = max(1.0, current_shape_info['size'] * factor)
        if 'reshape' in chosen_transforms_for_this_shape:
            orig_r = compute_shape_radius(current_shape_info); orig_color = current_shape_info['color']; orig_angle = current_shape_info.get('angle',0.0); orig_z = current_shape_info['z']
            possible_new_types = [s_type for s_type in SHAPES if s_type != current_shape_info['type']]
            if possible_new_types:
                new_type_str = random.choice(possible_new_types)
                reshaped_info: Dict[str, Any] = {'type': new_type_str, 'color': orig_color, 'angle': orig_angle, 'z': orig_z}
                if new_type_str in ('square', 'rectangle', 'oval'):
                    w = orig_r * 1.8; h = w if new_type_str == 'square' else w * random.uniform(0.6, 1.4)
                    reshaped_info.update(width=max(1.0,w), height=max(1.0,h))
                elif new_type_str == 'polygon': reshaped_info.update(size=max(1.0,orig_r), n=random.randint(5,8))
                else: reshaped_info.update(size=max(1.0, orig_r * (1.5 if new_type_str=='circle' else 1.2) )) # Heuristic sizing
                new_s[idx] = reshaped_info
    return new_s, new_o, transforms_applied_names, jittered_shape_attributes

def _draw_single_distractor(draw_context: ImageDraw.Draw, shape_info: Dict[str, Any], x_pos: float, y_pos: float):
    """Helper to draw one distractor shape, mirrors draw_composite logic."""
    typ, col = shape_info['type'], shape_info['color']
    ang = shape_info.get('angle', 0.0) 

    if typ == 'circle':
        r = shape_info['size'] / 2.0
        draw_context.ellipse([x_pos - r, y_pos - r, x_pos + r, y_pos + r], fill=col)
    elif typ == 'oval': 
        w, h = shape_info['width'] / 2.0, shape_info['height'] / 2.0
        max_dim = math.hypot(w,h) * 2.5 # Max dimension after rotation
        temp_canvas_size = int(max_dim)
        
        # Center oval in its temp canvas
        ox, oy = temp_canvas_size / 2, temp_canvas_size / 2
        
        temp_img = Image.new("RGBA", (temp_canvas_size, temp_canvas_size), (0,0,0,0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.ellipse((ox-w, oy-h, ox+w, oy+h), fill=col)
        if ang != 0: temp_img = temp_img.rotate(-math.degrees(ang), expand=True, resample=Image.BICUBIC, center=(ox,oy))
        

        paste_x = int(x_pos - temp_img.width / 2)
        paste_y = int(y_pos - temp_img.height / 2)
        
        if ang == 0 : # Only draw if no rotation, or implement complex polygon version
            draw_context.ellipse([x_pos - w, y_pos - h, x_pos + w, y_pos + h], fill=col)
            num_ellipse_pts = 36 
            ellipse_pts = []
            for i in range(num_ellipse_pts):
                theta_pt = 2 * math.pi * i / num_ellipse_pts
                ex = w * math.cos(theta_pt)
                ey = h * math.sin(theta_pt)
                # Rotate this ellipse point by `ang`
                rx = ex * math.cos(ang) - ey * math.sin(ang)
                ry = ex * math.sin(ang) + ey * math.cos(ang)
                ellipse_pts.append((x_pos + rx, y_pos + ry))
            if ellipse_pts: draw_context.polygon(ellipse_pts, fill=col)


    elif typ in ('square', 'rectangle'):
        w, h = shape_info['width'] / 2.0, shape_info['height'] / 2.0
        pts = [(x_pos - w, y_pos - h), (x_pos + w, y_pos - h),
               (x_pos + w, y_pos + h), (x_pos - w, y_pos + h)]
        if ang != 0: pts = [rotate_point(px, py, x_pos, y_pos, ang) for px, py in pts]
        draw_context.polygon(pts, fill=col)
    elif typ == 'triangle':
        s = shape_info['size']
        ht = s * math.sqrt(3) / 2.0
        verts = [(x_pos, y_pos - 2/3 * ht), (x_pos - s/2, y_pos + 1/3 * ht), (x_pos + s/2, y_pos + 1/3 * ht)]
        if ang != 0: verts = [rotate_point(px, py, x_pos, y_pos, ang) for px, py in verts]
        draw_context.polygon(verts, fill=col)
    elif typ == 'polygon':
        rad, n_sides = shape_info['size'], shape_info['n']
        current_angle = shape_info.get('angle', 0.0) # Polygon's inherent orientation + jitter
        verts = [(x_pos + rad * math.cos(current_angle + 2 * math.pi * i / n_sides),
                  y_pos + rad * math.sin(current_angle + 2 * math.pi * i / n_sides)) for i in range(n_sides)]
        draw_context.polygon(verts, fill=col)
    elif typ == 'line':
        line_len = shape_info['size'] / 2.0
        p1, p2 = (x_pos - line_len, y_pos), (x_pos + line_len, y_pos)
        if ang != 0:
            p1 = rotate_point(p1[0], p1[1], x_pos, y_pos, ang)
            p2 = rotate_point(p2[0], p2[1], x_pos, y_pos, ang)
        draw_context.line([p1, p2], fill=col, width=max(1, int(shape_info.get('size',10) / 10.0)))


def _add_distractors(
    base_image: Image.Image,
    main_object_center_coords: Tuple[float, float],
    main_object_approx_radius: float,
    canvas_size: int,
    *,
    gap_pixels: float = 10.0,
    num_distractors_range_tuple: Tuple[int, int] = (1, 3),
    allow_main_obj_overlap: bool = False,
    max_placement_tries: int = 50 
) -> Image.Image:
    
    img_with_dist = base_image.copy() 
    draw_on_img = ImageDraw.Draw(img_with_dist)
    
    placed_distractor_infos: List[Tuple[float, float, float]] = [] # (x_center, y_center, radius)
    num_distractors_to_place = random.randint(*num_distractors_range_tuple)

    for _ in range(num_distractors_to_place):
        distractor_s_info_list = _random_shape_list(canvas_size) 
        if not distractor_s_info_list: continue #shouldn't happen with current  _random_shape_list
        distractor_s_info = random.choice(distractor_s_info_list)
        distractor_s_radius = compute_shape_radius(distractor_s_info)

        is_placed = False
        for _try_idx in range(max_placement_tries):
            d_x = random.uniform(distractor_s_radius, canvas_size - distractor_s_radius)
            d_y = random.uniform(distractor_s_radius, canvas_size - distractor_s_radius)

            if not allow_main_obj_overlap:
                dist_to_main_obj_center = math.hypot(d_x - main_object_center_coords[0], d_y - main_object_center_coords[1])
                if dist_to_main_obj_center < (main_object_approx_radius + distractor_s_radius + gap_pixels):
                    continue 

            is_overlapping_other_distractors = False
            for prev_dx, prev_dy, prev_dr in placed_distractor_infos:
                if math.hypot(d_x - prev_dx, d_y - prev_dy) < (distractor_s_radius + prev_dr + gap_pixels / 2.0): # Smaller gap between distractors
                    is_overlapping_other_distractors = True; break
            if is_overlapping_other_distractors: continue
            
            _draw_single_distractor(draw_on_img, distractor_s_info, d_x, d_y)
            placed_distractor_infos.append((d_x, d_y, distractor_s_radius))
            is_placed = True; break 
        
    return img_with_dist

def generate_objreid_trial_data(
    canvas_size: int,
    conn1: bool, conn2: bool, no_affine: bool, 
    is_target_present_truth: bool, 
    add_distractors: bool, allow_distractor_overlap: bool, 
    output_dir_for_saving: Optional[str] = None, 
    run_id: str = "trial" 
) -> Tuple[Image.Image, Image.Image, List[Dict[str,Any]], Tuple[float,float,float,float], List[str], int, List[Dict[str,str]]]:
    # Returns: img1, img2, shapes_img1_metadata, (global_angle, global_scale, cx, cy), 
    #          jitter_transform_list, num_shapes_jittered, jittered_attrs_list

    center_x = center_y = canvas_size // 2
    shapes_img1, offsets_img1 = gen_shapes_offsets(canvas_size, require_conn=conn1)
    img1 = draw_composite(shapes_img1, offsets_img1, canvas_size, (center_x, center_y))

    angle_global_rad, scale_global = 0.0, 1.0
    final_offsets_for_img2_drawing = list(offsets_img1)
    img2_pre_jitter: Image.Image

    if no_affine: # Trial 3 specific
        img2_pre_jitter = img1.copy() # Image 2 starts identical before potential jitter
    else: # Trials 1, 9
        for _ in range(100):
            angle_global_rad = random.uniform(*GLOBAL_ROT_RANGE)
            scale_global = random.uniform(*GLOBAL_SCALE_RANGE)
            
            # Calculate offsets if this affine transform was applied to the composite *centered at origin*
            temp_offsets_after_global_affine = [
                (dx * scale_global * math.cos(angle_global_rad) - dy * scale_global * math.sin(angle_global_rad),
                 dx * scale_global * math.sin(angle_global_rad) + dy * scale_global * math.cos(angle_global_rad))
                for dx, dy in offsets_img1
            ]
            if composite_fits_canvas(shapes_img1, temp_offsets_after_global_affine, canvas_size) and \
               (not conn2 or composite_connected(shapes_img1, temp_offsets_after_global_affine, canvas_size)):
                final_offsets_for_img2_drawing = temp_offsets_after_global_affine
                break
        else: # Fallback if loop finishes (should be rare with enough attempts) - did not occur in practice
            print(f"Warning R{run_id}: Could not find fitting global affine after 100 tries. Using no-affine for img2 base.", file=sys.stderr)
            no_affine = True # Force no_affine path
            img2_pre_jitter = img1.copy()
        
        if not no_affine: # If affine was found and applied
             img2_pre_jitter = apply_affine(img1, scale_global, angle_global_rad, 
                                           (center_x, center_y), canvas_size)


    global_affine_tuple = (angle_global_rad, scale_global, center_x, center_y) # (ang, sc, cx_orig, cy_orig)

    applied_jitter_transforms_list: List[str] = []
    attributes_of_jittered_s: List[Dict[str,str]] = []
    num_s_jittered = 0
    
    shapes_for_img2_final_drawing = [copy.deepcopy(s) for s in shapes_img1]
    offsets_for_img2_final_drawing = final_offsets_for_img2_drawing # Start with globally transformed/original offsets

    if not is_target_present_truth: # "no" case -> apply jitter
        for _ in range(100): 
            jittered_s_list, jittered_o_list, applied_j_t_list, attrs_j_s_list = jitter_shapes(
                shapes_img1, # Jitter the base shapes
                final_offsets_for_img2_drawing, # With their globally transformed offsets
                canvas_size
            )
            if composite_fits_canvas(jittered_s_list, jittered_o_list, canvas_size) and \
               (not conn2 or composite_connected(jittered_s_list, jittered_o_list, canvas_size)):
                shapes_for_img2_final_drawing = jittered_s_list
                offsets_for_img2_final_drawing = jittered_o_list
                applied_jitter_transforms_list = applied_j_t_list
                attributes_of_jittered_s = attrs_j_s_list
                num_s_jittered = len(attributes_of_jittered_s)
                break
        else:
            print(f"Warning R{run_id}: Failed to apply valid jitter. Image 2 might be same as 'yes' case.", file=sys.stderr)
            # Fallback: img2 might not be jittered effectively
        
        img2_final = draw_composite(shapes_for_img2_final_drawing, offsets_for_img2_final_drawing, 
                                     canvas_size, (center_x, center_y))
    else: # "yes" case -> img2 is the globally transformed (or copied if no_affine) version
        img2_final = img2_pre_jitter

    if add_distractors:
        min_x_o, max_x_o, min_y_o, max_y_o = composite_bounds(shapes_for_img2_final_drawing, offsets_for_img2_final_drawing)
        obj_rad_approx = max(abs(min_x_o), abs(max_x_o), abs(min_y_o), abs(max_y_o))
        img2_final = _add_distractors(
            img2_final, (center_x, center_y), obj_rad_approx, canvas_size,
            allow_main_obj_overlap=allow_distractor_overlap
        )

    if output_dir_for_saving:
        os.makedirs(output_dir_for_saving, exist_ok=True)
        suffix = f"_{run_id}" if run_id else ""
        img1.save(os.path.join(output_dir_for_saving, f"img1{suffix}.png"))
        img2_final.save(os.path.join(output_dir_for_saving, f"img2{suffix}.png"))

    return (img1, img2_final, shapes_img1, 
            global_affine_tuple, applied_jitter_transforms_list, 
            num_s_jittered, attributes_of_jittered_s)

def generate_objreid_two_shot_examples(
    canvas_size: int,
    output_dir: Optional[str], 
    trial_conn1: bool, trial_conn2: bool, trial_no_affine: bool, 
    add_distractors: bool, allow_distractor_overlap: bool
) -> Tuple[Tuple[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fs_yes_dir = os.path.join(output_dir, "fs_yes")
        fs_no_dir = os.path.join(output_dir, "fs_no")
    else:
        fs_yes_dir = None
        fs_no_dir = None

    yes_i1, yes_i2, _, _, _, _, _ = generate_objreid_trial_data(
        canvas_size, trial_conn1, trial_conn2, trial_no_affine, True,
        add_distractors, allow_distractor_overlap, fs_yes_dir, "example"
    )
    no_i1, no_i2, _, _, _, _, _ = generate_objreid_trial_data(
        canvas_size, trial_conn1, trial_conn2, trial_no_affine, False,
        add_distractors, allow_distractor_overlap, fs_no_dir, "example"
    )
    return (yes_i1, yes_i2), (no_i1, no_i2)
