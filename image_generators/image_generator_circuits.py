import os
import sys
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict, Optional, Any

# ─────────── Constants for Image Generation ───────────
CANVAS_W, CANVAS_H = 800, 600
BREAD_W, BREAD_H = 200, 320
BREAD_LEFT = (CANVAS_W - BREAD_W) // 2
BREAD_TOP = (CANVAS_H - BREAD_H) // 2
PORT_RADIUS = 10
COMP_MIN_PORTS = 1  # Default, can be overridden by args
COMP_MAX_PORTS = 3  # Default, can be overridden by args
WIRE_WIDTH = 4
WIRE_COLOURS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
EXTENDED_WIRE_COLOURS = WIRE_COLOURS + [
    '#FF1493', '#00FFFF', '#FFD700', '#ADFF2F', '#FF00FF', '#1E90FF',
    '#D2691E', '#8A2BE2', '#00FA9A', '#DC143C', '#7FFF00', '#BDB76B',
    '#FF8C00', '#48D1CC', '#C71585', '#7CFC00', '#BA55D3', '#20B2AA'
]
COMPONENT_COLOURS = [
    "#ffd0d0", "#d0ffd0", "#d0d0ff", "#fff0d0",
    "#d0f0ff", "#f0d0ff", "#e0ffe0", "#ffe0ff",
]
FONT = ImageFont.load_default()

def segments_intersect(seg1: Tuple[Tuple[int, int], Tuple[int, int]],
                       seg2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
    """
    Check proper intersection (excluding shared endpoints) between two line segments.
    Each segment is ((x1,y1),(x2,y2)).
    """
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    p, q, r, s = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    o1, o2 = orient(p, q, r), orient(p, q, s)
    o3, o4 = orient(r, s, p), orient(r, s, q)

    # proper intersection if orientations straddle each other
    return (o1 * o2 < 0) and (o3 * o4 < 0)

def _draw_breadboard(draw: ImageDraw.Draw) -> List[Tuple[int, int]]:
    """Draw breadboard and return list of port-centre coords (len==10),
       with port numbers labelled just outside each circle."""
    draw.rectangle(
        [BREAD_LEFT, BREAD_TOP, BREAD_LEFT + BREAD_W, BREAD_TOP + BREAD_H],
        fill="#e0e0e0", outline="black", width=2
    )

    ports = []
    col_x = [BREAD_LEFT + BREAD_W * 0.25, BREAD_LEFT + BREAD_W * 0.75]
    spacing = BREAD_H / 6
    for c, x_coord in enumerate(col_x):
        for i in range(5):
            y_coord = BREAD_TOP + spacing * (i + 1)
            ports.append((int(x_coord), int(y_coord)))

    for idx, (x, y) in enumerate(ports, start=1):
        draw.ellipse(
            [x - PORT_RADIUS, y - PORT_RADIUS, x + PORT_RADIUS, y + PORT_RADIUS],
            fill="white", outline="black", width=2
        )
        txt = str(idx)
        tw, th = draw.textbbox((0, 0), txt, font=FONT)[2:]
        draw.text((x + PORT_RADIUS + 4, y - th / 2), txt, fill="black", font=FONT)
    return ports

def _random_component_bbox(side: str, comp_w: int, comp_h: int) -> Tuple[int, int, int, int]:
    """Return (left,top,right,bottom) for a component placed on given side."""
    margin = 20
    if side == 'left':
        left = margin
        top = random.randint(margin, CANVAS_H - comp_h - margin)
    elif side == 'right':
        left = CANVAS_W - comp_w - margin
        top = random.randint(margin, CANVAS_H - comp_h - margin)
    elif side == 'top':
        left = random.randint(margin, CANVAS_W - comp_w - margin)
        top = margin
    else:  # bottom
        left = random.randint(margin, CANVAS_W - comp_w - margin)
        top = CANVAS_H - comp_h - margin
    return (left, top, left + comp_w, top + comp_h)

def _draw_component(
    draw: ImageDraw.Draw,
    label: str,
    n_ports: int,
    existing_bbxs: List[Tuple[int, int, int, int]],
    *,
    fill_color: str = "#d0d0ff"
) -> Tuple[Dict[str, Any], str]:
    """Draw one rectangular component, return its metadata + side chosen."""
    comp_w, comp_h = 100, 60

    for _ in range(100):
        side = random.choice(["left", "right", "top", "bottom"])
        bbox = _random_component_bbox(side, comp_w, comp_h)
        l, t, r, b = bbox
        if all(r < L or l > R or b < T or t > B for (L, T, R, B) in existing_bbxs):
            break
    else:
        raise RuntimeError("Could not place component after 100 tries")

    draw.rectangle(bbox, fill=fill_color, outline="black", width=2)
    tw, th = draw.textbbox((0, 0), label, font=FONT)[2:]
    draw.text(((l + r - tw) / 2, (t + b - th) / 2), label, font=FONT, fill="black")

    term_side = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[side]
    ports_coords = []
    if term_side in ("left", "right"):
        x = l if term_side == "left" else r
        ys = np.linspace(t + 10, b - 10, n_ports)
        for y_coord in ys:
            ports_coords.append((int(x), int(y_coord)))
            draw.ellipse(
                [x - PORT_RADIUS, y_coord - PORT_RADIUS, x + PORT_RADIUS, y_coord + PORT_RADIUS],
                fill="white", outline="black", width=2
            )
    else:
        y = t if term_side == "top" else b
        xs = np.linspace(l + 10, r - 10, n_ports)
        for x_coord in xs:
            ports_coords.append((int(x_coord), int(y)))
            draw.ellipse(
                [x_coord - PORT_RADIUS, y - PORT_RADIUS, x_coord + PORT_RADIUS, y + PORT_RADIUS],
                fill="white", outline="black", width=2
            )
    return {"label": label, "bbox": bbox, "ports": ports_coords}, side

def do_segments_collide_strict(seg1: Tuple[Tuple[int,int],Tuple[int,int]],
                               seg2: Tuple[Tuple[int,int],Tuple[int,int]],
                               wire_width: int) -> bool:
    """Checks if two H/V wire segments visually collide, considering wire width."""
    if segments_intersect(seg1, seg2):
        return True

    (x1_s1, y1_s1), (x2_s1, y2_s1) = seg1
    (x1_s2, y1_s2), (x2_s2, y2_s2) = seg2

    ww_half_low = (wire_width - 1) // 2
    ww_half_high = wire_width // 2

    rect1_l, rect1_r, rect1_b, rect1_t = 0,0,0,0
    s1_is_h = (y1_s1 == y2_s1)
    s1_is_v = (x1_s1 == x2_s1)
    if s1_is_h:
        rect1_l, rect1_r = min(x1_s1, x2_s1), max(x1_s1, x2_s1)
        rect1_b, rect1_t = y1_s1 - ww_half_low, y1_s1 + ww_half_high
    elif s1_is_v:
        rect1_b, rect1_t = min(y1_s1, y2_s1), max(y1_s1, y2_s1)
        rect1_l, rect1_r = x1_s1 - ww_half_low, x1_s1 + ww_half_high
    else: return False # Diagonal, not handled by strict H/V collision

    rect2_l, rect2_r, rect2_b, rect2_t = 0,0,0,0
    s2_is_h = (y1_s2 == y2_s2)
    s2_is_v = (x1_s2 == x2_s2)
    if s2_is_h:
        rect2_l, rect2_r = min(x1_s2, x2_s2), max(x1_s2, x2_s2)
        rect2_b, rect2_t = y1_s2 - ww_half_low, y1_s2 + ww_half_high
    elif s2_is_v:
        rect2_b, rect2_t = min(y1_s2, y2_s2), max(y1_s2, y2_s2)
        rect2_l, rect2_r = x1_s2 - ww_half_low, x1_s2 + ww_half_high
    else: return False # Diagonal

    no_horizontal_overlap = rect1_r <= rect2_l or rect1_l >= rect2_r
    no_vertical_overlap = rect1_t <= rect2_b or rect1_b >= rect2_t
    
    return not (no_horizontal_overlap or no_vertical_overlap)


def _orthogonal_polyline(p0: Tuple[int,int], p1: Tuple[int,int], force_direction: Optional[str] = None) -> List[Tuple[int,int]]:
    """Return a two-segment (L-shaped) polyline between p0→p1."""
    x0, y0 = p0
    x1, y1 = p1

    horizontal_first_chosen: bool
    if force_direction == "h_first":
        horizontal_first_chosen = True
    elif force_direction == "v_first":
        horizontal_first_chosen = False
    else:
        horizontal_first_chosen = random.random() < 0.5
    
    mid = (x1, y0) if horizontal_first_chosen else (x0, y1)
    return [p0, mid, p1]

def _diagonal_polyline(
        p0: Tuple[int,int],
        p1: Tuple[int,int],
        existing: List[Tuple[Tuple[int,int],Tuple[int,int]]]
    ) -> List[Tuple[int,int]]:
    """Two-segment diagonal path p0→mid→p1. Detours if parallel overlap."""
    margin = PORT_RADIUS + 2
    x0, y0 = p0
    x1, y1 = p1

    slope1 = 1 if abs(x1 - x0) >= abs(y1 - y0) else -1
    slope2 = -1 / slope1 # type: ignore

    b1 = y0 - slope1 * x0
    b2 = y1 - slope2 * x1
    xi = (b2 - b1) / (slope1 - slope2)
    yi = slope1 * xi + b1

    xi = min(max(xi, margin), CANVAS_W - margin)
    yi = min(max(yi, margin), CANVAS_H - margin)
    mid = (int(xi), int(yi))
    path = [p0, mid, p1]

    def seg_slope(seg):
        (ax, ay), (bx, by) = seg
        dx, dy = bx - ax, by - ay
        return dy / dx if dx != 0 else None

    eps = 1e-6
    for new_seg_idx in range(len(path) - 1):
        new_s = (path[new_seg_idx], path[new_seg_idx+1])
        s_new_slope = seg_slope(new_s)
        for old_seg in existing:
            s_old_slope = seg_slope(old_seg)
            parallel = (s_new_slope is None and s_old_slope is None) or \
                       (s_new_slope is not None and s_old_slope is not None and abs(s_new_slope - s_old_slope) < eps)
            if parallel:
                b3 = mid[1] - slope2 * mid[0]
                b4 = y1 - slope1 * x1
                xi2 = (b4 - b3) / (slope2 - slope1)
                yi2 = slope2 * xi2 + b3
                xi2 = min(max(xi2, margin), CANVAS_W - margin)
                yi2 = min(max(yi2, margin), CANVAS_H - margin)
                mid2 = (int(xi2), int(yi2))
                return [p0, mid, mid2, p1]
    return path

def generate_circuit_image(
    *,
    min_components: int,
    max_components: int,
    min_ports: int, # Min ports per component
    max_ports: int, # Max ports per component
    min_wires: int, # Min breadboard-to-component wires
    max_wires: int, # Max breadboard-to-component wires
    min_cc_wires: int, # Min component-to-component wires
    max_cc_wires: int, # Max component-to-component wires
    wire_color_mode: str = "default",
    no_wire_crossing: bool = False,
) -> Tuple[Image.Image, int, str, Dict[int, str], Dict[int, Dict[str, Any]]]:
    """Build a synthetic circuit and return its data."""
    generated_single_color: Optional[str] = None
    if wire_color_mode == "single":
        generated_single_color = random.choice(WIRE_COLOURS)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(img)
    bread_ports_coords = _draw_breadboard(draw)

    n_total_components = random.randint(min_components, max_components)
    components_data: List[Dict[str, Any]] = []
    occupied_bboxes: List[Tuple[int,int,int,int]] = [(BREAD_LEFT, BREAD_TOP, BREAD_LEFT + BREAD_W, BREAD_TOP + BREAD_H)]

    for i in range(n_total_components):
        n_ports_i = random.randint(min_ports, max_ports)
        comp_meta, _ = _draw_component(
            draw, f"C{i+1}", n_ports_i, occupied_bboxes,
            fill_color=random.choice(COMPONENT_COLOURS)
        )
        occupied_bboxes.append(comp_meta["bbox"])
        components_data.append(comp_meta)

    all_wire_segments: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
    bb_to_comp_map: Dict[int, str] = {}
    wire_details: Dict[int, Dict[str, Any]] = {}
    used_component_ports = set() # Stores (comp_label, port_coord_tuple)

    available_breadboard_ports = list(range(1, 11))
    random.shuffle(available_breadboard_ports)
    
    # Max possible wires considering available component ports
    total_component_ports = sum(len(c['ports']) for c in components_data)
    max_possible_wires = min(max_wires, len(available_breadboard_ports), total_component_ports)
    num_bb_wires_to_draw = random.randint(min_wires, max_possible_wires)
    bb_wires_drawn_count = 0

    for bb_port_num in available_breadboard_ports:
        if bb_wires_drawn_count >= num_bb_wires_to_draw:
            break
        p0_bb = bread_ports_coords[bb_port_num - 1]
        random.shuffle(components_data) # Randomize component choice
        connection_made = False
        for comp_obj in components_data:
            free_ports_on_comp = [p for p in comp_obj["ports"] if (comp_obj["label"], tuple(p)) not in used_component_ports]
            if not free_ports_on_comp:
                continue
            
            p1_comp = random.choice(free_ports_on_comp)
            current_path = None
            path_found = False

            if no_wire_crossing:
                for direction_choice in ["h_first", "v_first"]: # Try both orthogonal strategies
                    poly_ortho = _orthogonal_polyline(p0_bb, p1_comp, force_direction=direction_choice)
                    collides = any(do_segments_collide_strict(seg, old_seg, WIRE_WIDTH)
                                   for seg_idx in range(len(poly_ortho)-1)
                                   for old_seg in all_wire_segments
                                   for seg in [(poly_ortho[seg_idx], poly_ortho[seg_idx+1])]) # type: ignore
                    if not collides:
                        current_path = poly_ortho
                        path_found = True
                        break
            else: # Allow crossings, use diagonal default
                current_path = _diagonal_polyline(p0_bb, p1_comp, all_wire_segments)
                path_found = True
            
            if path_found and current_path:
                used_component_ports.add((comp_obj["label"], tuple(p1_comp)))
                
                wire_col: str
                if wire_color_mode == "single" and generated_single_color:
                    wire_col = generated_single_color
                elif wire_color_mode == "unique":
                    wire_col = EXTENDED_WIRE_COLOURS[len(bb_to_comp_map) % len(EXTENDED_WIRE_COLOURS)]
                else:
                    wire_col = random.choice(WIRE_COLOURS)
                
                draw.line(current_path, fill=wire_col, width=WIRE_WIDTH)
                
                path_len = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(current_path, current_path[1:]))
                euclid = math.hypot(p1_comp[0] - p0_bb[0], p1_comp[1] - p0_bb[1])
                crossings = 0
                for seg_idx in range(len(current_path) -1):
                    new_segment = (current_path[seg_idx], current_path[seg_idx+1])
                    for old_s in all_wire_segments:
                        if segments_intersect(new_segment, old_s):
                            crossings +=1
                
                bb_to_comp_map[bb_port_num] = comp_obj["label"]
                wire_details[bb_port_num] = {"length": path_len, "euclid_dist": euclid, "crossings": crossings, "color": wire_col}
                for seg_idx in range(len(current_path) -1):
                    all_wire_segments.append((current_path[seg_idx], current_path[seg_idx+1]))
                
                bb_wires_drawn_count += 1
                connection_made = True
                break # Break from components loop, bb_port connected
        # If no connection was made for this bb_port (e.g. no_wire_crossing and no path found), it remains unconnected.

    # Component-to-component wires (visual clutter, not part of the task query)
    num_cc_wires_to_draw = random.randint(min_cc_wires, max_cc_wires)
    cc_wires_drawn = 0
    
    all_comp_ports_list = [] # List of {'label': str, 'coord': tuple, 'id': tuple}
    for comp_obj in components_data:
        for port_coord in comp_obj['ports']:
            all_comp_ports_list.append({'label': comp_obj['label'], 'coord': port_coord, 'id': (comp_obj['label'], tuple(port_coord))})

    for _ in range(num_cc_wires_to_draw * 5): # Try multiple times
        if cc_wires_drawn >= num_cc_wires_to_draw:
            break
        
        available_for_cc_connection = [p for p in all_comp_ports_list if p['id'] not in used_component_ports]
        if len(available_for_cc_connection) < 2:
            break # Not enough free ports left

        c1_port_info = random.choice(available_for_cc_connection)
        # Ensure c2 is from a different component
        c2_options = [p for p in available_for_cc_connection if p['label'] != c1_port_info['label']]
        if not c2_options:
            continue # No other component port available
        
        c2_port_info = random.choice(c2_options)
        p1_cc, p2_cc = c1_port_info['coord'], c2_port_info['coord']
        
        current_cc_path = None
        cc_path_found = False
        if no_wire_crossing:
            for direction_choice in ["h_first", "v_first"]:
                poly_ortho_cc = _orthogonal_polyline(p1_cc, p2_cc, force_direction=direction_choice)
                collides_cc = any(do_segments_collide_strict(seg, old_seg, WIRE_WIDTH)
                                  for seg_idx in range(len(poly_ortho_cc)-1)
                                  for old_seg in all_wire_segments
                                  for seg in [(poly_ortho_cc[seg_idx], poly_ortho_cc[seg_idx+1])]) # type: ignore
                if not collides_cc:
                    current_cc_path = poly_ortho_cc
                    cc_path_found = True
                    break
        else:
            current_cc_path = _diagonal_polyline(p1_cc, p2_cc, all_wire_segments)
            cc_path_found = True
            
        if cc_path_found and current_cc_path:
            used_component_ports.add(c1_port_info['id'])
            used_component_ports.add(c2_port_info['id'])
            
            cc_wire_col: str
            if wire_color_mode == "single" and generated_single_color:
                cc_wire_col = generated_single_color
            elif wire_color_mode == "unique":
                cc_wire_col = EXTENDED_WIRE_COLOURS[(len(bb_to_comp_map) + cc_wires_drawn) % len(EXTENDED_WIRE_COLOURS)]
            else:
                cc_wire_col = random.choice(WIRE_COLOURS)
            
            draw.line(current_cc_path, fill=cc_wire_col, width=WIRE_WIDTH)
            for seg_idx in range(len(current_cc_path) -1):
                all_wire_segments.append((current_cc_path[seg_idx], current_cc_path[seg_idx+1]))
            cc_wires_drawn +=1

    if not bb_to_comp_map: # No queryable wires were drawn
        print("Warning: No breadboard wires were successfully placed for query generation.", file=sys.stderr)
        return img, 1, "C1" if components_data else "N/A", {}, {}


    query_port_num = random.choice(list(bb_to_comp_map.keys()))
    correct_comp_label = bb_to_comp_map[query_port_num]

    return img, query_port_num, correct_comp_label, bb_to_comp_map, wire_details

def generate_two_shot_examples(
    odir: str = "few_shot_examples",
    **gen_kwargs,
) -> List[Tuple[Image.Image, str, str]]:
    """
    Produce two canonical examples (saved under `odir`) and return:
      [(image, prompt_text, correct_component_answer_format), …]
    """
    os.makedirs(odir, exist_ok=True)
    examples = []
    fs_gen_kwargs = {
        'min_components': gen_kwargs.get('min_components', 4),
        'max_components': gen_kwargs.get('max_components', 5),
        'min_ports': gen_kwargs.get('min_ports', 1),
        'max_ports': gen_kwargs.get('max_ports', 2),
        'min_wires': gen_kwargs.get('min_wires', 3),
        'max_wires': gen_kwargs.get('max_wires', 5),
        'min_cc_wires': gen_kwargs.get('min_cc_wires', 1),
        'max_cc_wires': gen_kwargs.get('max_cc_wires', 2),
        'wire_color_mode': gen_kwargs.get('wire_color_mode', 'default'),
        'no_wire_crossing': gen_kwargs.get('no_wire_crossing', False),
    }

    for i in range(2):
        img, qport, comp_label, _, _ = generate_circuit_image(**fs_gen_kwargs) 
        img.save(f"{odir}/two_shot_{i}.png")
        prompt = (
            f"Which component does the wire from port {qport} on the breadboard, which is the gray rectangle with numbered ports, connect to? "
            "A wire is a series of connected, same colored lines that go from the center of a port, represented on the screen as a white circle, to another port. Each wire only connects two ports, one at either end. "
            "A wire will NEVER turn at the same spot that it intersects another wire, and wires do not change colors. "
            "Answer with the component label in curly braces, e.g {C0}."
        )
        examples.append((img, prompt, f"After following the wire, it connects to {comp_label}. {{{comp_label}}}"))
    return examples