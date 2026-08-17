# SimulationRenderGT.py
"""
Rendering utilities for the *current* simulation architecture.

Why this exists
---------------
The older SimulationFunctions.save_game_state_to_image(...) renders by iterating
`player._territories`, which can drift from the current source-of-truth (GlobalState)
depending on how states are transformed during ML/GT simulations.

This module renders directly from:
- GlobalState (owner "A"/"D" + troop counts per territory index)
- Board.node_to_territory_dict (territory index -> Territory with ._name)
- SimulationFunctions.territory_coords (image-specific coordinates)

It uses the same base map image (e.g. "Game_Board.jpg") and writes an annotated PNG.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


territory_coords = {
    # North America
    'Alaska': (70, 90), 'Northwest Territory': (210, 110), 'Greenland': (380, 50), 
    'Alberta': (160, 180), 'Ontario': (220, 180), 'Quebec': (310, 180), 
    'Western United States': (180, 250), 'Eastern United States': (280, 300),
    
    # South America
    'Central America': (180, 365), 'Venezuela': (250, 440), 'Brazil': (360, 500),
    'Peru': (280, 560), 'Argentina': (270, 700),
    
    # Africa
    'North Africa': (540, 500), 'Egypt': (620, 460), 'East Africa': (680, 540),
    'Congo': (620, 600), 'South Africa': (630, 685), 'Madagascar': (745, 700),
    
    # Europe
    'Iceland': (490, 130), 'Scandinavia': (600, 120), 'Great Britain': (460, 220),
    'Northern Europe': (580, 240), 'Ukraine': (700, 180), 'Western Europe': (500, 370),
    'Southern Europe': (580, 320),
    
    # Asia
    'Siberia': (850, 90), 'Yakutsk': (960, 75), 'Kamchatka': (1050, 75), 'Ural': (800, 140), 
    'Irkutsk': (950, 170), 'Mongolia': (950, 260), 'Japan': (1095, 260), 'Afghanistan': (790, 280),
    'China': (900, 320), 'Middle East': (700, 380), 'India': (840, 400), 'Siam': (960, 440),
    
    # Australia
    'Indonesia': (990, 570), 'New Guinea': (1080, 540), 'Western Australia': (1010, 700),
    'Eastern Australia': (1100, 650)
    }


def save_global_state_to_image(
    *,
    global_state: GlobalState,
    players: Sequence[Players.Player],
    base_map_image_path: str,
    output_image_path: str,
    territory_coords_override: Optional[dict] = None,
    radius: int = 10,
) -> None:
    """
    Draw current game state from GlobalState onto a background map.

    Parameters
    ----------
    global_state:
        GlobalState with nodes indexed by Board territory index (1..N), possibly with dummy 0.
        owners are expected to be "A" or "D".
    players:
        Players list. By convention:
            players[0] corresponds to owner "A"
            players[1] corresponds to owner "D"
        Colors are taken from player._color if present, else from a default palette.
    base_map_image_path:
        Path to background image (Game_Board.jpg).
    output_image_path:
        Where to save the annotated image (PNG recommended).
    territory_coords_override:
        Optional dict overriding SimulationFunctions.territory_coords.
    radius:
        Circle radius for territory markers.
    """
    if Image is None:
        raise ImportError("PIL (Pillow) is required for rendering. Please `pip install pillow`.")

    coords = territory_coords_override or territory_coords

    base = Image.open(base_map_image_path).convert("RGBA")
    draw = ImageDraw.Draw(base)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()


    palette = [
        (220, 20, 60),   # crimson
        (65, 105, 225),  # royal blue
        (46, 139, 87),   # sea green
        (255, 140, 0),   # dark orange
        (148, 0, 211),   # dark violet
        (0, 139, 139),   # dark cyan
    ]

    # owner->color mapping (A=players[0], D=players[1])
    owner_color: Dict[str, Tuple[int, int, int]] = {}
    if len(players) >= 1:
        owner_color["A"] = getattr(players[0], "_color", None) or palette[0]
    else:
        owner_color["A"] = palette[0]
    if len(players) >= 2:
        owner_color["D"] = getattr(players[1], "_color", None) or palette[1]
    else:
        owner_color["D"] = palette[1]

    # Render each actual board territory using Board mapping.
    for idx in sorted(int(i) for i in Board.node_to_territory_dict.keys()):
        if idx <= 0:
            continue
        if idx >= len(global_state.nodes):
            # state might be shorter in some contexts; skip gracefully
            continue

        terr = Board.node_to_territory_dict[idx]
        name = terr._name
        xy = coords.get(name) or coords.get(str(name).title())
        if xy is None:
            continue

        node = global_state.nodes[idx]
        owner = node.owner
        troops = int(node.troops)

        color = owner_color.get(owner, (128, 128, 128))

        # marker dot
        r = int(radius)
        draw.ellipse(
            (xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r),
            fill=color + (180,),
            outline=(0, 0, 0, 220),
        )

        # label: "Troops" (optionally include name if you prefer)
        label = f"{troops}"
        # slight offset so it doesn't overlap the circle
        draw.text((xy[0] + r + 2, xy[1] - r), label, fill=(0, 0, 0, 255), font=font)

    base.save(output_image_path)
