"""Draw the Connect 4 animation used in the README.

Depth-3 alpha-beta (playing second) against the rule-based opponent, using
this project's own game, agents and search statistics. At every alpha-beta
turn the same position is also searched with plain Minimax, so the panel can
show how many positions pruning saved.

Run from the repository root:

    python scripts/readme_animation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.default_opponent import DefaultOpponent  # noqa: E402
from src.agents.minimax_agent import MinimaxAgent  # noqa: E402
from src.games.connect4 import Connect4  # noqa: E402

OUT = Path("docs/assets/connect4-alpha-beta.gif")

PAPER = (244, 240, 230)
INK = (28, 27, 25)
MUTED = (125, 117, 102)
RULE = (205, 197, 179)
BOARD = (63, 94, 115)
HOLE = (244, 240, 230)
SEARCH = (165, 69, 47)
OPPONENT = (201, 162, 39)
KICKER = (122, 74, 31)
LIGHT = (214, 205, 186)


def font(size: int, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSerif"
    if bold:
        name += "-Bold"
    for folder in ("/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"):
        path = Path(folder) / f"{name}.ttf"
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def winning_cells(game: Connect4) -> list[tuple[int, int]]:
    board, rows, cols = game.board, game.rows, game.cols
    for r in range(rows):
        for c in range(cols):
            player = board[r][c]
            if player == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player for rr, cc in cells):
                    return cells
    return []


def play(seed: int = 0, depth: int = 3):
    game = Connect4()
    search = MinimaxAgent("alpha-beta", use_alpha_beta=True, depth_limit=depth)
    plain = MinimaxAgent("minimax", use_alpha_beta=False, depth_limit=depth)
    opponent = DefaultOpponent(seed=seed)
    search_player = -1

    history = []
    while not game.is_terminal():
        board_before = [row.copy() for row in game.board]
        if game.current_player == search_player:
            action = search.select_action(game)
            plain.select_action(game)
            stats = (search.last_search_stats.nodes_visited, plain.last_search_stats.nodes_visited)
            who = "search"
        else:
            action = opponent.select_action(game)
            stats = None
            who = "opponent"
        landing = max(r for r in range(game.rows) if board_before[r][action] == 0)
        game.apply_action(action)
        history.append((who, action, landing, [row.copy() for row in game.board], stats))
    return game, history, search_player


def render(game, history, search_player, depth: int = 3) -> None:
    cell, pad = 64, 10
    bw, bh = game.cols * cell + pad * 2, game.rows * cell + pad * 2
    margin, header = 44, 128
    panel_x = margin + bw + 56
    width, height = panel_x + 470, header + bh + 110
    colour = {search_player: SEARCH, -search_player: OPPONENT}
    win = set(winning_cells(game))
    total_ab = sum(s[0] for *_, s in history if s)
    total_mm = sum(s[1] for *_, s in history if s)
    max_mm = max(s[1] for *_, s in history if s)

    def frame(board, move_index, falling=None, highlight=False):
        image = Image.new("RGB", (width, height), PAPER)
        d = ImageDraw.Draw(image)
        d.text((margin, 28), "ALPHA-BETA SEARCH AT CONNECT 4", font=font(15, mono=True, bold=True), fill=KICKER)
        d.text(
            (margin, 52),
            f"Depth-{depth} alpha-beta (red) against the rule-based opponent (gold)",
            font=font(22),
            fill=INK,
        )
        x0, y0 = margin, header
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=14, fill=BOARD)
        for r in range(game.rows):
            for c in range(game.cols):
                cx, cy = x0 + pad + c * cell, y0 + pad + r * cell
                value = board[r][c]
                fill = colour[value] if value else HOLE
                d.ellipse([cx + 6, cy + 6, cx + cell - 6, cy + cell - 6], fill=fill)
                if highlight and (r, c) in win:
                    d.ellipse([cx + 3, cy + 3, cx + cell - 3, cy + cell - 3], outline=INK, width=4)
        if falling:
            r_float, c, value = falling
            cx, cy = x0 + pad + c * cell, y0 + pad + r_float * cell
            d.ellipse([cx + 6, cy + 6, cx + cell - 6, cy + cell - 6], fill=colour[value])

        px, py = panel_x, header
        d.text((px, py), "MOVE", font=font(13, mono=True, bold=True), fill=MUTED)
        d.text((px, py + 18), f"{move_index} / {len(history)}", font=font(34, bold=True), fill=INK)

        shown = [h for h in history[:move_index] if h[4]]
        ab_sum = sum(h[4][0] for h in shown)
        mm_sum = sum(h[4][1] for h in shown)
        d.text((px, py + 82), "POSITIONS SEARCHED SO FAR", font=font(13, mono=True, bold=True), fill=MUTED)
        d.text((px, py + 102), f"{ab_sum:,}", font=font(30, bold=True), fill=SEARCH)
        d.text((px + 170, py + 114), "alpha-beta", font=font(14, mono=True), fill=SEARCH)
        d.text((px, py + 140), f"{mm_sum:,}", font=font(30, bold=True), fill=MUTED)
        d.text((px + 170, py + 152), "plain minimax", font=font(14, mono=True), fill=MUTED)

        d.text((px, py + 200), "PER ALPHA-BETA MOVE", font=font(13, mono=True, bold=True), fill=MUTED)
        chart_top, chart_h = py + 226, 150
        bar_w = 380 / max(1, sum(1 for h in history if h[4]))
        d.line([px, chart_top + chart_h, px + 390, chart_top + chart_h], fill=RULE, width=1)
        for i, h in enumerate(shown):
            ab, mm = h[4]
            x = px + i * bar_w
            mm_h = chart_h * mm / max_mm
            ab_h = chart_h * ab / max_mm
            d.rectangle([x + 1, chart_top + chart_h - mm_h, x + bar_w - 2, chart_top + chart_h], fill=LIGHT)
            d.rectangle([x + 1, chart_top + chart_h - ab_h, x + bar_w - 2, chart_top + chart_h], fill=SEARCH)
        d.text(
            (px, chart_top + chart_h + 10),
            "full bar: minimax   red: alpha-beta",
            font=font(13, mono=True),
            fill=MUTED,
        )
        if highlight:
            saved = 1 - total_ab / total_mm
            d.text((px, chart_top + chart_h + 44), "Alpha-beta wins.", font=font(20, bold=True), fill=SEARCH)
            d.text(
                (px, chart_top + chart_h + 72),
                f"Same moves, {saved:.0%} fewer positions searched.",
                font=font(16),
                fill=INK,
            )
        return image

    frames, durations = [], []
    empty = [[0] * game.cols for _ in range(game.rows)]
    frames.append(frame(empty, 0))
    durations.append(900)
    previous = empty
    for index, (who, action, landing, board, _) in enumerate(history, start=1):
        value = board[landing][action]
        for step in (0.0, 0.45, 0.8):
            frames.append(frame(previous, index, falling=(landing * step, action, value)))
            durations.append(45)
        frames.append(frame(board, index))
        durations.append(420 if who == "search" else 300)
        previous = board
    frames.append(frame(previous, len(history), highlight=True))
    durations.append(3500)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    palette = [f.convert("P", palette=Image.ADAPTIVE, colors=48) for f in frames]
    palette[0].save(OUT, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True)
    print(f"{OUT}: {len(history)} moves, alpha-beta {total_ab:,} vs minimax {total_mm:,} positions")


if __name__ == "__main__":
    final_game, moves, seat = play()
    render(final_game, moves, seat)
