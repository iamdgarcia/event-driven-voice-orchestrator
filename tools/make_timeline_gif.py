"""Render the cascade-vs-event-driven comparison as an animated GIF.

Data source: the canonical bench cell (RTT=1000ms, tools=4) from
bench/bench_turn_latency.py — cascade serializes 4 tool round-trips (4250ms
total) while the event-driven runtime dispatches them in parallel (1125ms) and
acks a mid-pending user turn at ~50ms. Every timestamp below matches that
bench's timeline; regenerate with:

    python tools/make_timeline_gif.py

Output: docs/img/cascade_vs_event_driven.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- canonical cell (matches bench_turn_latency.py: rtt=1000, tools=4) -------
RTT_MS = 1000
N_TOOLS = 4

# Cascade (A): model issues one call per batch; filler while pending.
# Marks (ms): send -> call1@180, filler@~700; call k+1 issued right after result k.
CASCADE = {
    "user_in": 0,
    "calls": [180 + i * RTT_MS for i in range(N_TOOLS)],  # call issued
    "results": [180 + (i + 1) * RTT_MS for i in range(N_TOOLS)],  # tool done
    "filler": 780,
    "answer": 180 + N_TOOLS * RTT_MS,  # 4180
    "answer_done": 4250,
    "ack": None,
}

# Event-driven (B/C): one parallel batch; ack at ~50ms into the pending window.
EVENT = {
    "user_in": 0,
    "calls": [180] * N_TOOLS,
    "results": [1180] * N_TOOLS,
    "filler": 700,
    "ack": 50,  # turn-2 (mid-pending) acked at 50ms after its send at t=50
    "ack_done": 260,
    "answer": 1180,
    "answer_done": 1156 + 100,
}

TOTAL_MS = 4500
MS_PER_FRAME = 100  # 25ms conversation time per drawn frame @ 4x density -> readable
FRAME_MS = 1000 // 10

W, H = 780, 460
PANEL_W = W // 2
BG = (18, 18, 24)
PANEL_BG = (24, 24, 32)
TXT = (232, 232, 240)
DIM = (140, 140, 155)
USER_COL = (86, 156, 214)
TOOL_QUEUED = (120, 120, 130)
TOOL_RUN = (230, 170, 60)
TOOL_DONE = (90, 190, 110)
ANSWER_COL = (190, 120, 220)
ACK_COL = (120, 200, 230)
RED = (220, 90, 90)

try:
    F = ImageFont.truetype("consola.ttf", 15)
    FB = ImageFont.truetype("consolab.ttf", 17)
    FT = ImageFont.truetype("consolab.ttf", 21)
except OSError:
    F = FB = FT = ImageFont.load_default()


def _fnt(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("consolab.ttf" if bold else "consola.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_frame(t_ms: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2 - 150, 8), "cascade vs event-driven turn", fill=TXT, font=FT)

    d.rectangle([4, 40, PANEL_W - 6, H - 8], fill=PANEL_BG, outline=(60, 60, 72))
    d.rectangle([PANEL_W + 2, 40, W - 4, H - 8], fill=PANEL_BG, outline=(60, 60, 72))
    d.text((20, 50), "CASCADE (serial)", fill=(240, 150, 150), font=FT)
    d.text((PANEL_W + 18, 50), "EVENT-DRIVEN (parallel)", fill=(150, 240, 180), font=FT)
    d.text((24, 76), f"wall clock {t_ms:5d} ms", fill=DIM, font=F)
    d.text((PANEL_W + 22, 76), f"wall clock {t_ms:5d} ms", fill=DIM, font=F)

    _draw_panel(d, CASCADE, 20, t_ms, cascade=True)
    _draw_panel(d, EVENT, PANEL_W + 20, t_ms, cascade=False)
    return img


def _bubble(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, col, label: str):
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=col, outline=(0, 0, 0))
    d.text((x + 10, y + (h - 15) // 2), label, fill=(20, 20, 26), font=F)


def _draw_panel(d: ImageDraw.ImageDraw, plan: dict, ox: int, t: int, *, cascade: bool):
    # user message in
    if t >= plan["user_in"]:
        _bubble(d, ox, 110, 180, 26, USER_COL, "user: check my orders")
    # turn-2 (event-driven only): arrives mid-pending
    if not cascade and t >= 50:
        _bubble(d, ox, 144, 250, 26, USER_COL, "user: and one more thing")

    # tool boxes
    ty = 190
    for i, (cs, rs) in enumerate(zip(plan["calls"], plan["results"])):
        x = ox + i * 88
        if t < cs:
            col, label = (50, 50, 58), f"T{i + 1}"
            txtcol = DIM
        elif t < rs:
            col, label, txtcol = TOOL_RUN, f"T{i + 1} rtt", (20, 20, 26)
        else:
            col, label, txtcol = TOOL_DONE, f"T{i + 1} ok", (20, 20, 26)
        d.rounded_rectangle([x, ty, x + 78, ty + 40], radius=6, fill=col, outline=(80, 80, 90))
        d.text((x + 12, ty + 12), label, fill=txtcol, font=F)
        if t >= cs and t < rs:
            w = int(78 * (t - cs) / RTT_MS)
            d.rectangle([x, ty + 38, x + max(w, 2), ty + 40], fill=TOOL_RUN)

    # model reentry bar
    rey = 246
    if not cascade:
        active = any(cs <= t < rs for cs, rs in zip(plan["calls"], plan["results"]))
        d.rounded_rectangle(
            [ox, rey, ox + (4 * 88 - 10), rey + 18],
            radius=4,
            fill=(70, 120, 90) if active else (46, 46, 56),
        )
        d.text((ox + 8, rey + 1), "model reentry: 1x RTT then answer", fill=TXT, font=F)
    else:
        k_done = sum(1 for r in plan["results"] if t >= r)
        d.rounded_rectangle(
            [ox, rey, ox + (4 * 88 - 10), rey + 18],
            radius=4,
            fill=(120, 70, 70) if 0 < k_done < N_TOOLS else (46, 46, 56),
        )
        d.text(
            (ox + 8, rey + 1),
            f"model reentry #{k_done + 1} (serial)",
            fill=TXT if k_done else DIM,
            font=F,
        )

    # filler
    if t >= plan["filler"] and t < plan["answer"]:
        _bubble(d, ox, 280, 210, 26, (150, 150, 160), "bot: One moment. [filler]")

    # turn-2 ack (event-driven only)
    if (
        not cascade
        and plan.get("ack") is not None
        and plan["ack"] + 50 <= t < plan["ack_done"] + 250
    ):
        _bubble(d, ox + 60, 316, 200, 26, ACK_COL, "bot: Te escucho... [ack 0ms]")

    # final answer
    if t >= plan["answer"]:
        _bubble(d, ox, 352, 280, 26, ANSWER_COL, "bot: both orders shipped. ETA 2 days.")

    # totals stamp
    if t >= plan["answer_done"]:
        col = (240, 150, 150) if cascade else (150, 240, 180)
        d.text((ox, 396), f"TOTAL {plan['answer_done']} ms", fill=col, font=FT)
        if not cascade:
            d.text((ox, 424), "turn-2 acked at 0 ms (mid-pending)", fill=ACK_COL, font=F)
    elif t >= plan["answer"]:
        d.text(
            (ox, 396),
            "user heard nothing useful yet" if cascade else "answer landed at 1x RTT",
            fill=DIM,
            font=F,
        )


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cascade_vs_event_driven.gif"

    frames: list[Image.Image] = []
    t = 0
    while t <= TOTAL_MS:
        frames.append(draw_frame(t))
        t += MS_PER_FRAME
    frames += [frames[-1]] * 8  # hold the final state

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )

    size = out.stat().st_size
    with Image.open(out) as g:
        n = getattr(g, "n_frames", 1)
    print(f"gif: {out}")
    print(f"frames: {n}  size: {size / 1024:.0f} KiB")
    ok = size < 3_000_000 and n > 30
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
