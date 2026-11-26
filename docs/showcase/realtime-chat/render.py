"""Replay a stream recorded by capture.py at its recorded speed, as an animated WebP of a chat window.

    python render.py stream.json chat.webp --pack UD-IQ3_XXS

Every frame shows the text that had arrived by that moment of the recording; nothing is sped up or smoothed.
The closing line gives the server's token count and the measured time.
"""
import argparse
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1000, 600, 25
BG, PANEL, INK, DIM, ACCENT, USER_BG, CODE_BG = "#0f1719", "#16232a", "#e8eef0", "#8fa3ab", "#76b900", "#24414d", "#0b1214"
FONTS = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


SANS, SANS_B, MONO = font("segoeui.ttf", 21), font("segoeuib.ttf", 21), font("consola.ttf", 18)
SMALL, SMALL_B, TITLE = font("segoeui.ttf", 16), font("segoeuib.ttf", 18), font("segoeuib.ttf", 22)


def wrap(text, fnt, width):
    """Greedy wrap that keeps leading spaces (code) and breaks over-long words."""
    if not text:
        return [""]
    indent = len(text) - len(text.lstrip(" "))
    words, lines, line = re.split(r"(\s+)", text.strip(" ")), [], " " * indent
    for word in words:
        trial = line + word
        if fnt.getlength(trial) <= width or not line.strip():
            line = trial
        else:
            lines.append(line.rstrip())
            line = " " * indent + word.lstrip()
    lines.append(line.rstrip())
    return lines


def layout(text, width):
    """Markdown-lite: fenced code in monospace on a dark band, headings and **bold** lines in bold, the rest plain."""
    rows, code = [], False
    for raw in text.split("\n"):
        if raw.strip().startswith("```"):
            code = not code
            rows.append(("gap", "", 6))
            continue
        if code:
            for part in wrap(raw.replace("\t", "    "), MONO, width - 24):
                rows.append(("code", part, 23))
            continue
        line = raw.replace("**", "").replace("`", "")
        bold = raw.lstrip().startswith("#")
        line = line.lstrip("# ") if bold else line
        if not line.strip():
            rows.append(("gap", "", 10))
            continue
        for part in wrap(line, SANS_B if bold else SANS, width):
            rows.append(("bold" if bold else "text", part, 30))
    return rows


def frame(prompt_rows, rows, header, status, cursor):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=PANEL)
    d.text((24, 16), "Basalt RTX Spark", font=TITLE, fill=ACCENT)
    d.text((222, 20), header, font=SMALL, fill=DIM)
    # user bubble, right-aligned
    bw = max(SANS.getlength(r) for r in prompt_rows) + 32
    by = 76
    bh = 30 * len(prompt_rows) + 20
    d.rounded_rectangle([W - 24 - bw, by, W - 24, by + bh], 12, fill=USER_BG)
    for i, r in enumerate(prompt_rows):
        d.text((W - 24 - bw + 16, by + 10 + 30 * i), r, font=SANS, fill=INK)
    # assistant text, scrolled so the newest line stays visible
    top, bottom = by + bh + 22, H - 52
    total = sum(h for _, _, h in rows)
    y = top - max(0, total - (bottom - top))
    last = (24, top)
    for kind, text, h in rows:
        if y + h > top - 4 and y < bottom:
            if kind == "code":
                d.rectangle([24, y - 2, W - 24, y + h - 2], fill=CODE_BG)
                d.text((36, y), text, font=MONO, fill="#c9e6a3")
                last = (36 + MONO.getlength(text), y)
            elif kind != "gap":
                f = SANS_B if kind == "bold" else SANS
                d.text((24, y), text, font=f, fill=INK)
                last = (24 + f.getlength(text), y)
        y += h
    if cursor:
        d.rectangle([last[0] + 3, last[1] + 4, last[0] + 13, last[1] + 25], fill=ACCENT)
    # once the answer scrolls, keep the prompt bubble on top of it
    if total > bottom - top:
        d.rectangle([0, 56, W, top - 6], fill=BG)
        d.rounded_rectangle([W - 24 - bw, by, W - 24, by + bh], 12, fill=USER_BG)
        for i, r in enumerate(prompt_rows):
            d.text((W - 24 - bw + 16, by + 10 + 30 * i), r, font=SANS, fill=INK)
    d.rectangle([0, H - 44, W, H], fill=PANEL)
    d.text((24, H - 34), status, font=SMALL_B, fill=INK)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("out")
    ap.add_argument("--pack", default="UD-IQ3_XXS")
    ap.add_argument("--hold", type=float, default=3.5)
    a = ap.parse_args()
    rec = json.load(open(a.stream, encoding="utf-8"))
    ev, s = rec["events"], rec["summary"]
    header = f"Qwen3.8-Flash-Next · {a.pack} · RTX Spark N1X laptop · real time"
    prompt_rows = [r for line in rec["prompt"].split("\n") for r in wrap(line, SANS, 640)]
    end = ev[-1]["t"]
    frames, i, text = [], 0, ""
    n = int((end + a.hold) * FPS) + 1
    for k in range(n):
        t = k / FPS
        while i < len(ev) and ev[i]["t"] <= t:
            text += ev[i]["text"]
            i += 1
        if t < ev[0]["t"]:
            status = f"reading the prompt · {t:4.2f} s"
        elif t <= end:
            status = f"streaming · {t:4.2f} s"
        else:
            status = (f"{s['completion_tokens']} tokens in {s['total_s']:.2f} s  ·  {s['decode_tok_s']:.0f} tok/s decode  ·  "
                      f"first token {s['first_token_s']:.2f} s")
        frames.append(frame(prompt_rows, layout(text, W - 48), header, status, cursor=(t <= end and (k // 6) % 2 == 0)))
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, quality=72, method=4)
    frames[-1].save(os.path.splitext(a.out)[0] + "-poster.png")
    print(len(frames), "frames,", os.path.getsize(a.out) // 1024, "KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
