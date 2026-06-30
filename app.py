import os
import io
import base64
import json
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH = "model/best.pt"
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.35))
IMG_SIZE    = int(os.environ.get("IMG_SIZE", 640))
MAX_UPLOAD  = 16 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model tidak ditemukan: '{MODEL_PATH}'")

model      = YOLO(MODEL_PATH)
CLASS_NAMES = model.names
print(f"[✓] Model: {MODEL_PATH}  |  Classes: {CLASS_NAMES}")

# ──────────────────────────────────────────────
# PIECE MAP  →  sesuaikan key dengan nama kelas model
# ──────────────────────────────────────────────

PIECE_MAP = {
    # White
    "white-king":   {"unicode": "♔", "fen": "K", "side": "white"},
    "white-queen":  {"unicode": "♕", "fen": "Q", "side": "white"},
    "white-rook":   {"unicode": "♖", "fen": "R", "side": "white"},
    "white-bishop": {"unicode": "♗", "fen": "B", "side": "white"},
    "white-knight": {"unicode": "♘", "fen": "N", "side": "white"},
    "white-pawn":   {"unicode": "♙", "fen": "P", "side": "white"},

    # Black
    "black-king":   {"unicode": "♚", "fen": "k", "side": "black"},
    "black-queen":  {"unicode": "♛", "fen": "q", "side": "black"},
    "black-rook":   {"unicode": "♜", "fen": "r", "side": "black"},
    "black-bishop": {"unicode": "♝", "fen": "b", "side": "black"},
    "black-knight": {"unicode": "♞", "fen": "n", "side": "black"},
    "black-pawn":   {"unicode": "♟", "fen": "p", "side": "black"},

    # Kelas "bishop" tanpa warna (index 0 di model kamu) — kemungkinan
    # class duplikat/noise dari training. Diberi default warna putih
    # supaya tetap kebaca di papan, bukan jadi simbol "?".
    "bishop":       {"unicode": "♗", "fen": "B", "side": "white"},
}

def get_piece(label: str):
    # normalisasi: lowercase + spasi/underscore -> hyphen, biar fleksibel
    key = label.strip().lower().replace(" ", "-").replace("_", "-")
    return PIECE_MAP.get(label) or PIECE_MAP.get(key) or \
           {"unicode": "?", "fen": "?", "side": "unknown"}
# ──────────────────────────────────────────────
# BOARD GEOMETRY
# ──────────────────────────────────────────────
SQ   = 80          # px per square
PAD  = 40          # border padding
BOARD_PX = SQ * 8 + PAD * 2   # 720

# Modern flat palette
C_LIGHT   = (240, 217, 181)   # warm cream
C_DARK    = ( 96, 133, 100)   # sage green (not brown — modern)
C_BORDER  = ( 34,  40,  54)   # dark navy border
C_BG      = ( 22,  27,  40)   # outer bg
C_COORD   = (180, 180, 180)   # coordinate labels
C_WHITE_P = (255, 255, 255)   # white piece fill
C_BLACK_P = ( 20,  20,  30)   # black piece fill
C_SHADOW  = (  0,   0,   0, 60)

# Piece shapes: drawn with PIL as flat vector-style SVG-like shapes
# Each piece is a lambda(draw, cx, cy, sq, color, shadow_color)

def draw_circle(draw, cx, cy, r, fill, outline=None, width=2):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=width)

def draw_pawn(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    # base
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.5, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    # stem
    draw.rounded_rectangle([cx-s*0.22, cy-s*0.1, cx+s*0.22, cy+s*0.55], radius=3, fill=fill, outline=ol, width=2)
    # head
    draw_circle(draw, cx, int(cy-s*0.35), int(s*0.38), fill, ol, 2)

def draw_rook(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    # base
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    # body
    draw.rounded_rectangle([cx-s*0.5, cy-s*0.5, cx+s*0.5, cy+s*0.6], radius=4, fill=fill, outline=ol, width=2)
    # battlements (3 teeth)
    tw = s * 0.25
    for dx in [-s*0.38, 0, s*0.38]:
        draw.rectangle([cx+dx-tw*0.5, cy-s*0.85, cx+dx+tw*0.5, cy-s*0.48], fill=fill, outline=ol, width=2)

def draw_knight(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    # base
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    # body (leaning right)
    pts = [
        (cx-s*0.3, cy+s*0.55),
        (cx+s*0.45, cy-s*0.1),
        (cx+s*0.55, cy-s*0.6),
        (cx+s*0.1,  cy-s*0.85),
        (cx-s*0.2,  cy-s*0.55),
        (cx+s*0.1,  cy-s*0.25),
        (cx-s*0.3,  cy+s*0.1),
    ]
    draw.polygon(pts, fill=fill, outline=ol)
    # head circle
    draw_circle(draw, int(cx+s*0.2), int(cy-s*0.65), int(s*0.28), fill, ol, 2)

def draw_bishop(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    # base
    draw.rounded_rectangle([cx-s*0.65, cy+s*0.55, cx+s*0.65, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    # body diamond-ish
    pts = [(cx, cy-s*0.85), (cx+s*0.45, cy+s*0.0), (cx, cy+s*0.55), (cx-s*0.45, cy+s*0.0)]
    draw.polygon(pts, fill=fill, outline=ol)
    # top dot
    draw_circle(draw, cx, int(cy-s*0.88), int(s*0.14), fill, ol, 2)

def draw_queen(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    # base
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    # skirt
    pts = [(cx-s*0.6, cy+s*0.55), (cx-s*0.7, cy-s*0.1),
           (cx, cy+s*0.2), (cx+s*0.7, cy-s*0.1), (cx+s*0.6, cy+s*0.55)]
    draw.polygon(pts, fill=fill, outline=ol)
    # crown with 5 points
    for i, dx in enumerate([-s*0.5, -s*0.25, 0, s*0.25, s*0.5]):
        h = s*0.55 if i % 2 == 0 else s*0.4
        draw.ellipse([cx+dx-s*0.1, cy-s*0.1-h-s*0.1, cx+dx+s*0.1, cy-s*0.1-h+s*0.1], fill=fill, outline=ol, width=2)
    draw.rounded_rectangle([cx-s*0.6, cy-s*0.15, cx+s*0.6, cy+s*0.2], radius=3, fill=fill, outline=ol, width=2)

def draw_king(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    # base
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    # body
    draw.rounded_rectangle([cx-s*0.45, cy-s*0.5, cx+s*0.45, cy+s*0.6], radius=5, fill=fill, outline=ol, width=2)
    # cross vertical
    draw.rectangle([cx-s*0.1, cy-s*0.95, cx+s*0.1, cy-s*0.45], fill=fill, outline=ol, width=2)
    # cross horizontal
    draw.rectangle([cx-s*0.32, cy-s*0.8, cx+s*0.32, cy-s*0.6], fill=fill, outline=ol, width=2)

DRAW_FN = {
    "P": draw_pawn,   "p": draw_pawn,
    "R": draw_rook,   "r": draw_rook,
    "N": draw_knight, "n": draw_knight,
    "B": draw_bishop, "b": draw_bishop,
    "Q": draw_queen,  "q": draw_queen,
    "K": draw_king,   "k": draw_king,
}

# ──────────────────────────────────────────────
# RENDER BOARD PNG
# ──────────────────────────────────────────────
def render_board_png(board_list: list) -> bytes:
    """Render flat modern chess board PNG. Returns bytes."""
    size = BOARD_PX
    img = Image.new("RGB", (size, size), C_BG)
    draw = ImageDraw.Draw(img, "RGBA")

    # Border rectangle
    draw.rounded_rectangle([PAD-6, PAD-6, size-PAD+6, size-PAD+6],
                            radius=6, fill=C_BORDER)

    # Squares
    for row in range(8):
        for col in range(8):
            x0 = PAD + col * SQ
            y0 = PAD + row * SQ
            color = C_LIGHT if (row + col) % 2 == 0 else C_DARK
            draw.rectangle([x0, y0, x0+SQ, y0+SQ], fill=color)

    # Coordinate labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font = ImageFont.load_default()

    files = "abcdefgh"
    ranks = "87654321"
    for i in range(8):
        # file labels bottom
        fx = PAD + i * SQ + SQ // 2 - 4
        draw.text((fx, size - PAD + 8), files[i], fill=C_COORD, font=font)
        # rank labels left
        ry = PAD + i * SQ + SQ // 2 - 7
        draw.text((PAD - 22, ry), ranks[i], fill=C_COORD, font=font)

    # Build lookup col,row → piece
    lookup = {}
    for p in board_list:
        lookup[(p["col"], p["row"])] = p

    # Draw pieces
    for (col, row), piece in lookup.items():
        cx = PAD + col * SQ + SQ // 2
        cy = PAD + row * SQ + SQ // 2

        is_white = piece["side"] == "white"
        fill  = C_WHITE_P if is_white else C_BLACK_P
        ol    = (60, 60, 60) if is_white else (200, 200, 200)

        # shadow
        shadow_draw = ImageDraw.Draw(img, "RGBA")
        shadow_draw.ellipse([cx-SQ*0.3+3, cy+SQ*0.36+3, cx+SQ*0.3+3, cy+SQ*0.44+3],
                             fill=(0, 0, 0, 50))

        fen_char = piece.get("fen", "?")
        fn = DRAW_FN.get(fen_char)
        if fn:
            fn(draw, cx, cy, SQ, fill, ol)
        else:
            # fallback: circle with "?"
            draw_circle(draw, cx, cy, int(SQ*0.35), fill, ol, 2)
            draw.text((cx-4, cy-7), "?", fill=ol, font=font)

    # Subtle grid lines
    for i in range(9):
        x = PAD + i * SQ
        draw.line([x, PAD, x, PAD + 8*SQ], fill=(0,0,0,30), width=1)
        draw.line([PAD, x, PAD + 8*SQ, x], fill=(0,0,0,30), width=1)

    # Watermark
    draw.text((size-110, size-18), "Chess Digitizer", fill=(80,80,100), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

# ──────────────────────────────────────────────
# BOARD LOGIC
# ──────────────────────────────────────────────
def estimate_board_bounds(dets, img_w, img_h):
    if not dets:
        return 0, 0, img_w, img_h
    x1s=[d["bbox"][0] for d in dets]; y1s=[d["bbox"][1] for d in dets]
    x2s=[d["bbox"][2] for d in dets]; y2s=[d["bbox"][3] for d in dets]
    bx=float(np.percentile(x1s,5)); by=float(np.percentile(y1s,5))
    bx2=float(np.percentile(x2s,95)); by2=float(np.percentile(y2s,95))
    px=(bx2-bx)*0.06; py=(by2-by)*0.06
    return max(0,bx-px), max(0,by-py), min(img_w,bx2+px)-max(0,bx-px), min(img_h,by2+py)-max(0,by-py)

def build_board(dets, img_w, img_h):
    bx,by,bw,bh = estimate_board_bounds(dets, img_w, img_h)
    board = {}
    for det in dets:
        x1,y1,x2,y2 = det["bbox"]
        cx=(x1+x2)/2; cy=(y1+y2)/2
        col = max(0,min(7,int((cx-bx)/bw*8)))
        row = max(0,min(7,int((cy-by)/bh*8)))
        key = f"{col},{row}"
        if key not in board or det["confidence"] > board[key]["confidence"]:
            piece = get_piece(det["label"])
            board[key] = {**piece, "col":col,"row":row,
                          "confidence":det["confidence"],"label":det["label"]}
    placed = sorted(board.values(), key=lambda x:(x["row"],x["col"]))

    # FEN
    fen_rows=[]
    for r in range(8):
        empty=0; row_str=""
        for c in range(8):
            k=f"{c},{r}"
            if k in board:
                if empty: row_str+=str(empty); empty=0
                row_str+=board[k]["fen"]
            else: empty+=1
        if empty: row_str+=str(empty)
        fen_rows.append(row_str)
    fen = "/".join(fen_rows) + " w KQkq - 0 1"
    return placed, fen

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
PALETTE=[(52,211,153),(251,191,36),(239,68,68),(59,130,246),(167,85,247),(236,113,44)]
def cls_color(i): c=PALETTE[i%len(PALETTE)]; return c

def load_image(b):
    pil=Image.open(io.BytesIO(b)).convert("RGB")
    bgr=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
    return pil,bgr

def draw_detections(bgr, results):
    out=bgr.copy()
    for box in results[0].boxes:
        x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
        cid=int(box.cls[0]); conf=float(box.conf[0])
        label=CLASS_NAMES.get(cid,str(cid)); color=cls_color(cid)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        txt=f"{label} {conf:.0%}"
        (tw,th),bl=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        ty=max(y1-4,th+4)
        cv2.rectangle(out,(x1,ty-th-4),(x1+tw+6,ty+bl),color,-1)
        cv2.putText(out,txt,(x1+3,ty-2),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
    return out

def to_b64(img_bytes_or_bgr, is_bgr=False):
    if is_bgr:
        ok,buf=cv2.imencode(".jpg",img_bytes_or_bgr,[cv2.IMWRITE_JPEG_QUALITY,88])
        return base64.b64encode(buf).decode()
    return base64.b64encode(img_bytes_or_bgr).decode()

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", classes=list(CLASS_NAMES.values()))

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error":"Tidak ada file 'image'"}),400
    file=request.files["image"]
    if not file.filename:
        return jsonify({"error":"Nama file kosong"}),400
    conf_req=float(request.args.get("conf",CONF_THRESH))
    raw=file.read()
    try: pil,bgr=load_image(raw)
    except Exception as e: return jsonify({"error":f"Gagal baca: {e}"}),400

    h,w=bgr.shape[:2]
    results=model.predict(source=pil,imgsz=IMG_SIZE,conf=conf_req,verbose=False)

    dets=[]; counts={}
    for box in results[0].boxes:
        cid=int(box.cls[0]); conf=float(box.conf[0])
        label=CLASS_NAMES.get(cid,str(cid))
        x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
        color=cls_color(cid)
        dets.append({"label":label,"confidence":round(conf,4),"bbox":[x1,y1,x2,y2],
                     "color":"#{:02x}{:02x}{:02x}".format(*color)})
        counts[label]=counts.get(label,0)+1

    board_list, fen = build_board(dets, w, h)

    # Render digital board PNG
    board_png = render_board_png(board_list)

    ann_bgr  = draw_detections(bgr, results)

    return jsonify({
        "success":          True,
        "total_detections": len(dets),
        "counts":           counts,
        "detections":       dets,
        "image_annotated":  to_b64(ann_bgr, is_bgr=True),
        "image_original":   to_b64(bgr,     is_bgr=True),
        "board_png":        to_b64(board_png),          # ← PNG papan digital
        "board":            board_list,
        "fen":              fen,
        "image_size":       {"width":w,"height":h},
    })

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)