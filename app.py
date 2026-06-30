import os
import io
import base64
import json
import time
import threading
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file, Response
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
# STATE TERBARU UNTUK OVERLAY / API (dipakai OBS Browser Source)
# ──────────────────────────────────────────────
LATEST_LOCK = threading.Lock()
LATEST = {
    "board_png": None,   # bytes PNG papan digital terbaru
    "fen": None,
    "board": [],
    "counts": {},
    "total_detections": 0,
    "updated_at": 0,      # epoch seconds, dipakai client utk polling/refresh
}

def set_latest(board_png_bytes, fen, board_list, counts, total):
    with LATEST_LOCK:
        LATEST["board_png"] = board_png_bytes
        LATEST["fen"] = fen
        LATEST["board"] = board_list
        LATEST["counts"] = counts
        LATEST["total_detections"] = total
        LATEST["updated_at"] = time.time()

def get_latest():
    with LATEST_LOCK:
        return dict(LATEST)

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

    "bishop":       {"unicode": "♗", "fen": "B", "side": "white"},
}

def get_piece(label: str):
    key = label.strip().lower().replace(" ", "-").replace("_", "-")
    return PIECE_MAP.get(label) or PIECE_MAP.get(key) or \
           {"unicode": "?", "fen": "?", "side": "unknown"}

# ──────────────────────────────────────────────
# BOARD GEOMETRY
# ──────────────────────────────────────────────
SQ   = 80
PAD  = 40
BOARD_PX = SQ * 8 + PAD * 2

C_LIGHT   = (240, 217, 181)
C_DARK    = ( 96, 133, 100)
C_BORDER  = ( 34,  40,  54)
C_BG      = ( 22,  27,  40)
C_COORD   = (180, 180, 180)
C_WHITE_P = (255, 255, 255)
C_BLACK_P = ( 20,  20,  30)

def draw_circle(draw, cx, cy, r, fill, outline=None, width=2):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=width)

def draw_pawn(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.5, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    draw.rounded_rectangle([cx-s*0.22, cy-s*0.1, cx+s*0.22, cy+s*0.55], radius=3, fill=fill, outline=ol, width=2)
    draw_circle(draw, cx, int(cy-s*0.35), int(s*0.38), fill, ol, 2)

def draw_rook(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    draw.rounded_rectangle([cx-s*0.5, cy-s*0.5, cx+s*0.5, cy+s*0.6], radius=4, fill=fill, outline=ol, width=2)
    tw = s * 0.25
    for dx in [-s*0.38, 0, s*0.38]:
        draw.rectangle([cx+dx-tw*0.5, cy-s*0.85, cx+dx+tw*0.5, cy-s*0.48], fill=fill, outline=ol, width=2)

def draw_knight(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
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
    draw_circle(draw, int(cx+s*0.2), int(cy-s*0.65), int(s*0.28), fill, ol, 2)

def draw_bishop(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    draw.rounded_rectangle([cx-s*0.65, cy+s*0.55, cx+s*0.65, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    pts = [(cx, cy-s*0.85), (cx+s*0.45, cy+s*0.0), (cx, cy+s*0.55), (cx-s*0.45, cy+s*0.0)]
    draw.polygon(pts, fill=fill, outline=ol)
    draw_circle(draw, cx, int(cy-s*0.88), int(s*0.14), fill, ol, 2)

def draw_queen(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    pts = [(cx-s*0.6, cy+s*0.55), (cx-s*0.7, cy-s*0.1),
           (cx, cy+s*0.2), (cx+s*0.7, cy-s*0.1), (cx+s*0.6, cy+s*0.55)]
    draw.polygon(pts, fill=fill, outline=ol)
    for i, dx in enumerate([-s*0.5, -s*0.25, 0, s*0.25, s*0.5]):
        h = s*0.55 if i % 2 == 0 else s*0.4
        draw.ellipse([cx+dx-s*0.1, cy-s*0.1-h-s*0.1, cx+dx+s*0.1, cy-s*0.1-h+s*0.1], fill=fill, outline=ol, width=2)
    draw.rounded_rectangle([cx-s*0.6, cy-s*0.15, cx+s*0.6, cy+s*0.2], radius=3, fill=fill, outline=ol, width=2)

def draw_king(draw, cx, cy, sq, fill, ol):
    s = sq * 0.42
    draw.rounded_rectangle([cx-s*0.7, cy+s*0.55, cx+s*0.7, cy+s*0.9], radius=4, fill=fill, outline=ol, width=2)
    draw.rounded_rectangle([cx-s*0.45, cy-s*0.5, cx+s*0.45, cy+s*0.6], radius=5, fill=fill, outline=ol, width=2)
    draw.rectangle([cx-s*0.1, cy-s*0.95, cx+s*0.1, cy-s*0.45], fill=fill, outline=ol, width=2)
    draw.rectangle([cx-s*0.32, cy-s*0.8, cx+s*0.32, cy-s*0.6], fill=fill, outline=ol, width=2)

DRAW_FN = {
    "P": draw_pawn,   "p": draw_pawn,
    "R": draw_rook,   "r": draw_rook,
    "N": draw_knight, "n": draw_knight,
    "B": draw_bishop, "b": draw_bishop,
    "Q": draw_queen,  "q": draw_queen,
    "K": draw_king,   "k": draw_king,
}

def render_board_png(board_list: list, transparent: bool = False) -> bytes:
    """Render flat modern chess board PNG. Returns bytes.
    transparent=True menghasilkan PNG dengan background alpha=0 di luar
    papan (cocok untuk OBS Browser/Image Source dengan area kosong tembus)."""
    size = BOARD_PX
    mode = "RGBA" if transparent else "RGB"
    bg = (0, 0, 0, 0) if transparent else C_BG
    img = Image.new(mode, (size, size), bg)
    draw = ImageDraw.Draw(img, "RGBA")

    draw.rounded_rectangle([PAD-6, PAD-6, size-PAD+6, size-PAD+6],
                            radius=6, fill=C_BORDER)

    for row in range(8):
        for col in range(8):
            x0 = PAD + col * SQ
            y0 = PAD + row * SQ
            color = C_LIGHT if (row + col) % 2 == 0 else C_DARK
            draw.rectangle([x0, y0, x0+SQ, y0+SQ], fill=color)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font = ImageFont.load_default()

    files = "abcdefgh"
    ranks = "87654321"
    for i in range(8):
        fx = PAD + i * SQ + SQ // 2 - 4
        draw.text((fx, size - PAD + 8), files[i], fill=C_COORD, font=font)
        ry = PAD + i * SQ + SQ // 2 - 7
        draw.text((PAD - 22, ry), ranks[i], fill=C_COORD, font=font)

    lookup = {}
    for p in board_list:
        lookup[(p["col"], p["row"])] = p

    for (col, row), piece in lookup.items():
        cx = PAD + col * SQ + SQ // 2
        cy = PAD + row * SQ + SQ // 2

        is_white = piece["side"] == "white"
        fill  = C_WHITE_P if is_white else C_BLACK_P
        ol    = (60, 60, 60) if is_white else (200, 200, 200)

        shadow_draw = ImageDraw.Draw(img, "RGBA")
        shadow_draw.ellipse([cx-SQ*0.3+3, cy+SQ*0.36+3, cx+SQ*0.3+3, cy+SQ*0.44+3],
                             fill=(0, 0, 0, 50))

        fen_char = piece.get("fen", "?")
        fn = DRAW_FN.get(fen_char)
        if fn:
            fn(draw, cx, cy, SQ, fill, ol)
        else:
            draw_circle(draw, cx, cy, int(SQ*0.35), fill, ol, 2)
            draw.text((cx-4, cy-7), "?", fill=ol, font=font)

    for i in range(9):
        x = PAD + i * SQ
        draw.line([x, PAD, x, PAD + 8*SQ], fill=(0,0,0,30), width=1)
        draw.line([PAD, x, PAD + 8*SQ, x], fill=(0,0,0,30), width=1)

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

    # Render papan digital (transparan, dipakai untuk overlay /board.png)
    board_png = render_board_png(board_list, transparent=True)

    ann_bgr  = draw_detections(bgr, results)

    # Simpan state terbaru → dipakai endpoint /board.png, /board, /api/board (OBS)
    set_latest(board_png, fen, board_list, counts, len(dets))

    return jsonify({
        "success":          True,
        "total_detections": len(dets),
        "counts":           counts,
        "detections":       dets,
        "image_annotated":  to_b64(ann_bgr, is_bgr=True),
        "image_original":   to_b64(bgr,     is_bgr=True),
        "board_png":        to_b64(board_png),
        "board":            board_list,
        "fen":              fen,
        "image_size":       {"width":w,"height":h},
    })

# ──────────────────────────────────────────────
# ENDPOINT UNTUK OBS / OVERLAY
# ──────────────────────────────────────────────
@app.route("/board.png")
def board_png_endpoint():
    """
    Mengembalikan PNG papan digital TERBARU (hasil deteksi terakhir).
    Pakai ini di OBS sebagai 'Image Source' (akan butuh refresh manual/plugin),
    atau lebih baik pakai /board (HTML) sebagai 'Browser Source' karena
    auto-refresh sendiri tanpa perlu plugin tambahan.
    """
    latest = get_latest()
    png = latest["board_png"]
    if png is None:
        # papan kosong default kalau belum ada deteksi sama sekali
        png = render_board_png([], transparent=True)
    resp = send_file(io.BytesIO(png), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

@app.route("/board")
def board_overlay_page():
    """
    Halaman HTML ringan, background transparan, yang otomatis
    me-refresh gambar papan setiap beberapa ratus ms.
    Tambahkan URL ini sebagai 'Browser Source' di OBS:
        http://<host>:<port>/board
    Centang 'Shutdown source when not visible' = OFF agar tetap update.
    """
    return render_template("board_overlay.html")

@app.route("/api/board")
def api_board():
    """
    JSON state papan terbaru (fen, board, updated_at) — berguna kalau kamu
    mau bikin overlay custom sendiri (mis. lewat OBS browser source lain,
    atau stream-deck/Lichess broadcast bridge) tanpa harus parse gambar.
    """
    latest = get_latest()
    return jsonify({
        "fen": latest["fen"],
        "board": latest["board"],
        "counts": latest["counts"],
        "total_detections": latest["total_detections"],
        "updated_at": latest["updated_at"],
        "board_png_url": "/board.png",
    })

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False,threaded=True)