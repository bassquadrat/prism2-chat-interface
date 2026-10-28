"""
Live PRISM2 chat demo on .svs whole-slide images: pick a slide, see a preview, type a prompt,
read the answer.

Slides are tiled and embedded with Virchow2 the same way as embed_wsi.py (20x, 0.5 mpp, 224 px,
tissue only), then PRISM2's get_response() answers the prompt for the whole slide. That's the
same call PRISM2 uses for zero-shot report generation.

Embedding a slide is slow the first time. After that it stays in memory, and on disk with
--cache-dir (same {slide_id}_tiles.pt format as embed_wsi.py, so slides embedded with it load
right away). Each question is answered on its own, the model doesn't see earlier messages.


Usage:
    python demo_server.py --image-root /path/to/svs_folder [--cache-dir /path/to/tile_cache]
then open the printed URL.
"""

from __future__ import annotations

import argparse
import hmac
import io
import json
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import openslide
import torch
from transformers import AutoModel, AutoProcessor

sys.path.append(str(Path(__file__).parent))
from embed_wsi import embed_tiles_batch, extract_tiles_svs, get_best_level, load_virchow2   # noqa: E402

TARGET_MPP = 0.5      # PRISM2 paper: 20x
TILE_SIZE = 224
MIN_TISSUE = 0.65
MAX_BODY_BYTES = 1_000_000
MAX_PROMPT_CHARS = 2000
PREVIEW_SIZE = (480, 480)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>PRISM2 demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 1.5rem auto; padding: 0 1rem; }
  h2 { margin-bottom: .2rem; }
  .note { color: #666; font-size: .85rem; margin-bottom: 1rem; }
  select { width: 100%; padding: .5rem; font-size: 1rem; }
  #preview { display: none; max-height: 220px; max-width: 100%; margin: .75rem 0; border: 1px solid #ccc; border-radius: 4px; }
  #chat { border: 1px solid #ddd; border-radius: 8px; padding: .75rem; height: 380px; overflow-y: auto; background: #fafafa; }
  .msg { margin: .5rem 0; padding: .6rem .8rem; border-radius: 8px; white-space: pre-wrap; max-width: 85%; }
  .user { background: #dbeafe; margin-left: auto; }
  .bot { background: #ececec; }
  .meta { color: #888; font-size: .75rem; margin-top: .25rem; }
  #row { display: flex; gap: .5rem; margin-top: .75rem; }
  textarea { flex: 1; padding: .5rem; font-size: 1rem; resize: vertical; }
  button { padding: 0 1.4rem; font-size: 1rem; cursor: pointer; }
</style></head><body>
<h2>PRISM2 demo</h2>
<div class="note">Each question is answered independently — no memory of earlier messages.
First question on a new slide takes a while (tiling + embedding).</div>
<select id="slide"><option>loading slides...</option></select>
<img id="preview" alt="slide preview">
<div id="chat"></div>
<div id="row">
  <textarea id="prompt" rows="2" placeholder="Type a prompt, e.g. Generate a pathology report for this specimen. (Enter = send)"></textarea>
  <button id="send">Send</button>
</div>
<script>
const $ = id => document.getElementById(id);
const token = new URLSearchParams(location.search).get("token") || "";
const api = (path, opts = {}) => fetch(path, {...opts, headers: {"X-Demo-Token": token, "Content-Type": "application/json"}});

function addMsg(cls, text, meta) {
  const d = document.createElement("div"); d.className = "msg " + cls; d.textContent = text;
  if (meta) { const m = document.createElement("div"); m.className = "meta"; m.textContent = meta; d.appendChild(m); }
  $("chat").appendChild(d); $("chat").scrollTop = $("chat").scrollHeight; return d;
}
function showPreview() {
  const s = $("slide").value; if (!s) return;
  $("preview").style.display = "block";
  $("preview").src = "/preview?slide=" + encodeURIComponent(s) + "&token=" + encodeURIComponent(token);
}
async function loadSlides() {
  const r = await api("/slides"); const j = await r.json();
  $("slide").innerHTML = "";
  for (const s of j.slides) { const o = document.createElement("option"); o.value = s.path; o.textContent = s.name; $("slide").appendChild(o); }
  showPreview();
}
$("slide").onchange = showPreview;

async function send() {
  const prompt = $("prompt").value.trim(); if (!prompt || $("send").disabled) return;
  $("prompt").value = ""; $("send").disabled = true;
  addMsg("user", prompt);
  const wait = addMsg("bot", "working...");
  const start = Date.now();
  const timer = setInterval(() => { wait.textContent = "working... " + Math.round((Date.now() - start) / 1000) + " s"; }, 1000);
  try {
    const r = await api("/chat", {method: "POST", body: JSON.stringify({slide: $("slide").value, prompt})});
    const j = await r.json();
    wait.textContent = j.error ? "Error: " + j.error : j.answer;
    if (!j.error) { const m = document.createElement("div"); m.className = "meta"; m.textContent = j.seconds.toFixed(1) + " s"; wait.appendChild(m); }
  } catch (e) { wait.textContent = "Error: " + e; }
  clearInterval(timer); $("send").disabled = false; $("prompt").focus();
}
$("send").onclick = send;
$("prompt").onkeydown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
loadSlides();
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live PRISM2 chat demo on .svs slides")
    parser.add_argument("--image-root", required=True, type=Path,
                         help="folder containing the .svs slides (searched recursively); nothing outside it is served")
    parser.add_argument("--cache-dir", type=Path, default=None,
                         help="optional folder for {slide_id}_tiles.pt embeddings — reused if present, "
                              "written after embedding a new slide")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                         help="max length of each answer (default: 200)")
    parser.add_argument("--batch-size", type=int, default=64, help="tiles per GPU forward pass (default: 64)")
    return parser.parse_args()


class Demo:
    def __init__(self, image_root: Path, cache_dir: Path | None, max_new_tokens: int, batch_size: int) -> None:
        self.image_root = image_root.resolve()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()   # one GPU job at a time
        self.embeddings: dict[Path, torch.Tensor] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.virchow2, self.transform = load_virchow2(self.device)
        self.prism2 = (
            AutoModel.from_pretrained("paige-ai/Prism2", trust_remote_code=True, torch_dtype="auto")
            .eval()
            .to(self.device)
        )
        self.processor = AutoProcessor.from_pretrained("paige-ai/Prism2", trust_remote_code=True)

    def list_slides(self) -> list[dict[str, str]]:
        paths = sorted(p for p in self.image_root.rglob("*") if p.suffix.lower() == ".svs" and p.is_file())
        return [{"path": p.relative_to(self.image_root).as_posix(), "name": p.name} for p in paths]

    def resolve_slide(self, rel: str) -> Path:
        path = (self.image_root / rel).resolve()
        if not path.is_relative_to(self.image_root) or path.suffix.lower() != ".svs" or not path.is_file():
            raise ValueError("slide not found")
        return path

    def thumbnail_png(self, rel: str) -> bytes:
        slide = openslide.OpenSlide(str(self.resolve_slide(rel)))
        try:
            buf = io.BytesIO()
            slide.get_thumbnail(PREVIEW_SIZE).convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
        finally:
            slide.close()

    def _slide_embeddings(self, path: Path) -> torch.Tensor:
        if path in self.embeddings:
            return self.embeddings[path]
        cache_file = self.cache_dir / f"{path.stem}_tiles.pt" if self.cache_dir else None
        if cache_file and cache_file.exists():
            embs = torch.load(cache_file, map_location="cpu", weights_only=True)["features"]
        else:
            slide = openslide.OpenSlide(str(path))
            level, _, read_size = get_best_level(slide, TARGET_MPP, TILE_SIZE)
            tiles = extract_tiles_svs(slide, level, TILE_SIZE, read_size, MIN_TISSUE)
            slide.close()
            if not tiles:
                raise ValueError("no tissue tiles found in this slide")
            embs = embed_tiles_batch(tiles, self.virchow2, self.transform, self.device, self.batch_size)
            if cache_file:
                torch.save({"features": embs, "label": "Unknown"}, cache_file)
        self.embeddings[path] = embs
        return embs

    def chat(self, rel: str, prompt: str) -> str:
        path = self.resolve_slide(rel)
        with self.lock:
            embs = self._slide_embeddings(path)
            batch = self.processor(tile_embeddings=[embs]).to(self.device)
            with torch.autocast("cuda", torch.bfloat16), torch.inference_mode():
                response = self.prism2.get_response(**batch, prompt=prompt, max_new_tokens=self.max_new_tokens)
        return response[0]


def make_handler(demo: Demo) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:   # request lines include the token, so don't log them
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _guard(self, query: dict[str, list[str]]) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0].lower()
            supplied = query.get("token", [""])[0] or self.headers.get("X-Demo-Token", "")
            if host not in {"localhost", "127.0.0.1"} or not hmac.compare_digest(supplied.encode(), demo.token.encode()):
                self._send(403, b"forbidden", "text/plain")
                return False
            return True

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if not self._guard(query):
                return
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/slides":
                self._send_json(200, {"slides": demo.list_slides()})
            elif url.path == "/preview":
                try:
                    self._send(200, demo.thumbnail_png(query.get("slide", [""])[0]), "image/png")
                except Exception:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            url = urlparse(self.path)
            if not self._guard(parse_qs(url.query)):
                return
            if url.path != "/chat":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > MAX_BODY_BYTES:
                    raise ValueError("request too large")
                data = json.loads(self.rfile.read(length))
                prompt = str(data["prompt"])[:MAX_PROMPT_CHARS]
                start = time.time()
                answer = demo.chat(str(data["slide"]), prompt)
                self._send_json(200, {"answer": answer, "seconds": time.time() - start})
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                print(f"chat error: {e!r}")
                self._send_json(500, {"error": "inference failed — see server log"})

    return Handler


def main() -> None:
    args = parse_args()
    demo = Demo(args.image_root, args.cache_dir, args.max_new_tokens, args.batch_size)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(demo))
    print(f"Slides under {demo.image_root}: {len(demo.list_slides())}")
    print(f"Ready — open http://localhost:{args.port}/?token={demo.token}")
    server.serve_forever()


if __name__ == "__main__":
    main()
