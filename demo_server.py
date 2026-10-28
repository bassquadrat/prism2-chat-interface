"""
Live PRISM2 chat demo on .svs whole-slide images: pick a slide, see a preview, type a prompt,
read the answer.

Slides are tiled and embedded with Virchow2 the same way as embed_wsi.py (20x, 0.5 mpp, 224 px,
tissue only), then PRISM2's get_response() answers the prompt for the whole slide. That's the
same call PRISM2 uses for zero-shot report generation.

Embedding a slide is slow the first time. After that it stays in memory, and on disk with
--cache-dir (same {slide_id}_tiles.pt format as embed_wsi.py, so slides embedded with it load
right away). Each question is answered on its own, the model doesn't see earlier messages.

--model smolvlm is for local tests without an NVIDIA GPU: it skips Virchow2 and PRISM2 and
asks SmolVLM-256M (a small general vision-language model) about the slide thumbnail instead.
Its answers say nothing about PRISM2, so the page title says which model is running.

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
from PIL import Image
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

sys.path.append(str(Path(__file__).parent))
from embed_wsi import embed_tiles_batch, extract_tiles_svs, get_best_level, load_virchow2   # noqa: E402

TARGET_MPP = 0.5      # PRISM2 paper: 20x
TILE_SIZE = 224
MIN_TISSUE = 0.65
MAX_BODY_BYTES = 1_000_000
MAX_PROMPT_CHARS = 2000
PREVIEW_SIZE = (480, 480)
SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
TITLES = {"prism2": "PRISM2 demo", "smolvlm": "Local test: SmolVLM-256M"}

PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


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
    parser.add_argument("--model", choices=list(TITLES), default="prism2",
                         help="smolvlm = small local test model on the thumbnail, runs on Mac/CPU (default: prism2)")
    return parser.parse_args()


class Demo:
    def __init__(self, image_root: Path, cache_dir: Path | None, max_new_tokens: int, batch_size: int,
                 model: str = "prism2") -> None:
        self.image_root = image_root.resolve()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()   # one GPU job at a time
        self.embeddings: dict[Path, torch.Tensor] = {}
        self.model = model

        if model == "smolvlm":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Device: {self.device}")
            self.processor = AutoProcessor.from_pretrained(SMOLVLM)
            self.smolvlm = AutoModelForImageTextToText.from_pretrained(SMOLVLM).eval().to(self.device)
            return

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

    def _thumbnail(self, path: Path) -> Image.Image:
        slide = openslide.OpenSlide(str(path))
        try:
            return slide.get_thumbnail(PREVIEW_SIZE).convert("RGB")
        finally:
            slide.close()

    def thumbnail_png(self, rel: str) -> bytes:
        buf = io.BytesIO()
        self._thumbnail(self.resolve_slide(rel)).save(buf, format="PNG")
        return buf.getvalue()

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
            if self.model == "smolvlm":
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=text, images=[self._thumbnail(path)], return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    out = self.smolvlm.generate(**inputs, max_new_tokens=self.max_new_tokens)
                new_tokens = out[:, inputs["input_ids"].shape[1]:]
                return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
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
                page = PAGE.replace("PRISM2 demo", TITLES[demo.model])
                self._send(200, page.encode(), "text/html; charset=utf-8")
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
    demo = Demo(args.image_root, args.cache_dir, args.max_new_tokens, args.batch_size, args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(demo))
    print(f"Slides under {demo.image_root}: {len(demo.list_slides())}")
    print(f"Ready — open http://localhost:{args.port}/?token={demo.token}")
    server.serve_forever()


if __name__ == "__main__":
    main()
