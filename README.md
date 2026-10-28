# PRISM2 Chat Interface

A chat interface for [PRISM2](https://huggingface.co/paige-ai/Prism2), Paige AI's pathology
vision-language foundation model. Pick a whole-slide image, ask a question and get an answer.

Built for a live demo in a talk about PRISM2, published in Nature Medicine as
[End-to-end multimodal pathology foundation model with clinical dialogue](https://www.nature.com/articles/s41591-026-04521-4)
(Vorontsov et al., 2026).

## How it works

1. The `.svs` slide is tiled and embedded with
   [Virchow2](https://huggingface.co/paige-ai/Virchow2): 224 px tiles at 20x (0.5 mpp), tissue
   only. Tile size and magnification follow the
   [PRISM2 paper](https://www.nature.com/articles/s41591-026-04521-4) (Methods), but tissue detection is
   not the same: the paper uses a trained segmentation model, here a simple HSV color
   threshold stands in for it.
2. The tile embeddings go to PRISM2's `get_response()` together with your prompt. PRISM2 uses
   the same call for its zero-shot report generation.
3. Embeddings are computed once per slide and cached in memory, and optionally on disk, so
   follow-up questions are fast. Each question is still answered on its own: the model doesn't
   see earlier messages.

`embed_wsi.py` does the same tiling and embedding as a standalone CLI, for batch-embedding a
folder of slides without the web app.

## Setup

```bash
pip install torch
pip install -r requirements.txt --no-build-isolation
```

`flash-attn` (required by PRISM2's model code) is built against the installed torch, so torch has
to be installed first. It only installs on Linux with an NVIDIA GPU and the build can take a while.

You also need:
- An NVIDIA GPU with CUDA. Both models are large and it hasn't been tested on CPU.
- Access to `paige-ai/Prism2` and `paige-ai/Virchow2` on Hugging Face. Both are gated, so
  request access on the model pages and run `huggingface-cli login`.
- The OpenSlide system library (`apt install openslide-tools` or `brew install openslide`),
  which `openslide-python` needs.

## Usage

```bash
python demo_server.py --image-root /path/to/folder_of_svs_files [--cache-dir /path/to/tile_cache]
```

Then open the URL it prints. It contains the access token, so copy the whole URL, not just
host and port.

Other flags: `--port` (default 8000), `--max-new-tokens` (default 200, max answer length),
`--batch-size` (default 64, tiles per GPU forward pass).
