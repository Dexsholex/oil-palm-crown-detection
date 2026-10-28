---
title: Oil Palm Counter
emoji: 🌴
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: agpl-3.0
short_description: Count oil palms in drone images, in your browser
models:
  - dexsholex/oil-palm-crown-detector
tags:
  - object-detection
  - remote-sensing
  - drone
  - agriculture
  - oil-palm
  - tree-counting
  - onnx
---

# Oil Palm Counter

Count oil palms in a drone image, in your browser. The image is never uploaded: the
[oil palm crown detector](https://huggingface.co/dexsholex/oil-palm-crown-detector) runs on the
visitor's own computer with [ONNX Runtime Web](https://onnxruntime.ai/), on the graphics card
(WebGPU) where available and on the processor otherwise.

This folder is a static page with no server or build step:

- `index.html`: the page.
- `demo.js`: loads the model, cuts the image into overlapping 1280 px tiles as the full
  pipeline does, runs the detector on each tile, merges palms seen by two tiles, and draws the
  boxes, the count and a CSV.

Source, pipeline and evaluation: https://github.com/Dexsholex/oil-palm-crown-detection (the
`demo/` folder). To run it locally, serve this folder with any static web server, for example
`python -m http.server`, and open the address it prints.

Built by Oluwasola Oyinloye ([ORCID 0009-0003-4990-9944](https://orcid.org/0009-0003-4990-9944)).
Cite: [doi:10.5281/zenodo.23704654](https://doi.org/10.5281/zenodo.23704654). Collaboration:
[o.oyin1115@gmail.com](mailto:o.oyin1115@gmail.com).
