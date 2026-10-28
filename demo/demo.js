// Oil palm counter: runs the oil palm crown detector inside the browser with ONNX Runtime Web.
// Nothing is uploaded. The image is read, tiled and processed on the visitor's own computer;
// the only downloads are the runtime (from jsDelivr) and the model (from Hugging Face).
//
// The steps mirror the Python pipeline: cut the image into overlapping 1280 px tiles, detect
// palms in each tile, keep each palm only in the tile that "owns" it, then merge duplicates.

const MODEL_URL = "https://huggingface.co/dexsholex/oil-palm-crown-detector/resolve/main/oil_palm_crown.onnx";
const MODEL_CACHE = "oil-palm-model-v1";   // rename if the model file on Hugging Face is replaced
const MODEL_BYTES = 45250560;              // fallback for the download progress bar
const TILE = 1280, OVERLAP = 256;          // same tiling as the training/production pipeline
const TILE_IOU = 0.45;                     // overlap removal within a tile (pipeline setting)
const MERGE_IOU = 0.3;                     // duplicate merge across tiles (pipeline setting)
const MIN_CONF = 0.05;                     // run once at the slider's floor, then filter
const MAX_DET = 300;                       // per tile, as Ultralytics
const TRAIN_GSD = 2.13;                    // cm per pixel of the training imagery
const MAX_PIXELS = 50e6;                   // keeps memory use sensible in a browser tab
const PAD = 114 / 255;                     // grey padding for tiles smaller than 1280 px
const DISPLAY_MAX = 2400;                  // longest side of the result picture

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const $ = (id) => document.getElementById(id);
const els = {
  drop: $("drop"), file: $("file"), fileName: $("file-name"), conf: $("conf"), confValue: $("conf-value"),
  gsd: $("gsd"), run: $("run"), status: $("status"), progress: $("progress"), result: $("result"),
  canvas: $("canvas"), count: $("count"), details: $("details"), csv: $("csv"), png: $("png"),
};

let sessionPromise = null;   // the model loads once and is reused
let backend = "";            // "webgpu" or "wasm", shown in the result details
let currentFile = null;
let busy = false;
let run = null;              // the last result: boxes at MIN_CONF plus what is needed to redraw

// ---------- model ----------

function getSession() {
  if (!sessionPromise) {
    sessionPromise = createSession().catch((err) => { sessionPromise = null; throw err; });
  }
  return sessionPromise;
}

async function createSession() {
  const bytes = await fetchModel();
  setStatus("Starting the model…");
  const forced = new URLSearchParams(location.search).get("backend");   // ?backend=wasm for testing
  const gpu = !forced && navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
  const order = forced ? [forced] : gpu ? ["webgpu", "wasm"] : ["wasm"];
  // On the processor a tile takes several seconds, so run it in a background worker and keep
  // the page responsive. (This has to be decided before the first model starts.)
  if (order[0] === "wasm") ort.env.wasm.proxy = true;
  for (const ep of order) {
    try {
      const session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
      backend = ep;
      return session;
    } catch (err) {
      console.warn(`Could not start the model with ${ep}:`, err);
    }
  }
  throw new Error("This browser could not start the model. Try a recent Chrome, Edge or Firefox.");
}

// Downloads the model once; later visits read it from the browser's cache storage.
async function fetchModel() {
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(MODEL_URL);
    if (hit) {
      setStatus("Loading the model from this browser…");
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch {
    cache = null;   // cache storage can be unavailable (private windows, file://)
  }

  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`The model download failed (HTTP ${res.status}). Check the connection and try again.`);
  const total = Number(res.headers.get("content-length")) || MODEL_BYTES;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setStatus(`Downloading the model (first visit only): ${mb(received)} of ${mb(total)} MB`, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }

  try {
    await cache?.put(MODEL_URL, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    // not fatal: the next visit downloads it again
  }
  return bytes;
}

// ---------- detection ----------

// Tile offsets covering 0..length with the pipeline's overlap; the last tile sits flush with the edge.
function tileStarts(length) {
  if (length <= TILE) return [0];
  const starts = [];
  for (let s = 0; s <= length - TILE; s += TILE - OVERLAP) starts.push(s);
  if (starts[starts.length - 1] + TILE < length) starts.push(length - TILE);
  return starts;
}

function iou(a, b) {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// Greedy non-max suppression: keep the highest score, drop boxes overlapping it by more than `limit`.
function nms(boxes, limit) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of boxes) {
    if (kept.every((k) => iou(k, box) <= limit)) kept.push(box);
  }
  return kept;
}

// The model returns 1×5×N: rows are box centre x, centre y, width, height and palm score.
function decode(output, conf, tileW, tileH) {
  const data = output.data, n = output.dims[2];
  const boxes = [];
  for (let i = 0; i < n; i++) {
    const score = data[4 * n + i];
    if (score <= conf) continue;
    const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
    boxes.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score });
  }
  const clamp = (v, max) => Math.min(Math.max(v, 0), max);
  return nms(boxes, TILE_IOU).slice(0, MAX_DET).map((b) => ({
    x1: clamp(b.x1, tileW), y1: clamp(b.y1, tileH), x2: clamp(b.x2, tileW), y2: clamp(b.y2, tileH), score: b.score,
  }));
}

async function detect(bitmap, scale) {
  const W = Math.round(bitmap.width * scale), H = Math.round(bitmap.height * scale);
  const work = document.createElement("canvas");
  work.width = W;
  work.height = H;
  const ctx = work.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, W, H);

  const session = await getSession();
  const xs = tileStarts(W), ys = tileStarts(H);
  const total = xs.length * ys.length;
  const margin = OVERLAP / 2;
  const plane = TILE * TILE;
  const all = [];
  let done = 0;
  const t0 = performance.now();

  for (const y0 of ys) {
    for (const x0 of xs) {
      const elapsed = (performance.now() - t0) / 1000;
      const eta = done ? ` (about ${Math.ceil((elapsed / done) * (total - done))} s left)` : "";
      setStatus(`Looking for palms: tile ${done + 1} of ${total}${eta}`, done / total);
      await nextFrame();

      const tw = Math.min(TILE, W - x0), th = Math.min(TILE, H - y0);
      const px = ctx.getImageData(x0, y0, tw, th).data;
      // A new buffer per tile: the background worker takes ownership of the one it is given.
      const input = new Float32Array(3 * plane).fill(PAD);
      for (let y = 0; y < th; y++) {
        for (let x = 0; x < tw; x++) {
          const i = (y * tw + x) * 4, j = y * TILE + x;
          input[j] = px[i] / 255;
          input[plane + j] = px[i + 1] / 255;
          input[2 * plane + j] = px[i + 2] / 255;
        }
      }
      const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, TILE, TILE]) };
      const output = (await session.run(feeds))[session.outputNames[0]];

      // Keep palms centred in the part of the tile it owns; the neighbouring tile sees border palms whole.
      const loX = x0 > 0 ? margin : 0, hiX = x0 + TILE < W ? TILE - margin : TILE + 1;
      const loY = y0 > 0 ? margin : 0, hiY = y0 + TILE < H ? TILE - margin : TILE + 1;
      for (const b of decode(output, MIN_CONF, tw, th)) {
        const cx = (b.x1 + b.x2) / 2, cy = (b.y1 + b.y2) / 2;
        if (cx >= loX && cx < hiX && cy >= loY && cy < hiY) {
          all.push({ x1: b.x1 + x0, y1: b.y1 + y0, x2: b.x2 + x0, y2: b.y2 + y0, score: b.score });
        }
      }
      done++;
    }
  }

  // Back to the original image's pixels, then merge palms seen by two tiles.
  const boxes = nms(all, MERGE_IOU).map((b) => ({
    x1: b.x1 / scale, y1: b.y1 / scale, x2: b.x2 / scale, y2: b.y2 / scale, score: b.score,
  }));
  return { boxes, tiles: total, seconds: (performance.now() - t0) / 1000 };
}

// ---------- page ----------

async function countPalms() {
  if (!currentFile) return;
  busy = true;
  els.run.disabled = true;
  try {
    let bitmap;
    try {
      bitmap = await createImageBitmap(currentFile, { colorSpaceConversion: "none" });
    } catch {
      throw new Error("This browser cannot read that file. Use a JPG or PNG image; for GeoTIFF orthomosaics, see “Whole orthomosaics” below.");
    }

    const gsd = parseFloat(els.gsd.value);
    const hasGsd = gsd > 0;
    const scale = hasGsd ? gsd / TRAIN_GSD : 1;
    const workPixels = bitmap.width * scale * bitmap.height * scale;
    if (workPixels > MAX_PIXELS) {
      const size = scale > 1
        ? `At ${gsd} cm/px it would be enlarged ${scale.toFixed(1)}× to match the model's 2.13 cm/px, which is too large for the browser demo.`
        : `This image is too large for the browser demo (${bitmap.width} × ${bitmap.height} px).`;
      throw new Error(`${size} Crop it to a smaller area, or use QGIS or the Python pipeline for whole orthomosaics.`);
    }

    const result = await detect(bitmap, scale);
    run = { ...result, bitmap, gsd: hasGsd ? gsd : null, name: currentFile.name };
    els.result.hidden = false;
    render();
    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err));
  } finally {
    busy = false;
    els.run.disabled = !currentFile;
  }
}

// Redraws from the stored result, so the confidence slider updates instantly.
function render() {
  if (!run) return;
  const conf = Number(els.conf.value);
  const boxes = run.boxes.filter((b) => b.score > conf);
  const { bitmap } = run;

  const k = Math.min(1, DISPLAY_MAX / Math.max(bitmap.width, bitmap.height));
  const canvas = els.canvas;
  canvas.width = Math.round(bitmap.width * k);
  canvas.height = Math.round(bitmap.height * k);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, Math.round(Math.max(canvas.width, canvas.height) / 700));
  ctx.strokeStyle = "#ffd600";
  for (const b of boxes) ctx.strokeRect(b.x1 * k, b.y1 * k, (b.x2 - b.x1) * k, (b.y2 - b.y1) * k);

  els.count.textContent = `${boxes.length.toLocaleString()} ${boxes.length === 1 ? "palm" : "palms"} detected`;
  const lines = [
    `${bitmap.width.toLocaleString()} × ${bitmap.height.toLocaleString()} px image, ${run.tiles} ${run.tiles === 1 ? "tile" : "tiles"}, confidence above ${conf.toFixed(2)}`,
    `Ran on this computer (${backend === "webgpu" ? "graphics card, WebGPU" : "processor, WebAssembly"}) in ${run.seconds.toFixed(1)} s`,
  ];
  if (run.gsd) {
    const ha = (bitmap.width * run.gsd / 100) * (bitmap.height * run.gsd / 100) / 10000;
    lines.splice(1, 0, `About ${ha.toFixed(2)} ha in the image at ${run.gsd} cm/px: ${Math.round(boxes.length / ha).toLocaleString()} palms per hectare of image (includes any roads or bare ground)`);
  }
  els.details.innerHTML = lines.map((l) => `<li>${l}</li>`).join("");
}

function downloadCsv() {
  const conf = Number(els.conf.value);
  const rows = [["palm_id", "x_centre_px", "y_centre_px", "x_min", "y_min", "x_max", "y_max", "confidence"]];
  run.boxes.filter((b) => b.score > conf).forEach((b, i) => {
    rows.push([i + 1, (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x1, b.y1, b.x2, b.y2].map((v, j) => (j ? v.toFixed(1) : v))
      .concat(b.score.toFixed(3)));
  });
  save(new Blob([rows.map((r) => r.join(",")).join("\n") + "\n"], { type: "text/csv" }), `${baseName()}_palms.csv`);
}

function downloadPng() {
  els.canvas.toBlob((blob) => save(blob, `${baseName()}_palms.png`), "image/png");
}

function save(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function chooseFile(file) {
  if (!file) return;
  currentFile = file;
  els.fileName.textContent = file.name;
  els.run.disabled = busy;
  // Start the model download now, so it is ready (or nearly) when "Count palms" is pressed.
  getSession().then(
    () => { if (!busy) setStatus("Model ready. Press “Count palms”."); },
    (err) => setStatus(err.message),
  );
}

function setStatus(text, fraction) {
  els.status.textContent = text;
  els.progress.hidden = fraction === undefined;
  if (fraction !== undefined) els.progress.value = fraction;
}

const mb = (bytes) => (bytes / 1e6).toFixed(1);
const baseName = () => run.name.replace(/\.[^.]+$/, "");
// Lets the browser draw the progress text between tiles. Hidden tabs do not draw, so there it only
// yields (a message, which browsers do not slow down), and the count carries on in the background.
const nextFrame = () => new Promise((resolve) => {
  if (document.hidden) {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(null);
  } else {
    requestAnimationFrame(() => setTimeout(resolve));
    setTimeout(resolve, 250);   // in case the tab is hidden before the next frame
  }
});

els.file.addEventListener("change", () => chooseFile(els.file.files[0]));
els.drop.addEventListener("dragover", (e) => { e.preventDefault(); els.drop.classList.add("over"); });
els.drop.addEventListener("dragleave", () => els.drop.classList.remove("over"));
els.drop.addEventListener("drop", (e) => {
  e.preventDefault();
  els.drop.classList.remove("over");
  chooseFile(e.dataTransfer.files[0]);
});
els.conf.addEventListener("input", () => { els.confValue.textContent = Number(els.conf.value).toFixed(2); render(); });
els.run.addEventListener("click", countPalms);
els.csv.addEventListener("click", downloadCsv);
els.png.addEventListener("click", downloadPng);
