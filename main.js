'use strict';

// A 2026 rebuild of ml4a-ofx ConvnetViewer: live-browse the activations of a
// convnet. MobileNetV1 (Keras-converted tfjs Layers model) so intermediate
// layer outputs can be tapped with tf.model().

const MODEL_URL =
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
const INPUT_SIZE = 224;
// Layers whose activations are near-duplicates of a neighbor, or not activations at all.
const SKIP_CLASSES = new Set(['InputLayer', 'ZeroPadding2D', 'BatchNormalization', 'Dropout']);
const GAP = 1; // px between tiles in the activation grid
const BG = 24; // grid background gray level
const TOP_K = 12; // number of predictions shown

const els = {
  video: document.getElementById('video'),
  layerList: document.getElementById('layerList'),
  inputTitle: document.getElementById('inputTitle'),
  stageBar: document.getElementById('stageBar'),
  infoBtn: document.getElementById('infoBtn'),
  infoPanel: document.getElementById('infoPanel'),
  layerInfo: document.getElementById('layerInfo'),
  grid: document.getElementById('grid'),
  gridWrap: document.getElementById('gridWrap'),
  preview: document.getElementById('preview'),
  preds: document.getElementById('preds'),
  status: document.getElementById('status'),
  fps: document.getElementById('fps'),
  pauseBtn: document.getElementById('pauseBtn'),
  webcamBtn: document.getElementById('webcamBtn'),
  fileBtn: document.getElementById('fileBtn'),
  normBtn: document.getElementById('normBtn'),
  weightsBtn: document.getElementById('weightsBtn'),
  fileInput: document.getElementById('fileInput'),
  zoomInBtn: document.getElementById('zoomInBtn'),
  zoomOutBtn: document.getElementById('zoomOutBtn'),
  zoomFitBtn: document.getElementById('zoomFitBtn'),
  zoomLabel: document.getElementById('zoomLabel'),
  magnifier: document.getElementById('magnifier'),
  inspLabel: document.getElementById('inspLabel'),
  inspector: document.getElementById('inspector'),
};
const gridCtx = els.grid.getContext('2d');
const previewCtx = els.preview.getContext('2d');
const inspectorCtx = els.inspector.getContext('2d');

let baseModel = null;
let layers = []; // {name, className, shape: [h, w, c]}
const subModels = new Map(); // layer name -> tf.LayersModel (weights shared with baseModel)
let selected = null;
let running = true;
let needsFrame = false; // draw one frame even while paused (e.g. after layer change)
let source = null; // 'webcam' | 'image'
let imageCanvas = null; // offscreen canvas holding a loaded image
let frozenFrame = null; // webcam frame captured at pause time — scrubbing layers while
                        // paused must re-render THIS frame, not the still-live camera
let perChannelNorm = true;
let gridLayout = { key: null, cols: 0, rows: 0 };
let zoomMode = 'fit'; // 'fit' | 'manual'
let zoomScale = 1; // CSS px per native canvas px when manual
let inspect = null; // {layer, ch} — hovered channel shown in the floating magnifier
const lastPointer = { x: 0, y: 0 };
let showWeights = false; // main grid shows learned kernels instead of activations
let weightsDirty = true; // weights grid needs recomposing (layer/mode/inspect/resize changed)
const weightsCache = new Map(); // conv layer name -> prepared display data
const frameTimes = [];
let lastPredictions = [];
let lastDraw = null; // {layer, w, h, c, litFraction} — also used by the test hook

// ---------------------------------------------------------------- status

function setStatus(msg, isError) {
  els.status.textContent = msg;
  els.status.hidden = !msg; // panel only appears while loading or on errors
  els.status.classList.toggle('error', !!isError);
  if (isError) console.error(msg);
}

function updateInputTitle() {
  const src = source === 'webcam' ? 'webcam' : source === 'image' ? 'image' : '…';
  els.inputTitle.textContent = `input — ${src} (224×224)`;
  els.webcamBtn.classList.toggle('toggled', source === 'webcam');
  els.fileBtn.classList.toggle('toggled', source === 'image');
  // Pause and FPS only mean something for a live source; a still image is already frozen.
  els.pauseBtn.disabled = source === 'image';
  els.fps.hidden = source !== 'webcam';
  if (source === 'webcam') frameTimes.length = 0; // don't average across the image-mode gap
  if (source === 'image' && !running) {
    running = true;
    els.pauseBtn.textContent = 'Pause';
    els.pauseBtn.classList.remove('toggled');
  }
}

// ---------------------------------------------------------------- model

async function loadModel() {
  setStatus('loading MobileNetV1 weights (~17 MB)…');
  baseModel = await tf.loadLayersModel(MODEL_URL);
  layers = baseModel.layers
    .filter((l) => {
      const s = l.outputShape;
      return Array.isArray(s) && s.length === 4 && !SKIP_CLASSES.has(l.getClassName());
    })
    .map((l) => ({
      name: l.name,
      className: l.getClassName(),
      shape: l.outputShape.slice(1), // [h, w, c]
    }));
  computeLayerMeta();
  buildLayerList();
  selectLayer('conv1_relu');
  // Warm up the GPU pipeline so the first real frame isn't a multi-second stall.
  tf.tidy(() => subModelFor(selected).predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])));
  setStatus(`model ready — ${layers.length} viewable layers, backend: ${tf.getBackend()}`);
}

// Per-layer facts for the ⓘ panel: shapes, params, receptive field.
const layerMeta = new Map();

function computeLayerMeta() {
  let rf = 1; // receptive field side, in input pixels
  let jump = 1; // input pixels between neighboring activations
  let global = false;
  for (const l of baseModel.layers) {
    const cn = l.getClassName();
    let kernel = null;
    if (cn === 'Conv2D' || cn === 'DepthwiseConv2D') {
      const cfg = l.getConfig();
      const k = cfg.kernelSize[0];
      const s = cfg.strides[0];
      rf += (k - 1) * jump;
      jump *= s;
      kernel = { k, s };
    } else if (cn === 'GlobalAveragePooling2D') {
      global = true;
    }
    let inShape = null;
    try { inShape = l.input.shape; } catch { /* input layers have no inbound tensor */ }
    layerMeta.set(l.name, {
      className: cn,
      kernel,
      rf: global ? null : Math.min(rf, INPUT_SIZE),
      inShape,
      outShape: l.outputShape,
      params: l.countParams(),
    });
  }
}

function layerExplanation(name) {
  if (name === 'conv1') {
    return 'A standard 3×3 convolution over the RGB input, with stride 2. The network\'s first feature extractor: 32 learned filters that respond to edges and color contrasts. (Several of its filters died during training — the flat gray tiles in weights view.)';
  }
  if (/_relu$/.test(name)) {
    return 'ReLU6 activation — the gate after the preceding convolution (and its batch-norm): negative values become 0, and values are capped at 6. What you see here is which features fired, and where.';
  }
  if (/^conv_dw_\d+$/.test(name)) {
    return 'Depthwise 3×3 convolution: each channel is filtered spatially on its own, with its own small kernel — spatial filtering with no mixing across channels. This is one half of MobileNet\'s split of an ordinary convolution into two cheaper steps.';
  }
  if (/^conv_pw_\d+$/.test(name)) {
    return 'Pointwise 1×1 convolution: at every pixel, all input channels are mixed into new combinations — channel mixing with no spatial context. Equivalent to a fully-connected layer applied at each location; this is the other half of MobileNet\'s split, and where most of the parameters live.';
  }
  if (name === 'reshape_1') {
    return 'The embedding: global average pooling has collapsed each 7×7 feature map to a single number, leaving one 1024-dimensional vector that summarizes the whole image. This is the "feature extraction" output people reuse for transfer learning.';
  }
  if (name === 'conv_preds') {
    return 'The classifier: a 1×1 convolution applied to the 1×1 embedding — mathematically identical to a fully-connected layer — mapping 1024 features to 1000 ImageNet class scores.';
  }
  if (name === 'act_softmax') {
    return 'Softmax: turns the 1000 raw class scores into probabilities that sum to 1. One (or a few) lit pixels = the network\'s answer; the top-12 panel reads out the same numbers.';
  }
  return 'Part of the MobileNetV1 pipeline.';
}

const fmtShape = (s) => (Array.isArray(s) ? s.filter((d) => d != null).join('×') : String(s));

function renderInfo() {
  const m = layerMeta.get(selected);
  if (!m) return;
  const idx = layers.findIndex((l) => l.name === selected);
  const stats = [m.inShape
    ? `in ${fmtShape(m.inShape)} → out ${fmtShape(m.outShape)}`
    : `out ${fmtShape(m.outShape)}`];
  if (m.kernel && m.kernel.s > 1) stats.push(`stride ${m.kernel.s} — downsamples resolution ${m.kernel.s}×`);
  stats.push(m.params ? `${m.params.toLocaleString()} learned parameters` : 'no learned parameters');
  stats.push(m.rf === null
    ? 'receptive field: the whole image (after global pooling)'
    : `receptive field ≈ ${m.rf}×${m.rf} px of the ${INPUT_SIZE}×${INPUT_SIZE} input`);
  els.infoPanel.innerHTML =
    `<b>${selected}</b> <span class="dim">· ${m.className.toLowerCase()} · viewable layer ${idx + 1} of ${layers.length}</span>` +
    `<p>${layerExplanation(selected)}</p>` +
    `<div class="meta">${stats.join('<br>')}</div>`;
  els.infoPanel.style.top = `${els.stageBar.offsetHeight + 8}px`;
}

function toggleInfo() {
  els.infoPanel.hidden = !els.infoPanel.hidden;
  els.infoBtn.classList.toggle('toggled', !els.infoPanel.hidden);
  if (!els.infoPanel.hidden) renderInfo();
}

function subModelFor(name) {
  // Submodels share weights with baseModel, so they are cached and never disposed.
  if (!subModels.has(name)) {
    subModels.set(name, tf.model({
      inputs: baseModel.inputs,
      outputs: [baseModel.getLayer(name).output, baseModel.outputs[0]],
    }));
  }
  return subModels.get(name);
}

// ---------------------------------------------------------------- layer list UI

function buildLayerList() {
  els.layerList.textContent = '';
  for (const l of layers) {
    const btn = document.createElement('button');
    btn.className = 'layer-btn';
    btn.dataset.layer = l.name;
    const shape = document.createElement('span');
    shape.className = 'shape';
    shape.textContent = `${l.shape[0]}×${l.shape[1]}×${l.shape[2]}`;
    btn.textContent = l.name;
    btn.appendChild(shape);
    btn.addEventListener('click', () => selectLayer(l.name));
    els.layerList.appendChild(btn);
  }
}

function selectLayer(name) {
  if (!layers.some((l) => l.name === name)) name = layers[0].name;
  if (inspect && inspect.layer !== name) clearInspect(); // channel indices don't map across layers
  selected = name;
  const l = layers.find((x) => x.name === name);
  els.layerInfo.textContent =
    `${name}  (${l.className.toLowerCase()})  —  ${l.shape[0]}×${l.shape[1]} × ${l.shape[2]} channels`;
  for (const btn of els.layerList.children) {
    const active = btn.dataset.layer === name;
    btn.classList.toggle('active', active);
    if (active) btn.scrollIntoView({ block: 'nearest' });
  }
  needsFrame = true;
  weightsDirty = true;
  if (!els.infoPanel.hidden) renderInfo();
  // First visit to a layer compiles new WebGL shaders (can take ~1s); dim the
  // stale grid until the new layer's first frame is drawn.
  els.grid.style.opacity = '0.3';
}

function stepLayer(delta) {
  const i = layers.findIndex((l) => l.name === selected);
  const next = Math.min(layers.length - 1, Math.max(0, i + delta));
  selectLayer(layers[next].name);
}

// ---------------------------------------------------------------- input sources

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
    });
    els.video.srcObject = stream;
    await new Promise((res) => { els.video.onloadedmetadata = res; });
    await els.video.play();
    source = 'webcam';
    updateInputTitle();
    setStatus('');
  } catch (err) {
    setStatus(`no webcam (${err.name}) — use "Load image…" or drop an image`, true);
  }
}

function useImage(img) {
  imageCanvas = imageCanvas || document.createElement('canvas');
  imageCanvas.width = img.naturalWidth || img.width;
  imageCanvas.height = img.naturalHeight || img.height;
  imageCanvas.getContext('2d').drawImage(img, 0, 0);
  source = 'image';
  needsFrame = true;
  updateInputTitle();
  setStatus('');
}

function loadImageFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => { useImage(img); URL.revokeObjectURL(url); };
  img.src = url;
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => { useImage(img); resolve(); };
    img.onerror = () => reject(new Error(`could not load ${url}`));
    img.src = url;
  });
}

function currentSource() {
  if (source === 'webcam' && els.video.readyState >= 2) {
    if (!running && frozenFrame && frozenFrame.width) {
      return { el: frozenFrame, w: frozenFrame.width, h: frozenFrame.height };
    }
    return { el: els.video, w: els.video.videoWidth, h: els.video.videoHeight };
  }
  if (source === 'image' && imageCanvas) {
    return { el: imageCanvas, w: imageCanvas.width, h: imageCanvas.height };
  }
  return null;
}

// ---------------------------------------------------------------- zoom / pan / inspect

function effectiveScale() {
  const rect = els.grid.getBoundingClientRect();
  return rect.width > 0 && els.grid.width > 0 ? rect.width / els.grid.width : 1;
}

function applyZoom() {
  const gw = els.grid.width || 1;
  const gh = els.grid.height || 1;
  els.grid.style.maxWidth = 'none';
  els.grid.style.maxHeight = 'none';
  if (zoomMode === 'fit') {
    // Scale up OR down so the grid fills the wrap, whatever the layer's native size.
    // Sized twice: the first pass can make scrollbars disappear, changing the avail space.
    for (let pass = 0; pass < 2; pass++) {
      const cs = getComputedStyle(els.gridWrap);
      const availW = els.gridWrap.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      const availH = els.gridWrap.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
      const scale = Math.max(0.05, Math.min(availW / gw, availH / gh));
      els.grid.style.width = `${Math.floor(gw * scale)}px`;
    }
    els.zoomLabel.textContent = 'fit';
  } else {
    els.grid.style.width = `${Math.round(gw * zoomScale)}px`;
    els.zoomLabel.textContent = `${zoomScale >= 10 ? zoomScale.toFixed(0) : zoomScale.toFixed(1)}×`;
  }
}

function setZoom(scale, cx, cy) {
  const wrap = els.gridWrap;
  const before = els.grid.getBoundingClientRect();
  if (cx === undefined) {
    const wr = wrap.getBoundingClientRect();
    cx = wr.left + wr.width / 2;
    cy = wr.top + wr.height / 2;
  }
  const ax = (cx - before.left) / Math.max(1, before.width);
  const ay = (cy - before.top) / Math.max(1, before.height);
  zoomMode = 'manual';
  zoomScale = Math.min(64, Math.max(0.25, scale));
  applyZoom();
  // keep the point under the cursor stationary
  const after = els.grid.getBoundingClientRect();
  wrap.scrollLeft += (after.left + ax * after.width) - cx;
  wrap.scrollTop += (after.top + ay * after.height) - cy;
}

function zoomFit() {
  zoomMode = 'fit';
  applyZoom();
}

const curScale = () => (zoomMode === 'fit' ? effectiveScale() : zoomScale);

// Magnifying-glass on click: the clicked channel is drawn enlarged in a floating
// box pinned next to where you clicked. Click the same tile (or Esc) to close.
const inspectAnchor = { x: 0, y: 0 };

function inspectAt(cx, cy) {
  if (!lastDraw || lastDraw.layer !== selected) return;
  const { w, h, c, cols } = lastDraw;
  if (c === 1) return; // single-tile layers: nothing to magnify
  const rect = els.grid.getBoundingClientRect();
  const nx = ((cx - rect.left) / rect.width) * els.grid.width;
  const ny = ((cy - rect.top) / rect.height) * els.grid.height;
  const col = Math.floor((nx - GAP) / (w + GAP));
  const row = Math.floor((ny - GAP) / (h + GAP));
  const ch = row * cols + col;
  if (col < 0 || col >= cols || row < 0 || ch >= c) return;
  if (inspect && inspect.layer === selected && inspect.ch === ch) {
    clearInspect(); // clicking the magnified tile again closes it
    return;
  }
  inspect = { layer: selected, ch };
  inspectAnchor.x = cx;
  inspectAnchor.y = cy;
  needsFrame = true;
  weightsDirty = true;
}

function positionMagnifier(cx, cy) {
  const r = els.magnifier.getBoundingClientRect();
  let x = cx + 20;
  let y = cy + 20;
  if (x + r.width > innerWidth - 8) x = cx - r.width - 20;
  if (y + r.height > innerHeight - 8) y = cy - r.height - 20;
  els.magnifier.style.left = `${Math.max(4, x)}px`;
  els.magnifier.style.top = `${Math.max(4, y)}px`;
}

function setInspectorVisible(v) {
  if (els.magnifier.hidden !== v) return;
  els.magnifier.hidden = !v;
}

function clearInspect() {
  if (!inspect) return;
  inspect = null;
  setInspectorVisible(false);
  needsFrame = true; // redraw the grid without the highlight box
  weightsDirty = true;
}

function drawInspector(vals, w, h, c, cols, rgb, label) {
  const ch = inspect.ch;
  if (els.inspector.width !== w || els.inspector.height !== h) {
    els.inspector.width = w;
    els.inspector.height = h;
  }
  const img = inspectorCtx.createImageData(w, h);
  const d = img.data;
  let k = 0;
  for (let y = 0; y < h; y++) {
    let vi = y * w * c + ch;
    for (let x = 0; x < w; x++) {
      if (rgb) {
        const b = vi * 3;
        d[k] = vals[b] | 0;
        d[k + 1] = vals[b + 1] | 0;
        d[k + 2] = vals[b + 2] | 0;
      } else {
        d[k] = d[k + 1] = d[k + 2] = vals[vi] | 0;
      }
      d[k + 3] = 255;
      k += 4;
      vi += c;
    }
  }
  inspectorCtx.putImageData(img, 0, 0);
  els.inspLabel.textContent = `${label} · #${ch} · ${w}×${h}`;
  setInspectorVisible(true);
  positionMagnifier(inspectAnchor.x, inspectAnchor.y); // stays pinned by the clicked tile
  // highlight the inspected tile in the grid
  const tx = GAP + (ch % cols) * (w + GAP);
  const ty = GAP + ((ch / cols) | 0) * (h + GAP);
  gridCtx.strokeStyle = '#6ee7a0';
  gridCtx.lineWidth = 1;
  gridCtx.strokeRect(tx - 0.5, ty - 0.5, w + 1, h + 1);
}

// ---------------------------------------------------------------- weights view

// ReLU/softmax layers have no weights; show the nearest conv at or above the selection.
function weightsSourceFor(name) {
  const all = baseModel.layers;
  for (let i = all.findIndex((l) => l.name === name); i >= 0; i--) {
    const cn = all[i].getClassName();
    if (cn === 'Conv2D' || cn === 'DepthwiseConv2D') return all[i];
  }
  return null;
}

// Kernels are mapped symmetrically: 0 → mid-gray (128), ±max|w| → black/white.
// perKernel scales each kernel by its own max|w| (shows every kernel's structure);
// otherwise one global scale (shows relative strength). Pointwise is always global.
async function weightsFor(layer, perKernel) {
  const cacheKey = `${layer.name}:${perKernel ? 'k' : 'g'}`;
  if (weightsCache.has(cacheKey)) return weightsCache.get(cacheKey);
  const kernel = layer.getWeights()[0]; // [kh, kw, in, out]
  const [kh, kw, cin, cout] = kernel.shape;
  const raw = await kernel.data();
  let maxAbs = 1e-9;
  for (let i = 0; i < raw.length; i++) maxAbs = Math.max(maxAbs, Math.abs(raw[i]));

  // per-kernel max|w|, indexed by the kernel id each flat index belongs to
  const kernelOf = (i, nKernels) => i % nKernels; // last tensor dim is out (or in-channel for dw's [..,cin,1])
  const scaleFor = (i, nKernels, maxes) =>
    127 / (perKernel ? maxes[kernelOf(i, nKernels)] : maxAbs);
  const kernelMaxes = (nKernels) => {
    const m = new Float32Array(nKernels).fill(1e-9);
    for (let i = 0; i < raw.length; i++) {
      const k = kernelOf(i, nKernels);
      m[k] = Math.max(m[k], Math.abs(raw[i]));
    }
    return m;
  };

  let entry;
  if (layer.getClassName() === 'DepthwiseConv2D') {
    // one kh×kw kernel per input channel; raw [kh,kw,cin,1] already matches the [h,w,c] grid layout
    const maxes = perKernel ? kernelMaxes(cin) : null;
    const vals = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) vals[i] = 128 + raw[i] * scaleFor(i, cin, maxes);
    entry = { vals, w: kw, h: kh, c: cin, rgb: false,
              desc: `${layer.name} weights — ${kh}×${kw} depthwise kernel per channel × ${cin}` };
  } else if (kh === 1 && kw === 1) {
    // pointwise: show the whole thing as one in×out connection-matrix heatmap
    const scale = 127 / maxAbs;
    const vals = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) vals[i] = 128 + raw[i] * scale;
    entry = { vals, w: cout, h: cin, c: 1, rgb: false,
              desc: `${layer.name} weights — 1×1 kernels as a ${cin} (in, rows) × ${cout} (out, cols) matrix` };
  } else if (cin === 3) {
    // spatial kernels over RGB input (conv1): render each filter in color
    const maxes = perKernel ? kernelMaxes(cout) : null;
    const vals = new Float32Array(kh * kw * cout * 3);
    for (let y = 0; y < kh; y++) {
      for (let x = 0; x < kw; x++) {
        for (let o = 0; o < cout; o++) {
          for (let ci = 0; ci < 3; ci++) {
            const i = ((y * kw + x) * 3 + ci) * cout + o;
            vals[((y * kw + x) * cout + o) * 3 + ci] = 128 + raw[i] * scaleFor(i, cout, maxes);
          }
        }
      }
    }
    entry = { vals, w: kw, h: kh, c: cout, rgb: true,
              desc: `${layer.name} weights — ${kh}×${kw}×3 (RGB) kernels × ${cout} filters` };
  } else {
    // general spatial conv (not in MobileNetV1): average over input channels
    const scale = 127 / maxAbs;
    const vals = new Float32Array(kh * kw * cout);
    for (let y = 0; y < kh; y++) {
      for (let x = 0; x < kw; x++) {
        for (let o = 0; o < cout; o++) {
          let s = 0;
          for (let ci = 0; ci < cin; ci++) s += raw[((y * kw + x) * cin + ci) * cout + o];
          vals[(y * kw + x) * cout + o] = 128 + (s / cin) * scale;
        }
      }
    }
    entry = { vals, w: kw, h: kh, c: cout, rgb: false,
              desc: `${layer.name} weights — ${kh}×${kw} kernels (mean over ${cin} in-channels) × ${cout}` };
  }
  weightsCache.set(cacheKey, entry);
  return entry;
}

async function drawWeights() {
  const srcLayer = weightsSourceFor(selected);
  if (!srcLayer) return;
  const wEntry = await weightsFor(srcLayer, perChannelNorm);
  els.layerInfo.textContent =
    wEntry.desc + (srcLayer.name !== selected ? `  (nearest conv above ${selected})` : '');
  renderGrid(wEntry.vals, wEntry.w, wEntry.h, wEntry.c, selected,
    { rgb: wEntry.rgb, inspLabel: `${srcLayer.name} weights` });
}

function toggleWeights() {
  showWeights = !showWeights;
  els.weightsBtn.classList.toggle('toggled', showWeights);
  weightsDirty = true;
  needsFrame = true;
  els.grid.style.opacity = '0.3';
  if (!showWeights) selectLayer(selected); // restores the activation layer-info text
}

// ---------------------------------------------------------------- per-frame pipeline

async function processFrame(src) {
  const layerName = selected; // frame is computed & labeled with this, even if selection changes mid-flight
  const model = subModelFor(layerName);
  const outputs = tf.tidy(() => {
    const img = tf.browser.fromPixels(src.el);
    const s = Math.min(src.h, src.w);
    const cropped = img.slice([(src.h - s) >> 1, (src.w - s) >> 1, 0], [s, s, 3]);
    const input = tf.image.resizeBilinear(cropped, [INPUT_SIZE, INPUT_SIZE])
      .div(127.5).sub(1).expandDims(0);
    return model.predict(input);
  });
  const [act, scores] = outputs;
  try {
    if (!showWeights) await drawActivations(act, layerName);
    updatePredictions(await scores.data());
  } finally {
    act.dispose();
    scores.dispose();
  }
  const s = Math.min(src.h, src.w);
  previewCtx.drawImage(src.el, (src.w - s) / 2, (src.h - s) / 2, s, s, 0, 0, INPUT_SIZE, INPUT_SIZE);
}

function gridFor(c, w, h, layerName) {
  const key = `${layerName}|${c}x${w}x${h}|${els.gridWrap.clientWidth}x${els.gridWrap.clientHeight}`;
  if (gridLayout.key === key) return gridLayout;
  const aspect = Math.max(0.2, els.gridWrap.clientWidth / Math.max(1, els.gridWrap.clientHeight));
  let best = null;
  for (let cols = 1; cols <= c; cols++) {
    const rows = Math.ceil(c / cols);
    const a = (cols * (w + GAP)) / (rows * (h + GAP));
    const score = Math.abs(Math.log(a / aspect));
    if (!best || score < best.score) best = { key, cols, rows, score };
  }
  gridLayout = best;
  return best;
}

async function drawActivations(act, layerName) {
  const [, h, w, c] = act.shape;
  const norm = tf.tidy(() => {
    const a = act.squeeze([0]);
    // Per-channel norm is degenerate on 1×1 maps (min == max → all black); use layer norm there.
    const dims = (perChannelNorm && !(h === 1 && w === 1)) ? [0, 1] : [0, 1, 2];
    const mn = a.min(dims, true);
    const mx = a.max(dims, true);
    return a.sub(mn).div(mx.sub(mn).add(1e-6)).mul(255);
  });
  let vals;
  try {
    vals = await norm.data(); // row-major [h, w, c]
  } finally {
    norm.dispose();
  }
  renderGrid(vals, w, h, c, layerName, {});
}

// Composes value tiles (activations or kernels) into the main grid canvas.
// vals is 0..255, laid out [h, w, c] — or [h, w, c, rgb] when opts.rgb is set.
function renderGrid(vals, w, h, c, layerName, { rgb = false, inspLabel = layerName } = {}) {
  const mode = showWeights ? 'w' : 'a';
  const { cols, rows } = gridFor(c, w, h, `${layerName}:${mode}`);
  const gw = cols * (w + GAP) + GAP;
  const gh = rows * (h + GAP) + GAP;
  if (els.grid.width !== gw || els.grid.height !== gh) {
    els.grid.width = gw;
    els.grid.height = gh;
    applyZoom(); // manual zoom sizes the canvas from its native dimensions
  }
  const img = gridCtx.createImageData(gw, gh);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = d[i + 1] = d[i + 2] = BG;
    d[i + 3] = 255;
  }
  let lit = 0;
  for (let ch = 0; ch < c; ch++) {
    const tx = GAP + (ch % cols) * (w + GAP);
    const ty = GAP + ((ch / cols) | 0) * (h + GAP);
    for (let y = 0; y < h; y++) {
      let di = ((ty + y) * gw + tx) * 4;
      let vi = y * w * c + ch;
      for (let x = 0; x < w; x++) {
        if (rgb) {
          const b = vi * 3;
          const r = vals[b] | 0;
          const g = vals[b + 1] | 0;
          const bl = vals[b + 2] | 0;
          if (Math.max(r, g, bl) > 16) lit++;
          d[di] = r; d[di + 1] = g; d[di + 2] = bl;
        } else {
          const v = vals[vi] | 0;
          if (v > 16) lit++;
          d[di] = d[di + 1] = d[di + 2] = v;
        }
        di += 4;
        vi += c;
      }
    }
  }
  gridCtx.putImageData(img, 0, 0);
  if (inspect && inspect.layer === layerName && inspect.ch < c) {
    drawInspector(vals, w, h, c, cols, rgb, inspLabel);
  }
  if (layerName === selected) els.grid.style.opacity = '1';
  lastDraw = { layer: layerName, w, h, c, cols, mode, litFraction: lit / (w * h * c) };
}

function updatePredictions(scores) {
  const top = Array.from(scores)
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, TOP_K)
    .map(({ p, i }) => ({ label: IMAGENET_LABELS[i] || `class ${i}`, p }));
  lastPredictions = top;
  els.preds.textContent = '';
  for (const { label, p } of top) {
    const row = document.createElement('div');
    row.className = 'pred';
    const pct = (p * 100).toFixed(1);
    row.innerHTML =
      `<div class="label"><span class="name"></span><span class="pct">${pct}%</span></div>` +
      `<div class="bar"><div style="width:${Math.max(1, p * 100)}%"></div></div>`;
    row.querySelector('.name').textContent = label;
    els.preds.appendChild(row);
  }
}

// ---------------------------------------------------------------- main loop

function currentFps() {
  if (frameTimes.length < 2) return 0;
  const dt = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / (frameTimes.length - 1);
  return 1000 / dt;
}

let consecutiveErrors = 0;

async function loop() {
  const src = currentSource();
  if (baseModel) {
    // A still image only needs re-processing when something changed; webcam runs live.
    const wantFrame = src && ((source === 'webcam' && running) || needsFrame);
    needsFrame = false;
    let didWork = false;
    try {
      if (wantFrame) {
        await processFrame(src);
        didWork = true;
        if (source === 'webcam') {
          frameTimes.push(performance.now());
          if (frameTimes.length > 30) frameTimes.shift();
          els.fps.textContent = `${currentFps().toFixed(0)} fps`;
        }
      }
      if (showWeights && weightsDirty) {
        weightsDirty = false;
        await drawWeights(); // weights are static — drawn only when layer/mode/inspect changes
        didWork = true;
      }
      if (didWork) consecutiveErrors = 0;
    } catch (err) {
      console.error(err);
      if (++consecutiveErrors >= 3) {
        setStatus(`frame error: ${err.message}`, true);
        running = false;
        els.pauseBtn.textContent = 'Paused ▶';
        els.pauseBtn.classList.add('toggled');
        snapshotWebcam();
      }
    }
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- controls

function snapshotWebcam() {
  if (source !== 'webcam' || !els.video.videoWidth) return;
  frozenFrame = frozenFrame || document.createElement('canvas');
  frozenFrame.width = els.video.videoWidth;
  frozenFrame.height = els.video.videoHeight;
  frozenFrame.getContext('2d').drawImage(els.video, 0, 0);
}

function togglePause() {
  if (els.pauseBtn.disabled) return; // no pausing a still image
  running = !running;
  els.pauseBtn.textContent = running ? 'Pause' : 'Paused ▶';
  els.pauseBtn.classList.toggle('toggled', !running);
  if (!running) {
    snapshotWebcam();
    needsFrame = true; // re-render once from the snapshot so the display matches it exactly
  }
}

function toggleNorm() {
  perChannelNorm = !perChannelNorm;
  els.normBtn.textContent = perChannelNorm ? 'Norm: per-channel' : 'Norm: layer';
  needsFrame = true;
  weightsDirty = true; // in weights mode this switches per-kernel vs global scaling
}

els.pauseBtn.addEventListener('click', togglePause);
els.normBtn.addEventListener('click', toggleNorm);
els.weightsBtn.addEventListener('click', toggleWeights);
els.webcamBtn.addEventListener('click', () => {
  if (els.video.srcObject) {
    source = 'webcam';
    updateInputTitle();
  } else {
    startWebcam();
  }
});
els.fileInput.addEventListener('change', (e) => loadImageFile(e.target.files[0]));
els.zoomInBtn.addEventListener('click', () => setZoom(curScale() * 1.5));
els.zoomOutBtn.addEventListener('click', () => setZoom(curScale() / 1.5));
els.zoomFitBtn.addEventListener('click', zoomFit);
els.infoBtn.addEventListener('click', toggleInfo);

// pinch / Ctrl+scroll zooms toward the cursor; plain scroll pans as usual
els.gridWrap.addEventListener('wheel', (e) => {
  if (!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  setZoom(curScale() * Math.exp(-e.deltaY * 0.01), e.clientX, e.clientY);
}, { passive: false });

// drag to pan; a click without movement toggles the magnifier on that tile
let drag = null;
els.gridWrap.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  drag = {
    x: e.clientX, y: e.clientY,
    sl: els.gridWrap.scrollLeft, st: els.gridWrap.scrollTop,
    moved: false, onCanvas: e.target === els.grid,
  };
  els.gridWrap.setPointerCapture(e.pointerId);
});
els.gridWrap.addEventListener('pointermove', (e) => {
  lastPointer.x = e.clientX;
  lastPointer.y = e.clientY;
  if (!drag) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 5) drag.moved = true;
  if (drag.moved) {
    e.preventDefault();
    els.gridWrap.scrollLeft = drag.sl - dx;
    els.gridWrap.scrollTop = drag.st - dy;
  }
});
els.gridWrap.addEventListener('pointerup', (e) => {
  const d = drag;
  drag = null;
  if (d && !d.moved && d.onCanvas) inspectAt(e.clientX, e.clientY);
});
els.gridWrap.addEventListener('pointercancel', () => { drag = null; });

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePause(); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); stepLayer(1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); stepLayer(-1); }
  else if (e.key === 'n' || e.key === 'N') toggleNorm();
  else if (e.key === 'w' || e.key === 'W') toggleWeights();
  else if (e.key === 'i' || e.key === 'I') toggleInfo();
  else if (e.key === '+' || e.key === '=') setZoom(curScale() * 1.5);
  else if (e.key === '-' || e.key === '_') setZoom(curScale() / 1.5);
  else if (e.key === '0') zoomFit();
  else if (e.key === 'Escape') {
    if (!els.infoPanel.hidden) toggleInfo();
    else clearInspect();
  }
});

window.addEventListener('resize', () => {
  if (zoomMode === 'fit') applyZoom();
  needsFrame = true; // recompute the grid layout for the new window shape
  weightsDirty = true;
});

document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragover'); });
document.addEventListener('dragleave', () => document.body.classList.remove('dragover'));
document.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('dragover');
  loadImageFile(e.dataTransfer.files[0]);
});

// ---------------------------------------------------------------- boot

const ready = (async () => {
  document.getElementById('predsTitle').textContent = `top-${TOP_K} predictions`;
  await tf.ready();
  await loadModel();
  // Start on the bundled sample image — the camera permission prompt only
  // appears when the visitor explicitly clicks "Webcam".
  await loadImageFromUrl('assets/test-cat.jpg').catch(() => {});
  loop();
})();
ready.catch((err) => setStatus(`failed to start: ${err.message}`, true));

// Programmatic hooks for automated testing (and console tinkering).
window.__viewer = {
  ready,
  layers: () => layers.map((l) => l.name),
  selectLayer,
  loadImageFromUrl,
  setZoom,
  zoomFit,
  toggleWeights,
  inspectChannel: (ch) => { inspect = { layer: selected, ch }; needsFrame = true; weightsDirty = true; },
  clearInspect,
  state: () => ({
    layer: selected,
    source,
    running,
    backend: tf.getBackend(),
    predictions: lastPredictions,
    draw: lastDraw,
    zoom: zoomMode === 'fit' ? 'fit' : zoomScale,
    showWeights,
    inspect,
    fps: currentFps(),
  }),
};
