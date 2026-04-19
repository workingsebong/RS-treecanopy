const canvas = document.getElementById("labelCanvas");
const canvasWrap = document.getElementById("canvasWrap");
const ctx = canvas.getContext("2d");
const tileList = document.getElementById("tileList");
const tileStats = document.getElementById("tileStats");
const tileName = document.getElementById("tileName");
const saveState = document.getElementById("saveState");
const cursorInfo = document.getElementById("cursorInfo");
const shapeInfo = document.getElementById("shapeInfo");
const emptyState = document.getElementById("emptyState");

const buttons = {
  prev: document.getElementById("prevBtn"),
  next: document.getElementById("nextBtn"),
  draw: document.getElementById("drawBtn"),
  finish: document.getElementById("finishBtn"),
  undo: document.getElementById("undoBtn"),
  delete: document.getElementById("deleteBtn"),
  fit: document.getElementById("fitBtn"),
  save: document.getElementById("saveBtn"),
};

const state = {
  tiles: [],
  tileIndex: -1,
  image: new Image(),
  imageLoaded: false,
  shapes: [],
  drawing: false,
  currentPoints: [],
  selectedShape: -1,
  selectedVertex: -1,
  hoveredImagePoint: null,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dirty: false,
  panning: false,
  draggingVertex: false,
  autoPlotting: false,
  lastAutoPointAt: 0,
  lastPointer: null,
  spaceDown: false,
};

let resizeFrame = null;
const AUTO_POINT_MIN_IMAGE_DISTANCE = 6;
const AUTO_POINT_MIN_MS = 35;

function api(path, options = {}) {
  return fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  }).then(async (response) => {
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || `Request failed: ${response.status}`);
    }
    return data;
  });
}

function setDirty(value) {
  state.dirty = value;
  saveState.textContent = value ? "Unsaved changes" : "Saved";
  saveState.classList.toggle("dirty", value);
}

function currentTile() {
  return state.tiles[state.tileIndex] || null;
}

function updateShapeInfo() {
  const suffix = state.drawing ? `, drawing ${state.currentPoints.length} points` : "";
  shapeInfo.textContent = `${state.shapes.length} polygons${suffix}`;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const nextWidth = Math.max(1, Math.round(rect.width * dpr));
  const nextHeight = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== nextWidth) canvas.width = nextWidth;
  if (canvas.height !== nextHeight) canvas.height = nextHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function scheduleResizeCanvas() {
  if (resizeFrame !== null) return;
  resizeFrame = window.requestAnimationFrame(() => {
    resizeFrame = null;
    resizeCanvas();
  });
}

function canvasSize() {
  return {
    width: canvas.width / (window.devicePixelRatio || 1),
    height: canvas.height / (window.devicePixelRatio || 1),
  };
}

function fitImage() {
  if (!state.imageLoaded) return;
  const size = canvasSize();
  const scaleX = size.width / state.image.naturalWidth;
  const scaleY = size.height / state.image.naturalHeight;
  state.scale = Math.min(scaleX, scaleY) * 0.96;
  state.offsetX = (size.width - state.image.naturalWidth * state.scale) / 2;
  state.offsetY = (size.height - state.image.naturalHeight * state.scale) / 2;
  draw();
}

function screenToImage(x, y) {
  return {
    x: (x - state.offsetX) / state.scale,
    y: (y - state.offsetY) / state.scale,
  };
}

function imageToScreen(point) {
  return {
    x: state.offsetX + point.x * state.scale,
    y: state.offsetY + point.y * state.scale,
  };
}

function eventPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const size = canvasSize();
  const scaleX = rect.width > 0 ? size.width / rect.width : 1;
  const scaleY = rect.height > 0 ? size.height / rect.height : 1;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function insideImage(point) {
  return (
    point.x >= 0 &&
    point.y >= 0 &&
    point.x < state.image.naturalWidth &&
    point.y < state.image.naturalHeight
  );
}

function clampPoint(point) {
  return {
    x: Math.max(0, Math.min(state.image.naturalWidth - 1, point.x)),
    y: Math.max(0, Math.min(state.image.naturalHeight - 1, point.y)),
  };
}

function pointDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function pointInPolygon(point, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const xi = points[i][0];
    const yi = points[i][1];
    const xj = points[j][0];
    const yj = points[j][1];
    const intersects = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function hitVertex(screenPoint) {
  const radius = 9;
  for (let s = state.shapes.length - 1; s >= 0; s--) {
    const points = state.shapes[s].points;
    for (let v = 0; v < points.length; v++) {
      const p = imageToScreen({ x: points[v][0], y: points[v][1] });
      if (pointDistance(screenPoint, p) <= radius) {
        return { shape: s, vertex: v };
      }
    }
  }
  return null;
}

function hitShape(imagePoint) {
  for (let i = state.shapes.length - 1; i >= 0; i--) {
    if (pointInPolygon(imagePoint, state.shapes[i].points)) return i;
  }
  return -1;
}

function drawPolygon(points, options) {
  if (!points.length) return;
  ctx.save();
  ctx.beginPath();
  points.forEach((point, index) => {
    const screen = imageToScreen({ x: point[0], y: point[1] });
    if (index === 0) ctx.moveTo(screen.x, screen.y);
    else ctx.lineTo(screen.x, screen.y);
  });
  if (options.closed) ctx.closePath();
  ctx.fillStyle = options.fill;
  ctx.strokeStyle = options.stroke;
  ctx.lineWidth = options.lineWidth;
  if (options.dashed) ctx.setLineDash([8, 6]);
  if (options.closed) ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);

  points.forEach((point, index) => {
    const screen = imageToScreen({ x: point[0], y: point[1] });
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, index === 0 ? 5 : 4, 0, Math.PI * 2);
    ctx.fillStyle = options.vertexFill;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  });
  ctx.restore();
}

function draw() {
  const size = canvasSize();
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "#dfe7e1";
  ctx.fillRect(0, 0, size.width, size.height);

  if (!state.imageLoaded) return;

  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    state.image,
    state.offsetX,
    state.offsetY,
    state.image.naturalWidth * state.scale,
    state.image.naturalHeight * state.scale
  );
  ctx.restore();

  const tile = currentTile();
  if (tile && (tile.valid_width < tile.width || tile.valid_height < tile.height)) {
    const start = imageToScreen({ x: tile.valid_width, y: 0 });
    ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
    ctx.fillRect(start.x, state.offsetY, (tile.width - tile.valid_width) * state.scale, tile.height * state.scale);
  }

  state.shapes.forEach((shape, index) => {
    const selected = index === state.selectedShape;
    drawPolygon(shape.points, {
      closed: true,
      fill: selected ? "rgba(31, 111, 235, 0.18)" : "rgba(22, 122, 74, 0.18)",
      stroke: selected ? "#1f6feb" : "#0d5f37",
      lineWidth: selected ? 3 : 2,
      vertexFill: selected ? "#1f6feb" : "#167a4a",
    });
  });

  if (state.drawing && state.currentPoints.length) {
    const points = state.currentPoints.map((p) => [p.x, p.y]);
    if (state.hoveredImagePoint) {
      points.push([state.hoveredImagePoint.x, state.hoveredImagePoint.y]);
    }
    drawPolygon(points, {
      closed: false,
      fill: "rgba(179, 92, 0, 0.08)",
      stroke: "#b35c00",
      lineWidth: 2,
      vertexFill: "#b35c00",
      dashed: true,
    });
  }
}

function startDrawing() {
  state.drawing = true;
  state.currentPoints = [];
  state.autoPlotting = false;
  state.lastAutoPointAt = 0;
  state.selectedShape = -1;
  state.selectedVertex = -1;
  buttons.draw.textContent = "Drawing";
  updateShapeInfo();
  draw();
}

function cancelDrawing() {
  state.drawing = false;
  state.currentPoints = [];
  state.autoPlotting = false;
  state.lastAutoPointAt = 0;
  buttons.draw.textContent = "Start Polygon";
  updateShapeInfo();
  draw();
}

function addDrawingPoint(imagePoint, force = false) {
  if (!state.drawing) return false;
  const lastPoint = state.currentPoints[state.currentPoints.length - 1];
  const now = window.performance.now();
  if (!force && lastPoint) {
    const farEnough = pointDistance(imagePoint, lastPoint) >= AUTO_POINT_MIN_IMAGE_DISTANCE;
    const slowEnough = now - state.lastAutoPointAt >= AUTO_POINT_MIN_MS;
    if (!farEnough || !slowEnough) return false;
  }
  state.currentPoints.push(imagePoint);
  state.lastAutoPointAt = now;
  updateShapeInfo();
  draw();
  return true;
}

function finishPolygon() {
  if (!state.drawing || state.currentPoints.length < 3) return;
  state.shapes.push({
    id: `tree_${Date.now()}`,
    label: "tree",
    points: state.currentPoints.map((p) => [Number(p.x.toFixed(3)), Number(p.y.toFixed(3))]),
  });
  state.selectedShape = state.shapes.length - 1;
  state.selectedVertex = -1;
  state.currentPoints = [];
  state.autoPlotting = false;
  state.lastAutoPointAt = 0;
  buttons.draw.textContent = "Drawing";
  setDirty(true);
  updateShapeInfo();
  draw();
}

function undoAction() {
  if (state.drawing) {
    state.currentPoints.pop();
  } else if (state.selectedShape >= 0) {
    state.shapes.splice(state.selectedShape, 1);
    state.selectedShape = -1;
    setDirty(true);
  } else if (state.shapes.length) {
    state.shapes.pop();
    setDirty(true);
  }
  updateShapeInfo();
  draw();
}

function deleteSelected() {
  if (state.selectedShape < 0) return;
  state.shapes.splice(state.selectedShape, 1);
  state.selectedShape = -1;
  state.selectedVertex = -1;
  setDirty(true);
  updateShapeInfo();
  draw();
}

async function saveLabels() {
  const tile = currentTile();
  if (!tile) return;
  saveState.textContent = "Saving...";
  const payload = {
    image: tile.image,
    image_width: state.image.naturalWidth,
    image_height: state.image.naturalHeight,
    class_name: "tree",
    shapes: state.shapes,
  };
  const result = await api(`/api/labels?image=${encodeURIComponent(tile.image)}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  tile.saved_shapes = result.shape_count;
  tile.updated_at = result.updated_at;
  setDirty(false);
  renderTileList();
}

function maybeDiscardChanges() {
  if (!state.dirty) return true;
  return window.confirm("Current tile has unsaved changes. Continue without saving?");
}

async function loadTile(index) {
  if (index < 0 || index >= state.tiles.length) return;
  if (!maybeDiscardChanges()) return;
  state.tileIndex = index;
  state.imageLoaded = false;
  state.shapes = [];
  state.currentPoints = [];
  state.drawing = false;
  state.autoPlotting = false;
  state.lastAutoPointAt = 0;
  state.selectedShape = -1;
  state.selectedVertex = -1;
  setDirty(false);
  renderTileList();

  const tile = currentTile();
  tileName.textContent = `${tile.selection_rank}. ${tile.tile_id || tile.image}`;
  emptyState.classList.remove("hidden");
  emptyState.textContent = "Loading tile...";

  const labelDoc = await api(`/api/labels?image=${encodeURIComponent(tile.image)}`);
  state.shapes = (labelDoc.shapes || []).map((shape) => ({
    id: shape.id || `tree_${Date.now()}`,
    label: "tree",
    points: shape.points || [],
  }));

  state.image.onload = () => {
    state.imageLoaded = true;
    emptyState.classList.add("hidden");
    buttons.draw.textContent = "Start Polygon";
    resizeCanvas();
    fitImage();
    updateShapeInfo();
  };
  state.image.onerror = () => {
    emptyState.classList.remove("hidden");
    emptyState.textContent = "Could not load tile image.";
  };
  state.image.src = `/images/${encodeURIComponent(tile.image)}?cache=${Date.now()}`;
}

function renderTileList() {
  tileList.innerHTML = "";
  const labeled = state.tiles.filter((tile) => tile.saved_shapes > 0).length;
  tileStats.textContent = `${state.tiles.length} tiles, ${labeled} with saved polygons`;
  state.tiles.forEach((tile, index) => {
    const item = document.createElement("li");
    item.className = "tile-item";
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === state.tileIndex ? "active" : "";
    button.innerHTML = `
      <strong>${tile.selection_rank}. ${tile.tile_id || tile.image}</strong>
      <span>${tile.saved_shapes || 0} saved polygons, ${tile.width}x${tile.height}</span>
    `;
    button.addEventListener("click", () => loadTile(index));
    item.appendChild(button);
    tileList.appendChild(item);
  });
}

function zoomAt(screenPoint, factor) {
  const before = screenToImage(screenPoint.x, screenPoint.y);
  state.scale = Math.max(0.08, Math.min(12, state.scale * factor));
  state.offsetX = screenPoint.x - before.x * state.scale;
  state.offsetY = screenPoint.y - before.y * state.scale;
  draw();
}

function panBy(dx, dy) {
  if (!state.imageLoaded) return;
  state.offsetX += dx;
  state.offsetY += dy;
  draw();
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.imageLoaded) return;
  canvas.setPointerCapture(event.pointerId);
  const screen = eventPoint(event);
  const rawImagePoint = screenToImage(screen.x, screen.y);
  const imagePoint = clampPoint(rawImagePoint);

  if (event.button === 1 || event.button === 2 || state.spaceDown) {
    state.panning = true;
    state.lastPointer = screen;
    return;
  }

  if (state.drawing) {
    if (state.currentPoints.length >= 3) {
      const firstScreen = imageToScreen(state.currentPoints[0]);
      if (pointDistance(screen, firstScreen) < 12) {
        state.autoPlotting = false;
        finishPolygon();
        return;
      }
    }
    if (insideImage(rawImagePoint)) {
      addDrawingPoint(imagePoint, true);
      state.autoPlotting = true;
    }
    return;
  }

  const vertexHit = hitVertex(screen);
  if (vertexHit) {
    state.selectedShape = vertexHit.shape;
    state.selectedVertex = vertexHit.vertex;
    state.draggingVertex = true;
    draw();
    return;
  }

  const shapeHit = insideImage(rawImagePoint) ? hitShape(imagePoint) : -1;
  state.selectedShape = shapeHit;
  state.selectedVertex = -1;
  draw();
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.imageLoaded) return;
  const screen = eventPoint(event);
  const rawImagePoint = screenToImage(screen.x, screen.y);
  const imagePoint = clampPoint(rawImagePoint);
  state.hoveredImagePoint = imagePoint;
  cursorInfo.textContent = insideImage(rawImagePoint)
    ? `x: ${Math.round(imagePoint.x)}, y: ${Math.round(imagePoint.y)}`
    : "x: -, y: -";

  if (state.panning && state.lastPointer) {
    state.offsetX += screen.x - state.lastPointer.x;
    state.offsetY += screen.y - state.lastPointer.y;
    state.lastPointer = screen;
    draw();
    return;
  }

  if (state.draggingVertex && state.selectedShape >= 0 && state.selectedVertex >= 0) {
    state.shapes[state.selectedShape].points[state.selectedVertex] = [
      Number(imagePoint.x.toFixed(3)),
      Number(imagePoint.y.toFixed(3)),
    ];
    setDirty(true);
    draw();
    return;
  }

  if (state.drawing && state.autoPlotting && insideImage(rawImagePoint)) {
    if (!addDrawingPoint(imagePoint)) draw();
    return;
  }

  if (state.drawing) draw();
});

canvas.addEventListener("pointerup", () => {
  state.panning = false;
  state.draggingVertex = false;
  state.autoPlotting = false;
  state.lastPointer = null;
});

canvas.addEventListener("pointercancel", () => {
  state.panning = false;
  state.draggingVertex = false;
  state.autoPlotting = false;
  state.lastPointer = null;
});

canvas.addEventListener("dblclick", (event) => {
  event.preventDefault();
  finishPolygon();
});

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});

canvas.addEventListener(
  "wheel",
  (event) => {
    if (!state.imageLoaded) return;
    event.preventDefault();
    const screen = eventPoint(event);
    zoomAt(screen, event.deltaY < 0 ? 1.12 : 0.88);
  },
  { passive: false }
);

window.addEventListener("resize", scheduleResizeCanvas);

if ("ResizeObserver" in window) {
  const canvasObserver = new ResizeObserver(scheduleResizeCanvas);
  canvasObserver.observe(canvasWrap);
}

window.addEventListener("keydown", async (event) => {
  if (event.code === "Space") {
    state.spaceDown = true;
    event.preventDefault();
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    await saveLabels();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoAction();
    return;
  }
  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
    event.preventDefault();
    const step = event.shiftKey ? 120 : 48;
    if (event.key === "ArrowLeft") panBy(step, 0);
    if (event.key === "ArrowRight") panBy(-step, 0);
    if (event.key === "ArrowUp") panBy(0, step);
    if (event.key === "ArrowDown") panBy(0, -step);
    return;
  }
  if (event.key.toLowerCase() === "s") {
    event.preventDefault();
    await saveLabels();
    return;
  }
  if (event.key === "Enter") {
    finishPolygon();
    return;
  }
  if (event.key === "Escape") {
    cancelDrawing();
    return;
  }
  if (event.key === "Backspace" || event.key === "Delete") {
    event.preventDefault();
    if (state.drawing) undoAction();
    else deleteSelected();
    return;
  }
  if (event.key.toLowerCase() === "n") {
    loadTile(state.tileIndex + 1);
  }
  if (event.key.toLowerCase() === "p") {
    loadTile(state.tileIndex - 1);
  }
});

window.addEventListener("keyup", (event) => {
  if (event.code === "Space") state.spaceDown = false;
});

buttons.prev.addEventListener("click", () => loadTile(state.tileIndex - 1));
buttons.next.addEventListener("click", () => loadTile(state.tileIndex + 1));
buttons.draw.addEventListener("click", startDrawing);
buttons.finish.addEventListener("click", finishPolygon);
buttons.undo.addEventListener("click", undoAction);
buttons.delete.addEventListener("click", deleteSelected);
buttons.fit.addEventListener("click", fitImage);
buttons.save.addEventListener("click", () => saveLabels().catch((error) => alert(error.message)));

async function init() {
  resizeCanvas();
  try {
    const data = await api("/api/tiles");
    state.tiles = data.tiles || [];
    renderTileList();
    if (state.tiles.length) {
      await loadTile(0);
    } else {
      emptyState.textContent = "No candidate images found.";
    }
  } catch (error) {
    emptyState.textContent = error.message;
  }
}

init();
