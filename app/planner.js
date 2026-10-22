/* Walking-route planner for the Gale Marswalk Planner. Works in the browser (window.MarswalkPlanner) and in Node
   (module.exports) so it can be tested on its own: `node --test app/tests/`.

   The terrain is a grid of square cells (`g` metres each) with heights H (metres), a slope per cell (degrees) and a
   validity flag. A cell is off-limits when it has no data, is steeper than the walker's limit, or its extra cost is
   Infinity (for example a rock too tall to step over). The route minimises walking time, with speed set by grade. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MarswalkPlanner = api;
})(typeof self !== 'undefined' ? self : this, function () {
  const VFLAT = 3.0 / 3.6;   // m/s on flat ground in a suit (example value, not a NASA EVA rule)

  /** Tobler's hiking rule, scaled so flat ground = VFLAT. grade = rise / run (uphill positive). */
  function walkSpeed(grade) {
    return VFLAT * Math.exp(-3.5 * Math.abs(grade + 0.05)) / Math.exp(-3.5 * 0.05);
  }
  const VMAX = walkSpeed(-0.05);

  /** Binary min-heap on parallel typed arrays (fast for ~1 M cells). */
  function Heap(capacity) {
    let key = new Float64Array(capacity), val = new Int32Array(capacity), size = 0;
    return {
      get size() { return size; },
      push(k, v) {
        if (size === key.length) { const k2 = new Float64Array(size * 2), v2 = new Int32Array(size * 2); k2.set(key); v2.set(val); key = k2; val = v2; }
        let i = size++;
        while (i > 0) { const p = (i - 1) >> 1; if (key[p] <= k) break; key[i] = key[p]; val[i] = val[p]; i = p; }
        key[i] = k; val[i] = v;
      },
      pop() {
        const top = val[0], k = key[--size], v = val[size];
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1; let m = i, mk = k;
          if (l < size && key[l] < mk) { m = l; mk = key[l]; }
          if (r < size && key[r] < mk) { m = r; }
          if (m === i) break;
          key[i] = key[m]; val[i] = val[m]; i = m;
        }
        key[i] = k; val[i] = v;
        return top;
      },
    };
  }

  /**
   * A* search for the quickest walk from a to b ({x, y} in metres from the grid's top-left corner).
   * grid: { H: Float32Array, valid: Uint8Array, slope: Float32Array, cols, rows, g }
   * cost (optional): Float32Array of extra time multipliers per cell (1 = normal, Infinity = blocked).
   * Returns { cells, time, dist, up, down, maxs, prof, hazardCells } or { error }.
   */
  function planRoute(grid, a, b, maxSlope, cost) {
    const C = grid.cols, R = grid.rows, g = grid.g, n = C * R, H = grid.H;
    const cell = (p) => {
      const x = Math.round(p.x / g), y = Math.round(p.y / g);
      return x < 0 || y < 0 || x >= C || y >= R ? -1 : y * C + x;
    };
    const ia = cell(a), ib = cell(b);
    const blocked = (i) => !grid.valid[i] || grid.slope[i] > maxSlope || (cost && cost[i] === Infinity);
    if (ia < 0 || ib < 0) return { error: 'That point is outside the map.' };
    if (blocked(ia) || blocked(ib)) return { error: 'That point is steeper than your limit, blocked by a hazard, or has no data. Pick another spot.' };
    const gScore = new Float64Array(n).fill(Infinity), came = new Int32Array(n).fill(-1), closed = new Uint8Array(n);
    const bx = ib % C, by = (ib / C) | 0;
    const h = (i) => Math.hypot((i % C) - bx, ((i / C) | 0) - by) * g / VMAX;
    const heap = Heap(1 << 16);
    gScore[ia] = 0; heap.push(h(ia), ia);
    const N8 = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    while (heap.size) {
      const i = heap.pop();
      if (closed[i]) continue;
      closed[i] = 1;
      if (i === ib) break;
      const x = i % C, y = (i / C) | 0;
      for (const [dx, dy] of N8) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= C || ny >= R) continue;
        const j = ny * C + nx;
        if (closed[j] || blocked(j)) continue;
        const d = (dx && dy ? Math.SQRT2 : 1) * g;
        const t = gScore[i] + (d / walkSpeed((H[j] - H[i]) / d)) * (cost ? cost[j] : 1);
        if (t < gScore[j]) { gScore[j] = t; came[j] = i; heap.push(t + h(j), j); }
      }
    }
    if (!isFinite(gScore[ib])) return { error: 'No route stays under that slope limit. Raise the limit or pick closer points.' };
    const cells = [];
    for (let i = ib; i !== -1; i = came[i]) cells.push(i);
    cells.reverse();
    let dist = 0, up = 0, down = 0, maxs = 0, hazardCells = 0;
    const prof = [[0, H[cells[0]]]];
    for (let k = 1; k < cells.length; k++) {
      const i = cells[k - 1], j = cells[k], dh = H[j] - H[i];
      const run = Math.hypot((j % C) - (i % C), ((j / C) | 0) - ((i / C) | 0)) * g;
      dist += Math.hypot(run, dh);
      if (dh > 0) up += dh; else down -= dh;
      maxs = Math.max(maxs, grid.slope[j]);
      if (cost && cost[j] > 1) hazardCells++;
      prof.push([dist, H[j]]);
    }
    return { cells, time: gScore[ib], dist, up, down, maxs, prof, hazardCells };
  }

  /** Slope (degrees) per cell from heights, by central differences. Cells without data get 99. */
  function slopes(H, valid, cols, rows, g) {
    const out = new Float32Array(cols * rows);
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      if (!valid[i]) { out[i] = 99; continue; }
      const xl = x > 0 && valid[i - 1] ? i - 1 : i, xr = x < cols - 1 && valid[i + 1] ? i + 1 : i;
      const yu = y > 0 && valid[i - cols] ? i - cols : i, yd = y < rows - 1 && valid[i + cols] ? i + cols : i;
      const dx = xr === xl ? 0 : (H[xr] - H[xl]) / ((xr - xl) * g);
      const dy = yd === yu ? 0 : (H[yd] - H[yu]) / (((yd - yu) / cols) * g);
      out[i] = Math.atan(Math.hypot(dx, dy)) * 180 / Math.PI;
    }
    return out;
  }

  return { planRoute, walkSpeed, slopes, VFLAT };
});
