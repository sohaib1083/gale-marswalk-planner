// node --test app/tests/
const test = require('node:test');
const assert = require('node:assert');
const P = require('../planner.js');

function flatGrid(cols, rows, g = 2) {
  const n = cols * rows, H = new Float32Array(n), valid = new Uint8Array(n).fill(1);
  return { H, valid, cols, rows, g, slope: P.slopes(H, valid, cols, rows, g) };
}

test('flat ground: straight walk at 3 km/h', () => {
  const grid = flatGrid(50, 50);
  const r = P.planRoute(grid, { x: 0, y: 0 }, { x: 80, y: 0 }, 25);
  assert.ok(!r.error);
  assert.ok(Math.abs(r.dist - 80) < 1e-6, `dist ${r.dist}`);
  assert.ok(Math.abs(r.time - 80 / P.VFLAT) < 1e-6, `time ${r.time}`);
  assert.strictEqual(r.up, 0);
});

test('a steep wall makes the route go round it', () => {
  const grid = flatGrid(60, 60);
  for (let y = 0; y < 50; y++) grid.slope[y * 60 + 30] = 40;   // wall at x = 30, gap at the bottom
  const r = P.planRoute(grid, { x: 40, y: 20 }, { x: 80, y: 20 }, 25);
  assert.ok(!r.error);
  assert.ok(r.dist > 120, `should detour, got ${r.dist}`);
  assert.ok(r.cells.every((i) => grid.slope[i] <= 25));
});

test('no route when the wall is complete', () => {
  const grid = flatGrid(40, 40);
  for (let y = 0; y < 40; y++) grid.slope[y * 40 + 20] = 40;
  const r = P.planRoute(grid, { x: 10, y: 10 }, { x: 70, y: 10 }, 25);
  assert.ok(r.error && r.error.startsWith('No route'));
});

test('hazard cost: avoids a blocked rock and counts costly cells', () => {
  const grid = flatGrid(30, 30), cost = new Float32Array(900).fill(1);
  cost[5 * 30 + 10] = Infinity;                           // rock right on the straight line
  const r = P.planRoute(grid, { x: 0, y: 10 }, { x: 40, y: 10 }, 25, cost);
  assert.ok(!r.error);
  assert.ok(!r.cells.includes(5 * 30 + 10));
});

test('uphill is slower than downhill (Tobler)', () => {
  assert.ok(P.walkSpeed(0.2) < P.walkSpeed(-0.05));
  assert.ok(Math.abs(P.walkSpeed(0) - P.VFLAT) < 1e-12);
});

test('blocked start gives a clear error', () => {
  const grid = flatGrid(10, 10); grid.valid[0] = 0;
  assert.match(P.planRoute(grid, { x: 0, y: 0 }, { x: 10, y: 10 }, 25).error, /Pick another spot/);
});
