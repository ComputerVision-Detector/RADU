import '../src/web/frontend/editor-model.js';

const M = globalThis.RaduModel;
const assert = (condition, message) => { if (!condition) throw new Error(message); };

Deno.test('grouping, page switching and attached element data remain intact', () => {
  const doc = {
    image: { width: 800, height: 600 },
    elements: [
      { id: 'start', type: 'Box', shape: 'terminator', bbox: [20, 20, 160, 80] },
      { id: 'end', type: 'Box', shape: 'process', bbox: [300, 200, 460, 280] },
    ],
    relations: [], warnings: [],
  };
  M.normalize(doc);
  M.group(doc, ['start', 'end']);
  assert(M.expandGroup(doc, ['start']).length === 2, 'group selection lost a member');
  const second = M.addPage(doc);
  assert(doc.active_page === second && doc.elements.length === 0, 'new page is not blank');
  M.switchPage(doc, 'page-1');
  assert(doc.elements.length === 2 && doc.elements[0].group_id, 'switching pages lost elements');
});

Deno.test('translation moves bbox, freehand points and both endpoint formats', () => {
  const element = {
    bbox: [10, 20, 70, 80], start: [10, 20], end: { x: 70, y: 80 },
    points: [{ x: 30, y: 40 }],
  };
  M.translate(element, 5, -3);
  assert(element.bbox[0] === 15 && element.bbox[1] === 17, 'bbox did not move');
  assert(element.start.x === 15 && element.end.y === 77, 'arrow endpoints did not move');
  assert(element.points[0].x === 35 && element.points[0].y === 37, 'pen points did not move');
});

Deno.test('shape resize and connector boundary use updated geometry', () => {
  const node = { type: 'Box', shape: 'decision', bbox: [100, 100, 220, 180] };
  const oldPoint = M.boundary(node, { x: 400, y: 140 });
  node.bbox = M.resize(node.bbox, 'e', 60, 0, { width: 800, height: 600 });
  const newPoint = M.boundary(node, { x: 400, y: 140 });
  assert(newPoint.x > oldPoint.x && node.bbox[2] === 280, 'connector did not follow resize');
  const reverse = M.resize([200, 200, 300, 260], 'nw', -30, -20, { width: 800, height: 600 });
  assert(reverse[0] === 170 && reverse[1] === 180, 'northwest handle failed');
});

Deno.test('crossing lines remain separate relation objects', () => {
  const first = { id: 'one', start: { x: 50, y: 0 }, end: { x: 50, y: 100 } };
  const second = { id: 'two', start: { x: 0, y: 50 }, end: { x: 100, y: 50 } };
  for (const [mode, command] of [['none', ' L '], ['arc', ' Q '], ['gap', ' M '], ['sharp', ' L ']]) {
    assert(M.jumpPath(second, [first], mode).includes(command), `${mode} crossing missing`);
  }
  assert(first.id === 'one' && second.id === 'two', 'line jump changed relation identity');
});
