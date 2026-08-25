const isImg = (n) => n.type === 'element' && n.tagName === 'img';
const isBlank = (n) => n.type === 'text' && !n.value.trim();
const isFigure = (n) => n.type === 'element' && n.tagName === 'figure';
const isTable = (n) => n.type === 'element' && n.tagName === 'table';

/** A paragraph holding nothing but one image, with its alt text as caption. */
function toFigure(node) {
  if (node.type !== 'element' || node.tagName !== 'p') return null;

  const meaningful = node.children.filter((c) => !isBlank(c));
  if (meaningful.length !== 1 || !isImg(meaningful[0])) return null;

  const img = meaningful[0];
  const alt = img.properties?.alt;
  const children = [img];

  if (alt) {
    children.push({
      type: 'element',
      tagName: 'figcaption',
      properties: {},
      children: [{ type: 'text', value: alt }],
    });
  }

  return {
    type: 'element',
    tagName: 'figure',
    properties: { className: ['figure'] },
    children,
  };
}

/**
 * Promote lone images to captioned figures, and lay out runs of adjacent
 * figures as a grid.
 *
 * Figures that sit together are almost always meant to be compared, and
 * comparison is impossible when each one is a screen tall and the reader has
 * to hold the previous one in memory.
 *
 * Images inside table cells are left alone — their row and column headers
 * already label them.
 *
 * The walk is an explicit single pass rather than `visit`, because the grid
 * this creates is itself a node holding a run of adjacent figures: a
 * general-purpose traversal descends into it and wraps them again, and again.
 */
function walk(node) {
  if (!node.children || !node.children.length) return;

  const converted = node.children.map((child) => {
    const fig = toFigure(child);
    if (fig) return fig;

    // A wide table must scroll inside its own box. Left bare, it pushes the
    // whole page sideways on a phone — measured at 108px of horizontal
    // overflow on a 375px viewport before this wrapper existed.
    if (isTable(child)) {
      walk(child);
      return {
        type: 'element',
        tagName: 'div',
        properties: { className: ['table-scroll'] },
        children: [child],
      };
    }

    walk(child);
    return child;
  });

  const out = [];
  let run = [];

  const flush = () => {
    if (run.length > 1) {
      out.push({
        type: 'element',
        tagName: 'div',
        properties: { className: ['figure-grid'] },
        children: run,
      });
    } else {
      out.push(...run);
    }
    run = [];
  };

  for (const child of converted) {
    if (isFigure(child)) {
      run.push(child);
    } else if (run.length && isBlank(child)) {
      // Whitespace separating figures in a run; drop it rather than break the run.
    } else {
      flush();
      out.push(child);
    }
  }
  flush();

  node.children = out;
}

export default function rehypeFigure() {
  return (tree) => walk(tree);
}
