import { visit } from 'unist-util-visit';

const isImg = (n) => n.type === 'element' && n.tagName === 'img';
const isBlank = (n) => n.type === 'text' && !n.value.trim();

/**
 * Turn a paragraph containing nothing but one image into a <figure>, using the
 * image's alt text as a visible <figcaption>.
 *
 * Images inside table cells are left alone — those are already labelled by
 * their row and column headers.
 */
export default function rehypeFigure() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'p' || !parent || index === undefined) return;

      const meaningful = node.children.filter((c) => !isBlank(c));
      if (meaningful.length !== 1 || !isImg(meaningful[0])) return;

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

      parent.children[index] = {
        type: 'element',
        tagName: 'figure',
        properties: { className: ['figure'] },
        children,
      };
    });
  };
}
