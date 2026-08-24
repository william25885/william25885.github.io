// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeFigure from './src/lib/rehype-figure.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://william25885.github.io',
  // No `base`: this is a user site served from the domain root.
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, rehypeFigure],
    }),
    // Dual themes: Shiki emits both palettes, and global.css switches to the
    // dark one under `prefers-color-scheme: dark`.
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
});
