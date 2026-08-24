# william25885.github.io

Personal site and project portfolio — <https://william25885.github.io>

Built with [Astro](https://astro.build). Project pages are Markdown in a content
collection; maths is rendered at build time with `remark-math` + `rehype-katex`.
No client-side JavaScript is shipped.

## Development

```bash
npm install
npm run dev        # dev server
npm run build      # static build to dist/
npm run preview    # serve the build locally
```

## Layout

```
src/
├── components/          Nav, Footer, ProjectCard, TagList
├── layouts/             BaseLayout, ProjectLayout
├── content/projects/    one Markdown file per project
├── pages/               home, /projects, /projects/[id], /projects/tag/[tag]
├── styles/global.css    design tokens and all site styling
└── content.config.ts    content collection schema
public/
├── media/               figures used by project pages
└── resume.pdf
```

Deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.
