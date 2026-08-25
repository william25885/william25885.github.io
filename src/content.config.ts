import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const projects = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    tags: z.array(z.string()),
    category: z.enum(['research', 'systems']),
    order: z.number(),
    repo: z.string().url().optional(),
    liveDemo: z.string().url().optional(),
    demoComponent: z.string().optional(),
    demoData: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { projects };
