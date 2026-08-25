---
title: "LINE Travel Bot — Conversational Travel Planner"
summary: "A travel assistant on LINE that collects preferences through multi-step dialogue and generates personalised itineraries with Gemini, backed by an OAuth-protected analytics dashboard."
tags: ["Next.js", "TypeScript", "Prisma", "PostgreSQL", "Gemini API", "LINE Bot SDK", "LLM", "Full-Stack", "Vercel"]
category: systems
order: 8
repo: "https://github.com/william25885/Line-Travel-Bot"
liveDemo: "https://wp1141-mrbm.vercel.app"
---

## What it does

A travel planner that lives inside LINE. It asks about destination, trip length, budget, theme and
month through ordinary conversation, then generates a day-by-day itinerary with attractions, timing,
food suggestions, Google Maps links and travel tips.

Behind it sits an admin dashboard with conversation history, search and analytics.

The bot is live: **[@083lhmmz](https://line.me/R/ti/p/@083lhmmz)** on LINE.

## The state problem

A LINE webhook is stateless. Each message arrives as an independent HTTP POST, with no notion that
the user said something thirty seconds ago — but a planning conversation is inherently stateful,
because "5 days" only means anything if you remember you just asked how long the trip was.

Conversation state is therefore persisted per user and advanced explicitly: which slots are filled
(country, days, budget, theme, month), which is being asked for now, and what has been collected so
far. Every message loads that state, interprets the message against it, and writes it back.

Making the state a database row rather than in-memory session data is what allows deployment to a
serverless platform at all — no instance holds a conversation, so any instance can serve any message.
It also makes the admin dashboard possible for free: an inspectable conversation is a side effect of
having made state durable in the first place.

## Two paths through the same flow

The bot works in two modes, and the fallback is not a degraded afterthought:

**Rule-based.** Slots are filled by matching structured replies. Deterministic, free, and always
available.

**LLM-augmented** (when `GEMINI_API_KEY` is set). Gemini extracts preferences from free-form
language — "I want to go to Japan for 5 days" resolves to `country: 日本, days: 5天`; "plan a beach
trip in March" fills `month` and `theme` together — and generates the final itinerary.

Without an API key the bot still runs end to end, just without natural-language understanding or
generated itineraries. Building it this way means a quota exhaustion, an outage or a missing key
degrades the experience instead of breaking the product, and it keeps the conversation logic
testable without calling a paid API.

Gemini is used with `gemini-2.0-flash-exp` as primary and `gemini-1.5-pro` as fallback, behind
structured prompts with retries and timeouts.

## Validation

LLM output is not trusted directly. Extracted preferences are checked against the expected format
before entering conversation state, and invalid input produces a concrete example rather than an
error. This is the boundary that keeps a hallucinated field from silently corrupting a conversation
that the user then has to restart.

## Admin dashboard

Google OAuth through NextAuth, then:

- All conversations, filterable by user ID or conversation state
- Full detail per conversation: messages, extracted preferences, generated recommendation
- Analytics: popular destinations, distribution across conversation states, and planning completion
  rate

Completion rate is the metric that actually drives changes. It shows the proportion of conversations
that reach a finished itinerary, and the state distribution shows *where* the others stopped — which
localises a drop-off to a specific question rather than leaving it as a general sense that people
give up.

## Stack

Next.js 16 with the App Router, serving both the dashboard and the webhook through API routes.
PostgreSQL on Neon via Prisma, with the schema as the single source of truth for conversation state.
TypeScript throughout, Tailwind for styling, deployed on Vercel.

One repository holds the bot and the dashboard, which matters because they share the conversation
schema — a change to what the bot stores is a change to what the dashboard reads, and keeping them
together makes that a single typed edit rather than a coordination problem.
