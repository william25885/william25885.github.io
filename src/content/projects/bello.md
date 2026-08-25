---
title: "Bello — Social Meetup & Real-time Chat System"
summary: "A full-stack platform for organising physical meetups, with real-time private and group chat, a friend system and an admin console, on a normalised PostgreSQL schema deployed serverless."
tags: ["Vue 3", "Flask", "PostgreSQL", "JWT", "Google OAuth", "Vercel", "Full-Stack"]
category: systems
order: 7
repo: "https://github.com/william25885/Bello"
liveDemo: "https://bello-tw.vercel.app/"
---

## What it does

Bello organises **physical** meetups — lunch, coffee, dinner, drinks, language exchange. A user
hosts an event, others join it, and the platform handles everything around that: discovering people,
agreeing on a place, and talking before and after.

Two roles: users host and join meetups, chat, and manage friends; administrators moderate meetups
and users and can inspect chat history when something is reported.

## Data model

The schema is normalised to **fourth normal form**, which is the decision the rest of the system
rests on.

That matters here more than it would in most applications, because the domain is dense with
many-to-many relationships that all touch each other: users to meetups (attendance), users to users
(friendship, and friend *requests*, which are directional and stateful), users to conversations,
conversations to messages. Denormalising any of them — a participant list stored on the meetup row,
a friends array on the user row — makes the write path cheap and then makes every subsequent
question expensive and every concurrent update a race. Keeping the join tables explicit means
attendance, friendship state and message ownership each have exactly one authoritative
representation.

The cost is more joins on read, which is the right thing to pay for on a platform this size.

## Backend

Flask, split by domain rather than by layer — `auth`, `profile`, `meeting`, `chat`, `friend`,
`admin` each own their endpoints, with `DB_utils.py` and `jwt_utils.py` shared underneath. A change
to friend-request semantics touches one directory.

**Authentication** supports both password login (bcrypt-hashed) and Google OAuth, issuing a JWT in
either case, so the rest of the API only ever deals with one kind of credential regardless of how
the user signed in. Authorisation is enforced by `@require_auth` and `@require_admin` decorators at
the endpoint boundary — the check sits on the route rather than inside handler logic, which is what
makes it auditable by reading the route table.

## Frontend

Vue 3 with Vue Router and Bootstrap, built with Vite. Route-level navigation guards keep unauthorised
users out of pages, backed by the server-side decorators — the guards are for user experience, the
decorators are the actual security boundary.

Two details worth mentioning because they came from real failures rather than a feature list:

**Explicit loading states.** Pages track their own load status so data does not flicker in
piecemeal as requests resolve. Without this, a chat view visibly assembles itself, which reads as
broken.

**Embedded browser detection.** Links opened inside LINE or Facebook run in an in-app webview where
Google OAuth is blocked outright. The app detects this and tells the user to open the page in a real
browser — an entirely predictable failure that produces a completely mystifying error if unhandled,
and the single most common way real users first hit the app.

Avatar upload uses Cropper.js so images are cropped client-side before upload rather than stored
oversized and resized on read.

## Real-time features

- One-to-one private chat
- Group chat rooms scoped to a meetup
- Online status shown throughout the friend list, private chat list and search results
- Friend requests with send / accept / reject states
- Google Places autocomplete for meetup locations, with map display

Meetups can be password-protected, which is how a private gathering coexists with a public listing.

## Admin console

Administrators can browse and search all meetups and users, cancel or end meetups, delete users, and
page through both private and meetup conversation lists with message counts and last-activity
timestamps.

Chat inspection is a deliberate product decision rather than an oversight: a platform that arranges
in-person meetings between strangers needs a way to act on a harassment report, and that requires
being able to read the conversation it refers to.

## Deployment

Frontend as a static build and backend as **Vercel Serverless Functions**, against a Neon-hosted
PostgreSQL database.

The serverless choice fits the traffic shape — meetup activity is bursty and mostly idle — but it
constrains the backend: no in-process state survives between requests, so nothing can be cached in a
module-level variable and every request re-establishes its database connection. Writing the Flask
app as if each request were the first one it had ever seen is what makes it deployable this way.
