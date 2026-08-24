---
title: "SMART-D — LLM-Guided Multi-Agent Reinforcement Learning"
summary: "A cooperative multi-agent PPO framework that uses a frozen LLM as a training-time macro-planner, then distils the guidance away so the deployed policy carries no inference cost."
tags: ["Multi-Agent RL", "PPO", "LLM Planning", "Self-Distillation", "Overcooked-AI", "vLLM", "Reinforcement Learning", "PyTorch", "LLM"]
category: research
order: 2
repo: "https://github.com/william25885/SMART-D"
context: "Deep Reinforcement Learning final project, National Taiwan University"
---

## The problem

Cooperative multi-agent tasks with sparse rewards fail in a characteristic way. In Overcooked-AI's
*Forced Coordination* layout, two agents are placed in separate halves of the kitchen and neither
can complete a dish alone — one must hand ingredients across a counter for the other to plate and
serve. No reward arrives until a full soup is delivered, and no single agent's exploration can ever
produce that event.

Standard PPO does not recover from this. Both agents wander, neither is reinforced, and the policies
settle into a stable non-cooperative equilibrium — a coordination collapse, not a slow start.

The exploration problem here is not really about state coverage. It is that the agents have no
notion of the *task decomposition*: fetch onion, pass onion, place in pot, plate, serve. An LLM,
asked in text, describes that decomposition immediately.

## The approach

SMART-D uses a frozen instruction-tuned LLM (Qwen2.5-1.5B-Instruct, served through vLLM) as a
**macro-planner during training only**. The loop has three parts:

1. **State-to-text summariser** — converts the Overcooked grid state into a compact textual
   description of what each agent holds, what the pots contain, and what is reachable.
2. **LLM subgoal planner** — returns a structured JSON subgoal per agent ("pick up onion", "hand
   over to partner"). This is a macro-action over many environment steps, not a per-step action.
3. **PPO learner** — a single network serving as both teacher and student, trained with the
   environment reward shaped by progress toward the current subgoal.

## The deployment problem, and the distillation schedule

Querying an LLM at every decision point is not a policy anyone can deploy: it puts a
billion-parameter forward pass inside the control loop. The guidance has to end up *inside* the
network.

SMART-D handles this with a coupled schedule. The teacher/student split lives in one network: the
teacher pathway sees the LLM subgoal, the student pathway does not. Beginning at the halfway point
of training, a self-distillation term pulls the student's action distribution toward the teacher's,

$$
\mathcal{L}_{\mathrm{SD}} = \lambda_{\mathrm{sd}} \, D_{\mathrm{KL}}\!\left( \pi_{\mathrm{teacher}}(\cdot \mid s, g) \,\|\, \pi_{\mathrm{student}}(\cdot \mid s) \right)
$$

while the LLM's reward-shaping contribution is annealed toward zero over the same window. The two
schedules are deliberately coupled: the shaping term is only withdrawn as fast as the student
absorbs the behaviour it was producing.

By the end of training the LLM can be **removed entirely** — the deployed policy is an ordinary PPO
network with zero inference-time dependency on the language model.

## Measuring whether the distillation worked

The training loop logs a single diagnostic that makes success falsifiable:

```
[ 100/1000] steps=320,000  reward=12.34  λ_sd=0.0123  loss=0.0456  ent=0.123
  [eval] return=180.00  soups=9.00  gap=12.50
```

`gap` is the evaluation return **with** the LLM in the loop minus the return **without** it. A large
gap means the policy still depends on guidance it will not have at deployment. A gap approaching
zero means the behaviour has been internalised. Watching `gap` shrink while return stays high is the
actual claim being made; either half alone would be easy to fake.

## Results

SMART-D was evaluated against a standard PPO baseline across the `cramped_room`,
`coordination_ring` and `forced_coordination` layouts, and outperformed it on all three. The
qualitative result is the one that matters: on *Forced Coordination*, standard PPO collapses to the
non-cooperative equilibrium and SMART-D does not — the agents learn the hand-off.

> **On numbers.** Per-layout return and soup-delivery figures are not yet published in the
> repository. The comparison above is reported qualitatively rather than with a table, because a
> results table should come from a fixed, re-runnable evaluation protocol, and that protocol is
> still being tidied up alongside the code.

## What I would do differently

The most fragile part of the system is the summariser. It is hand-written, and every fact it omits
is a fact the planner cannot condition on — an LLM planner is only as good as the state description
it receives, and debugging a bad plan almost always meant finding something missing from the text,
not something wrong with the model. A learned or schema-driven summariser would remove the largest
source of silent failure.
