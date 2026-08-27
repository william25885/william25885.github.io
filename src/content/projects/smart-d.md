---
title: "SMART-D — LLM-Guided Multi-Agent Reinforcement Learning"
summary: "A cooperative multi-agent framework that uses a frozen LLM as a training-time macro-planner, then distils the guidance into the policy so the LLM can be removed entirely at deployment."
tags: ["Multi-Agent RL", "PPO", "LLM Planning", "Self-Distillation", "Overcooked-AI", "vLLM", "Reinforcement Learning", "PyTorch", "LLM"]
category: research
order: 2
repo: "https://github.com/william25885/SMART-D"
---

> **A three-person project.** SMART-D was built with Bryan Andersen and Cheng Ru Shu at National
> Taiwan University, and written up as *SMART-D: Semantic Macro-Planning for Multi-Agent
> Reinforcement and Task Distillation*. **My share was the implementation of SMART-D itself, the
> Overcooked-AI evaluation environment, the rebuttal experiment, and the Methodology section of the
> paper.** The Standard PPO baseline, the online action-controller variant, and the related-work
> survey were my collaborators'.

## The problem

Cooperative multi-agent tasks with sparse rewards fail in a characteristic way. In Overcooked-AI's
*Forced Coordination* layout, a physical barrier bisects the kitchen and neither agent can reach the
serving counter alone — one must hand ingredients across a counter for the other to plate and serve.
Reward arrives only on delivery, +20 per soup, and no single agent's exploration can ever produce
that event.

Standard PPO does not merely learn slowly here. It scores **0.00 ± 0.00**: across three seeds and
six million environment steps, it never delivers a single soup. The agents settle into a stable
non-cooperative equilibrium — a coordination collapse, not a slow start.

The exploration problem is not really about state coverage. The agents have no notion of the *task
decomposition*: fetch onion, hand it across, place in pot, wait, plate, serve. An LLM, asked in
text, describes that decomposition immediately.

## Results

Three seeds, $6.0 \times 10^6$ training steps each, evaluated on environment seed 42 over 20
episodes. Reported as mean ± standard deviation of the final five checkpoints, averaged across seeds.

| Kitchen layout | Standard PPO | SMART-D |
|---|---|---|
| Cramped Room | 171.67 ± 3.45 | **216.07 ± 2.26** |
| Coordination Ring | 147.80 ± 3.58 | **201.00 ± 14.45** |
| Forced Coordination | 0.00 ± 0.00 | **117.47 ± 8.20** |

The third row is the result the framework exists for. A gain from 171 to 216 on *Cramped Room* is a
better policy; a gain from **exactly zero** to 117 on *Forced Coordination* is a different kind of
claim — the baseline never finds the reward at all, so there is nothing for it to improve on.

The *Cramped Room* standard deviation is also worth noting: 2.26 against the baseline's 3.45. The
semantic anchor does not only raise the score, it makes runs agree with each other.

## The macro-planner

The planner $\Phi$ runs in three stages.

**State summariser.** Each raw Overcooked state is translated into a structured natural-language
description: layout name, current timestep and remaining horizon, each player's position,
orientation and held object, the status of every pot (empty, filling with $n/3$ onions, cooking
with $c$ steps left, or ready), and the locations of onion dispensers, dish dispensers and serving
counters.

**LLM call.** The summary goes to a frozen LLM under a fixed system prompt, which constrains the
output to a JSON object over eight legal subgoals:

$$
\begin{aligned}
\Sigma_g = \{\; &\texttt{GET\_ONION},\; \texttt{PUT\_ONION},\; \texttt{GET\_DISH},\; \texttt{PICK\_SOUP}, \\
&\texttt{DELIVER},\; \texttt{PLACE\_ON\_COUNTER},\; \texttt{PICKUP\_FROM\_COUNTER},\; \texttt{IDLE} \;\}
\end{aligned}
$$

Serving through **vLLM** is what makes this dependable rather than best-effort: its
`response_format=json_object` guard enforces structured decoding, so the planner receives parseable
output instead of prose that happens to contain JSON. The model is Qwen2.5-1.5B-Instruct.

Calls are made every $K = 20$ environment steps rather than every step. This is the design decision
the whole framework rests on, and it does two things at once: it amortises inference cost, and it
holds the subgoal fixed long enough to act as a *stationary anchor* while both agents update their
policies against it.

**Subgoal encoder.** The JSON is encoded into a fixed 18-dimensional vector,

$$
g(j,t) = \big[\; \mathrm{onehot}(z^1) \;\|\; \mathrm{onehot}(z^2) \;\|\; u \;\|\; \tfrac{H-t}{H} \;\big]
$$

where $u \in [0,1]$ is an urgency scalar and $H = 400$ is the episode horizon. **Both agents receive
the same encoding**, which is how a centralised plan becomes a shared anchor across two
independently-acting policies without any communication channel between them.

## One network, two roles

The teacher and the student are not two networks. They are the *same* MLP $\pi_\theta$, evaluated
twice per minibatch with different conditioning:

$$
\text{Teacher:} \quad \ell^{\mathrm{T}}, V^{\mathrm{T}} = \pi_\theta(o, g), \qquad
\text{Student:} \quad \ell^{\mathrm{S}} = \pi_\theta(o, \mathbf{0}_{18})
$$

The subgoal embedding acts as a conditioning switch: supply $g$ and the network is the LLM-guided
teacher, supply the zero vector $\mathbf{0}_{18}$ and it is the unguided student. At deployment $g$
is permanently $\mathbf{0}_{18}$, and the language model is never queried.

This avoids the parameter doubling of standard knowledge distillation, and it means improvements to
the shared backbone reach the student immediately rather than having to be transferred.

The distillation loss is a KL between the two, with the teacher logits **detached**:

$$
\mathcal{L}_{\mathrm{SD}}(\theta) = \mathrm{KL}\Big( \mathrm{softmax}\big(\mathrm{sg}[\ell^{\mathrm{T}}]\big) \;\Big\|\; \mathrm{softmax}\big(\ell^{\mathrm{S}}\big) \Big)
$$

The stop-gradient is essential rather than cosmetic. Without it, gradients flow back into the
teacher branch too, and the uninformed student starts dragging the teacher toward itself — the
objective degenerates into self-teaching and the LLM's guidance is averaged away.

The composite update is

$$
\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{PPO}}(\theta) + c_v \mathcal{L}_V(\theta) - c_H \mathcal{H}(\theta) + \lambda_{\mathrm{SD}}(k)\, \mathcal{L}_{\mathrm{SD}}(\theta)
$$

with $c_v = 0.5$, $c_H = 0.01$, and gradients clipped to a global norm of 0.5.

## The coupled schedule

Two schedules run against each other over $N$ training iterations.

The reward-shaping weight holds, then decays:

$$
w_{\mathrm{shp}}(k) = \begin{cases} 1, & k/N \le 0.5 \\ 1 - 2\,(k/N - 0.5), & k/N > 0.5 \end{cases}
$$

and the distillation coefficient is silent, then ramps:

$$
\lambda_{\mathrm{SD}}(k) = \begin{cases} 0, & k/N \le 0.5 \\ \lambda^\star \cdot \min\big(1,\, 2(k/N - 0.5)\big), & k/N > 0.5 \end{cases}
$$

with $\lambda^\star = 0.1$.

Note what the shaping term actually is: Overcooked's built-in dense events — onion-in-pot,
dish-pickup, soup-pickup — not a reward for following the LLM's subgoal. The LLM shapes *behaviour*
through the conditioning vector, not through the reward.

The result is a clean two-phase curriculum. Phase one is shaped-reward PPO on the teacher branch
only, letting the LLM organise joint exploration without polluting a student that has nothing to
learn from yet. Phase two transfers the guided skill into the zero-subgoal branch while the shaping
signal fades, so the student is distilled against a teacher that is itself being re-aligned to the
sparse task reward.

## Measuring whether the distillation worked

Every 50 training iterations, 20 evaluation episodes are played **twice** under deterministic
actions: once with the subgoal embedding zeroed, once with live LLM subgoals refreshed every 50
steps rather than the 20 used in training. The distillation gap is the difference,

$$
\Delta_{\mathrm{SD}} = \bar{R}_{\mathrm{LLM}} - \bar{R}_{0}
$$

A large gap means the policy still depends on guidance it will not have at deployment. A gap
approaching zero while return stays high means the behaviour has been internalised. Either half
alone would be easy to fake — a student that has learned nothing and a teacher that has also learned
nothing agree perfectly.

## Ablations

**LLM calling frequency $K$**, on *Cramped Room*:

| $K$ | Score |
|---|---|
| 10 | **219.60 ± 0.55** |
| 20 (used) | 216.80 ± 4.66 |
| 30 | 206.00 ± 9.70 |

Halving the interval buys about three points and doubles the number of LLM calls; $K = 30$ loses ten.
$K = 20$ is the compromise, and the monotone trend confirms the planner is doing real work rather
than acting as an expensive random perturbation.

**Distillation coefficient $\lambda^\star$**:

| $\lambda^\star$ | Score |
|---|---|
| 0.01 | 197.4 ± 4.72 |
| 0.1 (used) | **216.80 ± 4.66** |
| 0.5 | 204.8 ± 15.12 |

Too little and the student never bridges the sparse-reward bottleneck; too much and the KL term
dominates the objective, penalising the local exploration needed to discover the action pipeline.

**LLM backbone**, and the most interesting of the three:

| Method | Backbone | Score |
|---|---|---|
| Standard PPO | — | 171.67 ± 3.45 |
| SMART-D | Qwen2.5-0.5B-Instruct | 215.20 ± 2.42 |
| SMART-D | Qwen2.5-1.5B-Instruct | **216.07 ± 2.26** |
| SMART-D | Qwen2.5-3B-Instruct | 214.53 ± 2.99 |

A sixfold range of model size moves the score by less than two points, while every variant beats the
baseline by more than forty. Whatever the LLM is contributing, it is not something only a large model
possesses — which is the strongest available evidence that the framework extracts a genuine task
abstraction rather than exploiting one model's prompt-specific behaviour.

## Limitations

Stated in the paper, and worth repeating:

- **No zero-shot cross-map transfer.** The method is evaluated per layout with layout-specific
  training. It does not adapt to an unseen kitchen topology.
- **Two agents only.** Scaling past two, or introducing a human partner, is untested.
- **The LLM still misbehaves.** Semantic hallucination and output-format mismatch occur despite the
  structured-decoding guard.

## A negative result worth keeping

A companion experiment inserted an LLM directly into the action-selection pathway — an oracle branch
inside $\varepsilon$-greedy, choosing primitive actions rather than subgoals.

It helped early and then stalled. Two mechanisms explain it. First, **no persistent intent**: every
call is conditioned on the instantaneous state, so recommendations oscillate between conflicting
micro-decisions exactly when the policy needs to converge. Second, **abstraction mismatch**: a
counter handoff does not factor cleanly into language tokens, and the LLM's reasoning is reliable at
the strategic layer ("pick up an onion") rather than at grid kinematics.

That failure is the argument for the main design. Ask the LLM to *plan* over a temporally abstract
subgoal space, not to *act* — which decouples LLM frequency from control frequency, suppresses
action oscillation by holding subgoals fixed over $K$ steps, and meets the model at the level of
abstraction where it is actually dependable.
