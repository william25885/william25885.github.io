---
title: "Scene-Specialized DQN for Multi-Level Generalization"
summary: "A mixture-of-experts agent for the LevDoom Seek and Slay benchmark — five level-specialised DQN experts routed by a learned scene classifier, compressed to fit a deployment size budget."
tags: ["Deep RL", "Rainbow DQN", "Mixture of Experts", "RND", "ViZDoom", "Model Compression", "Reinforcement Learning", "PyTorch"]
category: research
order: 3
repo: "https://github.com/william25885/levdoom-scene-specialized-dqn"
---

## The benchmark and the design decision

LevDoom's *Seek and Slay* task asks a single policy to perform across five levels that differ in
map layout, enemy composition and visual style. The intended difficulty is generalisation: one
agent, five distributions.

Training a single network across all five caused interference — strategies that work in a small
room actively hurt in a large maze, and the shared weights could not hold both. So the problem was
restructured: instead of forcing one policy to generalise, **five level-specialised experts** are
trained independently and a lightweight classifier routes between them.

This trades parameter efficiency for per-level performance. The size budget lost to five networks
is recovered afterwards through half-precision compression.

## Results

Mean kills across 5 fixed evaluation seeds per level:

| Level | Environment ID | Map | Mean kills |
|---|---|---|---|
| 0 | `SeekAndSlayLevel0-v0` | default | 35.60 |
| 1 | `SeekAndSlayLevel1_6-v0` | mixed_enemies | 17.60 |
| 2 | `SeekAndSlayLevel3_1-v0` | blue_mixed_resized | 10.80 |
| 3 | `SeekAndSlayLevel2_3-v0` | red_mixed_enemies | 14.80 |
| 4 | `SeekAndSlayLevel4-v0` | complete | 8.40 |

The descending trend tracks the benchmark's difficulty gradient. Level 4 — a large maze with
sparsely distributed enemies — stayed the bottleneck despite double the training budget and the
strongest exploration incentives.

## Routing

Each level has a distinct visual identity: wall colours, enemy sprites, map scale. A small CNN
(three convolutional layers → global average pool → linear, roughly 53 KB) identifies the level from
raw frames.

At episode start the classifier **accumulates logits over the first 3 steps**, averages them, and
takes the argmax. The selected expert is then **locked for the rest of the episode**.

Locking is the part worth defending. It costs 3 essentially random actions out of a 1250-step
episode — 0.24% of the budget — and in exchange removes any possibility of mid-episode expert
switching. A switch would hand a fresh network a frame stack assembled under a different policy,
discarding exactly the temporal context the expert depends on. Paying 0.24% to make that failure
impossible is a good trade.

## Observation handling

- Input is **4 stacked RGB frames** at 84×84 (12 channels), stored as `uint8` and normalised at
  sample time.
- RGB is kept rather than converted to grayscale. Level identity is signalled largely by colour —
  `blue_mixed_resized`, `red_mixed_enemies` — and grayscale conversion would destroy the very
  feature the router depends on.

## Algorithm — Rainbow-Lite

Four Rainbow components, chosen for sample efficiency under a fixed step budget:

| Component | Purpose | Setting |
|---|---|---|
| Dueling architecture | Separates state value from action advantage — useful in the many states where the agent is just navigating and no action is clearly better | hidden dim 512 |
| Double DQN | Decouples action selection from evaluation to reduce Q-value overestimation | — |
| Prioritised experience replay | Samples high-TD-error transitions more often, so rare kill events are learned from faster | $\alpha = 0.6$, $\beta: 0.4 \to 1.0$ |
| N-step returns | Propagates delayed kill rewards back to the movement actions that set them up | $n = 3$ |

Distributional RL (C51) and NoisyNets are deliberately not implemented — hence "Rainbow-Lite".

The dueling head computes

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')
$$

Loss is Huber, weighted by importance-sampling weights, with gradient clipping at 10.0.

## Exploration — two curiosity signals, scaled per level

Level 4 is a large maze where enemies are sparse; $\varepsilon$-greedy alone leaves the agent
circling in corners. Two intrinsic rewards address this:

**Random Network Distillation** — a frozen randomly-initialised target network and a trained
predictor. Prediction error on a state is a novelty signal, clipped to $[0, 5]$. This is *global*:
it rewards states novel with respect to everything seen so far.

**Visual episodic memory** — a per-episode anti-stuck mechanism. Each frame is downsampled, quantised
to 8 intensity levels, and hashed:

```python
grid = gray[::12, ::12].astype(np.uint8) // 32
return hash(grid.tobytes())
```

A bonus of $1/\sqrt{n_{\text{visits}}}$ discourages revisiting; the table clears at episode end.
Compared to training a separate density model this costs essentially nothing.

**The scaling is the actual design decision.** Intrinsic rewards are not applied uniformly — they
are switched off where exploration is not the bottleneck:

| Levels | RND weight | Episodic weight | Rationale |
|---|---|---|---|
| 0, 1 | 0.0 | 0.0 | Small maps; enemies are found without incentive. Curiosity would only dilute the kill signal. |
| 2, 3 | 1.0 | 0.5 | Moderate map size. |
| 4 | 5.0 | 2.0 | Large maze, sparse enemies — exploration dominates the difficulty. |

Shaped reward per step:

$$
r = r_{\text{env}} + 50\,\Delta\text{kills} + 0.1\,\Delta\text{health} + w_{\text{rnd}}\, r_{\text{RND}} + w_{\text{ep}}\, r_{\text{episodic}}
$$

clipped to $[-30, 100]$.

## Level 4 — the exploration bottleneck

Level 4 got double the training budget (4M vs 2M steps) and the strongest curiosity weights, and
still returned the lowest kill count.

> The following are **post-hoc hypotheses derived from the code, not verified experimentally.**
> They are stated here because the failure is real and unexplained, and a plausible unverified
> explanation is more useful to a reader than silence — provided it is labelled as such.

*Hypothesis A — intrinsic reward may dominate the kill signal.* At `rnd_scale = 5.0` the novelty
bonus can reach 25 per step and is paid *every* step, whereas a kill pays 50 once. Accumulated over
1250 steps, exploration reward can plausibly exceed total kill reward, biasing the policy toward
wandering over engaging.

*Hypothesis B — hash collisions in the episodic memory.* Sampling every 12th pixel of a 240×320
frame yields roughly a 20×27 grid at 8 intensity levels. In a large maze with repeated wall
textures, distinct locations may hash identically — in which case the anti-stuck penalty fires in
the wrong places and actively discourages entering unexplored regions. Small maps, with fewer and
more visually distinct locations, would not show this.

Distinguishing the two is straightforward and worth doing: log the ratio of intrinsic to extrinsic
return per episode, and count distinct hashes against distinct ground-truth positions.

## Model compression

Five experts plus the classifier exceeded the deployment size budget at single precision. All
weights are downcast to FP16:

```python
compressed = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
```

Total payload: **31.69 MB**, with no measured change in behaviour. Action selection is `argmax` over
Q-values, so only the *ordering* of Q-values matters, and FP16 truncation sits far below the margin
separating actions in practice.

## Training

| | |
|---|---|
| Steps per expert | 2,000,000 (Levels 0–3) · 4,000,000 (Level 4) |
| Total environment steps | **12,000,000** |
| Replay capacity | 200,000 |
| Warmup | 25,000 steps |
| Batch size | 64 |
| Optimiser | Adam, lr 1e-4 |
| Target sync | every 8,000 steps |
| Update frequency | every 4 steps |
| $\varepsilon$ schedule | 1.0 → 0.05, linear over 50% of the step budget |
| Checkpointing | best mean kills over 5 evaluation episodes, every 50,000 steps |

Each expert is trained independently from random initialisation. No weights are shared or
transferred between levels.

## Reference

The environment is [LevDoom](https://github.com/TTomilin/LevDoom), built on ViZDoom (Tomilin,
Dai, Fang and Pechenizkiy, *IEEE Conference on Games*, 2022).
