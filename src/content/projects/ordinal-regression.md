---
title: "Go Rank Prediction as Ordinal Regression"
summary: "Predicting a Go player's rank from 1D to 9D out of game records at 80.65% accuracy — treating rank as an ordered quantity rather than nine unrelated classes, with predictions pooled across views, seeds and games."
tags: ["Ordinal Regression", "CORAL", "Transformer", "BiLSTM", "CatBoost", "Ensemble", "PyTorch"]
category: research
order: 5
repo: "https://github.com/william25885/Go-Rank-Prediction"
context: "Machine Learning, National Taiwan University"
---

## The task

Given the record of Go games played at an unknown rank, identify the rank: 1D through 9D. Each test
sample is a file holding several games, all played at the same rank, and one rank must be predicted
for the file as a whole.

**Held-out test accuracy: 80.65%**, sixth on the class leaderboard. Random guessing over nine
classes scores 11%.

## Two accuracies, and why they differ

The cross-validated accuracy over individual games is **42.8%**. The accuracy over test files is
**80.65%**. Both are correct; they measure different things, and the distance between them is where
most of the engineering went.

A single game is weak evidence. A 5D playing carefully and a 6D playing loosely produce records that
genuinely resemble each other, and no model recovers a distinction the input does not contain. What
the task actually asks is easier: a file aggregates several games, so the same underlying signal
arrives several times with independent noise.

The system exploits that at three levels, all of them averaging distributions rather than voting on
labels:

1. **Views within a game.** Sequence models see 120 moves, but games run longer. Rather than pick a
   window, six overlapping crops — opening, end, centre, and quarter positions — are scored and
   averaged.
2. **Seeds within a model.** Each sequence architecture is trained twice under different seeds, and
   the two posteriors are averaged.
3. **Games within a file.** Each game produces a full posterior over the nine ranks from the
   meta-learner; the file's prediction is the argmax of their mean.

Averaging posteriors rather than taking a majority vote matters at the third step in particular: a
game the model finds ambiguous contributes a flat distribution and moves the file-level answer very
little, whereas a hard vote would give it the same weight as a confident one.

## What the input already contains

Worth being precise about, because it changes what the model is doing. Each move carries:

| Group | Dims | Source |
|---|---|---|
| Policy | 9 | Nine supervised models, each trained on games of one rank — the probability this move is played, according to each |
| Value | 9 | The same nine models' predicted win rates for the position |
| Rank probabilities | 9 | A dedicated rank model's estimate that the position belongs to each rank |
| Strength | 1 | A relative playing-strength score |
| KataGo analysis | 3 | Win rate, territory lead, and uncertainty |

So a per-move rank estimate is *already an input feature*. This is not a system that infers rank
from raw board positions — it is one that **aggregates and calibrates existing per-move rank signals
across a game and across games**, which is a narrower problem and an honest description of it.

That framing also explains why the tabular model reaches 20.3% on its own while the sequence models
reach 38.5%: the signal is not in any single move's rank estimate but in how those estimates behave
over the course of a game.

## Feature construction

The per-move vector is expanded from 30 raw values to **79 dimensions**:

| Group | Dims | Content |
|---|---|---|
| Base | 30 | Policy, Value, RankP (9 each) + strength, win rate, lead, uncertainty |
| Derived | 6 | Per-vector maxima and entropies |
| First differences | 37 | Move-to-move change across all features |
| Second differences | 3 | Curvature of the key scalar signals |
| Positional | 3 | Colour, normalised move index |

The differences are the part that matters. A rank signal lives in the *trajectory*: how sharply the
lead swings, when policy entropy spikes, whether uncertainty resolves or accumulates. First and
second differences put that in front of the model directly rather than leaving it to be inferred
through a recurrence.

Win rate and lead are given from Black's perspective, so both are sign-flipped on White's moves —
without that, the same position reads as opposite evidence depending on who is to play.

## Keeping the ordering: the CORAL head

Rank is ordered. Flat classification discards that: mistaking a 6D for a 7D and mistaking a 1D for a
9D cost the same under cross-entropy.

Each sequence model therefore carries **two heads** on the same pooled representation: a
classification head and a CORAL ordinal head. CORAL replaces "which of nine classes?" with eight
nested binary questions — *is the rank above 1D? above 2D?* … — and recovers class probabilities by
differencing adjacent cumulative terms:

$$
P(y = k) = \sigma(\theta_{k-1}) - \sigma(\theta_k)
$$

Because the questions are nested, an error of one rank costs one flipped decision and an error of
five costs five. The ordering is in the loss surface rather than hoped for.

The objective mixes three terms:

$$
\mathcal{L} = 0.28\,\mathcal{L}_{\text{CE}} + 0.70\,\mathcal{L}_{\text{BCE}} + 0.02\,\mathcal{L}_{\text{margin}}
$$

- $\mathcal{L}_{\text{BCE}}$ — the CORAL term over the eight cumulative targets, carrying most of the
  weight.
- $\mathcal{L}_{\text{CE}}$ — cross-entropy **weighted by inverse class frequency**, since without
  reweighting the model drifts toward the crowded middle ranks.
- $\mathcal{L}_{\text{margin}}$ — a hinge penalty pushing adjacent thresholds apart. CORAL assumes
  ordered thresholds but does not enforce it, and when they cross the differencing step yields
  negative probabilities. Two percent of the loss weight removes that failure mode.

At prediction time the heads are blended $0.7$ CORAL + $0.3$ softmax — the ordinal head is better
calibrated about *where* on the scale a player sits, the classification head sharper at committing
to one rank.

## Models

**Sequence.** A small Transformer and a BiLSTM over the 120×79 move sequence, both carrying the dual
head. Out-of-fold accuracy 38.5% and 36.5% respectively, per game.

**Tabular.** CatBoost over statistical summaries — mean, standard deviation, min, max, median,
quartiles, skewness of every feature — computed separately over **opening, midgame and endgame**.
This deliberately discards move order for robustness, while the segmentation keeps the one ordering
fact that matters most: strength is not uniform across a game's phases. 20.3% per game.

**Meta-learner.** Logistic regression stacking the three models' out-of-fold probabilities with 10
side features (policy/value/rank entropies, log game length, win-rate standard deviation, mean
absolute lead, mean uncertainty, per-phase policy entropy).

The ensemble is worth more than its parts: 42.8% per game against the best component's 38.5%,
*including* a component that scores 20.3% alone. A weak model still contributes when its errors are
uncorrelated with the strong ones, and the tabular model's are — it cannot see move order at all, so
it fails on different games. Dropping it for being weak in isolation would have cost accuracy.

## What is not measured

Two things I would want before claiming more than the leaderboard number does.

**The ordinal objective is never tested against its alternative.** CORAL was chosen so that
near-misses cost less than gross errors, but the task is scored on exact-match accuracy — which
treats being wrong by one rank and wrong by five identically, and so cannot see the property CORAL
was selected for. Mean absolute rank error, and an ablation against a plain cross-entropy head,
would show whether it earned its place. The two heads are trained together and blended at inference,
so their separate contributions were never isolated.

**The stacking figure is optimistic.** The 42.8% comes from fitting the meta-learner and scoring it
on the same rows. The three base models' contributions are genuinely out-of-fold, and a logistic
regression over 37 features has little capacity to overfit, so the bias should be small — but it is
not zero, and the number is labelled out-of-fold in the code when it is not. The 80.65% is unaffected:
that is a held-out test set scored by the organisers.
