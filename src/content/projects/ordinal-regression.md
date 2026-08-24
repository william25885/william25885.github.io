---
title: "Go Rank Prediction as Ordinal Regression"
summary: "Predicting a Go player's rank from 1D to 9D out of game records — treating rank as an ordered quantity rather than nine unrelated classes, over a stacked ensemble of sequence and tabular models."
tags: ["Ordinal Regression", "Transformer", "BiLSTM", "CatBoost", "Ensemble", "Feature Engineering", "PyTorch"]
category: research
order: 5
repo: "https://github.com/william25885/Go-Rank-Prediction"
context: "National Taiwan University"
---

## The question

Given the move sequence of a Go game, how strong is the player? Ranks run 1D through 9D — nine
categories, but not nine *arbitrary* categories. They are ordered, and the distance between them is
meaningful: mistaking a 6D for a 7D is a near-miss, mistaking a 1D for a 9D is a total failure.

Standard multi-class cross-entropy cannot express that. It assigns the same loss to both errors,
because it treats the labels as an unordered set. Discarding the ordering is discarding most of what
the label structure tells you — so the central modelling decision here is to keep it, using an
ordinal-regression objective (CORAL) rather than flat classification.

## Input representation

Each game is a sequence of moves, and each move carries a rich feature vector derived from engine
analysis: a 9-dimensional policy vector, a 9-dimensional value vector, a 9-dimensional rank
probability vector, plus scalar strength, win rate, lead and uncertainty.

That base is expanded to a **79-dimensional per-move feature vector**:

| Group | Dims | Content |
|---|---|---|
| Base | 30 | Policy, Value, RankP (9 each) + strength, win rate, lead, uncertainty |
| Derived | 6 | Per-vector maxima and entropies |
| First differences | 37 | Move-to-move change across all features |
| Second differences | 3 | Curvature of the key scalar signals |
| Positional | 3 | Colour, normalised move index |

The differences are the part that matters. A strong player is not identified by any single position
evaluation — it is identified by how the evaluation *moves* under their play. A rank signal lives in
the trajectory: how sharply the lead swings, how often the policy entropy spikes, whether
uncertainty resolves or accumulates. First and second differences make that trajectory directly
visible to the model rather than something it must infer through a recurrence.

Sequences are truncated to 120 moves.

## Keeping the ordering: the CORAL head

Each sequence model carries **two heads** on the same pooled representation: a standard
classification head (`ce_logits`) and an ordinal head (`ord_logits`) implementing CORAL.

CORAL replaces "which of nine classes?" with eight nested binary questions — *is the rank above 1D?
above 2D?* … *above 8D?* — sharing one representation, with monotonically ordered thresholds. Class
probabilities are recovered by differencing adjacent cumulative probabilities:

$$
P(y = k) = \sigma(\theta_{k-1}) - \sigma(\theta_k)
$$

with the endpoints taken as $1 - \sigma(\theta_0)$ and $\sigma(\theta_{K-2})$. Because the questions
are nested, an error of one rank costs one flipped binary decision and an error of five costs five —
the ordering is built into the loss surface rather than hoped for.

The training objective mixes three terms:

$$
\mathcal{L} = 0.28\,\mathcal{L}_{\text{CE}} + 0.70\,\mathcal{L}_{\text{BCE}} + 0.02\,\mathcal{L}_{\text{margin}}
$$

- $\mathcal{L}_{\text{BCE}}$ — the CORAL term, binary cross-entropy over the eight cumulative targets,
  carrying most of the weight.
- $\mathcal{L}_{\text{CE}}$ — ordinary cross-entropy, **weighted by inverse class frequency**. Rank
  distributions are heavily imbalanced, and without reweighting the model drifts toward the crowded
  middle ranks.
- $\mathcal{L}_{\text{margin}}$ — a hinge penalty, $\mathrm{relu}(m - (\theta_j - \theta_{j+1}))$,
  that pushes adjacent thresholds apart. CORAL assumes ordered thresholds but does not enforce it;
  when they cross, the differencing step yields negative probabilities. This term costs 2% of the
  loss weight and removes that failure mode.

At prediction time the two heads are blended, $0.7$ CORAL + $0.3$ softmax. The ordinal head is the
better-calibrated estimator of *where* on the scale a player sits; the classification head is
sharper at picking a specific rank. Neither alone was as good as the mix.

## Models

Three model families, each seeing the game differently:

**Sequence models.** A small Transformer and a BiLSTM over the 120×79 move sequence, both carrying
the dual head above. These see order and context — the Transformer attends across the whole game,
the BiLSTM reads it in both directions.

**Tabular model.** CatBoost (with `HistGradientBoostingClassifier` as fallback) over statistical
summaries of each game: mean, standard deviation, min, max, median, quartiles and skewness of every
feature, computed separately over **opening, midgame and endgame segments**. This deliberately throws
away move order to gain robustness, and the segmentation preserves the one ordering fact that
matters most — playing strength is not uniform across a game's phases.

**Meta-learner.** Logistic regression stacking the sequence and tabular predictions together with 10
side features (policy/value/rank entropies, log game length, win-rate standard deviation, mean
absolute lead, mean uncertainty, and per-phase policy entropy).

Everything is fit under 5-fold cross-validation.

## Results

| Model | Accuracy |
|---|---|
| Tabular (CatBoost / HGBT) | ~20.3% |
| BiLSTM | ~36.5% |
| Transformer | ~38.5% |
| **Stacked ensemble** | **~42.8%** |

Nine-way classification has a 11.1% random baseline, so the ensemble runs at roughly four times
chance.

The interesting number is the ensemble's margin. It beats its best component by 4.3 points while
including a component that scores only 20.3% on its own — barely twice chance. A weak model still
contributes if its errors are uncorrelated with the strong ones, and the tabular model's errors
should be: it cannot see move order at all, so it fails on entirely different games than the
sequence models do. Dropping it for being weak in isolation would have cost accuracy.

## Multi-view inference

Long games are unstable to score. A single 120-move window samples one slice of a game that may run
far longer, and which slice you take moves the prediction.

At inference, sequence models are therefore evaluated over multiple views of each game and their
predictions are pooled before reaching the meta-learner. This is variance reduction on essentially
free compute — the same weights, applied several times — and it matters most on exactly the long
games where a single view is least representative.

## What limits the result

Accuracy near 43% on a nine-way ordered problem is well above chance but far from resolved, and the
ceiling is probably in the labels rather than the model. Rank is a property of a *player*, measured
over hundreds of games, while the input here is a *single* game. A strong player playing loosely and
a weak player playing carefully produce genuinely similar records — the label is noisy with respect
to the input, and no amount of modelling recovers information the input does not contain.

The gap in the evaluation is that accuracy is the wrong metric for the model that was actually
built. A prediction wrong by one rank and a prediction wrong by five score identically under it —
which is precisely the distinction the CORAL objective was chosen to capture. Mean absolute rank
error, reported alongside an ablation against flat cross-entropy, would show whether the ordinal
head earned its place. That comparison has not been run, so the claim here is about the design, not
a measured gain.
