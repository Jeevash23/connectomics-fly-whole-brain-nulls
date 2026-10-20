# Design: from the paper's methods to the code

The study followed a protocol frozen on 16 October 2026, before any null wiring was run; file digests were recorded at the freeze. Two dated amendments followed, described at the end. The paper's Methods and Appendix A are the full description; this file says where each part lives in the code.

## Targets

- **What they are:** the 164 tested predictions of Shiu et al. 2024, in `data/derived/targets.csv`. Each row has a category, an experimental outcome (`truth`: 57 yes, 107 no) and Shiu et al.'s published v630 prediction (`shiu_pred`).
- **How they are built:** `scripts/build_targets.py` builds them from the CSV export of the supplementary spreadsheet (`scripts/export_sheets.py`). Every category recounts to their Supplementary Table 10.

## The model

- **The simulator:** `code/lif.py`, a NumPy and Numba reimplementation of Shiu et al.'s Brian2 model (`model.py`, repository commit 91bdd1e7). It uses:
  - the same equations and constants;
  - exact integration at 0.1 ms;
  - a 1.8 ms synaptic delay;
  - Poisson input adding 250 × w_syn to the membrane potential, with no refractory period for driven neurons;
  - silencing that removes all of a neuron's output.
- **Trials:** each experiment runs 30 trials of 1,000 ms.
- **Common random numbers:** an experiment uses the same Poisson seed on every wiring, derived from its name (`experiments.seed_for`). Trial *t* seeds its own stream with seed + *t*.
- **Known deviation:** at high input rates the model fires faster than Shiu et al. report. On v630 with right sugar input, MN9 fires at 80.5 Hz against their 65.7 Hz at 100 Hz input, and at 22.4 Hz against 19.4 Hz at 50 Hz.
  - The sugar receptor neurons themselves match.
  - The model was kept as specified, and the effect on item predictions is reported in the paper.
- **Input sets and readout neurons:** `code/inputs.py`, from Shiu et al.'s repository files. The screen cell types are read from `sez_neurons.pickle` with an unpickler that allows no classes.

## Wirings

- **The measured wirings:** `code/wiring.py`.
  - **v630:** Shiu et al.'s own model files (127,400 neurons, 14,687,178 connections).
  - **v783:** FlyWire v783 (144,837 cells, 15,091,902 connections), built by `scripts/build_v783.py`. Every cell has a cell-type label and a super class; 6,845 cells without a fine type, mostly putative glia, carry one of 30 coarse labels. Signs are Shiu et al.'s excitatory or inhibitory assignment.
- **The null families:** `code/nulls.py` (`make`).
  - Every family keeps the neuron set and the number of connections.
  - Each draw is seeded by family, version and draw number: `numpy.random.default_rng([family, version, draw])`.
  - N3 and N4 need cell types and super classes, so they run on v783 only.
  - `nulls.check` reports what each family kept, the share of measured pairs retained, and the self-loops and repeated pairs created; `results/null_checks_draw0.json` holds draw 0 of every family.
- **Item neurons on v783** resolve by unchanged FlyWire root ID.
  - The second MN9 is the other neuron labelled MN9.
  - Items whose neurons are absent from v783 are not measurable and are left out of v783 scores: 157 of the 164 items remain.

## Prediction rules

These are in `code/experiments.py`. Where Shiu et al. state a rule, theirs is used; the rules for 1D, 3BC, 4A and 5B were chosen in the protocol.

| Category | Experiment | Prediction |
| --- | --- | --- |
| 1C | Right or left sugar receptor neurons at 100 Hz | Contralateral MN9 rate above ipsilateral |
| 1D | Right sugar at 200 Hz | Neuron rate above 0 Hz |
| 1F | Right sugar at 50 Hz, neuron silenced | Required if MN9 ≤ 0.8 × control |
| 2A | Cell type at 50 Hz | MN9 above 0 Hz |
| 3BC | Right sugar at 100 Hz or 200 Hz, with bitter or Ir94e at 200 Hz | Inhibits if MN9 ≤ 0.1 × MN9 with the same sugar alone |
| 4A | Water at 260 Hz | Neuron rate above 0 Hz |
| 4C | Water at 160 Hz, neuron silenced | Required if MN9 ≤ 0.8 × control |
| 5B | All Johnston's-organ neurons at 220 Hz | Neuron rate above 0 Hz |
| 5D | All Johnston's-organ neurons at 160 Hz, aBN1 or aBN2 silenced | Required if aDN1 ≤ 0.8 × control |
| 5I | JO-CE or JO-F at 150 Hz | aDN1 above 0 Hz |

If a control rate is 0 Hz, "required" and "inhibits" cannot be shown, and the prediction is "no".

## Endpoints and analysis

- **Per run:** `code/score.py` gives balanced accuracy, the mean of the hit rate on "yes" items and the correct-rejection rate on "no" items, for:
  - all measurable items;
  - the optogenetic screen (category 2A);
  - the other items.
- **Per version:** `code/analyse.py` compares the measured wiring with each family's mean. The 95% interval comes from 10,000 item bootstraps, resampling items within each category, with the draws held fixed (seed 20261016). It also gives:
  - each family's range;
  - the number of draws at or above the measured wiring;
  - for every item, the share of draws that predict it correctly.
- **Primary endpoint:** pooled balanced accuracy on v783, measured minus each family's mean.
- **Interpretation:** the protocol fixed a practical margin of 0.05 balanced accuracy.
  - "Clearly above" a family means the interval excludes 0 and the difference is at least 0.05.

## Amendments

- **Amendment 1, 16 October 2026** (after the measured v630 run was scored, before any null run).
  - One screen cell type is spelled "asteroid" in Supplementary Table 3 but "Asteroid" in `sez_neurons.pickle`, so screen items are matched case-insensitively.
  - No experiment, rule or simulation changed.
- **Amendment 2, 18 October 2026** (after the 10 draws per family on v783 and 5 on v630 had finished, before any analysis was run or read).
  - The draws rose from 10 per family on v783 and 5 on v630 to 100 per family.
  - The finished runs became draws 0–9 and 0–4 of the 100. New draws used the same code, seed rule, 30 trials and 10 threads.
  - Before any new draw, the rebuilt v783 wiring and draw 0 of every family were checked against the earlier runs, and one finished run was re-run with an identical output.
- **Erratum.** The frozen protocol text gives 58 "yes" outcomes; the frozen target table, which every score uses, has 57. No result is affected.

## Post hoc analyses (not in the protocol)

These were added on 19 October 2026, after an independent reading of an earlier draft. They are labelled post hoc in the paper (section 3.5 and Appendix B).

- **`posthoc/analyses.py` → `posthoc/posthoc.json`** reads the saved firing rates of every published run and changes nothing. It gives:
  - "yes" predictions and active neurons per family;
  - the N3 contrast split item by item;
  - requirement items split by whether their control run fires;
  - a bootstrap that resamples items sharing a stimulus or control run together;
  - the 80% requirement threshold moved to 70% and 90%.
- **`posthoc/seeds.py`** reruns a wiring with a shifted set of Poisson input seeds, and nothing else changes. `posthoc/seeds_summary.py` → `posthoc/seeds.json` compares:
  - the measured v783 wiring under nine further seed sets;
  - N3 draws 0–9 under one further seed set.
- **`posthoc/n3_simple.py`** builds N3s: each N3 draw with every repeated pair and self-loop removed by swapping postsynaptic ends within a block, keeping everything N3 keeps. It ran draws 0–99 with the published input seeds. `posthoc/n3s_analysis.py` → `posthoc/n3s.json` compares N3s with N3 and with the measured wiring.
  - The protocol, the analysis and three predictions were frozen with file hashes before the runs. The predictions were that N3s scores at least as well as N3, that its water control is silent less often, and that the measured wiring exceeds it by less.
  - All three failed. N3s averages 0.818 against N3's 0.821 on the same draws, and the measured wiring exceeds it by +0.061.
