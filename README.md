# Cell-type wiring carries most of a whole-brain *Drosophila* model's accuracy

Code and results for the preprint *Cell-type wiring carries most of a whole-brain Drosophila model's accuracy: a benchmark against graded null wirings*, by Richard Jeevash (link to be added when it is posted).

## The question

Shiu et al. (2024) built a leaky integrate-and-fire model of the whole fly brain from the FlyWire connectome and compared 164 of its predictions with experiments. Agreement on one wiring does not show which of its properties the predictions need. This repository:
- reruns the model on the measured wiring and on 100 draws from each of four null-wiring families, each keeping a different part of the measured wiring's structure (N3 and N4 each keep everything N2 keeps, plus one further property; N1 keeps every connected pair but moves strengths and signs between them);
- scores every wiring on the same 164 tests by balanced accuracy.

| Family | Keeps | Removes |
| --- | --- | --- |
| N1 weight shuffle | Every connected pair, and the multiset of signed synapse counts | Which pair has which strength and sign |
| N2 degree-preserving | Each neuron's in-degree, out-degree, total output synapses and output sign | Partners and cell-type structure |
| N3 cell-type block | As N2, plus edge and synapse counts between every pair of cell types | Which neurons within the types are partners |
| N4 boundary | Edges out of sensory neurons and into motor, descending and visceral/circulatory neurons; N2's permutation inside | Interior wiring |

## Results

On FlyWire v783, 157 of the 164 items can be measured.

| Wiring | Balanced accuracy | Measured minus family mean (95% interval) | Draws reaching the measured score |
| --- | --- | --- | --- |
| Measured | 0.879 | — | — |
| N1 weight shuffle | 0.578 | +0.301 (+0.245 to +0.354) | 0 of 100 |
| N2 degree-preserving | 0.501 | +0.378 (+0.325 to +0.428) | 0 of 100 |
| N4 boundary | 0.724 | +0.155 (+0.095 to +0.215) | 0 of 100 |
| N3 cell-type block | 0.821 | +0.058 (+0.026 to +0.088) | 0 of 100 |

- **Null families:** values are family means over 100 draws.
- **Intervals:** item bootstraps (10,000 resamples within each category, with the draws held fixed).
- **The optogenetic screen alone:** cell-type block wiring matches the measured wiring (+0.006, −0.007 to +0.021).
- **v630 (Shiu et al.'s own wiring, all 164 items):** the measured wiring scores 0.870, against 0.574 for N1 and 0.502 for N2.
- **Full numbers** are in [results/v783/analysis.json](results/v783/analysis.json) and [results/v630/analysis.json](results/v630/analysis.json).

**Post hoc analyses** (not in the frozen protocol; [posthoc/](posthoc/)) qualify two of these results:
- **N1 and N2 leave the network largely silent** at the model's fixed synaptic weight. N2 draws predict a response for 0.2 of the 157 items on average, against 54 for the measured wiring.
- **The excess over N3 is narrow and moves with the analysis choices.**
  - It is 0.039 with a 70% requirement threshold and 0.064 with 90%.
  - Half of it comes from five water-requirement items that share one control run, which is silent in 48 of the 100 N3 draws. The other half comes from items chosen without the model.
  - Averaging over input seeds raises it slightly.
  - FlyWire's cell types are fine (half hold two cells), so N3 mainly exchanges partners between members of a type.
  - Removing the repeated pairs and self-loops that N3's permutation creates (N3s, specified and frozen before it ran) leaves it essentially unchanged: +0.061.

The paper's reading: fine cell-type wiring carries most of the model's accuracy on these tests; exact partners may add a little, but the evidence is narrow.

## Layout

```
code/      the simulator, wirings, null families, prediction rules, scoring and analysis
scripts/   downloads, the v783 builder, the target rebuild, checks and the full-run script
data/      sources.json (every input, pinned and hashed) and derived/targets.csv (the 164 items)
results/   runs.csv (all 602 published runs), analysis.json for each wiring version, and null_checks_draw0.json
           (what each family kept, for draw 0)
posthoc/   the post hoc analyses and seed runs, with their outputs (posthoc.json, seeds.json)
docs/      DESIGN.md: how the paper's methods map to the code
```

## Requirements

- Python 3.12 with the packages in `requirements.txt`: the versions used for the published runs.
- About 4 GB of memory and 10 CPU threads per simulation run, so four runs at a time need about 16 GB and 40 threads.
- About 150 MB of downloads.

## Reproducing

**Set up and download the inputs:**

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py      # Shiu et al.'s v630 model files, about 90 MB
python scripts/build_v783.py      # the FlyWire v783 wiring, about 61 MB downloaded
```

**Quick check, a few minutes:**
1. Rebuild five wirings and compare them with the published hashes.
2. Re-run one complete simulation (N1, draw 0) and compare its output.
3. Score it.

```bash
python scripts/verify.py wirings --version v783 --runs M N1_00 N2_00 N3_00 N4_00
(cd code && NUMBA_NUM_THREADS=10 python run_wiring.py --version v783 --wiring N1 --draw 0)
python scripts/verify.py rates
python code/score.py results/v783/N1_00
```

**The full benchmark:**

```bash
scripts/run_all.sh v783 100 4     # 401 runs, 4 at a time
scripts/run_all.sh v630 100 4     # 201 runs
```

- Each script ends by running `code/analyse.py`, which writes `results/<version>/analysis.json`; the published file is first kept as `analysis.published.json`. If any run fails, the analysis is not run; run the script again, and it skips finished runs.
- For scale: the published runs used cloud machines with 96 cores, running 10 threads per run. The median run time on v783 was 72 s for N1, 277 s for N2, 405 s for N3 and 364 s for N4. On v630 it was 80 s for N1 and 292 s for N2.

**The post hoc analyses** need the per-run outputs: run the full benchmark, or download them from the release (see [Release files](#release-files)) and unpack them at the top of the repository.

```bash
python posthoc/analyses.py        # writes posthoc/posthoc.json
python posthoc/cell_types.py      # writes posthoc/cell_types.json (needs only the v783 wiring)
```

The seed runs are 19 full simulations: the measured v783 wiring with seed sets 1–9, and N3 draws 0–9 with seed set 1. Run them from `code/`, then summarise:

```bash
cd code
for k in 1 2 3 4 5 6 7 8 9; do NUMBA_NUM_THREADS=10 python ../posthoc/seeds.py --version v783 --wiring M --seed-set $k --out ../posthoc/seeds/v783/M_seed$k; done
for d in 0 1 2 3 4 5 6 7 8 9; do NUMBA_NUM_THREADS=10 python ../posthoc/seeds.py --version v783 --wiring N3 --draw $d --seed-set 1 --out ../posthoc/seeds/v783/N3_0${d}_seed1; done
cd .. && python posthoc/seeds_summary.py   # writes posthoc/seeds.json
```

N3 without repeated pairs (N3s) is 100 full simulations, each building its wiring first, which takes a few minutes:

```bash
cd code
for d in $(seq 0 99); do NUMBA_NUM_THREADS=10 python ../posthoc/n3_simple.py --run $d --out ../posthoc/n3s/v783/N3s_$(printf %02d $d); done
cd .. && python posthoc/n3s_analysis.py    # writes posthoc/n3s.json
```

**Rebuilding the 164 targets** from Shiu et al.'s supplementary spreadsheet is optional, because `data/derived/targets.csv` is included:

```bash
python scripts/fetch_data.py --optional
python scripts/export_sheets.py && python scripts/build_targets.py
python scripts/recount_shiu_supplement.py     # recounts their published accuracy: 150 of 164 (91%), 49 of 58 without the screen (84%)
```

## Reproducibility notes

- **The code is the code that produced the published runs.** `code/` changes only file paths and comments.
- **What was tested** on a separate copy of this repository:
  - the rebuilt v783 file is identical, array by array, to the one the published runs used;
  - eleven rebuilt wirings match their published hashes;
  - a complete re-run of v783 N1 draw 0 reproduced its published output exactly;
  - `code/analyse.py` run on the published v783 runs gives a byte-identical `analysis.json`;
  - `data/derived/targets.csv`, rebuilt from the spreadsheet, is identical to the included file;
  - the null-family checks are identical to `results/null_checks_draw0.json` (rerun with `python scripts/null_checks.py --out /tmp/checks.json` and compare).
- **Thread count:** each simulated trial seeds its own random stream, so results should not depend on the number of threads.
  - The published runs used 10 threads (`NUMBA_NUM_THREADS=10`).
  - Earlier checks found identical outputs on x86-64 and arm64 cloud machines.
- **Two simulator modes:** `scripts/check_dense_vs_active.py` confirms that the fast mode used for every run gives the same spike counts as the reference mode that updates every neuron at every step.
- **Self-loops and repeated pairs** created by the null permutations are kept, as in the paper.
- **The simulator** is a NumPy and Numba reimplementation of Shiu et al.'s Brian2 model.
  - At high input rates its firing rates run above the published ones: MN9 fires at 80.5 Hz against 65.7 Hz at 100 Hz sugar input.
  - See [docs/DESIGN.md](docs/DESIGN.md) and the paper.
- **Full per-run outputs** (195 MB unpacked) are not in the repository's files; they are attached to the release (see below). `results/runs.csv` gives each run's output hash and scores.

## Release files

The [v1.0 release](https://github.com/Jeevash23/connectomics-fly-whole-brain-nulls/releases/tag/v1.0) has three attachments:

| File | Size | Contents |
| --- | --- | --- |
| `per-run-outputs.tar.gz` | 9.3 MB | `rates.json`, `run.json` and `scores.json` of all 602 published runs (unpacks into `results/v783/` and `results/v630/`), and of the 119 post hoc runs (unpacks into `posthoc/seeds/` and `posthoc/n3s/`) |
| `frozen-protocols.tar.gz` | 10 kB | The benchmark's protocol and its freeze record (hashes of the protocol, code and inputs at the freeze and at each amendment), and the N3s protocol with its freeze record, byte for byte, with a short README |
| `SHA256SUMS` | | Checksums of the two archives |

```bash
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz      # at the top of the repository
python scripts/verify.py rates       # every published rates.json against results/runs.csv
```

With the outputs unpacked, `code/analyse.py v783`, `code/analyse.py v630` and the post hoc scripts above regenerate `results/*/analysis.json` and the files in `posthoc/` byte for byte (checked on 20 October 2026; `posthoc/cell_types.py` reads the v783 file that `scripts/build_v783.py` builds).

## Data and licences

- **Code:** MIT licence ([LICENSE](LICENSE)).
- **Not redistributed here.** The scripts download these from their original sources, at pinned versions:
  - the FlyWire v630 and v783 connectivity tables as distributed with Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model; the code there is MIT, and the FlyWire data are for non-commercial use under CC BY-NC 4.0);
  - FlyWire v783 cell types and super classes, from the connectome data curation by Alexander Bates for the SJCABS connectome tutorial (github.com/sjcabs/fly_connectome_data_tutorial), under the same FlyWire terms.
- **The targets:** `data/derived/targets.csv` is rebuilt from the supplementary tables of Shiu et al. 2024 (*Nature*, open access, CC BY).
- **Model outputs:** the outputs in `results/` are computed from FlyWire data, so FlyWire's non-commercial terms apply to them.
- **Citing the data:** please cite the FlyWire papers when you use these data:
  - Dorkenwald et al. 2024, *Nature* 634:124–138;
  - Schlegel et al. 2024, *Nature* 634:139–152;
  - Eckstein et al. 2024, *Cell* 187:2574–2594, for transmitter predictions;
  - Shiu et al. 2024, *Nature* 634:210–219, for the model.

## Citation

Please cite the preprint (the link will be added here when it is posted) and, for the code, this repository; [CITATION.cff](CITATION.cff) has the details.
