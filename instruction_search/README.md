# Validation-Guided Instruction Search

A minimal implementation of the instruction search of the paper (Sec. 3.2, supplement Sec. B). The captioner, the
critic and the refiner are one frozen Qwen2-VL-7B-Instruct, each steered by a natural-language instruction. The
search chooses the three instructions by the validation R1 of a proxy retriever trained on the captions they
produce, under caption-quality guards, and scores the selected pipeline once on the locked test splits.

| File | Role |
|---|---|
| `search.py` | the search: proposal, screening, evaluation, selection, locked test |
| `proposer.py` | proposals from the history of evaluated instructions (critique, then rewrite) |
| `kernel.py` | caption kernel: stage-1 sampling, critic feedback, refinement |
| `guards.py` | instruction constraints, development statistics, guards, pick rule |
| `instructions/` | the initial instructions and the ones the search selected in the paper |
| `keys/` | the 100,000 SYNTH-PEDES training images (`train_100k.txt.gz`) and the 1,000 development images (`dev_1k.txt`, 263 identities, a subset of the training images) |

## What the search does

The instructions are optimized in pipeline order: captioner (rounds `R00`..), then critic on the captions of the
best captioner (`C1`..), then refiner on the feedback of the best critic (`F1`..). Round 0 of each stage is its
initial instruction. Every later round:

1. **Propose.** The frozen model, without an image, sees every instruction of the stage evaluated so far, sorted by
   validation score and with its development statistics, plus the proposals the guards rejected and why. It lists
   three weaknesses of the best instruction and rewrites it (8 samples). The refiner's proposer also sees the best
   captioner instructions and the critic whose feedback the refiner receives. Proposals that break a hard
   constraint (one sentence, visible facts only, placeholders, length) are dropped.
2. **Screen.** Up to 3 valid proposals per attempt caption the 1,000 development images. A proposal that violates a guard is dropped
   (table below); after 3 attempts without a survivor, the captioner tries a minimal repair of proposals that failed
   only format guards. Among the survivors, the one with the most body regions per caption is kept (tie: fewest
   outputs cut). If none survives, the proposal with the fewest violations is still evaluated but is **not eligible**.
3. **Evaluate.** The kept instruction captions all 100,000 images (K = 8 captions per image), the proxy retriever is
   trained, and the utility V(π) is the best macro validation R1 over the epochs (tie: macro validation mAP).

`final` selects the eligible configuration with the highest V, writes `final/selection.json` and its SHA-256 lock,
and only then runs the test command, once per system. Test numbers are never read by the search.

Guards, relative to the statistic g⁰ of the stage's initial instruction on the development images:

| Statistic | Constraint |
|---|---|
| Claims contradicted by a visual verifier (share) ↓ | ≤ g⁰ + 0.02 (only with `verifier_cmd`) |
| Body regions per caption ↑ | ≥ g⁰ − 0.25 |
| Outputs cut at the token budget (share) ↓ | ≤ g⁰ + 0.05 |
| Single-sentence captions (share) ↑ | ≥ min(c, g⁰ − 0.02), c = 0.95 (captioner), 0.898 (critic, refiner) |
| Valid samples, captioner (share) ↑ | ≥ g⁰ − 0.05 |
| Real rewrites among the training captions, critic and refiner (share) ↑ | ≥ 0.90 |
| Images with K valid captions (share) ↑ | ≥ 0.95 |

## Running

Requirements: `torch`, `transformers` (tested with 4.57), `Pillow`, and a GPU that holds the 7B model in float16
(about 16 GB plus activations; lower `--batch` in `kernel_args` if memory is short).

**1. Proxy retriever.** The search calls your trainer and tester through two commands in the config; the paths in
braces are filled in by the search.

- `train_cmd` gets `{captions}` (JSON: image key → list of K captions), `{run_dir}`, and optionally `{keys}` and
  `{image_root}`. It must validate after every epoch and write `{run_dir}/val.jsonl`, one line per epoch:
  `{"epoch": 1, "R1": {"CUHK-PEDES": 41.2, "ICFG-PEDES": 25.0, "RSTPReid": 27.1}, "mAP": {...}}`.
  It must not load the test splits.
- `test_cmd` gets `{run_dir}`, `{epoch}` (the validation-selected epoch) and `{out}`, and writes its test metrics
  to `{out}` as JSON.

In the paper the proxy is a CLIP ViT-B/16 trained with the SDM loss of IRRA on the captioned images (batch 64,
learning rate 1e-5, at most 30 epochs, stopped after 6 epochs without a new best macro validation R1), evaluated
on the benchmarks without fine-tuning. Validation uses the official CUHK-PEDES and RSTPReid validation splits and,
because ICFG-PEDES has none, 1,000 identities held out from the ICFG-PEDES training split:

```bash
python search.py carve-icfg-val --anno /path/to/ICFG-PEDES/ICFG-PEDES.json --out icfg_val_ids.json
```

**2. Config.** Copy `config_example.json` and set `image_root`, `train_cmd` and `test_cmd`. Paths are relative to the
directory you run from. `verifier_cmd` (optional) gets `{captions}`, `{image_root}` and `{out}` and writes
`{"contradiction": <share of contradicted attribute claims>}`; without it the contradiction guard is skipped.
`test_also` names further runs to score once after the lock (e.g. the initial instructions).

**3. Search.**

```bash
cd instruction_search
python search.py captioner --config my_config.json
python search.py critic    --config my_config.json
python search.py refiner   --config my_config.json
python search.py final     --config my_config.json      # lock, then test once
```

Every step checks its outputs, so an interrupted command can simply be run again. Each run keeps its proposals,
screened candidates, guard results and validation history under `work_dir/runs/<name>/`. The caption kernel can also
be run on its own:

```bash
python kernel.py caption --instruction instructions/captioner_R03.txt --keys keys/dev_1k.txt --image_root DIR --out caps.json
python kernel.py critic  --instruction instructions/critic_C5.txt --keys keys/dev_1k.txt --image_root DIR --captions caps.json --out fb.json
python kernel.py refine  --instruction instructions/refiner_initial.txt --keys keys/dev_1k.txt --image_root DIR --feedback fb.json --out refined.json
```

## Result in the paper

The selected pipeline (paper configuration C5) uses the captioner instruction `captioner_R03.txt`, the critic
instruction `critic_C5.txt` and the initial refiner instruction. Validation-selected checkpoints, test scored once
(macro average over CUHK-PEDES, ICFG-PEDES and RSTPReid):

| Captions, instructions | Test R1 ↑ | Test mAP ↑ | Val R1 ↑ |
|---|---|---|---|
| Single-pass, initial | 33.58 | 25.82 | 30.58 |
| Single-pass, optimized | 34.82 | 26.07 | 32.12 |
| Refined, initial | 38.99 | 28.62 | 36.23 |
| Refined, optimized | **40.05** | **28.89** | **36.38** |

Each configuration was trained once; the +1.06 of optimized over initial instructions is a point estimate, while
the gain of critique-and-refine (+5.2 to +5.4) is far larger (supplement Sec. B).

## Differences from the runs of the paper

- Generation uses Hugging Face `transformers` with the decoding settings of the paper (`kernel.DECODE`); the paper's
  critic and refiner passes ran the same settings in vLLM. Samples are not bit-identical.
- Body regions are counted with a word lexicon (`guards.REGION_WORDS`); the paper used an attribute parser over the
  same vocabulary, so the values can differ slightly. The contradiction guard needs your own verifier.
- Only Qwen2-VL plays the critic and the refiner. The paper also tried LLaVA-1.6 in these roles, so run names here
  need not match the configuration names of the paper.
- Rounds run one after another. In the paper some rounds overlapped in time and were proposed before the preceding
  round had finished, and the first four captioner rounds were screened with a CLIP identity score instead of the
  region count.
