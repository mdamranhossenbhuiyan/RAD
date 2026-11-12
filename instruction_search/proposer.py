#!/usr/bin/env python3
"""Instruction proposals from the history of evaluated instructions (step (i) of the search, Sec. 3.2).

The proposer is the frozen Qwen2-VL-7B-Instruct, used without an image. A single OPRO-style request ("here are the
scored instructions, write a better one") made the 7B model copy the best instruction, so each proposal is made in
two steps, as in ProTeGi ("textual gradients"):
  1. critique: the model sees every evaluated instruction of the stage with its validation score and development
     statistics, and the instructions rejected by the guards with the reason, and lists weaknesses of the best one;
  2. rewrite: it rewrites the best instruction to fix them (assistant turn pre-filled with <INS>).
For the refiner, both steps also see the earlier stages (the best captioner instructions and the critic whose
feedback the refiner receives). Proposals are checked with guards.check_instruction and de-duplicated. The captioner
stage has a third mode, repair: a minimal rewrite of a screened instruction that failed only format guards.

    python proposer.py --stage captioner --history history.json --out proposals.json --n 8 --seed 1
"""
import argparse
import json
import re

from guards import check_instruction

# ---------------------------------------------------------------------------------------------------------------
# Captioner (stage 1)
CAP_CONTEXT = """An image-captioning model receives one photo of a pedestrian together with an instruction and writes one caption about the person. The captions are used to train a person re-identification model that must find the right person from a text description. Captions are better when they are accurate and mention the visible details that distinguish this person from other people. The captioning model writes 8 different captions per photo and at most 40 tokens per caption; longer captions are cut off.

Every instruction must:
1. ask for exactly one sentence;
2. ask only for facts that are clearly visible, with no speculation or guessing;
3. forbid background or scene description;
4. be plain text of at most 120 words, without example captions.

Instructions tested so far, from worst to best. Score = retrieval accuracy (R1, %) of the re-identification model averaged over three validation sets; higher is better. Caption statistics on {n_dev} development photos: {stat_doc}.

{history}{rejected}"""

CAP_CRITIQUE = """{context}
Look at the best instruction (the last one above) and its statistics. List three specific weaknesses of that instruction that could stop its captions from distinguishing people, or that make the captions inaccurate or too long. Be concrete and brief: one line per weakness."""

CAP_REWRITE = """An image-captioning model receives one photo of a pedestrian together with an instruction and writes one caption that is used to train a person re-identification model.

Current instruction:
<INS>
{prompt}
</INS>

Weaknesses found in the current instruction:
{critique}

Rewrite the instruction so that it fixes these weaknesses. Make real changes, not just rewording. The new instruction must still ask for exactly one sentence, only clearly visible facts with no speculation or guessing, and no background. Keep it short, and do not include an example caption. Write only the new instruction inside <INS> and </INS>."""

CAP_REPAIR = """An image-captioning model receives one photo of a pedestrian together with an instruction and writes a caption about the person. It writes 8 captions per photo and stops after 40 tokens.

Instruction:
<INS>
{prompt}
</INS>

When this instruction was tested on 1,000 photos, its captions had these problems:
{fails}

Every caption must be exactly ONE sentence. Many of these captions went on with a second sentence about the person or the scene, or ran past the length limit.

Rewrite the instruction with the smallest change that fixes these problems: state the one-sentence rule clearly and strictly (a single sentence that ends with one period, never a second sentence), and ask for a caption short enough to finish well within the limit. Keep the rest of the instruction. Do not include an example caption. Write only the new instruction inside <INS> and </INS>."""

CAP_STATS = [  # (key, definition shown to the proposer, format in a history entry); only measured statistics are shown
    ("words", "words = words per caption", "words {:.1f}"),
    ("regions", "regions = body regions mentioned per caption (upper body, lower body, footwear, carried item, head, "
                "accessory, gender)", "regions {:.2f}"),
    ("contradiction", "errors = fraction of claims a visual checker found wrong (lower is better)", "errors {:.3f}"),
    ("idt_ap", "distinctiveness = how well a frozen CLIP model finds other photos of the same person from the caption "
               "(higher is better)", "distinctiveness {:.4f}"),
    ("cut", "cut = fraction of captions cut off at the length limit", "cut {:.3f}"),
]
CAP_ECHO_RE = re.compile(r"(?i)(\bscore\b|new instruction|instructions? above|current instruction|weakness|<\s*/?\s*INS\s*>|"
                         r"captioning model receives|120 words|example captions?|plain text)")

# ---------------------------------------------------------------------------------------------------------------
# Critic and refiner (stages 2 and 3); placeholders are shown to the proposer as [CAPTION] / [FEEDBACK]
PIPE = ("A captioning pipeline describes photos of pedestrians for training a text-based person re-identification model "
        "(the model must find the right person from a text description). Step 1 writes 8 captions per photo. Step 2, the "
        "CRITIC, sees the photo and one caption and writes short feedback about that caption. Step 3, the REFINER, sees "
        "the photo, the caption and the feedback and rewrites the caption as one sentence of at most 40 tokens. The "
        "refined captions are the training data. Captions are better when they are accurate and mention the visible "
        "details that distinguish this person from other people.")
TASK = {
    "critic": ("You are improving the instruction given to the CRITIC. [CAPTION] marks where the caption is inserted. "
               "The feedback is cut off after 96 tokens, so it must be short. The instruction must keep [CAPTION], ask "
               "for feedback based only on what is visible in the photo, and must not ask the critic to write a caption."),
    "refiner": ("You are improving the instruction given to the REFINER. [CAPTION] and [FEEDBACK] mark where the caption "
                "and the critic's feedback are inserted. The refined caption is cut off after 40 tokens. The instruction "
                "must keep [CAPTION] and [FEEDBACK], ask for exactly one sentence, and ask only for clearly visible facts."),
}
TPL_CONTEXT = """{pipe}

{task}

{upstream}Instructions tested so far, from worst to best. Score = retrieval accuracy (R1, %) of the re-identification model trained on the refined captions, averaged over three validation sets; higher is better. Statistics of the refined captions on {n_dev} development photos: {stat_doc}.

{history}{rejected}"""
TPL_CRITIQUE = """{context}
Look at the best instruction (the last one above) and its statistics. List three specific weaknesses of that instruction that could stop the refined captions from distinguishing people, or that make them inaccurate or too long.{extra} Be concrete and brief: one line per weakness."""
TPL_CRITIQUE_EXTRA = {"critic": "", "refiner": (" Take the earlier steps into account: what the best step-1 instruction asks "
                                                "the captions to contain, and what kind of feedback the critic marked USED gives.")}
TPL_REWRITE = """{pipe}

{task}

{upstream}Current instruction:
<INS>
{prompt}
</INS>

Weaknesses found in the current instruction:
{critique}

Rewrite the instruction so that it fixes these weaknesses. Make real changes, not just rewording.{extra} Keep the markers exactly as written ({markers}). Keep it short and do not include an example. Write only the new instruction inside <INS> and </INS>."""
TPL_REWRITE_EXTRA = {"critic": "", "refiner": (" Use what the earlier steps show: keep the details the best step-1 instruction "
                                               "asks for, and make the refiner act on the kind of feedback that critic gives.")}
TPL_STATS = [
    ("words", "words = words per caption", "words {:.1f}"),
    ("regions", "regions = body regions mentioned per caption", "regions {:.2f}"),
    ("contradiction", "errors = fraction of claims a visual checker found wrong (lower is better)", "errors {:.3f}"),
    ("cut", "cut = fraction of refined captions cut off at the length limit", "cut {:.3f}"),
    ("refined_share", "refined = fraction of training captions that are real refinements", "refined {:.2f}"),
]
TPL_ECHO_RE = re.compile(r"(?i)(\bscore\b|instructions? above|current instruction|weakness|<\s*/?\s*INS\s*>|re-identification model trained)")
UPSTREAM_ECHO_RE = re.compile(r"(?i:<\s*/?\s*(STEP1|CRITIC)\s*>|step-1 instruction|critic instruction)|\bUSED\b")


def to_view(t):
    return t.replace("{candidate}", "[CAPTION]").replace("{feedback}", "[FEEDBACK]")


def from_view(t, stage):
    t = re.sub(r"\[\s*FEEDBACK\s*\]", "{feedback}", re.sub(r"\[\s*CAPTION\s*\]", "{candidate}", t))
    if "{candidate}" not in t:                                  # a dropped placeholder is appended, and recorded
        t = t.rstrip() + "\n\nCandidate caption:\n{candidate}\n"
    if stage == "refiner" and "{feedback}" not in t:
        t = t.rstrip() + "\n\nFeedback:\n{feedback}\n\nRefined caption:"
    return t


def stat_doc(table, dev):
    return "; ".join(d for k, d, _ in table if k in dev)


def stat_line(table, dev):
    return "; ".join(f.format(dev[k]) for k, _, f in table if k in dev)


def entry(e, stage):
    s = f"Score: {e['score']:.2f}"
    if stage == "captioner" and e.get("val"):
        s += " (" + ", ".join(f"{b} {r:.1f}" for b, r in e["val"].items()) + ")"
    line = stat_line(CAP_STATS if stage == "captioner" else TPL_STATS, e["dev"])
    text = e["instruction"].strip() if stage == "captioner" else to_view(e["instruction"]).strip()
    return f"<INS>\n{text}\n</INS>\n{s}" + (f"; {line}" if line else "") + "\n"


def rejected_block(rejected, stage):
    if not rejected:
        return ""
    show = (lambda t: t.strip()) if stage == "captioner" else (lambda t: to_view(t).strip())
    return ("\nInstructions rejected before training, with the problem found:\n\n" +
            "\n".join(f"<INS>\n{show(r['instruction'])}\n</INS>\nRejected: {r['reason']}\n" for r in rejected[-4:]))


def upstream_blocks(up):
    """Earlier stages shown to the refiner proposer: (long block for the critique step, short block for the rewrite)."""
    L = [f"What the earlier steps showed. Step-1 instructions (the captioner's prompt), the {len(up['captioner'])} best tested, "
         "from worst to best. Score = retrieval accuracy (R1, %, averaged over three validation sets) of the "
         "re-identification model trained on that instruction's step-1 captions alone, without critic or refiner; the "
         "statistics are those defined below, measured on the step-1 captions. The refiner rewrites the captions of the "
         "last (best) one.", ""]
    for e in up["captioner"]:
        s = f"Score: {e['score']:.2f} (" + ", ".join(f"{b} {r:.1f}" for b, r in e["val"].items()) + ")"
        line = stat_line([x for x in TPL_STATS if x[0] != "refined_share"], e["dev"])
        L += ["<STEP1>", e["instruction"].strip(), "</STEP1>", s + (f"; {line}" if line else ""), ""]
    L += ["CRITIC instructions tested, from worst to best. Score = retrieval accuracy of the model trained on captions "
          "that the original refiner instruction rewrote from that critic's feedback; words, regions and errors are those "
          "of the refined captions. The refiner receives the feedback of the critic marked USED.", ""]
    for e in up["critic"]:
        line = stat_line(TPL_STATS[:3], e["dev"])
        s = f"Score: {e['score']:.2f}; critic model Qwen2-VL-7B (same model as step 1)" + (f"; {line}" if line else "")
        L += ["<CRITIC>", to_view(e["instruction"]).strip(), "</CRITIC>", s + ("; USED" if e["name"] == up["used"] else ""), ""]
    used = next(e for e in up["critic"] if e["name"] == up["used"])
    short = ("The step-1 instruction that gave the best captions (the refiner rewrites its captions):\n<STEP1>\n" +
             up["captioner"][-1]["instruction"].strip() + "\n</STEP1>\n\nThe critic instruction whose feedback the refiner "
             "receives:\n<CRITIC>\n" + to_view(used["instruction"]).strip() + "\n</CRITIC>\n\n")
    return "\n".join(L) + "\n", short


def parse(text):
    t = re.split(r"<\s*/\s*INS\s*>", text, flags=re.I)[0]
    t = re.split(r"<\s*INS\s*>", t, flags=re.I)[0]
    return t.strip().strip('"').strip() or None


class Proposer:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()

    def sample(self, text, prefill, n, temperature, max_new_tokens):
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        chat = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
        enc = self.proc(text=[chat], return_tensors="pt").to("cuda")
        with self.torch.no_grad():   # top_k must be passed: Qwen2-VL's generation_config pins top_k=1 (greedy)
            out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95, top_k=50,
                                      max_new_tokens=max_new_tokens, num_return_sequences=n,
                                      pad_token_id=self.proc.tokenizer.eos_token_id)
        return self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def propose(P, stage, H, n, seed, repair=False):
    """H: {history: [{name, instruction, score, val, dev}], rejected: [{instruction, reason}], previous: [...],
    n_dev, upstream (refiner only), repair_targets (repair only)}. Returns the proposals with their violations."""
    P.torch.manual_seed(seed)
    previous, props = list(H["previous"]), []

    def keep(raw, **info):
        p = parse(raw)
        if p is not None and stage != "captioner":
            p = from_view(p, stage)
        viol = ["empty proposer output"] if p is None else check_instruction(stage, p, previous)
        echo = CAP_ECHO_RE if stage == "captioner" else TPL_ECHO_RE
        if p and (echo.search(p) or (stage == "refiner" and UPSTREAM_ECHO_RE.search(p))):
            viol.append("copies text of the optimizer's own task description")
        props.append(dict(instruction=p, violations=viol, raw=raw, **info))
        if p and not viol:
            previous.append(p)

    if repair:
        for i in range(n):
            t = H["repair_targets"][i % len(H["repair_targets"])]
            text = CAP_REPAIR.format(prompt=t["instruction"].strip(), fails="\n".join(f"- {f}" for f in t["fails"]))
            keep(P.sample(text, "<INS>\n", 1, 0.7, 300)[0], repaired_from=t["instruction"])
        return props
    hist = sorted(H["history"], key=lambda e: e["score"])
    best = hist[-1]
    rej = rejected_block(H.get("rejected", []), stage)
    if stage == "captioner":
        ctx = CAP_CONTEXT.format(n_dev=f"{H['n_dev']:,}", stat_doc=stat_doc(CAP_STATS, best["dev"]),
                                 history="\n".join(entry(e, stage) for e in hist), rejected=rej)
        critique_text = CAP_CRITIQUE.format(context=ctx)
        rewrite = lambda c: CAP_REWRITE.format(prompt=best["instruction"].strip(), critique=c)  # noqa: E731
        max_rewrite = 300
    else:
        up, up_short = upstream_blocks(H["upstream"]) if stage == "refiner" else ("", "")
        ctx = TPL_CONTEXT.format(pipe=PIPE, task=TASK[stage], upstream=up, n_dev=f"{H['n_dev']:,}",
                                 stat_doc=stat_doc(TPL_STATS, best["dev"]),
                                 history="\n".join(entry(e, stage) for e in hist), rejected=rej)
        critique_text = TPL_CRITIQUE.format(context=ctx, extra=TPL_CRITIQUE_EXTRA[stage])
        markers = "[CAPTION]" if stage == "critic" else "[CAPTION] and [FEEDBACK]"
        rewrite = (lambda c: TPL_REWRITE.format(pipe=PIPE, task=TASK[stage], upstream=up_short, critique=c, markers=markers,
                                                prompt=to_view(best["instruction"]).strip(), extra=TPL_REWRITE_EXTRA[stage]))
        max_rewrite = 350
    for crit in P.sample(critique_text, "", n, 1.0, 200):
        keep(P.sample(rewrite(crit.strip()), "<INS>\n", 1, 0.7, max_rewrite)[0], critique=crit.strip(), edited_from=best["name"])
    return props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["captioner", "critic", "refiner"])
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--model_id", default="Qwen/Qwen2-VL-7B-Instruct")
    a = ap.parse_args()
    props = propose(Proposer(a.model_id), a.stage, json.load(open(a.history)), a.n, a.seed, a.repair)
    with open(a.out, "w") as f:
        json.dump(dict(stage=a.stage, seed=a.seed, repair=a.repair, proposals=props), f, indent=2)


if __name__ == "__main__":
    main()
