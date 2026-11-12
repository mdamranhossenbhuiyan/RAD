#!/usr/bin/env python3
"""The caption kernel q_pi(c | x) of Sec. 3.1: stage-1 sampling, critic feedback and refinement with one frozen
Qwen2-VL-7B-Instruct, steered by an instruction file. Decoding settings are those of the paper's runs.

  caption : K captions per image. Rounds of K+2 samples (at most 5 rounds); the first K valid, distinct samples are kept.
  critic  : one feedback per stage-1 caption (template with the placeholder {candidate}).
  refine  : one rewrite per caption (template with {candidate} and {feedback}). Invalid rewrites are replaced by
            stage-1 captions: refined captions first, then the stage-1 captions, de-duplicated, first K kept.

Every record counts the generations that ended without an end-of-sequence token ("cut" at the token budget).

    python kernel.py caption --instruction I.txt --keys keys.txt --image_root DIR --out captions.json
    python kernel.py critic  --instruction T.txt --keys keys.txt --image_root DIR --captions captions.json --out feedback.json
    python kernel.py refine  --instruction T.txt --keys keys.txt --image_root DIR --feedback feedback.json --out refined.json
"""
import argparse
import gzip
import json
import os
import os.path as op
import re

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MAX_IMAGE_SIDE = 672
K = 8
POOL = 12          # refine: size of the merged list before the first K are kept
DECODE = {
    "caption": dict(max_new_tokens=40, temperature=0.90, top_p=0.92, top_k=50, repetition_penalty=1.20, no_repeat_ngram_size=4),
    "critic": dict(max_new_tokens=96, temperature=0.40, top_p=0.90, top_k=50, repetition_penalty=1.05, no_repeat_ngram_size=4),
    "refine": dict(max_new_tokens=40, temperature=0.60, top_p=0.90, top_k=50, repetition_penalty=1.08, no_repeat_ngram_size=4),
}
FALLBACK = "Person wearing visible clothing with distinguishable colors and items."

# ---------------------------------------------------------------------------------------------------------------
# Cleaning and validity of a caption (same rules as the agentic captioning scripts of this repository)
BAD_PATTERNS = [
    r"\bif\s*\(\s*1\s*\)", r"when you look up something on the internet", r"a photo showing a scene with people or objects",
    r"\bscene with people or objects\b", r"\bthis image shows\b", r"\bthe image shows\b", r"\bthe image depicts\b",
    r"\bblurred lights\b", r"\bvenue\b", r"\bsubway station\b", r"\bmall\b", r"\bon the internet\b", r"\bpeople seated behind\b",
]
PERSON_TOKENS = ["man", "woman", "person", "individual", "figure", "girl", "boy", "child", "shirt", "t-shirt", "jacket",
                 "hoodie", "coat", "sweater", "top", "jersey", "blazer", "pants", "jeans", "trousers", "shorts", "skirt",
                 "leggings", "shoes", "sneakers", "boots", "sandals", "bag", "backpack", "purse", "hat", "cap", "scarf",
                 "front", "back", "side", "profile", "rear"]


def clean(s):
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.strip().replace("\n", " ").replace("\t", " ")).strip().strip("“”\"'")
    s = re.sub(r"\b(system|assistant|user)\b\s*:", "", s, flags=re.I).strip()
    s = re.sub(r"^\W+", "", s)
    for p in (",", ".", "!", "?"):
        s = re.sub(r"\s+" + re.escape(p), p, s)
    if s and s[-1] not in ".!?":
        s += "."
    return s


def is_valid(s):
    s0 = s.lower().strip()
    if not s0 or any(re.search(p, s0) for p in BAD_PATTERNS):
        return False
    return 5 <= len(s0.split()) <= 40 and any(t in s0 for t in PERSON_TOKENS)


def dedup_key(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower().strip())).strip()


def unique_valid(xs, limit):
    out, seen = [], set()
    for x in xs:
        x = clean(x)
        if x and is_valid(x) and dedup_key(x) not in seen:
            seen.add(dedup_key(x))
            out.append(x)
            if len(out) >= limit:
                break
    return out


def read_keys(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return [l.strip() for l in f if l.strip()]


# ---------------------------------------------------------------------------------------------------------------
class Kernel:
    def __init__(self, model_id=MODEL_ID, device="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    def image(self, path):
        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if max(w, h) > MAX_IMAGE_SIDE:
            s = MAX_IMAGE_SIDE / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
        return im

    def chat(self, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, images, texts, n, kind, seed):
        """n samples for each (image, text) pair -> per pair a list of (text, ended)."""
        torch = self.torch
        torch.manual_seed(seed)
        enc = self.proc(text=[self.chat(t) for t in texts], images=images, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=True, num_return_sequences=n,
                                      pad_token_id=self.proc.tokenizer.pad_token_id, **DECODE[kind])
        new = out[:, enc["input_ids"].shape[1]:]
        ended = [bool(set(row.tolist()) & self.eos) for row in new]
        txt = self.proc.batch_decode(new, skip_special_tokens=True)
        return [list(zip(txt[i * n:(i + 1) * n], ended[i * n:(i + 1) * n])) for i in range(len(texts))]


def run_caption(kern, keys, root, instruction, out, k, batch, seed, retry_passes=3):
    index = {x: i for i, x in enumerate(keys)}
    for attempt in range(retry_passes + 1):
        todo = [x for x in keys if x not in out or (len(out[x]["captions"]) < k and out[x]["attempt"] < attempt)]
        for s in range(0, len(todo), batch):
            kb = todo[s:s + batch]
            images = [kern.image(op.join(root, x)) for x in kb]
            pools, n_samp, n_cut = [[] for _ in kb], [0] * len(kb), [0] * len(kb)
            active = list(range(len(kb)))
            for rnd in range(5):
                if not active:
                    break
                res = kern.generate([images[i] for i in active], [instruction] * len(active), k + 2, "caption",
                                    seed * 1000003 + index[kb[0]] + 7919 * attempt + 7 * rnd)
                still = []
                for i, samples in zip(active, res):
                    n_samp[i] += len(samples)
                    n_cut[i] += sum(not e for _, e in samples)
                    pools[i] = unique_valid(pools[i] + [t for t, _ in samples], limit=4 * k)
                    if len(pools[i]) < k:
                        still.append(i)
                active = still
            for i, x in enumerate(kb):
                rec = dict(captions=pools[i][:k] or [FALLBACK], n_samples=n_samp[i], n_cut=n_cut[i], attempt=attempt)
                if x not in out or len(rec["captions"]) > len(out[x]["captions"]):
                    out[x] = rec
                else:
                    out[x]["attempt"] = attempt
            yield


def run_critic(kern, keys, root, template, src, out, k, batch, seed):
    index = {x: i for i, x in enumerate(keys)}
    todo = [x for x in keys if x not in out]
    for s in range(0, len(todo), batch):
        kb = todo[s:s + batch]
        pairs = [(x, c) for x in kb for c in src[x]["captions"][:k]]
        images = {x: kern.image(op.join(root, x)) for x in kb}
        res = kern.generate([images[x] for x, _ in pairs], [template.replace("{candidate}", c) for _, c in pairs], 1,
                            "critic", seed * 1000003 + index[kb[0]])
        for x in kb:
            out[x] = dict(captions=[], feedback=[], n_cut=0)
        for (x, c), [(f, ended)] in zip(pairs, res):
            out[x]["captions"].append(c)
            out[x]["feedback"].append(f.strip())
            out[x]["n_cut"] += not ended
        yield


def run_refine(kern, keys, root, template, src, out, k, batch, seed):
    index = {x: i for i, x in enumerate(keys)}
    todo = [x for x in keys if x not in out]
    for s in range(0, len(todo), batch):
        kb = todo[s:s + batch]
        pairs = [(x, c, f) for x in kb for c, f in zip(src[x]["captions"], src[x]["feedback"])]
        images = {x: kern.image(op.join(root, x)) for x in kb}
        res = kern.generate([images[x] for x, _, _ in pairs],
                            [template.replace("{candidate}", c).replace("{feedback}", f) for _, c, f in pairs], 1,
                            "refine", seed * 1000003 + index[kb[0]] + 13)
        refined = {x: [] for x in kb}
        cut = {x: 0 for x in kb}
        for (x, _, _), [(t, ended)] in zip(pairs, res):
            cut[x] += not ended
            if is_valid(clean(t)):
                refined[x].append(clean(t))
        for x in kb:
            merged = unique_valid(refined[x] + src[x]["captions"], limit=POOL)[:k] or src[x]["captions"][:k]
            ref = {dedup_key(r) for r in unique_valid(refined[x], limit=100)}
            out[x] = dict(captions=merged, n_refined=sum(dedup_key(c) in ref for c in merged),
                          n_cut=cut[x], n_calls=len(src[x]["captions"]))
        yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["caption", "critic", "refine"])
    ap.add_argument("--instruction", required=True, help="instruction (caption) or template file (critic, refine)")
    ap.add_argument("--keys", required=True, help="image paths relative to --image_root, one per line (.txt or .txt.gz)")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--captions", help="critic: output of the caption step")
    ap.add_argument("--feedback", help="refine: output of the critic step")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--batch", type=int, default=8, help="images per generate call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=50, help="batches between checkpoints of --out")
    a = ap.parse_args()
    keys = read_keys(a.keys)
    text = open(a.instruction).read()
    text = text.strip() if a.step == "caption" else text          # templates are used exactly as written
    out = json.load(open(a.out)) if op.exists(a.out) else {}       # resume
    kern = Kernel(a.model_id)
    if a.step == "caption":
        it = run_caption(kern, keys, a.image_root, text, out, a.k, a.batch, a.seed)
    elif a.step == "critic":
        it = run_critic(kern, keys, a.image_root, text, json.load(open(a.captions)), out, a.k, a.batch, a.seed)
    else:
        it = run_refine(kern, keys, a.image_root, text, json.load(open(a.feedback)), out, a.k, a.batch, a.seed)

    def save():
        os.makedirs(op.dirname(op.abspath(a.out)), exist_ok=True)
        with open(a.out + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(a.out + ".tmp", a.out)

    for i, _ in enumerate(it):
        if (i + 1) % a.save_every == 0:
            save()
    save()


if __name__ == "__main__":
    main()
