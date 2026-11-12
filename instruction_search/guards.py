"""Hard constraints on instructions, caption statistics of the development images, the guards g_j(pi) <= kappa_j of
Eq. (5) (supplement, Tab. "Guards") and the rule that picks one screened proposal per round."""
import re
import statistics

from kernel import is_valid

# ---------------------------------------------------------------------------------------------------------------
# Hard constraints every proposed instruction must meet before it is screened
ONE_SENTENCE_RE = re.compile(r"\b(one|1|single)\b[^.\n]{0,40}\bsentence\b", re.I)
FACTUAL_CAP_RE = re.compile(r"\b(factual|visible|image-grounded|grounded|only what (is|you can) see|observable)\b", re.I)
FACTUAL_TPL_RE = re.compile(r"\b(factual|visible|image-grounded|grounded|observable|hallucinat\w*|only what)\b", re.I)
NO_SPEC_RE = re.compile(r"(\b(no|not|without|avoid|never|don't|do not|nor)\b[^.\n]{0,80}\b(speculat\w*|guess\w*|infer\w*|"
                        r"assum\w*|hallucinat\w*|invent\w*|imagin\w*)|not visible|only if clearly visible)", re.I)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def check_instruction(stage, text, previous=()):
    """List of violations (empty = valid). stage: captioner | critic | refiner."""
    v, t = [], text.strip()
    if any(ord(c) > 0x2FFF for c in t):
        v.append("non-Latin characters")
    if norm(t) in {norm(p) for p in previous}:
        v.append("identical to an earlier instruction")
    if stage == "captioner":
        if not ONE_SENTENCE_RE.search(t):
            v.append("no explicit one-sentence output requirement")
        if not FACTUAL_CAP_RE.search(t):
            v.append("no factual / visible-only requirement")
        if not NO_SPEC_RE.search(t):
            v.append("no no-speculation clause")
        if not 20 <= len(t.split()) <= 150:
            v.append(f"{len(t.split())} words (allowed 20-150)")
        if re.search(r"[{}]|<[^<>\n]{1,30}>|\[[A-Z_ ]+\]", t):
            v.append("template placeholders or markup")
        if re.search(r"(?i)\b(example|e\.g\.)\s*:\s*\"?[A-Z][^\n]{40,}", t):
            v.append("contains a full example caption (would be copied verbatim)")
        return v
    need = ["{candidate}"] + (["{feedback}"] if stage == "refiner" else [])
    for ph in need:
        if t.count(ph) != 1:
            v.append(f"placeholder {ph} appears {t.count(ph)} times (needs exactly 1)")
    if stage == "critic" and "{feedback}" in t:
        v.append("critic template must not contain {feedback}")
    bare = t.replace("{candidate}", "").replace("{feedback}", "")
    if re.search(r"[{}]", bare):
        v.append("other template braces")
    if not 15 <= len(bare.split()) <= 220:
        v.append(f"{len(bare.split())} words (allowed 15-220)")
    if not FACTUAL_TPL_RE.search(bare):
        v.append("no visible-only / factuality requirement")
    if stage == "refiner" and not ONE_SENTENCE_RE.search(bare):
        v.append("no explicit one-sentence output requirement")
    return v


# ---------------------------------------------------------------------------------------------------------------
# Body regions a caption mentions: a lexicon over upper body, lower body, footwear, carried item, head, accessory and
# gender. (The paper counts regions with an attribute parser built on the same vocabulary; this lexicon count is a
# simpler stand-in and can differ slightly from the numbers in the paper.)
REGION_WORDS = {
    "upper": "shirt t-shirt tshirt tee tee-shirt top blouse jacket coat sweater hoodie sweatshirt vest cardigan polo jersey "
             "dress blazer windbreaker parka tank",
    "lower": "pants pant trousers jeans shorts skirt leggings capris slacks tights bottoms khakis",
    "footwear": "shoes shoe sneakers sneaker boots boot sandals sandal heels slippers flats loafers flip-flops",
    "carried": "backpack bag bags handbag purse suitcase umbrella satchel tote briefcase backpacks handbags",
    "head": "hat cap beanie helmet scarf headband",
    "accessory": "glasses sunglasses eyeglasses",
    "gender": "man men male boy guy gentleman he his him himself woman women female girl lady she her hers herself",
}
REGION_OF = {w: r for r, ws in REGION_WORDS.items() for w in ws.split()}
SENT_SPLIT_RE = re.compile(r"[.!?](?:\s+|$)")


def regions(caption):
    return len({REGION_OF[w] for w in re.findall(r"[a-z]+(?:-[a-z]+)*", caption.lower()) if w in REGION_OF})


def one_sentence(caption):
    return len([p for p in SENT_SPLIT_RE.split(caption.strip()) if p.strip()]) <= 1


def dev_stats(records, k, contradiction=None):
    """Statistics of the training captions of the development images (output of kernel.py caption or refine)."""
    caps = [c for r in records.values() for c in r["captions"][:k]]
    s = dict(words=statistics.mean(len(c.split()) for c in caps),
             regions=statistics.mean(regions(c) for c in caps),
             one_sentence=statistics.mean(one_sentence(c) for c in caps),
             images_with_k=statistics.mean(len(r["captions"]) >= k for r in records.values()))
    r0 = next(iter(records.values()))
    if "n_samples" in r0:          # captioner: cut = samples without end of sequence, among all samples drawn
        s["cut"] = sum(r["n_cut"] for r in records.values()) / max(sum(r["n_samples"] for r in records.values()), 1)
        s["valid"] = statistics.mean(is_valid(c) for c in caps)
    else:                          # critic / refiner: cut = rewrites without end of sequence
        s["cut"] = sum(r["n_cut"] for r in records.values()) / max(sum(r["n_calls"] for r in records.values()), 1)
        s["refined_share"] = statistics.mean(r["n_refined"] / max(len(r["captions"][:k]), 1) for r in records.values())
    if contradiction is not None:
        s["contradiction"] = contradiction
    return s


# ---------------------------------------------------------------------------------------------------------------
def guards(stage, s, base):
    """Violated guards of statistics s, relative to the statistics `base` of the stage's initial instruction."""
    f = []
    if "contradiction" in s and "contradiction" in base and s["contradiction"] > base["contradiction"] + 0.02:
        f.append(f"claim contradiction rate {s['contradiction']:.3f} > initial {base['contradiction']:.3f} + 0.02")
    if s["regions"] < base["regions"] - 0.25:
        f.append(f"only {s['regions']:.2f} body regions per caption (initial {base['regions']:.2f}, at most 0.25 fewer)")
    if s["cut"] > base["cut"] + 0.05:
        f.append(f"{s['cut']:.3f} of outputs cut at the token limit (initial {base['cut']:.3f} + 0.05)")
    c = 0.95 if stage == "captioner" else 0.898
    if s["one_sentence"] < min(c, base["one_sentence"] - 0.02):
        f.append(f"one-sentence rate {s['one_sentence']:.3f} below {min(c, base['one_sentence'] - 0.02):.3f}")
    if stage == "captioner" and s["valid"] < base["valid"] - 0.05:
        f.append(f"valid-caption rate {s['valid']:.3f} < initial {base['valid']:.3f} - 0.05")
    if stage != "captioner" and s["refined_share"] < 0.90:
        f.append(f"refined share {s['refined_share']:.3f} < 0.90")
    if s["images_with_k"] < 0.95:
        f.append(f"only {s['images_with_k']:.3f} of the images reach K valid captions")
    return f


REPAIRABLE = ("one-sentence rate", "cut at the token limit")      # failures the captioner's repair rewrite addresses


def pick_key(c):
    """Among proposals that pass every guard: most body regions per caption, then fewest outputs cut."""
    return (c["dev"]["regions"], -c["dev"]["cut"])
