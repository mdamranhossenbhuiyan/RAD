#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gc
import json
import time
import math
import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False


# =========================================================
# IO / Utils
# =========================================================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def basename_wo_ext(p: str) -> str:
    b = os.path.basename(p)
    if b.lower().endswith(".json"):
        return b[:-5]
    return os.path.splitext(b)[0]


def chunk_list(xs: List[Any], bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def ordered_dump(mapping: Dict[str, Any], keys_in_order: List[str]) -> OrderedDict:
    od = OrderedDict()
    for k in keys_in_order:
        if k in mapping:
            od[k] = mapping[k]
    return od


# =========================================================
# Paths / Images
# =========================================================

def parse_train_dirs(train_dir: Optional[str], train_dirs_csv: Optional[str]) -> List[str]:
    roots: List[str] = []
    if train_dir:
        roots.append(train_dir.strip())
    if train_dirs_csv:
        for r in train_dirs_csv.split(","):
            r = r.strip()
            if r:
                roots.append(r)

    seen = set()
    out = []
    for r in roots:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out


def _resolve_image_path_one_root(train_dir: str, rel_or_abs: str) -> Optional[str]:
    p = str(rel_or_abs).strip()
    if not p:
        return None

    if os.path.isabs(p) and os.path.exists(p):
        return p

    c1 = os.path.join(train_dir, p)
    if os.path.exists(c1):
        return c1

    base = os.path.basename(p)
    c2 = os.path.join(train_dir, base)
    if os.path.exists(c2):
        return c2

    if not base.lower().endswith(IMG_EXTS):
        for ext in IMG_EXTS:
            c3 = os.path.join(train_dir, base + ext)
            if os.path.exists(c3):
                return c3

    return None


def resolve_image_path(train_dirs: List[str], key: str) -> Optional[str]:
    for td in train_dirs:
        p = _resolve_image_path_one_root(td, key)
        if p is not None:
            return p
    return None


def scan_images_from_dirs(train_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for td in train_dirs:
        for root, _, fnames in os.walk(td):
            for f in fnames:
                if f.lower().endswith(IMG_EXTS):
                    rel = os.path.relpath(os.path.join(root, f), td)
                    files.append(rel)
    files.sort()
    return files


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def maybe_resize(im: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return im
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh))


# =========================================================
# Model Loader
# =========================================================

def _dtype_from_str(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_llava_single_gpu(model_id: str, device: str, model_dtype: str, use_fast: bool):
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=use_fast,
    )

    td = _dtype_from_str(model_dtype)

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=td,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return processor, model


# =========================================================
# Prompt Section (UNCHANGED)
# =========================================================

SYSTEM_STAGE1 = (
    "You write captions for TEXT-TO-IMAGE person re-identification. "
    "Be factual and image-grounded. Avoid background/scene commentary. "
    "Avoid repetition. Output ONE concise sentence only."
)

SYSTEM_CRITIC = (
    "You are a strict caption critic for TEXT-TO-IMAGE person re-identification. "
    "Give short, actionable feedback based on visible attributes only."
)

SYSTEM_REFINE = (
    "You refine person re-identification captions to maximize discriminative visual-text alignment. "
    "Be factual and concise. Avoid repetition and avoid background commentary. Output ONE concise sentence."
)


def stage1_prompt_with_style(ref_caption: str) -> str:
    return (
        "Write ONE concise sentence describing the person (no fixed prefix).\n"
        "Match the STYLE of the reference caption (do not copy exact words).\n"
        "Order: upper-body -> lower-body -> footwear -> accessories -> carried item -> hair/hood -> viewpoint.\n"
        "Mention patterns/logos/text only if clearly visible.\n"
        "No background commentary. No speculation.\n\n"
        f"Reference caption:\n{ref_caption}"
    )


def stage1_prompt_no_style() -> str:
    return (
        "Write ONE concise sentence describing the person (no fixed prefix).\n"
        "Order: upper-body -> lower-body -> footwear -> accessories -> carried item -> hair/hood -> viewpoint.\n"
        "Mention patterns/logos/text only if clearly visible.\n"
        "No background commentary. No speculation."
    )


def critic_prompt(ref_caption: str, candidate: str, use_style: bool) -> str:
    if use_style:
        return (
            "Critique the caption for ReID.\n\n"
            f"Reference style caption:\n{ref_caption}\n\n"
            f"Candidate caption:\n{candidate}\n\n"
            "Return 3-6 short bullet points only:\n"
            "- missing discriminative visible details (color/item)\n"
            "- missing accessories / carried item\n"
            "- repetition / redundant phrasing\n"
            "- hallucination / non-visible claims\n"
            "- viewpoint/occlusion if useful\n"
        )
    return (
        "Critique the caption for ReID.\n\n"
        f"Candidate caption:\n{candidate}\n\n"
        "Return 3-6 short bullet points only:\n"
        "- missing discriminative visible details (color/item)\n"
        "- missing accessories / carried item\n"
        "- repetition / redundant phrasing\n"
        "- hallucination / non-visible claims\n"
        "- viewpoint/occlusion if useful\n"
    )


def refine_prompt(ref_caption: str, candidate: str, feedback: str, use_style: bool) -> str:
    if use_style:
        return (
            "Rewrite into ONE concise sentence describing the person (no fixed prefix).\n"
            "Match the reference caption STYLE.\n"
            "Keep it short (18–28 words). No repetition. No background.\n"
            "Use only visible attributes.\n\n"
            f"Reference caption:\n{ref_caption}\n\n"
            f"Candidate caption:\n{candidate}\n\n"
            f"Feedback bullets:\n{feedback}\n\n"
            "Final refined caption:"
        )
    return (
        "Rewrite into ONE concise sentence describing the person (no fixed prefix).\n"
        "Keep it short (18–28 words). No repetition. No background.\n"
        "Use only visible attributes.\n\n"
        f"Candidate caption:\n{candidate}\n\n"
        f"Feedback bullets:\n{feedback}\n\n"
        "Final refined caption:"
    )


# =========================================================
# Chat / Generation
# =========================================================

def apply_chat(processor, prompts: List[str], system_prompt: str) -> List[str]:
    chats = []
    for p in prompts:
        full_prompt = f"{system_prompt}\n\n{p}"
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        try:
            chats.append(
                processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )
            )
        except Exception:
            chats.append(full_prompt)
    return chats


@torch.inference_mode()
def llava_generate_groups(
    processor,
    model,
    images: List[Image.Image],
    chat_texts: List[str],
    *,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[List[str]]:
    device = next(model.parameters()).device

    inputs = processor(
        text=chat_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    K = max(1, int(num_samples))

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        num_return_sequences=K,
        return_dict_in_generate=True,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
    )

    use_amp = (next(model.parameters()).dtype in (torch.float16, torch.bfloat16))
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model.generate(**inputs, **gen_kwargs)

    seqs = out.sequences
    gen_only = seqs[:, prompt_len:]
    decoded = processor.batch_decode(gen_only, skip_special_tokens=True)

    groups = []
    for i in range(0, len(decoded), K):
        groups.append(decoded[i:i + K])
    return groups


# =========================================================
# Caption Cleaning / Filtering
# =========================================================

BAD_PATTERNS = [
    r"\bif\s*\(\s*1\s*\)",
    r"when you look up something on the internet",
    r"a photo showing a scene with people or objects",
    r"\bscene with people or objects\b",
    r"\bthis image shows\b",
    r"\bthe image shows\b",
    r"\bthe image depicts\b",
    r"\bblurred lights\b",
    r"\bvenue\b",
    r"\bsubway station\b",
    r"\bmall\b",
    r"\bon the internet\b",
    r"\bpeople seated behind\b",
]

PERSON_REID_TOKENS = [
    "man", "woman", "person", "individual", "figure", "girl", "boy", "child",
    "shirt", "t-shirt", "jacket", "hoodie", "coat", "sweater", "top", "jersey", "blazer",
    "pants", "jeans", "trousers", "shorts", "skirt", "leggings",
    "shoes", "sneakers", "boots", "sandals",
    "bag", "backpack", "purse", "hat", "cap", "scarf",
    "front", "back", "side", "profile", "rear"
]

BAD_BACKGROUND_TOKENS = [
    "street", "venue", "subway", "mall", "background", "people", "lights", "stage", "escalator", "table"
]


def clean_caption_pure(s: str) -> str:
    if not s:
        return ""
    s = s.strip().replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("“”\"'")
    s = re.sub(r"\b(system|assistant|user)\b\s*:", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\W+", "", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+!", "!", s)
    s = re.sub(r"\s+\?", "?", s)
    if s and s[-1] not in ".!?":
        s += "."
    return s


def normalize_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_valid_reid_caption(s: str) -> bool:
    s0 = s.lower().strip()

    if not s0:
        return False

    for pat in BAD_PATTERNS:
        if re.search(pat, s0):
            return False

    nwords = len(s0.split())
    if nwords < 5 or nwords > 40:
        return False

    if not any(tok in s0 for tok in PERSON_REID_TOKENS):
        return False

    return True


def unique_keep_order(xs: List[str], limit: int) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        x = clean_caption_pure(x)
        if not x:
            continue
        if not is_valid_reid_caption(x):
            continue
        key = normalize_for_dedup(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= limit:
            break
    return out


def heuristic_reid_score(s: str) -> float:
    s0 = s.lower()
    score = 0.0

    upper = ["shirt", "t-shirt", "jacket", "hoodie", "coat", "sweater", "top", "jersey", "blazer"]
    lower = ["pants", "jeans", "trousers", "shorts", "skirt", "leggings"]
    foot = ["shoes", "sneakers", "boots", "sandals"]
    acc = ["bag", "backpack", "purse", "hat", "cap", "scarf"]
    view = ["front", "back", "side", "profile", "rear"]

    if any(x in s0 for x in upper):
        score += 2.0
    if any(x in s0 for x in lower):
        score += 2.0
    if any(x in s0 for x in foot):
        score += 2.0
    if any(x in s0 for x in acc):
        score += 2.0
    if any(x in s0 for x in view):
        score += 1.0

    if any(x in s0 for x in BAD_BACKGROUND_TOKENS):
        score -= 1.5

    nwords = len(s0.split())
    if 12 <= nwords <= 28:
        score += 1.0

    return score


# =========================================================
# Optional CLIP Reranker
# =========================================================

class CLIPRerankerSafeFP32:
    def __init__(
        self,
        model_name="ViT-L-14",
        pretrained="openai",
        device="cuda:0",
        mode="hybrid",
        alpha=0.35,
        lam=0.08,
        max_words=28,
        cache_size=512,
    ):
        self.device = torch.device(device)
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, force_quick_gelu=True
            )
        except TypeError:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.float().to(self.device)
        self.model.eval()

        self.mode = mode
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.max_words = int(max_words)
        self.cache_size = int(cache_size)
        self._neg_img_feat_cache: Optional[torch.Tensor] = None

    def _len_penalty(self, text: str) -> float:
        words = len(text.strip().split())
        if words <= self.max_words:
            return 0.0
        return float(words - self.max_words)

    @torch.inference_mode()
    def score(self, pil_images: List[Image.Image], candidates: List[List[str]]) -> torch.Tensor:
        B = len(pil_images)
        K = len(candidates[0])
        assert all(len(x) == K for x in candidates), "Candidates must be B x K"

        img_tensor = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device).float()
        img_feat = self.model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        if self.cache_size > 0 and self._neg_img_feat_cache is None:
            self._neg_img_feat_cache = img_feat.detach().clone()

        flat_caps = [c for row in candidates for c in row]
        txt_tokens = self.tokenizer(flat_caps).to(self.device)
        txt_feat = self.model.encode_text(txt_tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sims_batch = txt_feat @ img_feat.T
        pos_idx = torch.arange(B, device=self.device).repeat_interleave(K)
        pos_sim = sims_batch[torch.arange(B * K, device=self.device), pos_idx]

        if self.mode == "cosine":
            return pos_sim.view(B, K).detach().float().cpu()

        sims_neg = sims_batch.clone()
        sims_neg[torch.arange(B * K, device=self.device), pos_idx] = -1e9
        max_neg_batch = sims_neg.max(dim=1).values

        if self._neg_img_feat_cache is not None and self._neg_img_feat_cache.shape[0] > 1:
            sims_cache = txt_feat @ self._neg_img_feat_cache.T
            max_neg_cache = sims_cache.max(dim=1).values
            max_neg = torch.maximum(max_neg_batch, max_neg_cache)
        else:
            max_neg = max_neg_batch

        margin = pos_sim - max_neg

        if self.mode == "margin":
            return margin.view(B, K).detach().float().cpu()

        len_pen = torch.tensor(
            [self._len_penalty(t) for t in flat_caps],
            device=self.device,
            dtype=torch.float32
        )
        score = pos_sim + self.alpha * margin - self.lam * len_pen
        return score.view(B, K).detach().float().cpu()


# =========================================================
# Output Files
# =========================================================

def build_out_paths(base_name: str, out_dir: str, part_name: str) -> Dict[str, str]:
    return {
        "stage1": os.path.join(out_dir, f"{base_name}.{part_name}.stage1.json"),
        "critic": os.path.join(out_dir, f"{base_name}.{part_name}.critic.json"),
        "refine": os.path.join(out_dir, f"{base_name}.{part_name}.refine.json"),
        "selected": os.path.join(out_dir, f"{base_name}.{part_name}.selected.json"),
        "ranked": os.path.join(out_dir, f"{base_name}.{part_name}.ranked_candidates.json"),
    }


# =========================================================
# Pipeline Helpers
# =========================================================

def get_ref_loader(json_path: Optional[str], train_dirs: List[str]):
    if json_path is not None and os.path.isfile(json_path):
        ref_data = read_json(json_path)
        if not isinstance(ref_data, dict):
            raise ValueError("Reference JSON must be dict")

        all_keys = list(ref_data.keys())

        def get_ref_caption(k: str) -> str:
            v = ref_data.get(k, "")
            if isinstance(v, list) and len(v) > 0:
                return str(v[0]).strip()
            return str(v).strip()

        base_name = basename_wo_ext(json_path)
        mode = "WITH_JSON"
    else:
        all_keys = scan_images_from_dirs(train_dirs)

        def get_ref_caption(k: str) -> str:
            return ""

        base_name = "NOREF"
        mode = "NO_JSON"

    return all_keys, get_ref_caption, base_name, mode


def generate_until_min_valid(
    processor,
    model,
    image: Image.Image,
    ref_caption: str,
    use_ref_style: bool,
    min_valid: int,
    stage1_samples: int,
    stage1_max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_rounds: int = 3,
) -> Tuple[List[str], List[List[str]]]:
    valid_pool: List[str] = []
    raw_rounds: List[List[str]] = []

    for _ in range(max_rounds):
        p = stage1_prompt_with_style(ref_caption) if use_ref_style else stage1_prompt_no_style()
        chat = apply_chat(processor, [p], SYSTEM_STAGE1)

        groups = llava_generate_groups(
            processor,
            model,
            [image],
            chat,
            max_new_tokens=stage1_max_new_tokens,
            num_samples=stage1_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        raw = groups[0] if groups else []
        raw_rounds.append(raw)

        cleaned = [clean_caption_pure(x) for x in raw]
        cleaned = [x for x in cleaned if x.strip()]
        merged = valid_pool + cleaned
        valid_pool = unique_keep_order(merged, limit=max(min_valid, stage1_samples * max_rounds))

        if len(valid_pool) >= min_valid:
            break

    if len(valid_pool) < min_valid:
        fallback_sorted = sorted(valid_pool, key=heuristic_reid_score, reverse=True)
        valid_pool = fallback_sorted

    return valid_pool[:max(min_valid, len(valid_pool))], raw_rounds


# =========================================================
# Main Pipeline
# =========================================================

def run_agentic_shard(
    *,
    json_path: Optional[str],
    use_ref_style: bool,
    train_dirs: List[str],
    out_dir: str,
    model_id: str,
    device: str,
    model_dtype: str,
    use_fast: bool,
    max_image_side: int,
    batch_size: int,
    start_idx: int,
    end_idx: int,
    part_name: str,
    resume: bool,
    save_every_batch: bool,
    print_samples: bool,
    stage1_samples: int,
    stage1_max_new_tokens: int,
    critic_max_new_tokens: int,
    refine_samples: int,
    refine_max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    keep_unique_candidates: int,
    min_valid_captions: int,
    clip_rerank: bool,
    clip_model: str,
    clip_pretrained: str,
    clip_mode: str,
    clip_alpha: float,
    clip_penalty_lambda: float,
    clip_max_words: int,
    clip_cache_size: int,
):
    ensure_dir(out_dir)

    all_keys, get_ref_caption, base_name, mode = get_ref_loader(json_path, train_dirs)

    N = len(all_keys)
    s = max(0, int(start_idx))
    e = int(end_idx)
    if e < 0 or e > N:
        e = N
    if e <= s:
        raise ValueError(f"Invalid shard range: start={s}, end={e}, total={N}")

    shard_keys = all_keys[s:e]
    total = len(shard_keys)

    paths = build_out_paths(base_name, out_dir, part_name)

    def _load_if_exists(p: str) -> Dict[str, Any]:
        if resume and os.path.exists(p):
            try:
                obj = read_json(p)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {}

    stage1_out = _load_if_exists(paths["stage1"])
    critic_out = _load_if_exists(paths["critic"])
    refine_out = _load_if_exists(paths["refine"])
    selected_out = _load_if_exists(paths["selected"])
    ranked_out = _load_if_exists(paths["ranked"])

    processor, model = load_llava_single_gpu(
        model_id=model_id,
        device=device,
        model_dtype=model_dtype,
        use_fast=use_fast,
    )

    reranker = None
    if clip_rerank and _HAS_OPENCLIP:
        try:
            reranker = CLIPRerankerSafeFP32(
                model_name=clip_model,
                pretrained=clip_pretrained,
                device=device,
                mode=clip_mode,
                alpha=clip_alpha,
                lam=clip_penalty_lambda,
                max_words=clip_max_words,
                cache_size=clip_cache_size,
            )
            print(f"[clip] ON: {clip_model}/{clip_pretrained} mode={clip_mode}")
        except Exception as e:
            print(f"[clip] OFF(init fail): {e}")
            reranker = None

    print(f"\n[MODE] {mode}")
    print(f"[shard] {part_name} | keys {s}:{e} | total={total}")
    print(f"[ref_style] {use_ref_style}")
    print(f"[min_valid_captions] {min_valid_captions}")
    for k, p in paths.items():
        print(f"[out] {k}: {p}")

    t0 = time.time()
    done = 0
    missing = 0

    for bi, keys in enumerate(chunk_list(shard_keys, batch_size), start=1):
        keys_todo = [k for k in keys if k not in selected_out]
        if not keys_todo:
            continue

        images: List[Image.Image] = []
        keys_ok: List[str] = []
        refcaps_ok: List[str] = []

        for k in keys_todo:
            p = resolve_image_path(train_dirs, k)
            if p is None:
                missing += 1
                continue
            try:
                im = load_image(p)
                im = maybe_resize(im, max_image_side)
                images.append(im)
                keys_ok.append(k)
                rc = get_ref_caption(k).strip() if use_ref_style else ""
                refcaps_ok.append(rc)
            except Exception:
                missing += 1

        if not images:
            continue

        # -------------------------------------------------
        # Stage 1: generate until at least min_valid_captions
        # -------------------------------------------------
        cand_matrix: List[List[str]] = []
        stage1_raw_debug: List[List[List[str]]] = []

        for im, rc in zip(images, refcaps_ok):
            valid_caps, raw_rounds = generate_until_min_valid(
                processor=processor,
                model=model,
                image=im,
                ref_caption=rc,
                use_ref_style=use_ref_style,
                min_valid=min_valid_captions,
                stage1_samples=stage1_samples,
                stage1_max_new_tokens=stage1_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_rounds=3,
            )

            if len(valid_caps) < min_valid_captions:
                valid_caps = sorted(valid_caps, key=heuristic_reid_score, reverse=True)

            valid_caps = valid_caps[:max(min_valid_captions, keep_unique_candidates)]
            if len(valid_caps) == 0:
                valid_caps = ["Person wearing visible clothing with distinguishable colors and items."]

            cand_matrix.append(valid_caps)
            stage1_raw_debug.append(raw_rounds)

        for k, cands, raw_dbg in zip(keys_ok, cand_matrix, stage1_raw_debug):
            stage1_out[k] = {
                "candidates": cands,
                "num": len(cands),
                "raw_rounds": raw_dbg,
            }

        # -------------------------------------------------
        # Critic on ALL candidates
        # -------------------------------------------------
        critic_prompts_flat: List[str] = []
        critic_owner: List[Tuple[int, int]] = []

        for i, (rc, cands) in enumerate(zip(refcaps_ok, cand_matrix)):
            local_caps = cands[:min_valid_captions]
            for j, cand in enumerate(local_caps):
                critic_prompts_flat.append(critic_prompt(rc, cand, use_ref_style))
                critic_owner.append((i, j))

        critic_imgs = [images[i] for i, _ in critic_owner]
        chat_fb = apply_chat(processor, critic_prompts_flat, SYSTEM_CRITIC)

        fb_groups = llava_generate_groups(
            processor,
            model,
            critic_imgs,
            chat_fb,
            max_new_tokens=critic_max_new_tokens,
            num_samples=1,
            temperature=0.4,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            no_repeat_ngram_size=4,
        )

        feedback_per_image: List[List[str]] = [[] for _ in range(len(images))]
        for (i, j), grp in zip(critic_owner, fb_groups):
            fb = grp[0].strip() if grp else ""
            while len(feedback_per_image[i]) <= j:
                feedback_per_image[i].append("")
            feedback_per_image[i][j] = fb

        for i, k in enumerate(keys_ok):
            critic_out[k] = {
                "critic_pairs": [
                    {
                        "candidate": cand_matrix[i][j],
                        "feedback": feedback_per_image[i][j] if j < len(feedback_per_image[i]) else ""
                    }
                    for j in range(min(len(cand_matrix[i]), min_valid_captions))
                ]
            }

        # -------------------------------------------------
        # Refine ALL candidates
        # -------------------------------------------------
        refine_prompts_flat: List[str] = []
        refine_owner: List[Tuple[int, int]] = []

        for i, (rc, cands, fbs) in enumerate(zip(refcaps_ok, cand_matrix, feedback_per_image)):
            upto = min(len(cands), min_valid_captions)
            for j in range(upto):
                cand = cands[j]
                fb = fbs[j] if j < len(fbs) else ""
                refine_prompts_flat.append(refine_prompt(rc, cand, fb, use_ref_style))
                refine_owner.append((i, j))

        refine_imgs = [images[i] for i, _ in refine_owner]
        chat_rf = apply_chat(processor, refine_prompts_flat, SYSTEM_REFINE)

        rf_groups = llava_generate_groups(
            processor,
            model,
            refine_imgs,
            chat_rf,
            max_new_tokens=refine_max_new_tokens,
            num_samples=refine_samples,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
        )

        refined_per_image: List[List[str]] = [[] for _ in range(len(images))]
        for (i, _), grp in zip(refine_owner, rf_groups):
            vals = [clean_caption_pure(x) for x in grp]
            vals = [x for x in vals if is_valid_reid_caption(x)]
            refined_per_image[i].extend(vals)

        for i in range(len(images)):
            merged = refined_per_image[i] + cand_matrix[i]
            refined_per_image[i] = unique_keep_order(merged, limit=max(min_valid_captions, keep_unique_candidates))
            if len(refined_per_image[i]) == 0:
                refined_per_image[i] = cand_matrix[i][:max(1, min_valid_captions)]

        for i, k in enumerate(keys_ok):
            refine_out[k] = {
                "refined_candidates": refined_per_image[i]
            }

        # -------------------------------------------------
        # Ranking
        # -------------------------------------------------
        ranked_lists: Dict[str, List[Dict[str, Any]]] = {}

        if reranker is not None:
            Kmax = max(len(x) for x in refined_per_image)
            cand_mat = []
            for row in refined_per_image:
                if len(row) < Kmax:
                    row = row + [row[0]] * (Kmax - len(row))
                cand_mat.append(row)

            try:
                scores = reranker.score(images, cand_mat)
                for i, k in enumerate(keys_ok):
                    order = sorted(range(Kmax), key=lambda j: float(scores[i, j]), reverse=True)
                    ranked = []
                    seen = set()
                    r = 1
                    for j in order:
                        cap = cand_mat[i][j]
                        key_norm = normalize_for_dedup(cap)
                        if key_norm in seen:
                            continue
                        seen.add(key_norm)
                        ranked.append({
                            "rank": r,
                            "caption": cap,
                            "score": float(scores[i, j])
                        })
                        r += 1
                    ranked_lists[k] = ranked
            except Exception as e:
                print(f"[clip] score fail -> fallback heuristic: {e}")
                reranker = None

        if reranker is None:
            for i, k in enumerate(keys_ok):
                uniq = unique_keep_order(refined_per_image[i], limit=max(min_valid_captions, keep_unique_candidates))
                scored = sorted(
                    [{"caption": c, "score": heuristic_reid_score(c)} for c in uniq],
                    key=lambda x: x["score"],
                    reverse=True
                )
                ranked_lists[k] = [
                    {"rank": idx + 1, "caption": item["caption"], "score": float(item["score"])}
                    for idx, item in enumerate(scored)
                ]

        for k in keys_ok:
            ranked_out[k] = ranked_lists[k]
            chosen = [x["caption"] for x in ranked_lists[k][:2]]
            if len(chosen) == 0:
                chosen = ["Person wearing visible clothing with distinguishable colors and items."]
            while len(chosen) < 2:
                chosen.append(chosen[0])
            selected_out[k] = [clean_caption_pure(chosen[0]), clean_caption_pure(chosen[1])]

        done += len(keys_ok)

        if save_every_batch:
            write_json(ordered_dump(stage1_out, shard_keys), paths["stage1"])
            write_json(ordered_dump(critic_out, shard_keys), paths["critic"])
            write_json(ordered_dump(refine_out, shard_keys), paths["refine"])
            write_json(ordered_dump(selected_out, shard_keys), paths["selected"])
            write_json(ordered_dump(ranked_out, shard_keys), paths["ranked"])

        dt = time.time() - t0
        rate = done / max(dt, 1e-9)
        print(f"[{now_str()}] batch={bi} done={done}/{total} missing={missing} {rate:.2f} imgs/s")

        if print_samples and len(keys_ok) > 0:
            kk = keys_ok[0]
            print(f"\n[{kk}]")
            print("  stage1 first:", stage1_out[kk]["candidates"][0] if stage1_out[kk]["candidates"] else "")
            pairs = critic_out[kk].get("critic_pairs", [])
            if pairs:
                print("  critic first:", pairs[0]["feedback"][:140].replace("\n", " ") + "...")
            refs = refine_out[kk].get("refined_candidates", [])
            if refs:
                print("  refine first:", refs[0])
            print("  selected1:", selected_out[kk][0])
            print("  selected2:", selected_out[kk][1])

        for im in images:
            try:
                im.close()
            except Exception:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(ordered_dump(stage1_out, shard_keys), paths["stage1"])
    write_json(ordered_dump(critic_out, shard_keys), paths["critic"])
    write_json(ordered_dump(refine_out, shard_keys), paths["refine"])
    write_json(ordered_dump(selected_out, shard_keys), paths["selected"])
    write_json(ordered_dump(ranked_out, shard_keys), paths["ranked"])

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n✅ DONE")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print(f"Shard keys: {s}:{e} (total={total})")


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--json_path", type=str, default=None)
    ap.add_argument("--use_ref_style", action="store_true")

    ap.add_argument("--train_dir", type=str, default=None)
    ap.add_argument("--train_dirs", type=str, default=None)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--use_fast", action="store_true")
    ap.add_argument("--max_image_side", type=int, default=672)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=100000)
    ap.add_argument("--part_name", type=str, default="0K-100K")

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_every_batch", action="store_true")
    ap.add_argument("--no_print", action="store_true")

    ap.add_argument("--stage1_samples", type=int, default=14)
    ap.add_argument("--stage1_max_new_tokens", type=int, default=40)
    ap.add_argument("--critic_max_new_tokens", type=int, default=96)
    ap.add_argument("--refine_samples", type=int, default=1)
    ap.add_argument("--refine_max_new_tokens", type=int, default=40)

    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--keep_unique_candidates", type=int, default=12)
    ap.add_argument("--min_valid_captions", type=int, default=10)

    ap.add_argument("--clip_rerank", action="store_true")
    ap.add_argument("--clip_model", type=str, default="ViT-L-14")
    ap.add_argument("--clip_pretrained", type=str, default="openai")
    ap.add_argument("--clip_mode", type=str, default="hybrid", choices=["cosine", "margin", "hybrid"])
    ap.add_argument("--clip_alpha", type=float, default=0.35)
    ap.add_argument("--clip_penalty_lambda", type=float, default=0.08)
    ap.add_argument("--clip_max_words", type=int, default=28)
    ap.add_argument("--clip_cache_size", type=int, default=512)

    args = ap.parse_args()

    train_dirs = parse_train_dirs(args.train_dir, args.train_dirs)
    if not train_dirs:
        raise SystemExit("Provide --train_dir or --train_dirs")

    for td in train_dirs:
        if not os.path.isdir(td):
            raise SystemExit(f"Train dir not found: {td}")

    if args.use_ref_style and (not args.json_path or not os.path.isfile(args.json_path)):
        raise SystemExit("--use_ref_style requires a valid --json_path")

    run_agentic_shard(
        json_path=args.json_path,
        use_ref_style=args.use_ref_style,
        train_dirs=train_dirs,
        out_dir=args.out_dir,
        model_id=args.model_id,
        device=args.device,
        model_dtype=args.model_dtype,
        use_fast=args.use_fast,
        max_image_side=args.max_image_side,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        part_name=args.part_name,
        resume=args.resume,
        save_every_batch=args.save_every_batch,
        print_samples=(not args.no_print),
        stage1_samples=args.stage1_samples,
        stage1_max_new_tokens=args.stage1_max_new_tokens,
        critic_max_new_tokens=args.critic_max_new_tokens,
        refine_samples=args.refine_samples,
        refine_max_new_tokens=args.refine_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        keep_unique_candidates=args.keep_unique_candidates,
        min_valid_captions=args.min_valid_captions,
        clip_rerank=args.clip_rerank,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_mode=args.clip_mode,
        clip_alpha=args.clip_alpha,
        clip_penalty_lambda=args.clip_penalty_lambda,
        clip_max_words=args.clip_max_words,
        clip_cache_size=args.clip_cache_size,
    )


if __name__ == "__main__":
    main()
