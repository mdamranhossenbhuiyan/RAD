#!/usr/bin/env python3
"""Validation-guided instruction search (Sec. 3.2 and supplement B).

The three instructions are optimized in pipeline order (one pass of coordinate ascent): captioner, then critic on the
captions of the best captioner, then refiner on the feedback of the best critic. One round of a stage:
  (i)   propose instructions from the stage's history (proposer.py: critique -> rewrite with the frozen Qwen2-VL);
  (ii)  screen up to `dev_screen_max` valid proposals on the development images (kernel.py) and drop those that
        violate a guard (guards.py, relative to the stage's initial instruction); if none passes, retry, and for the
        captioner try a minimal format repair; among the survivors keep the one with the most body regions per
        caption (tie: fewest outputs cut). If nothing passes, the proposal with the fewest violations is still
        evaluated but is NOT eligible for selection;
  (iii) caption all training images with it, train the proxy retriever (your `train_cmd`) and measure the utility
        V(pi) = best macro validation R1 over the epochs (tie: macro validation mAP).
`final` selects the eligible configuration with the highest V, hash-locks the selection and only then runs your
`test_cmd`, once per system. Every step checks its outputs, so the same command resumes after an interruption.

    python search.py --config config.json captioner      # R00 (initial) .. R<n>
    python search.py --config config.json critic         # C1 (initial) .. C<n+1>
    python search.py --config config.json refiner        # F1 .. F<n>  (starting point: the best critic run)
    python search.py --config config.json final          # lock the selection, then score test once
    python search.py --config config.json all
    python search.py carve-icfg-val --anno ICFG-PEDES/ICFG-PEDES.json --out icfg_val_ids.json
"""
import argparse
import difflib
import hashlib
import json
import os
import os.path as op
import random
import shlex
import shutil
import subprocess
import sys
import time

import guards as G
from kernel import read_keys

HERE = op.dirname(op.abspath(__file__))
PREFIX = {"captioner": "R", "critic": "C", "refiner": "F"}
SHORT = {"CUHK-PEDES": "CUHK", "ICFG-PEDES": "ICFG", "RSTPReid": "RSTP"}


def read(p):
    with open(p) as f:
        return json.load(f)


def write(obj, p):
    os.makedirs(op.dirname(op.abspath(p)), exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(p + ".tmp", p)


def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Search:
    def __init__(self, cfg):
        self.c = cfg
        self.work = cfg["work_dir"]
        self.k = cfg.get("k", 8)
        self.py = cfg.get("python", sys.executable)
        self.model_id = cfg.get("model_id", "Qwen/Qwen2-VL-7B-Instruct")
        self.n_dev = len(read_keys(cfg["dev_keys"]))
        os.makedirs(self.work, exist_ok=True)

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(op.join(self.work, "search.log"), "a") as f:
            f.write(line + "\n")

    def sh(self, cmd, **fmt):
        if isinstance(cmd, str):                     # only the named placeholders are replaced
            for k, v in fmt.items():
                cmd = cmd.replace("{" + k + "}", shlex.quote(str(v)))
        self.log("run: " + (cmd if isinstance(cmd, str) else " ".join(map(str, cmd))))
        subprocess.run(cmd, shell=isinstance(cmd, str), check=True)

    def rdir(self, name):
        return op.join(self.work, "runs", name)

    def runs(self, stage):
        out = []
        for name in sorted(os.listdir(op.join(self.work, "runs"))) if op.isdir(op.join(self.work, "runs")) else []:
            p = op.join(self.rdir(name), "run.json")
            if name.startswith(PREFIX[stage]) and op.exists(p):
                out.append(read(p))
        return out

    def best(self, stage):
        ok = [r for r in self.runs(stage) if r["eligible"]]
        if not ok:
            raise SystemExit(f"no finished eligible {stage} run; run the {stage} stage first")
        return max(ok, key=lambda r: (r["V"], r["mAP"]))

    # ---- caption kernel ----
    def kernel(self, step, instruction, keys, out, src=None):
        if op.exists(out + ".done"):
            return read(out)
        cmd = [self.py, op.join(HERE, "kernel.py"), step, "--instruction", instruction, "--keys", keys,
               "--image_root", self.c["image_root"], "--out", out, "--k", str(self.k), "--model_id", self.model_id]
        if step == "critic":
            cmd += ["--captions", src]
        if step == "refine":
            cmd += ["--feedback", src]
        self.sh(cmd + self.c.get("kernel_args", []))
        open(out + ".done", "w").close()
        return read(out)

    def captions(self, stage, instruction, keys, d, split):
        """Training captions of one configuration on `keys` (split: dev | full); returns the final records."""
        os.makedirs(d, exist_ok=True)
        if stage == "captioner":
            return self.kernel("caption", instruction, keys, op.join(d, "captions.json"))
        if stage == "critic":
            cap = op.join(self.rdir(self.best("captioner")["name"]), split, "captions.json")
            self.kernel("critic", instruction, keys, op.join(d, "feedback.json"), cap)
            return self.kernel("refine", self.c["initial"]["refiner"], keys, op.join(d, "refined.json"), op.join(d, "feedback.json"))
        fb = op.join(self.rdir(self.best("critic")["name"]), split, "feedback.json")
        return self.kernel("refine", instruction, keys, op.join(d, "refined.json"), fb)

    def dev_screen(self, stage, instruction, d):
        recs = self.captions(stage, instruction, self.c["dev_keys"], d, "dev")
        contra = None
        if self.c.get("verifier_cmd"):      # optional visual verifier -> {"contradiction": share of contradicted claims}
            vp = op.join(d, "verifier.json")
            if not op.exists(vp):
                cp = op.join(d, "dev_train_captions.json")
                write({x: r["captions"][:self.k] for x, r in recs.items()}, cp)
                self.sh(self.c["verifier_cmd"], captions=cp, image_root=self.c["image_root"], out=vp)
            contra = read(vp)["contradiction"]
        return G.dev_stats(recs, self.k, contra)

    # ---- one configuration ----
    def base_stats(self, stage):
        if stage == "refiner":
            return self.best("critic")["dev"]
        return read(op.join(self.rdir(PREFIX[stage] + ("00" if stage == "captioner" else "1")), "run.json"))["dev"]

    def history(self, stage):
        H = [dict(name=r["name"], instruction=open(r["instruction"]).read(), score=r["V"], val=r["val"], dev=r["dev"])
             for r in self.runs(stage)]
        if stage == "refiner":       # the starting point: the best critic run, with the initial refiner instruction
            b = self.best("critic")
            H.append(dict(name=b["name"], instruction=open(self.c["initial"]["refiner"]).read(), score=b["V"], val=b["val"], dev=b["dev"]))
        rejected = [dict(instruction=s["instruction"], reason="; ".join(s["fails"]))
                    for r in self.runs(stage) for s in r["screened"] if s["fails"]]
        previous = [h["instruction"] for h in H] + [s["instruction"] for r in self.runs(stage) for s in r["screened"]]
        return H, rejected, previous

    def upstream(self):
        cap = sorted([r for r in self.runs("captioner") if r["eligible"]], key=lambda r: r["V"])[-3:]
        cri = sorted(self.runs("critic"), key=lambda r: r["V"])
        f = lambda r: dict(name=r["name"], instruction=open(r["instruction"]).read(), score=r["V"], val=r["val"], dev=r["dev"])
        return dict(captioner=[f(r) for r in cap], critic=[f(r) for r in cri], used=self.best("critic")["name"])

    def propose(self, stage, d, H, rejected, previous, seed, repair_targets=None):
        hp, pp = op.join(d, "history.json"), op.join(d, "proposals.json")
        if not op.exists(pp):
            h = dict(history=H, rejected=rejected, previous=previous, n_dev=self.n_dev)
            if stage == "refiner":
                h["upstream"] = self.upstream()
            if repair_targets:
                h["repair_targets"] = repair_targets
            write(h, hp)
            self.sh([self.py, op.join(HERE, "proposer.py"), "--stage", stage, "--history", hp, "--out", pp,
                     "--n", str(self.c.get("proposals", 8)), "--seed", str(seed), "--model_id", self.model_id]
                    + (["--repair"] if repair_targets else []))
        return [p for p in read(pp)["proposals"] if not p["violations"]]

    def screen(self, stage, props, d, base, screened, previous, rejected):
        for j, p in enumerate(props[:self.c.get("dev_screen_max", 3)]):
            cd = op.join(d, f"cand{j + 1}")
            ip = op.join(cd, "instruction.txt")
            os.makedirs(cd, exist_ok=True)
            if not op.exists(ip):
                with open(ip, "w") as f:
                    f.write(p["instruction"].strip() + "\n" if stage == "captioner" else p["instruction"])
            dev = self.dev_screen(stage, ip, op.join(cd, "dev"))
            fails = G.guards(stage, dev, base)
            screened.append(dict(instruction=open(ip).read(), path=ip, dev_dir=op.join(cd, "dev"), dev=dev, fails=fails))
            previous.append(p["instruction"])
            if fails:
                rejected.append(dict(instruction=p["instruction"], reason="; ".join(fails)))

    def choose(self, stage, name, r):
        d = self.rdir(name)
        cp = op.join(d, "choice.json")
        if op.exists(cp):
            return read(cp)
        base = self.base_stats(stage)
        H, rejected, previous = self.history(stage)
        screened = []
        for att in range(1, self.c.get("propose_attempts", 3) + 1):
            pd = op.join(d, f"propose_a{att}")
            props = self.propose(stage, pd, H, rejected, previous, 1000 * r + att)
            if stage == "captioner":      # screen the proposals that change the edited instruction most
                ref = max(H, key=lambda e: e["score"])["instruction"]
                props.sort(key=lambda p: difflib.SequenceMatcher(None, ref, p["instruction"]).ratio())
            self.screen(stage, props, pd, base, screened, previous, rejected)
            if any(not s["fails"] for s in screened):
                break
        fixable = [s for s in screened if s["fails"] and all(any(x in f for x in G.REPAIRABLE) for f in s["fails"])]
        if stage == "captioner" and fixable and not any(not s["fails"] for s in screened):
            targets = [dict(instruction=s["instruction"], fails=s["fails"])
                       for s in sorted(fixable, key=G.pick_key, reverse=True)[:2]]
            for att in range(1, self.c.get("repair_attempts", 2) + 1):
                pd = op.join(d, f"repair_a{att}")
                props = self.propose(stage, pd, H, rejected, previous, 5000 + 1000 * r + att, repair_targets=targets)
                self.screen(stage, props, pd, base, screened, previous, rejected)
                if any(not s["fails"] for s in screened):
                    break
        ok = [s for s in screened if not s["fails"]]
        if ok:
            chosen, rule = max(ok, key=G.pick_key), "passed every guard; most regions per caption, then fewest cut"
        elif screened:
            chosen = max(screened, key=lambda s: (-len(s["fails"]),) + G.pick_key(s))
            rule = "GUARD RELAXED: no proposal passed every guard; fewest violations. NOT eligible for selection"
        else:
            raise SystemExit(f"{name}: the proposer produced no valid instruction")
        choice = dict(chosen=chosen, rule=rule, eligible=bool(ok), screened=screened)
        write(choice, cp)
        self.log(f"{name}: {rule}")
        return choice

    def evaluate(self, stage, name, r=0):
        d = self.rdir(name)
        if op.exists(op.join(d, "run.json")):
            return read(op.join(d, "run.json"))
        os.makedirs(d, exist_ok=True)
        ip = op.join(d, "instruction.txt")
        if r == 0:                                   # the stage's initial instruction (eligible by definition)
            shutil.copyfile(self.c["initial"][stage], ip)
            choice = dict(eligible=True, chosen=dict(fails=[]), screened=[], rule="initial instruction")
            dev = self.dev_screen(stage, ip, op.join(d, "dev"))
        else:
            choice = self.choose(stage, name, r)
            shutil.copyfile(choice["chosen"]["path"], ip)
            if not op.exists(op.join(d, "dev")):
                shutil.copytree(choice["chosen"]["dev_dir"], op.join(d, "dev"))
            dev = choice["chosen"]["dev"]
        recs = self.captions(stage, ip, self.c["train_keys"], op.join(d, "full"), "full")
        cap = op.join(d, "train_captions.json")
        write({x: rec["captions"][:self.k] for x, rec in recs.items()}, cap)
        self.sh(self.c["train_cmd"], captions=cap, keys=self.c["train_keys"], image_root=self.c["image_root"],
                run_dir=op.join(d, "train"))
        V, mAP, epoch, val = utility(op.join(d, "train"))
        run = dict(name=name, stage=stage, instruction=ip, eligible=choice["eligible"], fails=choice["chosen"]["fails"],
                   rule=choice["rule"], screened=choice["screened"], dev=dev, V=V, mAP=mAP, epoch=epoch, val=val)
        write(run, op.join(d, "run.json"))
        self.log(f"{name}: V = {V:.2f} macro val R1 (epoch {epoch}), eligible {run['eligible']}")
        return run

    def stage(self, stage):
        n = self.c["rounds"][stage]
        if stage == "captioner":
            names = [f"R{r:02d}" for r in range(n + 1)]
        elif stage == "critic":
            names = [f"C{r + 1}" for r in range(n + 1)]
        else:
            names = [None] + [f"F{r}" for r in range(1, n + 1)]      # round 0 = the best critic run, already evaluated
        for r, name in enumerate(names):
            if name:
                self.evaluate(stage, name, r)

    # ---- final selection and locked test ----
    def final(self):
        fd = op.join(self.work, "final")
        sel = op.join(fd, "selection.json")
        if not op.exists(sel + ".lock"):
            cands = [dict(name=r["name"], stage=r["stage"], eligible=r["eligible"], V=r["V"], mAP=r["mAP"], epoch=r["epoch"],
                          run_dir=op.join(self.rdir(r["name"]), "train"), instruction_sha256=sha256(r["instruction"]))
                     for s in PREFIX for r in self.runs(s)]
            best = max([c for c in cands if c["eligible"]], key=lambda c: (c["V"], c["mAP"]))
            write(dict(rule="highest macro validation R1 (tie: mAP) among configurations that passed every guard; "
                            "test splits not used", selected=best, candidates=cands,
                       locked=time.strftime("%Y-%m-%d %H:%M:%S")), sel)
            with open(sel + ".lock", "w") as f:
                f.write(sha256(sel))
            self.log(f"selection locked: {best['name']} (V = {best['V']:.2f})")
        if open(sel + ".lock").read().strip() != sha256(sel):
            raise SystemExit("selection.json changed after it was locked; test refused")
        S = read(sel)
        by = {c["name"]: c for c in S["candidates"]}
        tp = op.join(fd, "test.json")
        T = read(tp) if op.exists(tp) else {}
        for name in [S["selected"]["name"]] + [x for x in self.c.get("test_also", []) if x != S["selected"]["name"]]:
            if name in T:
                continue                              # each system is scored once
            out = op.join(fd, f"test_{name}.json")
            self.sh(self.c["test_cmd"], run_dir=by[name]["run_dir"], epoch=by[name]["epoch"], out=out)
            T[name] = dict(read(out), epoch=by[name]["epoch"], V=by[name]["V"])
            write(T, tp)
        self.log(f"test scores in {tp}")


def utility(run_dir):
    """V(pi) of Eq. (4): best macro validation R1 over the logged epochs (tie: macro mAP). The trainer writes
    <run_dir>/val.jsonl, one line per epoch: {"epoch": e, "R1": {benchmark: value}, "mAP": {benchmark: value}}."""
    E = [json.loads(l) for l in open(op.join(run_dir, "val.jsonl")) if l.strip()]
    mean = lambda d: sum(d.values()) / len(d)
    b = max(E, key=lambda e: (mean(e["R1"]), mean(e["mAP"])))
    return mean(b["R1"]), mean(b["mAP"]), b["epoch"], {SHORT.get(k, k): v for k, v in b["R1"].items()}


def carve_icfg_val(anno, out, n_ids=1000, seed=2026):
    """ICFG-PEDES has no validation split: 1,000 identities drawn from its training identities (disjoint from test)."""
    annos = read(anno)
    ids = sorted({int(a["id"]) for a in annos if a["split"] == "train"})
    pick = sorted(random.Random(seed).sample(ids, n_ids))
    assert not set(pick) & {int(a["id"]) for a in annos if a["split"] == "test"}
    n_images = sum(a["split"] == "train" and int(a["id"]) in set(pick) for a in annos)
    write(dict(seed=seed, identities=pick, n_images=n_images), out)
    print(f"{len(pick)} identities -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["captioner", "critic", "refiner", "final", "all", "carve-icfg-val"])
    ap.add_argument("--config")
    ap.add_argument("--anno", help="carve-icfg-val: ICFG-PEDES/ICFG-PEDES.json")
    ap.add_argument("--out", help="carve-icfg-val: output file")
    a = ap.parse_args()
    if a.command == "carve-icfg-val":
        return carve_icfg_val(a.anno, a.out)
    s = Search(read(a.config))
    for cmd in (["captioner", "critic", "refiner", "final"] if a.command == "all" else [a.command]):
        s.final() if cmd == "final" else s.stage(cmd)


if __name__ == "__main__":
    main()
