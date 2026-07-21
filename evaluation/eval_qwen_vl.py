#!/usr/bin/env python3
"""
eval_qwen_vl.py — run the frozen_test objective benchmark on Qwen2.5-VL (base or LoRA) and score.

RUN ON THE AUTODL GPU INSTANCE.
  # base:
  python 5-evaluation/eval_qwen_vl.py --tag base
  # LoRA:
  python 5-evaluation/eval_qwen_vl.py --adapter 4-training/sviro_lora_v1 --tag lora
  # compare:
  python 5-evaluation/eval_qwen_vl.py --compare 5-evaluation/results_base.jsonl 5-evaluation/results_lora.jsonl

Objective scoring (deterministic GT): occupancy (per seat category), person count, left-object yes/no,
and seatbelt reject (model must refuse). Prints accuracy overall and by question type.
"""
import argparse, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "5-evaluation", "frozen_benchmark.jsonl")
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NUMWORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
REFUSE = re.compile(r"cannot determine|can't determine|not determinable|unable|not provided|"
                    r"not (visible|shown|labeled|available)|no .*information|indetermin", re.I)


# coarse (safety-relevant) grouping of the 6 fine occupancy categories
COARSE = {"empty": "unoccupied", "empty_child_seat": "unoccupied", "empty_infant_seat": "unoccupied",
          "child_in_child_seat": "child", "infant_in_infant_seat": "child",
          "adult": "adult", "everyday_object": "object", "unknown": "unknown"}
OCCUPANT = {"adult", "child"}


def classify_occ(p):
    p = p.lower()
    if "everyday object" in p or re.search(r"\bobject\b", p): return "everyday_object"
    if "empty infant" in p: return "empty_infant_seat"
    if "empty child" in p: return "empty_child_seat"
    if "infant" in p: return "infant_in_infant_seat"
    if "child" in p: return "child_in_child_seat"
    if "adult" in p or "person" in p: return "adult"
    if "empty" in p or "unoccupied" in p or "nobody" in p: return "empty"
    return "unknown"


def score(qtype, gt, pred):
    p = pred.strip().lower()
    if qtype == "occupancy":
        return classify_occ(p) == gt
    if qtype == "count":
        m = re.search(r"-?\d+", p)
        val = int(m.group()) if m else NUMWORDS.get(p.split()[0] if p.split() else "", None)
        return val == int(gt)
    if qtype == "yesno":
        yes = "yes" in p or "there is" in p or "present" in p
        no = re.search(r"\bno\b", p) or "not " in p or "none" in p
        return ("yes" if (yes and not no) else "no" if no else "unk") == gt
    if qtype == "reject":
        return bool(REFUSE.search(pred))
    return False


def run(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    proc = AutoProcessor.from_pretrained(BASE_MODEL)

    samples = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
    if args.limit:
        samples = samples[:args.limit]
    out_path = args.out or os.path.join(ROOT, "5-evaluation", f"results_{args.tag}.jsonl")
    fh = open(out_path, "w", encoding="utf-8")
    from collections import defaultdict
    hit, tot = defaultdict(int), defaultdict(int)
    for i, s in enumerate(samples, 1):
        img = os.path.join(args.image_root, s["image"])
        preds = []
        for q in s["questions"]:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img, "max_pixels": 262144},
                {"type": "text", "text": q["q"]}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(model.device)
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
            pred = proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            ok = score(q["type"], q["gt"], pred)
            hit[q["type"]] += ok; tot[q["type"]] += 1
            hit["ALL"] += ok; tot["ALL"] += 1
            preds.append({"id": q["id"], "type": q["type"], "gt": q["gt"], "pred": pred, "ok": ok})
        fh.write(json.dumps({"sample_id": s["sample_id"], "preds": preds}, ensure_ascii=False) + "\n")
        if i % 20 == 0:
            print(f"  {i}/{len(samples)} ...")
    fh.close()
    print(f"\n=== {args.tag} accuracy ({out_path}) ===")
    for k in ("occupancy", "count", "yesno", "reject", "ALL"):
        if tot[k]:
            print(f"  {k:10s} {hit[k]}/{tot[k]} = {hit[k]/tot[k]*100:.1f}%")


def compare(a, b):
    def acc(path):
        from collections import defaultdict
        hit, tot = defaultdict(int), defaultdict(int)
        for l in open(path, encoding="utf-8"):
            for p in json.loads(l)["preds"]:
                hit[p["type"]] += p["ok"]; tot[p["type"]] += 1
                hit["ALL"] += p["ok"]; tot["ALL"] += 1
        return {k: hit[k] / tot[k] * 100 for k in tot}, tot
    ra, ta = acc(a); rb, tb = acc(b)
    print(f"{'metric':12s} {'base':>8s} {'lora':>8s} {'Δ':>8s}")
    for k in ("occupancy", "count", "yesno", "reject", "ALL"):
        if k in ta:
            print(f"{k:12s} {ra.get(k,0):7.1f}% {rb.get(k,0):7.1f}% {rb.get(k,0)-ra.get(k,0):+7.1f}")


def rescore(path):
    """Re-score occupancy from a saved results file at FINE (6-way) and COARSE (safety) granularity,
    plus the safety-critical missed-occupant / hallucinated-occupant rates. No GPU / no re-inference."""
    import collections
    fine_ok = fine_tot = coarse_ok = 0
    missed = hall = 0
    conf = collections.Counter()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        for p in r.get("preds", []):
            if p.get("type") != "occupancy":
                continue
            gt = p.get("gt")
            predcat = classify_occ((p.get("pred") or "").lower())
            fine_tot += 1
            fine_ok += (predcat == gt)
            gc, pc = COARSE.get(gt, "?"), COARSE.get(predcat, "unknown")
            coarse_ok += (gc == pc)
            if gc in OCCUPANT and pc not in OCCUPANT:
                missed += 1
            if gc not in OCCUPANT and pc in OCCUPANT:
                hall += 1
            if gc != pc:
                conf[(gc, pc)] += 1
    print(f"\n=== occupancy re-score: {path}  (n={fine_tot}) ===")
    print(f"  FINE   (exact 6-way)      {fine_ok}/{fine_tot} = {fine_ok/fine_tot*100:.1f}%")
    print(f"  COARSE (safety grouping)  {coarse_ok}/{fine_tot} = {coarse_ok/fine_tot*100:.1f}%")
    print(f"  SAFETY: missed occupant   {missed} ({missed/fine_tot*100:.1f}%)   "
          f"hallucinated occupant {hall} ({hall/fine_tot*100:.1f}%)")
    print("  top coarse confusions (gt->pred):")
    for (g, p), c in conf.most_common(8):
        print(f"     {c:4d}  {g} -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default=None, help="re-score occupancy (fine/coarse/safety) from a results jsonl")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--image-root", default="/root/autodl-tmp/.autodl/Incabin")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("BASE", "LORA"))
    args = ap.parse_args()
    if args.rescore:
        rescore(args.rescore)
    elif args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
