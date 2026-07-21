#!/usr/bin/env python3
"""
build_driveact_cot_qa_llm.py — LLM generation for Drive&Act (driver-activity) GT facts.

Same 3-stage pipeline as SVIRO (generate -> cross-check -> validate), reusing llm_client and the
robust helpers, but with an ACTION-grounded prompt/validator. Balanced subsample (cap per coarse
activity) keeps the set comparable to SVIRO and controls cost.

RUN ON THE AUTODL INSTANCE.
Usage:  python scripts/build_driveact_cot_qa_llm.py --cap 120 --workers 5 --resume \
          --out 2-generation/raw_outputs/driveact_full_raw.jsonl
"""
import argparse, datetime, json, os, random, re, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sviro_cot_qa_llm as S  # reuse parse_json, call_retry, _norm_caps, llm_client

ROOT = S.ROOT
GT = os.path.join(ROOT, "1-gt_facts", "driveact", "driveact_a_column_co_driver_rgb_gt_facts.jsonl")
GT_REF = "1-gt_facts/driveact/driveact_a_column_co_driver_rgb_gt_facts.jsonl"
OUT = os.path.join(ROOT, "2-generation", "raw_outputs", "driveact_full_raw.jsonl")
PROMPT_VERSION = "driveact_v1"

SUPPORT = {"general_activity": "gt_supported", "seat_belt": "gt_supported",
           "door_status": "gt_supported", "out_of_position_candidate": "candidate"}
CANDIDATE = {"out_of_position_candidate"}
# forbidden: person attributes / counting others (not in single-view activity GT)
FORBIDDEN = re.compile(r"\b(gender|male|female|man|woman|age|years old|caucasian|asian|african|"
                       r"hispanic|latino|ethnicit\w*|emotion|happy|sad|angry|race)\b", re.I)
SAFE_NEG = re.compile(r"candidate|cannot|not confirm|needs? review|manual|vlm review|not provided|"
                      r"unable|is not|does not|not shown", re.I)

GEN_SYSTEM = """You are an in-cabin DRIVER-MONITORING annotator producing GROUNDED training data from the Drive&Act dataset.
ACTIVITY FACTS are the ONLY source of truth for what the person is doing.

Absolute rules:
1. Use ONLY the ACTIVITY FACTS: the coarse activity (midlevel.activity) and, if present, the fine-grained
   object interaction (objectlevel: activity + object + location). Do not invent other actions or objects.
2. NEVER claim personal identity, age, gender, emotion, ethnicity/appearance, or the number of other passengers,
   nor any activity not in ACTIVITY FACTS. If asked, refuse.
3. Single co-driver view — do NOT describe other seats' occupancy or count people.
4. suited_case "out_of_position_candidate" is a CANDIDATE only: phrase any out-of-position note as a candidate
   needing review, never a confirmed unsafe state.
5. If a fact is not in ACTIVITY FACTS, it is "not provided by GT". Never guess.
6. Caption: think State/Scene (what the person is doing: coarse + fine object/location) -> Risk (distraction or
   safety implication grounded in the activity, e.g. attention/hands away from driving) -> Decision (conservative
   monitoring action: flag distraction / remind / no action if compliant). Emit caption_prose as label-free prose.
7. EXACTLY 5 QA in this order: (1) Recognition — the coarse activity; (2) Recognition — the fine-grained object
   interaction (object + location) if objectlevel present, else another activity-recognition question;
   (3) Reasoning — the monitoring risk/implication grounded in the activity; (4) Decision — conservative action;
   (5) Reject — refuse an unsupported fact (identity/age/emotion/how-many-passengers).
   Each QA: question, answer, reason, capability (Recognition|Reasoning|Decision|Reject), use_case, gt_evidence
   (dotted refs, e.g. activity_facts.midlevel.activity=..., activity_facts.objectlevel.object=...).
8. use_case = the row's suited_case (general_activity | seat_belt | door_status | out_of_position_candidate),
   or "unsupported_reject" for the Reject item.
Output ONE strict JSON object, no prose outside JSON."""

GEN_USER = """ACTIVITY FACTS (authoritative JSON):
{gt}

suited_case: {sc} (support_level: {lvl})

Return ONE JSON object exactly:
{{"caption":{{"state_scene":"","risk":"","decision":""}},
  "caption_prose":"",
  "qa":[{{"question":"","answer":"","reason":"","capability":"","use_case":"","gt_evidence":[""]}}],
  "limitations_used":[""]}}"""

CHECK_SYSTEM = """You are a STRICT, INDEPENDENT fact-checker for driver-activity annotations.
Given ACTIVITY_FACTS and a DRAFT, return a corrected copy with the SAME JSON schema.
STRUCTURE: keep EVERY qa item (same count, order, capability, use_case); never delete/add/reorder; keep the Reject.
Correct only the TEXT that: contradicts the coarse/fine activity, object, or location; claims identity/age/gender/
emotion/appearance or counts other passengers; or states out_of_position as confirmed (must read as candidate).
Keep every gt_evidence. Output only the corrected JSON object."""
CHECK_USER = "Return corrected JSON with the same schema."


def load():
    return [json.loads(l) for l in open(GT, encoding="utf-8") if l.strip()]


def subsample(rows, cap, seed):
    rnd = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["activity_facts"]["midlevel"]["activity"]].append(r)
    out = []
    for _a, rs in by.items():
        rnd.shuffle(rs); out.extend(rs[:cap])
    rnd.shuffle(out)
    return out


def validate(row, rec):
    issues = []
    caps = [q.get("capability") for q in rec.get("qa", [])]
    if caps.count("Reject") != 1:
        issues.append(f"reject_count={caps.count('Reject')}")
    for n in ("Recognition", "Reasoning", "Decision"):
        if n not in caps:
            issues.append("missing_" + n)
    act = row["activity_facts"]["midlevel"]["activity"]
    sc = (row.get("suited_cases") or ["general_activity"])[0]
    for q in rec.get("qa", []):
        ans = q.get("answer") or ""
        low = ((q.get("question") or "") + " " + ans).lower()
        if q.get("capability") != "Reject" and FORBIDDEN.search(low):
            issues.append("forbidden_attribute")
        for ev in q.get("gt_evidence", []):
            m = re.match(r"activity_facts\.midlevel\.activity=(.+)", ev)
            if m and m.group(1).strip() != act:
                issues.append("bad_evidence")
        if sc in CANDIDATE and q.get("use_case") in CANDIDATE and q.get("capability") in ("Recognition", "Decision"):
            if not SAFE_NEG.search(ans):
                issues.append("candidate_not_flagged")
    return {"passed": not issues, "issues": sorted(set(issues))}


def process_row(row, models):
    gt = json.dumps(row, ensure_ascii=False)
    sc = (row.get("suited_cases") or ["general_activity"])[0]
    rec = {"sample_id": row["sample_id"], "prompt_version": PROMPT_VERSION, "models": models,
           "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "image": row["image"]["path"], "gt_facts_ref": GT_REF, "suited_cases": row.get("suited_cases", [])}
    try:
        draft = S.call_retry(S.llm_client.generate, GEN_SYSTEM,
                             GEN_USER.format(gt=gt, sc=sc, lvl=SUPPORT.get(sc, "gt_supported")))
        final = S.call_retry(S.llm_client.crosscheck, gt, json.dumps(draft, ensure_ascii=False),
                             CHECK_SYSTEM, CHECK_USER)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec
    S._norm_caps(draft); S._norm_caps(final)
    fcaps = [q.get("capability") for q in final.get("qa", [])]
    if (len(final.get("qa", [])) < len(draft.get("qa", []))
            or not (all(x in fcaps for x in ("Recognition", "Reasoning", "Decision")) and fcaps.count("Reject") == 1)):
        final["qa"] = draft.get("qa", [])
        rec["crosscheck_qa_fallback"] = True
    for k in ("caption", "caption_prose", "limitations_used"):
        if not final.get(k):
            final[k] = draft.get(k)
    rec.update({k: final.get(k) for k in ("caption", "caption_prose", "qa", "limitations_used")})
    rec["_draft"] = draft
    rec["qc"] = validate(row, rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=120, help="max samples per coarse activity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    rows = subsample(load(), args.cap, args.seed)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done_ids = set()
    if args.resume and os.path.exists(out_path):
        good = {}
        for line in open(out_path, encoding="utf-8"):
            try:
                o = json.loads(line); sid = o.get("sample_id")
            except Exception:  # noqa: BLE001
                continue
            if sid and not o.get("error"):
                good[sid] = json.dumps(o, ensure_ascii=False)
        done_ids = set(good)
        rows = [r for r in rows if r["sample_id"] not in done_ids]
        with open(out_path, "w", encoding="utf-8") as fh0:
            for l in good.values():
                fh0.write(l + "\n")
        print(f"resume: {len(done_ids)} good kept, {len(rows)} to (re)generate")
    models = {"gen": os.environ["GEN_MODEL"], "check": os.environ["CHECK_MODEL"]}
    n_total = len(rows); done = [0]; wlock = threading.Lock()
    fh = open(out_path, "a" if args.resume else "w", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_row, r, models) for r in rows]
            for fut in as_completed(futs):
                rec = fut.result()
                with wlock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                    done[0] += 1
                    tag = rec.get("error") or ("PASS" if rec.get("qc", {}).get("passed") else rec.get("qc", {}).get("issues"))
                    print(f"[{done[0]}/{n_total}] {rec['sample_id']} qc={tag}")
    finally:
        fh.close()
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    n_ok = sum(1 for r in recs if r.get("qc", {}).get("passed"))
    n_err = sum(1 for r in recs if r.get("error"))
    print(f"\ntotal in {out_path}: {len(recs)}  qc_passed={n_ok}  errors={n_err}")


if __name__ == "__main__":
    main()
