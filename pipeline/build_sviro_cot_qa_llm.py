#!/usr/bin/env python3
"""
build_sviro_cot_qa_llm.py — LLM three-stage in-cabin pilot generator.

RUN ON THE AUTODL INSTANCE (needs internet + .env). Pipeline per sample:
    GT Facts  ->  [GEN]  draft caption/QA JSON
              ->  [CHECK] independent model strips GT-contradictions / forbidden labels / over-claimed candidates
              ->  [SAFETY] near-free guard scan for forbidden in-cabin claims
              ->  [VALIDATE] deterministic local checks (no API)
              ->  raw JSONL

Implements 2-generation/prompts/v0_pilot.md + the migration-plan fixes:
  - prose caption external, structured JSON internal (G1)
  - drop `prediction`; optional `attention`
  - P1: child presence is GT-supported -> tag occupancy to a supported concept, not a candidate id
  - P2/P3/P4: context-aware reject, reject-evidence convention, real case-id mapping
  - P6: QA diversity + `reason` on every answer

Usage:  python scripts/build_sviro_cot_qa_llm.py --n 30
"""
import argparse, datetime, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client  # noqa: E402
import prompts  # noqa: E402  — prompt params from prompts/*.yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "1-gt_facts", "sviro_bmw_x5_random_train_gt_facts.jsonl")
GT_REF = "1-gt_facts/sviro_bmw_x5_random_train_gt_facts.jsonl"
OUT = os.path.join(ROOT, "2-generation", "raw_outputs", "sviro_pilot_llm_raw.jsonl")
_P = prompts.load("sviro")
PROMPT_VERSION = _P["version"]
USE_CASE_HELP  = _P["use_case_help"]
GEN_SYSTEM     = _P["gen_system"]
GEN_USER_TMPL  = _P["gen_user"]
CHECK_SYSTEM   = _P["check_system"]
CHECK_USER     = _P["check_user"]

HARD_FORBIDDEN = ["seatbelt", "seat belt", "belt", "pet", "dog", "cat", "door open", "door closed",
                  "asleep", "sleeping", "unconscious", "age ", "gender", "male", "female", "emotion"]
# orientation / ISOFIX words are allowed ONLY inside a refusal or candidate phrasing (SAFE_NEG);
# they are a violation only when ASSERTED as a confirmed fact.
ORIENTATION = re.compile(r"forward-facing|rear-facing|isofix|latched", re.I)
SAFE_NEG = re.compile(r"candidate|cannot|not confirm|not directly label|not labeled|does not|do not|"
                      r"no .*label|manual review|manual or vlm|vlm review|needs? review|review is|"
                      r"before driving|verify|manually|unable|not provided|not specify|is not", re.I)
# appearance / state inferences that are NOT in GT (replaces the flaky API safety scan)
APPEARANCE = re.compile(r"\b(caucasian|asian|african|hispanic|latino|ethnicit\w*)\b|"
                        r"appears?\s+(alert|awake|calm|tired|drowsy|asleep)|"
                        r"looks?\s+(alert|awake|tired|calm)|\b(emotion|happy|sad|angry|anxious)\b", re.I)
CANDIDATE_IDS = {"03_child_in_forward_facing_child_seat_candidates",
                 "04_rear_facing_child_seat_candidates",
                 "06_isofix_child_seat_orientation_candidates",
                 "11_out_of_position_candidates"}

# ---- allowed use_case vocabulary given to the model (P1/P4) ----






def load_rows():
    with open(GT, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def has(seats, v):
    return any(x == v for x in seats.values())


def sample_rows(rows, n):
    rnd = random.Random(0)
    idx = dict(enumerate(rows))

    def pool(pred):
        ids = [i for i, r in idx.items() if pred(r)]
        rnd.shuffle(ids)
        return ids

    def flagged(r):
        return any(p.get("pose_flags") for p in r.get("persons_summary", []))

    quotas = [
        (lambda r: all(v == "empty" for v in r["seat_states"].values()), 4),
        (lambda r: has(r["seat_states"], "adult"), 5),
        (lambda r: has(r["seat_states"], "infant_in_infant_seat"), 4),
        (lambda r: has(r["seat_states"], "child_in_child_seat"), 4),
        (lambda r: has(r["seat_states"], "everyday_object"), 4),
        (flagged, 6),
        (lambda r: len(r["suited_cases"]) >= 5, 3),
    ]
    picked, seen = [], set()
    for pred, q in quotas:
        c = 0
        for i in pool(pred):
            if i in seen:
                continue
            seen.add(i); picked.append(i); c += 1
            if c >= q or len(picked) >= n:
                break
        if len(picked) >= n:
            break
    for i in pool(lambda r: True):
        if len(picked) >= n:
            break
        if i not in seen:
            seen.add(i); picked.append(i)
    return [idx[i] for i in picked[:n]]


def cases_compact(row):
    return "\n".join(
        f"- {c['case_id']} | {c['case_name']} | {c['support_level']} | evidence={c['evidence']} | limitations={c['limitations']}"
        for c in row["suited_cases"])


def parse_json(text):
    if not text or not text.strip():
        raise ValueError("empty response")
    t = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError("no json object in response")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])  # first object; ignores trailing junk
    return obj


def call_retry(fn, *a, tries=5, **k):
    """Retry with exponential backoff + jitter; surface the raw response on failure."""
    last = None
    for attempt in range(tries):
        raw = None
        try:
            raw = fn(*a, **k)
            return parse_json(raw)
        except Exception as e:  # noqa: BLE001
            last = RuntimeError(f"{type(e).__name__}: {str(e)[:70]} | raw[:120]={(raw or '')[:120]!r}")
            time.sleep(2.0 * (2 ** attempt) + random.uniform(0, 1.5))
    raise last


_CAP_MAP = {"recognition": "Recognition", "reasoning": "Reasoning",
            "decision": "Decision", "reject": "Reject"}


def _norm_caps(obj):
    for q in obj.get("qa", []):
        c = (q.get("capability") or "").strip().lower()
        q["capability"] = _CAP_MAP.get(c, q.get("capability"))


def validate(row, rec):
    issues, seats = [], row["seat_states"]
    caps = [q.get("capability") for q in rec.get("qa", [])]
    if caps.count("Reject") != 1:
        issues.append(f"reject_count={caps.count('Reject')}")
    for need in ("Recognition", "Reasoning", "Decision"):
        if need not in caps:
            issues.append(f"missing_{need}")
    blob = ((rec.get("caption_prose") or "") + " " +
            " ".join((q.get("answer") or "") for q in rec.get("qa", []))).lower()
    for q in rec.get("qa", []):
        ans = q.get("answer") or ""
        low = ((q.get("question") or "") + " " + ans).lower()
        if q.get("capability") != "Reject":
            for f in HARD_FORBIDDEN:
                if re.search(rf"\b{re.escape(f.strip())}\b", low):
                    issues.append(f"forbidden:{f.strip()}")
            if APPEARANCE.search(low):
                issues.append("appearance_inference")
        if ORIENTATION.search(low) and not SAFE_NEG.search(ans):
            issues.append("asserted_orientation")
        for ev in q.get("gt_evidence", []):
            if ev.startswith("seat_states."):
                m = re.match(r"seat_states\.(\w+)=(.+)", ev)
                if m and seats.get(m.group(1)) != m.group(2):
                    issues.append(f"bad_evidence:{ev}")
        if q.get("use_case") in CANDIDATE_IDS and q.get("capability") in ("Recognition", "Decision"):
            if not SAFE_NEG.search(ans):
                issues.append(f"candidate_not_flagged:{q.get('use_case')}")
    return {"passed": not issues, "issues": sorted(set(issues))}


def process_row(row, models):
    gt_json = json.dumps(row, ensure_ascii=False)
    rec = {"sample_id": row["sample_id"], "prompt_version": PROMPT_VERSION,
           "models": models, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "image": row["image"]["path"], "gt_facts_ref": GT_REF,
           "suited_cases": [c["case_id"] for c in row["suited_cases"]]}
    try:
        draft = call_retry(llm_client.generate, GEN_SYSTEM,
                           GEN_USER_TMPL.format(gt=gt_json, cases=cases_compact(row)))
        final = call_retry(llm_client.crosscheck, gt_json, json.dumps(draft, ensure_ascii=False),
                           CHECK_SYSTEM, CHECK_USER)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec
    _norm_caps(draft); _norm_caps(final)
    # structure guard: cross-check must not drop/relabel qa items
    fcaps = [q.get("capability") for q in final.get("qa", [])]
    need_ok = (all(x in fcaps for x in ("Recognition", "Reasoning", "Decision"))
               and fcaps.count("Reject") == 1)
    if len(final.get("qa", [])) < len(draft.get("qa", [])) or not need_ok:
        final["qa"] = draft.get("qa", [])
        rec["crosscheck_qa_fallback"] = True
    for k in ("caption", "caption_prose", "limitations_used"):  # cross-check may null these out
        if not final.get(k):
            final[k] = draft.get(k)
    rec.update({k: final.get(k) for k in ("caption", "caption_prose", "qa", "limitations_used")})
    rec["_draft"] = draft  # keep pre-check draft for faithfulness QC
    rec["qc"] = validate(row, rec)  # deterministic checks incl. forbidden/appearance/orientation (no API)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="sample N rows (ignored if --all)")
    ap.add_argument("--all", action="store_true", help="use ALL GT rows (full generation)")
    ap.add_argument("--out", default=OUT, help="output jsonl path")
    ap.add_argument("--resume", action="store_true", help="skip sample_ids already present in --out")
    ap.add_argument("--split-file", default=None, help="3-splits/split_assignment.jsonl to restrict rows")
    ap.add_argument("--include-splits", default="train,val_dev", help="comma list of splits to generate")
    ap.add_argument("--workers", type=int, default=5, help="concurrent API workers (raise if stable, lower if 429/empty)")
    args = ap.parse_args()

    rows_all = load_rows()
    rows = rows_all if (args.all or args.n >= len(rows_all)) else sample_rows(rows_all, args.n)
    if args.split_file:  # keep frozen_test out of training data
        inc = {s.strip() for s in args.include_splits.split(",")}
        amap = {}
        for line in open(args.split_file, encoding="utf-8"):
            try:
                o = json.loads(line); amap[o["sample_id"]] = o["split"]
            except Exception:  # noqa: BLE001
                pass
        before = len(rows)
        rows = [r for r in rows if amap.get(r["sample_id"]) in inc]
        print(f"split filter {inc}: {before} -> {len(rows)} rows")
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done_ids = set()
    if args.resume and os.path.exists(out_path):
        good = {}  # keep only SUCCESSFUL records; error rows are retried
        for line in open(out_path, encoding="utf-8"):
            try:
                o = json.loads(line); sid = o.get("sample_id")
            except Exception:  # noqa: BLE001
                continue
            if sid and not o.get("error"):
                good[sid] = json.dumps(o, ensure_ascii=False)  # last good wins (dedupe)
        done_ids = set(good)
        rows = [r for r in rows if r["sample_id"] not in done_ids]
        with open(out_path, "w", encoding="utf-8") as fh0:  # rewrite: drop error rows, keep good
            for l in good.values():
                fh0.write(l + "\n")
        print(f"resume: {len(done_ids)} good kept, {len(rows)} to (re)generate")

    models = {"gen": os.environ["GEN_MODEL"], "check": os.environ["CHECK_MODEL"],
              "safety": os.environ.get("SAFETY_MODEL")}
    n_total = len(rows)
    done = [0]
    wlock = threading.Lock()
    # append when resuming (file already rewritten with good rows), else start fresh; stream to disk
    fh = open(out_path, "a" if args.resume else "w", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_row, row, models) for row in rows]
            for fut in as_completed(futs):
                rec = fut.result()
                with wlock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[0] += 1
                    tag = rec.get("error") or ("PASS" if rec.get("qc", {}).get("passed")
                                               else rec.get("qc", {}).get("issues"))
                    print(f"[{done[0]}/{n_total}] {rec['sample_id']} qc={tag}")
    finally:
        fh.close()

    recs = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    n_ok = sum(1 for r in recs if r.get("qc", {}).get("passed"))
    n_err = sum(1 for r in recs if r.get("error"))
    print(f"\ntotal in {out_path}: {len(recs)}  qc_passed={n_ok}  errors={n_err}  workers={args.workers}")


if __name__ == "__main__":
    main()
