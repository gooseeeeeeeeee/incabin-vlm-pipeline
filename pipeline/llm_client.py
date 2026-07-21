"""
llm_client.py — OpenAI-compatible client for the in-cabin generation pipeline.

Provider: Vultr Serverless Inference (ONE endpoint, ONE key, many models).
RUN LOCATION: on the AutoDL instance only (needs internet). Keys come from a .env that
is gitignored AND rsync-excluded, so they never reach the synced data disk.

Roles (mirrors the exterior pipeline, adapted to in-cabin):
  GEN    : text model, drafts caption/QA JSON from injected GT Facts (no image; VISION stage
           is off for the SVIRO pilot — GT Facts are authoritative).
  CHECK  : an INDEPENDENT text model (different vendor) that strips any claim contradicting
           GT / any forbidden label / any candidate stated as confirmed.
  SAFETY : optional near-free guard model, final scan for forbidden in-cabin claims.
  VISION : optional multimodal (Omni) model, only for future real cabin data.

Install:  pip install openai python-dotenv
"""
import base64
import mimetypes
import os
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

_CLIENT = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                 timeout=45, max_retries=1)

# global throttle shared across threads — keep total request rate under the provider's limit
_MIN_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "1.0"))
_rate_lock = threading.Lock()
_last_call = [0.0]


def _throttle():
    with _rate_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _chat(model, messages, temperature, json_mode=True, max_tokens=8192):
    _throttle()
    kw = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    msg = _CLIENT.chat.completions.create(**kw).choices[0].message
    content = (msg.content or "").strip()
    if not content:  # reasoning models sometimes leave content empty and put text in reasoning_content
        content = (getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", "") or "").strip()
    return content


def generate(system_prompt, user_prompt, temperature=0.4):
    """GEN: draft caption/QA JSON from GT-injected text prompt."""
    return _chat(os.environ["GEN_MODEL"],
                 [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
                 temperature)


def crosscheck(gt_facts_json, draft_json, system_prompt, user_prompt, temperature=0.0):
    """CHECK: independent model re-reads the draft against GT; returns corrected JSON."""
    return _chat(os.environ["CHECK_MODEL"],
                 [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"{user_prompt}\n\nGT_FACTS:\n{gt_facts_json}\n\nDRAFT:\n{draft_json}"}],
                 temperature)


def safety_scan(text, temperature=0.0):
    """SAFETY (optional): flag forbidden in-cabin claims (belt/age/gender/sleep/orientation/...)."""
    model = os.environ.get("SAFETY_MODEL")
    if not model:
        return None
    sys = ("You audit in-cabin annotations for FORBIDDEN claims only. "
           "ALLOWED (never flag): the occupancy categories adult, child, infant, person, empty, "
           "empty child seat, empty infant seat, everyday object — these are ground-truth labels, not age claims. "
           "FLAG only: a specific numeric age or age percentile, gender, emotion, personal identity, "
           "seatbelt worn/not worn, pet presence, vehicle door status, sleeping/unconsciousness, "
           "ISOFIX tension/latch, or a CONFIRMED forward/rear-facing seat orientation. "
           "Output JSON {\"violations\":[\"...\"]}. Empty list if none.")
    return _chat(model, [{"role": "system", "content": sys},
                         {"role": "user", "content": text}], temperature)


def generate_with_image(image_path, system_prompt, user_prompt, temperature=0.4):
    """Optional VISION stage (future real cabin data). Uses the Omni multimodal model."""
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as fh:
        data_url = f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"
    return _chat(os.environ["VISION_MODEL"],
                 [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": [
                      {"type": "text", "text": user_prompt},
                      {"type": "image_url", "image_url": {"url": data_url}}]}],
                 temperature)


if __name__ == "__main__":
    print("endpoint:", os.environ["LLM_BASE_URL"])
    print("GEN   =", os.environ["GEN_MODEL"])
    print("CHECK =", os.environ["CHECK_MODEL"])
    print("SAFETY=", os.environ.get("SAFETY_MODEL", "(off)"))
    # tiny live ping (costs ~nothing): ask GEN to echo JSON
    try:
        out = generate("Reply only JSON.", 'Return {"ok": true}.')
        print("live GEN reply:", out.strip()[:120])
    except Exception as e:
        print("live call failed:", type(e).__name__, str(e)[:200])
