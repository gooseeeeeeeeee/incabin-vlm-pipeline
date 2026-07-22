"""Prompt loader — prompts live in pipeline/prompts/*.yaml as editable parameters.

Each generation script does:
    import prompts
    P = prompts.load("sviro")          # or "driveact"
    GEN_SYSTEM = P["gen_system"]; ...
Edit the YAML to change a prompt — no code change, clean and independent per stage.
"""
import os, yaml
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

def load(name):
    with open(os.path.join(_DIR, f"{name}.yaml"), encoding="utf-8") as f:
        p = yaml.safe_load(f)
    # compose nested placeholder: {use_case_help} inside gen_system
    if p.get("use_case_help") and "{use_case_help}" in p.get("gen_system", ""):
        p["gen_system"] = p["gen_system"].replace("{use_case_help}", p["use_case_help"])
    return p
