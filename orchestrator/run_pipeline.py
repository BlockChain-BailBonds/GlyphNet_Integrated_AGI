#!/usr/bin/env python3
import os, json, datetime, subprocess

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
OUTPUT_DIR = os.path.join(ROOT, "output")
CODEX_PATH = os.path.join(ROOT, "codex/runtime_registry.json")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log(msg):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}")

def run_module(path, *args):
    if not os.path.exists(path):
        log(f"[WARN] Missing {path}")
        return None
    log(f"→ Running {path}")
    return subprocess.run(["python3", path, *args], cwd=ROOT)

def get_latest_output():
    f = os.path.join(OUTPUT_DIR, "rehydrated_entry.json")
    if os.path.exists(f):
        return json.load(open(f))
    return None

def register_to_codex(entry):
    ensure_dir(os.path.dirname(CODEX_PATH))
    data = {"entries": []}
    if os.path.exists(CODEX_PATH):
        data = json.load(open(CODEX_PATH))
    entry["registered_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    data["entries"].append(entry)
    with open(CODEX_PATH, "w") as f:
        json.dump(data, f, indent=2)
    log(f"[✓] Registered to Codex ({len(data['entries'])} total)")

def main():
    log("=== GlyphNet Integrated AGI Orchestrator ===")
    ensure_dir(OUTPUT_DIR)

    demo = os.path.join(ROOT, "examples/full_chain_demo.py")
    run_module(demo)

    entry = get_latest_output()
    if entry:
        register_to_codex(entry)
    else:
        log("[WARN] No output to register")

    emotion = os.path.join(ROOT, "emotion/Aletheis_Empatheos/main.py")
    if os.path.exists(emotion):
        run_module(emotion)

    log("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()

    # 4. Auto-commit Codex ledger and any new outputs
    log("Committing latest Codex ledger and outputs...")
    try:
        commit_message = f"Auto-commit GlyphNet run {datetime.datetime.utcnow().isoformat()}Z"
        subprocess.run(["git", "add", "-A"], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=ROOT)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
        log("[✓] Auto-commit and push complete.")
    except Exception as e:
        log(f"[WARN] Git commit/push failed: {e}")
