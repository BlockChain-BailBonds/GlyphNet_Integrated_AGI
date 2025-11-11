#!/usr/bin/env python3
import json, os, datetime
src = "output/rehydrated_entry.json"
dst = "codex/runtime_registry.json"

if not os.path.exists(src):
    raise FileNotFoundError(src)

entry = json.load(open(src))
entry["registered_at"] = datetime.datetime.now(datetime.UTC).isoformat()

if os.path.exists(dst):
    data = json.load(open(dst))
else:
    data = {"entries": []}

data["entries"].append(entry)
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    json.dump(data, f, indent=2)

print(f"[✓] Registered {src} -> {dst} ({len(data['entries'])} total)")
