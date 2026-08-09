"""Generate per-seat-count district files for the interactive map.

Emits maps/n<NN>.json (one FeatureCollection per stop) + outline.json (shared
boundary). Stops = 120..400 step 10, plus 215 (kept for the legacy preset).
Pure local computation — no network, no API/LLM tokens. Reuses build() from
build_geojson5 unchanged.

Usage:  venv/bin/python build_maps.py
"""
import os, json, time
import build_geojson5 as B

STEP = 5
STOPS = list(range(120, 401, STEP))
if 215 not in STOPS:
    STOPS.append(215)
STOPS = sorted(set(STOPS))

os.makedirs("maps", exist_ok=True)

# shared boundary once
json.dump(B.outline_ll(), open("outline.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("wrote outline.json")

total = 0
for n in STOPS:
    t = time.time()
    fc, ov = B.build(n)
    path = f"maps/n{n}.json"
    json.dump(fc, open(path, "w"), ensure_ascii=False, separators=(",", ":"))
    sz = os.path.getsize(path)
    total += sz
    print(f"n={n:3d}: {len(fc['features']):3d} feats  {sz/1024:6.1f} KB  overlap {ov:+.3f}%  {time.time()-t:4.1f}s")

print(f"done: {len(STOPS)} stops, {total/1024/1024:.1f} MB total")
