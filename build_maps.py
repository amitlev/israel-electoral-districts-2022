"""Generate the per-seat-count district files for the interactive map.

For each election in elections.json emits
    maps/<election>/n<N>.json   one FeatureCollection per seat count (stop)
    maps/<election>/meta.json   data statistics + seat tallies per stop (for the UI and for checks)
plus outline.json (shared boundary). Stops = 120..400 step 5, plus 215.

Pure local computation — no network, no API/LLM tokens. Each stop is seeded on
its own (see build_geojson5.build), so the output does not depend on build order
and stops are built in parallel.

Usage:  venv/bin/python build_maps.py            # every election
        venv/bin/python build_maps.py 2026       # just one
"""
import os, sys, json, time
from collections import Counter
from multiprocessing import Pool
import engine as E
import build_geojson5 as B

STEP = 5
STOPS = sorted(set(range(120, 401, STEP)) | {215})


def _init(eid):
    B.init(eid)


def _one(n):
    t = time.time()
    fc, ov = B.build(n)
    out = os.path.join(E.ROOT, "maps", B.data["election"]["id"], f"n{n}.json")
    json.dump(fc, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
    props = [f["properties"] for f in fc["features"]]
    return dict(n=n, feats=len(props), size=os.path.getsize(out), overlap=ov, secs=time.time() - t,
                parties=dict(Counter(p["wp"] for p in props)), blocs=dict(Counter(p["wb"] for p in props)))


def build_election(eid, procs):
    el = E.election(eid)
    os.makedirs(os.path.join(E.ROOT, "maps", eid), exist_ok=True)
    stats = E.load(eid)["stats"]
    print(f"== {eid}: {el['title']} ==")
    print(f"   valid {stats['valid']:,} | located {stats['located']:,} ({100*stats['located']/stats['valid']:.1f}%)"
          f" | double envelopes {stats['envelopes']:,} | other unlocated {stats['unlocated_other']:,}")
    with Pool(procs, initializer=_init, initargs=(eid,)) as pool:
        res = sorted(pool.imap_unordered(_one, STOPS), key=lambda r: r["n"])
    for r in res:
        if r["feats"] != r["n"]:
            raise SystemExit(f"n={r['n']}: built {r['feats']} districts")
        if abs(r["overlap"]) > 0.01:
            raise SystemExit(f"n={r['n']}: district overlap {r['overlap']:.3f}%")
    meta = dict(id=eid, title=el["title"], knesset=el["knesset"], stops=STOPS, stats=stats,
                seats={str(r["n"]): dict(parties=r["parties"], blocs=r["blocs"]) for r in res})
    json.dump(meta, open(os.path.join(E.ROOT, "maps", eid, "meta.json"), "w"), ensure_ascii=False, indent=1)
    total = sum(r["size"] for r in res)
    for n in (120, 215):
        r = next(x for x in res if x["n"] == n)
        print(f"   n={n} blocs:", "  ".join(f"{b} {c}" for b, c in sorted(r["blocs"].items(), key=lambda x: -x[1])))
    print(f"   {len(res)} stops, {total/1024/1024:.1f} MB, slowest stop {max(r['secs'] for r in res):.0f}s")


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(E.CONF["elections"], reverse=True)
    procs = max(1, min(8, (os.cpu_count() or 2) - 1))
    json.dump(B.outline_ll(), open(os.path.join(E.ROOT, "outline.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    t0 = time.time()
    for eid in ids:
        build_election(eid, procs)
    print(f"done in {time.time()-t0:.0f}s ({procs} processes)")
