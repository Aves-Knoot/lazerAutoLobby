#!/usr/bin/env python3
"""Plant a known farm signal in synthetic data, then check the model finds it.

Critically, the fake world includes a CONFOUND: some maps are simply very
popular (huge passcount) without overpaying. A naive frequency ranking picks
those up. The passcount offset should reject them.
"""
import numpy as np, pandas as pd, time
from osu_farm.core import connect

rng = np.random.default_rng(11)
DB = "test.db"
import os
if os.path.exists(DB): os.remove(DB)
conn = connect(DB)

N_MAPS, N_PLAYERS = 900, 1500
FARM = set(range(0, 40))        # overpays  -> model should rank these top
POPULAR = set(range(40, 80))    # just popular, fairly paid -> must NOT rank

maps = []
for i in range(N_MAPS):
    sr = float(np.clip(rng.normal(6.2, 1.1), 2.5, 9.5))
    length = int(np.clip(rng.normal(110, 45), 25, 400))
    base_pass = np.exp(rng.normal(9.0, 1.1))
    if i in POPULAR: base_pass *= 40          # the confound
    if i in FARM:    base_pass *= rng.uniform(0.7, 1.6)
    maps.append(dict(
        beatmap_id=i, sr=sr, hit_length=length,
        passcount=int(base_pass), playcount=int(base_pass * rng.uniform(3, 9)),
        max_combo=int(length * 6), ar=9.0, od=9.0, cs=4.0,
        count_circles=int(length * 4), count_sliders=int(length * 2),
    ))
mdf = pd.DataFrame(maps)

conn.executemany(
    "INSERT INTO beatmaps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    [(m["beatmap_id"], m["beatmap_id"], f"song{m['beatmap_id']}", "artist", "Insane",
      "ranked", m["sr"], m["cs"], m["ar"], m["od"], 5.0, 180.0,
      m["hit_length"] + 20, m["hit_length"], m["max_combo"], m["count_circles"],
      m["count_sliders"], 1, m["playcount"], m["passcount"], "2024-01-01", time.time())
     for m in maps])

players, scores, sid = [], [], 0
for u in range(N_PLAYERS):
    ppv = float(np.clip(rng.normal(4800, 900), 2000, 9000))
    players.append((u, f"p{u}", "US", u + 1, ppv, 50000, "2026-08-01", time.time()))
    skill = ppv / 1000.0

    # Selection: farm maps get chosen more often, popular maps get chosen more
    # often too (that's the confound), everything else uniform.
    w = np.ones(N_MAPS)
    w[list(FARM)] *= 9.0
    w[list(POPULAR)] *= 9.0
    fit = np.exp(-((mdf.sr.values - (skill * 0.85)) ** 2) / 1.5)
    w = w * fit
    chosen = rng.choice(N_MAPS, size=100, replace=False, p=w / w.sum())

    for pos, bid in enumerate(chosen, 1):
        m = maps[bid]
        acc = float(np.clip(rng.normal(97.5, 1.4), 90, 100))
        misses = int(rng.poisson(0.6))
        combo_ratio = float(np.clip(rng.beta(9, 1.2), 0.2, 1.0))
        # "True" pp function of the fake world.
        pp = (12.0 * m["sr"] ** 2.1
              * (acc / 100) ** 9
              * (0.55 + 0.45 * combo_ratio)
              * np.exp(-0.09 * misses)
              * (1 + 0.0012 * m["hit_length"]))
        if bid in FARM:
            pp *= 1.45                     # the planted overpayment
        pp *= rng.lognormal(0, 0.05)
        date = "2026-08-01" if rng.random() < 0.6 else "2026-03-01"
        scores.append((sid, u, bid, pos, float(pp), acc,
                       int(combo_ratio * m["max_combo"]), misses,
                       0, 0, 0, "NM", "S", date))
        sid += 1

conn.executemany("INSERT INTO players VALUES (?,?,?,?,?,?,?,?)", players)
conn.executemany("INSERT INTO scores VALUES (" + ",".join("?" * 14) + ")", scores)
conn.commit(); conn.close()
print(f"synthetic world: {N_PLAYERS} players, {len(scores):,} scores, {N_MAPS} maps")
print(f"  planted farm maps: {len(FARM)}   |   popularity confounds: {len(POPULAR)}\n")

# ------------------------------------------------------------------ evaluate
from osu_farm.analyze import farm_report, score_table, overrepresentation
from osu_farm.core import connect as c2

rep = farm_report(DB, top_n=120, post_rework_only=False)

def hit_rate(ids, truth): return sum(1 for i in ids if i in truth) / max(len(truth), 1)

top40 = rep.beatmap_id.head(40).tolist()
print(f"\n=== RESULTS ===")
print(f"planted farm maps recovered in top 40 : {hit_rate(top40, FARM):.0%}")
print(f"popularity confounds leaked into top 40: {sum(1 for i in top40 if i in POPULAR)}")

# Naive baseline: rank by raw appearance count, no passcount control.
conn = c2(DB)
naive = (pd.read_sql_query("SELECT beatmap_id, COUNT(*) n FROM scores "
                           "GROUP BY beatmap_id ORDER BY n DESC", conn)
         .beatmap_id.head(40).tolist())
conn.close()
print(f"\nnaive frequency baseline:")
print(f"  farm maps recovered  : {hit_rate(naive, FARM):.0%}")
print(f"  confounds leaked     : {sum(1 for i in naive if i in POPULAR)}")
print(f"\nquadrant counts:\n{rep.quadrant.value_counts().to_string()}")
