# osu! farm map finder

Finds beatmaps that pay more pp than their difficulty justifies, controlling for
popularity, and biased toward players who are actively adding to their top 100.

## Setup

```bash
pip install httpx numpy pandas statsmodels scikit-learn scipy rosu-pp-py

# https://osu.ppy.sh/home/account/edit -> OAuth -> New OAuth Application
export OSU_CLIENT_ID=12345
export OSU_CLIENT_SECRET=your_secret
```

## Running

```bash
python run.py collect --players 4000    # deep sample across all skill levels
python run.py coverage                  # check players/scores per pp bracket
python run.py maps                      # .osu files + strain features
python run.py report --top 200 --out farm_maps.csv
python run.py report --post-rework      # current-meta-only view
```

### Sampling depth

The GLOBAL rankings endpoint caps at page 200 -- 10,000 players, all very
high level. That pool cannot populate the lower pp brackets at all, so a farm
list built from it is only valid for the top of the game.

`collect` therefore defaults to walking COUNTRY rankings, which get their own
200-page allowance each. It fills per-bracket quotas (`DEFAULT_QUOTAS` in
collect.py) and stops sampling a country once its pp drops below `--pp-floor`.

```bash
python run.py collect --players 4000 --pp-floor 800   # reach further down
python run.py collect --global-only --pages 200       # old behaviour, top 10k
```

`collect_scores` also interleaves players across brackets, so a run you cut
short still covers every skill tier instead of only the strongest.

Every stage is resumable. Re-running `collect` skips players fetched in the last
14 days and beatmaps fetched in the last 30.

## Rate limiting

Defaults to 60 requests/minute against a documented ceiling of 60. Handles
429 with `Retry-After`, retries 5xx with backoff, and refreshes the OAuth token
automatically. Raise with `--rpm` if you're impatient, but the ceiling is a limit set by the devs who *will* remove your oauth key if you abuse it

## The rework problem, and how this works around it

The July 2026 rework replaced AR/HD bonuses with a density-based Reading skill,
made speed a harmonic sum, and nerfed repetitive acute-angle jumps. **No released
pp calculator implements it.** `rosu-pp-py` 4.0.2 is the newest on PyPI and
returns `None` for `reading`, `mechanical_difficulty`, and `consistency_factor`
on osu!standard -- it's still the old aim/speed/flashlight split.

So instead of simulating pp with a stale algorithm, `fit_pp_surrogate()` learns
the current pp function from live scores:

```
log(pp) ~ f(SR, accuracy, misses, combo_ratio, length, AR, OD, CS, mods)
```

Because osu! reprocessed every score during the rollout, the pp values the API
returns *are* post-rework. The fitted model is therefore the current algorithm,
recovered empirically. A map's mean residual is its **pp efficiency**: how much
it overpays relative to its own stats. On synthetic data the fit reaches
R² ≈ 0.98 on held-out scores.

Strain-shape features are computed separately with `rosu-pp` and are still valid,
because strain *shape* (spikiness, where the hard part sits) is a property of the
map rather than of the pp formula. The `acc_gradient` and `forgiveness_ratio`
fields in `pp_surface()` do depend on the formula -- treat them as a rough prior
until rosu-pp updates, then swap them in as primary signals.

## The three layers

**1. Mechanistic efficiency** (`features.py`) -- surrogate residual per map, plus
strain features: `aim_strain_gini` (is difficulty spiky?), `hardest_section_pos`
(0 = filter at the start, cheap retries; 1 = filter at the end, every failed
attempt costs the whole song), `late_difficulty_mass`, `slider_ratio`.

**2. Overrepresentation** (`analyze.py`) -- negative binomial rate model, fit
per pp bracket:

```
log E[appearances_m] = log(passcount_m) + log(n_players) + f(SR, length, mods) + eps_m
```

The offset is the popularity control. It asks whether a map appears in more top
100s than its *exposure* justifies, not whether it appears a lot. Dispersion is
estimated via an auxiliary Poisson fit. `eps_m` is empirical-Bayes shrunk with
pseudo-counts so low-passcount maps don't dominate.

**3. HITS farmer weighting** -- bipartite player↔map power iteration, seeded with
layer 1's efficiency scores to break the circularity between "who farms" and
"what's farm". Separates the farmer cluster from tournament and skill players
without hand-labeling anyone.

Final score is `0.5·z(efficiency) + 0.3·z(overrep) + 0.2·z(farm_authority)`. The
mechanistic axis leads deliberately -- it's the half that survives a meta shift.

## Quadrants

Each map is labeled by where it lands on the two axes:

| quadrant | meaning |
|---|---|
| `undiscovered` | efficient, not yet widely farmed — **the target** |
| `known_farm` | efficient and already saturated |
| `stale_farm` | overrepresented but no longer efficient — rework casualty |
| `ignore` | no signal either way |

## Activity weighting

Player weight comes from score dates, which arrive free with the top-100 pull —
no extra API calls. Recency-decayed with a 90-day half-life, then saturated
(`1 - exp(-mass/10)`). This measures pp-motivated activity rather than raw
playtime, which is closer to what you actually want to weight by. osu! also drops
players from the rankings after ~90 days idle, so the pool starts partly filtered.

## Validation

`python test_synthetic.py` builds a fake world with 40 planted farm maps and 40
pure-popularity confounds (40× passcount, fairly paid), then checks recovery:

```
planted farm maps recovered in top 40 : 78%
popularity confounds leaked           : 1

naive frequency baseline:
  farm maps recovered                 : 35%
  confounds leaked                    : 16
```

Run this after any change to the scoring weights.

## Known gaps

- `angle_entropy` is stubbed. Post-rework it matters a lot (repetitive acute
  angles are now nerfed, unpredictable ones buffed), but it needs hitobject
  positions parsed from the .osu file and angle sequences computed directly —
  rosu-pp doesn't expose it. This is the highest-value thing to add next.
- Mod buckets are collapsed to the pp-relevant set. Each `(map, mods)` pair is
  really a separate asset; the current model uses the dominant mod per map.
- No tournament-pool or Daily Challenge exclusion list yet. Both inflate exposure
  without being farm. Mappool data is public — worth maintaining a blocklist.
- Achievability (P(player of skill s hits accuracy a)) isn't modeled yet. The
  `estimated_unstable_rate` field on `PerformanceAttributes` is the hook for it.
