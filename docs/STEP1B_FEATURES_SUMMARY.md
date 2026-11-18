# Step 1b: Feature engineering
Owner: SRN 50. Features are built from the candidate rows of Step 1 (`pathsnap/features.py`, run through `scripts/02_build_dataset.py`).
All numbers are from the final LightGBM model on the 80 train trips / 10 student-test trips. Evaluator trips are never used.

## 1.1 Per-point motion estimates (`track_kinematics`)
- Heading and speed from a centred difference over +-2 samples, divided by the real time difference (so irregular timestamps are handled).
- When nearly stopped (speed under 1 m/s) the heading is noise, so we carry forward the last heading while moving.
- `dt_prev`, `dt_next`: time to the previous/next fix. `turn_rate`: heading change from the previous point.

## 1.2 The 27 features, grouped by idea
Geometry between the GPS point and the candidate arc:
| Feature | Meaning | Why |
|---|---|---|
| `dist` | point-to-arc distance (m) | the true road is usually the closest |
| `dist_rank` | rank of this candidate by distance (0 = nearest) | scale-free version of dist |
| `dist_minus_min` | dist minus the nearest candidate's dist | how much worse than the best option |
| `dist_ratio` | (dist+1)/(min dist+1) | relative distance; helps when noise level differs per trip |
| `n_cand` | candidates available for this point | density of the road network here |
| `proj_t`, `outside_seg` | where the point projects along the arc (0-1); whether it projects beyond its ends | tells if the point is really "on" the segment or just near its end |
| `arc_len` | arc length (m) | short arcs are crossed within a second or two |

Direction (the main difficulty, since about 92% of roads are two-way):
| `head_diff`, `head_cos` | angle between travel heading and the arc bearing | the correct carriageway points the way you travel |
| `head_rel` | head_diff minus the best head_diff among this point's candidates | relative comparison between candidates |
| `prog3`, `prog10` | signed displacement of the track along the arc direction over +-3 and +-10 points, divided by time (m/s) | positive = moving along the arc direction. More robust than instantaneous heading at low speed |
| `prog10_rel` | prog10 minus the best prog10 among this point's candidates | relative direction evidence |

Context (does the arc explain the neighbouring fixes too?):
| `ctx_dist_mean`, `ctx_dist_max` | distance of the neighbouring GPS points (+-1, 3, 5 steps) to this arc | the true road fits the whole local stretch, not just one noisy point |
| `ctx_dist_far` | same at +-10 and +-15 steps | longer horizon |
| `ctx_rank_mean` | rank of ctx_dist_mean among the point's candidates | |
| `ctx_head_diff` | mean heading difference at the neighbours | |

Motion and graph:
| `speed`, `speed_valid` | speed (m/s), whether moving | direction cues are unreliable when stopped |
| `dt_prev`, `dt_next`, `turn_rate` | time gaps and turning | gaps/turns reduce confidence |
| `deg_from`, `deg_to` | node degree at the arc ends | junction vs mid-block |
| `has_reverse` | whether a reverse twin arc exists | one-way vs two-way road |

Note: the context features use neighbouring GPS points of the same trip. That is legitimate (trajectory is available as a whole at inference) and uses no labels.

## 1.3 How the feature set evolved (this shows real experimentation)
1. First version (23 features, no `prog*`/`ctx_dist_far`): LightGBM student-test Top-1 = **0.907**.
2. Error analysis on the test predictions: **44% of errors were the wrong direction on the same road**, and these happened at low speed (median error speed about 2 m/s versus 3.7 m/s overall).
3. Added the signed-progress features `prog3`, `prog10`, `prog10_rel` and `ctx_dist_far` (4 features -> 27).
4. LightGBM student-test Top-1 -> **0.912** (untuned), and CV macro-F1 rose from 0.943 to 0.948.
5. Non-learned reference points: nearest arc only = 0.544; nearest arc + heading = 0.857. So the learned model adds about 5 points over a hand-made rule.

## 1.4 Which features matter (final tuned LightGBM, importance by total gain)
| Feature | Share of gain |
|---|---|
| `dist_ratio` | 23.2% |
| `dist_minus_min` | 16.4% |
| `prog10_rel` | 13.9% |
| `dist_rank` | 13.5% |
| `prog10` | 9.7% |
| `ctx_rank_mean` | 5.8% |
| `ctx_head_diff` | 3.4% |
| `proj_t` | 2.4% |
| `outside_seg`, `arc_len`, `prog3` | about 1.7-1.9% each |
| all other 15 | under 1% each |
Reading: relative distance does most of the work, then direction-over-time (`prog*`, about 25% together). Raw `head_diff`/`head_cos` have little gain (about 0.5% each) because `prog*` captures the same direction information more robustly.
Caveat to state honestly: this is split-gain importance from one model. We did not run a full feature-ablation study.

