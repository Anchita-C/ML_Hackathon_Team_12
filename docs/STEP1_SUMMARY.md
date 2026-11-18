# Step 1: Dataset exploration, preprocessing and candidate generation
Owner: SRN 50. All figures were recomputed by an independent script (`scripts/00_verify_step1.py`).

## 1. What Step 1 covers
1. Inventory of all trips and files.
2. Data-quality checks (missing values, time order, gaps, coordinates, network problems).
3. Converting coordinates to metres.
4. Building the "right answers" (which road each GPS point is on).
5. Generating candidate roads for every GPS point.
6. Plots and Folium maps.
Evaluator trips are never read. Only `train` (80 trips) and `student_test` (10 trips) are used.

## 2. Verified statistics
All numbers below were recomputed by an independent script (`scripts/00_verify_step1.py`, reads the raw files directly)
and compared with the EDA script (`01_eda.py`). They agree on points, nodes, arcs, time gaps, self-loops, duplicate arcs,
components and route breaks for **all 90 trips**.

### Inventory
| | Train | Student-test |
|---|---|---|
| Trips | 80 | 10 |
| Trips with all 4 files (track, nodes, arcs, route) | 80 | 10 |
| GPS points | 208,417 | 19,160 |
| Points per trip (median / min / max) | 1,529 / 361 / 19,313 | 1,420 / 262 / 3,958 |
| Road segments (arcs) per trip | 649 to 307,392 | 3,577 to 56,027 |

Typical trip extent is about 9.5 km. The trips are spread across the world, and each trip has its own road network
(ids are local to the trip).

### Data-quality checks
| Check | Train | Student-test |
|---|---|---|
| Missing (NaN) values | 0 | 0 |
| Timestamps not increasing | 0 | 0 |
| Longitude/latitude outside valid range | 0 | 0 |
| Route or arc ids out of range | 0 | 0 |
| Gaps over 10 s between GPS points | 590 | 27 |
| Largest gap | 153,064 s (trip 90) | 210 s |
| Average share of steps that are not exactly 1 s | 6.2% | 1.8% |
| Trips that start at t > 1 s | 16 of 90 trips overall | |
| Steps faster than 50 m/s (of 227,487 steps) | 1,541 (0.7%) | included |
| Self-loop roads | 22 | 2 |
| Duplicate roads (same from, to) | 1,775 | 128 |
| Roads that also exist in the opposite direction | 92.2% | 92.7% |
| Isolated nodes | 0 | 0 |
| Largest number of disconnected network pieces in one trip | 421 | 100 |
| Breaks in the reference route (consecutive arcs not connected) | 13 | 1 |

### Reference-route quality (labels)
- Each GPS point is matched to its road with a dynamic-programming alignment along the route (the vehicle never goes backwards).
- Typical (median) distance from a GPS point to its matched road: **4.7 m**; typical 95th percentile: 14.5 m.
- Points more than 50 m from the matched road are marked **unreliable**: **2.5% of train points, 0.07% of student-test points**.
- Two trips are the cause: **trip 33 (59% unreliable)** and **trip 54 (89%)**: the vehicle left the supplied route.
  Open `artifacts/eda/map_train_00000033_label_issue.html` to see it. Unreliable points are excluded from training.

### Candidate generation (12 nearest roads within 150 m)
- About 10.2 candidate rows per GPS point (train 2,142,302 rows; student-test 195,675 rows).
- The correct road is among the candidates for **99.88% of student-test points** (miss rate 0.12%; 0.07% on reliable labels).
- Train: miss rate 0.56% on reliable labels. 198 train points (0.1%) have no road within 150 m.
- Rows used for model training: 471,913 rows from 46,341 points (25% sample of reliable points, seed 33); 9.8% are positive.

## 3. What was done, and why
| Step | Why |
|---|---|
| Inventory and quality checks | Bad data gives bad results; also the task asks for it |
| Convert to metres with a local projection per trip | Degrees are not distances, and trips are worldwide |
| Remove self-loops from candidates, treat duplicate roads as the same road | They would split or confuse the labels |
| Build labels by route alignment | The data has no per-point road labels |
| Flag points over 50 m from the route | 50 m is far beyond normal GPS noise (typical 4.7 m): the car left the route |
| 12 candidates within 150 m | Turns "find the road among thousands" into a small ranking problem, with 99.9% coverage |
| Split by trip only | Points from one trip look alike; mixing them would inflate scores |

## 4. Honest limitations (say them yourself)
1. The labels are derived by us from the route; they are not independently verified ground truth.
2. Trips 33 and 54 contain stretches off the reference route.
3. The 12-candidate list misses the true road for 0.12% of student-test points.
4. The evaluator trips were never opened, so nothing is known about them.
