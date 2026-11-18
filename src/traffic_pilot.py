"""
Pilot run of the Week 4 test campaign: a small, fully reproducible version of the SIL campaign written in Python.

1. Traffic: a 3-lane, 1 km ring road with Indian mixed traffic (45% two-wheelers, 30% cars, 15% auto-rickshaws,
   10% buses/trucks) driven by the Intelligent Driver Model (IDM) with random acceleration noise, at four
   densities (15, 40, 70 and 100 PCU/km/lane). Scripted critical events (cut-ins and hard braking ahead of the
   ego vehicle) are injected about 40 times an hour.
2. Model under test: a statistical stand-in for the Saarthi Traffic Analytics Model (camera only). Detection
   depends on class, range, occlusion by vehicles in front and the condition; it misclassifies, mis-measures
   distance and produces false positives. Traffic state is estimated from detections; forward conflicts are
   flagged from a tracked time-to-collision (TTC).
3. Five conditions are applied to the same traffic (paired design): clear day, night, heavy rain, fog/dust,
   sensor fault (30% of the image blocked on the left, 2% dropped frames, +4% distance bias).
4. Metrics: recall/precision by class, MOTA, ID switches, distance error, density error, traffic-state macro-F1,
   conflict recall, false alerts per hour, latency (separate queueing model). 95% bootstrap intervals over runs.

The perception error model is an assumption (listed in PERCEPTION below). The pilot's purpose is to exercise
the metrics, statistics and go/no-go logic on realistic traffic, and to show how density and occlusion interact.

Run: python src/traffic_pilot.py      (about 4-6 minutes)
"""
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
SEED, RUNS, SIM_S, WARM_S, DT = 404, 6, 600, 120, 0.1
RING, LANES, FOV, EVAL_R = 1000.0, 3, 80.0, 60.0
DENSITIES = [15, 40, 70, 100]                     # PCU/km/lane
CLS = ["two-wheeler", "car", "auto-rickshaw", "bus/truck"]
MIX = np.array([0.45, 0.30, 0.15, 0.10]); LEN = np.array([2.0, 4.5, 3.0, 11.0]); PCU = np.array([0.5, 1.0, 1.0, 3.0])
# IDM parameters per class: desired speed (m/s), time gap (s), max accel, comfortable decel, jam gap (m)
V0 = np.array([15.0, 16.7, 11.1, 13.9]); TG = np.array([0.9, 1.2, 1.2, 1.6]); AM = np.array([1.8, 1.2, 0.9, 0.7])
BC = np.array([2.5, 2.0, 1.5, 1.5]); S0 = np.array([1.0, 2.0, 1.5, 2.5])
# occlusion: probability that a vehicle of class row hides a vehicle of class column behind it in the same lane
BLOCK = np.array([[0.15, 0.05, 0.08, 0.02],
                  [0.50, 0.35, 0.45, 0.05],
                  [0.40, 0.25, 0.30, 0.03],
                  [0.90, 0.85, 0.90, 0.30]])
ADJ_LANE_BLOCK = 0.5                               # viewing angle halves occlusion in the side lanes
PERCEPTION = {   # base detection prob per class, class multipliers, range fade start/end (m), distance noise, FP per frame
    "clear day":    dict(pd=[0.95, 0.98, 0.96, 0.99], mul=[1.00, 1.00, 1.00, 1.00], d0=60, d1=120, sig=0.030, bias=0.00, fp=0.02, drop=0.000, block_left=1.0),
    "night":        dict(pd=[0.95, 0.98, 0.96, 0.99], mul=[0.85, 0.95, 0.90, 0.97], d0=35, d1=95, sig=0.045, bias=0.00, fp=0.05, drop=0.000, block_left=1.0),
    "heavy rain":   dict(pd=[0.95, 0.98, 0.96, 0.99], mul=[0.90, 0.96, 0.93, 0.98], d0=45, d1=105, sig=0.040, bias=0.00, fp=0.08, drop=0.000, block_left=1.0),
    "fog / dust":   dict(pd=[0.95, 0.98, 0.96, 0.99], mul=[0.85, 0.93, 0.90, 0.96], d0=25, d1=75, sig=0.050, bias=0.00, fp=0.10, drop=0.000, block_left=1.0),
    "sensor fault": dict(pd=[0.95, 0.98, 0.96, 0.99], mul=[1.00, 1.00, 1.00, 1.00], d0=60, d1=120, sig=0.030, bias=0.04, fp=0.06, drop=0.020, block_left=0.3),
}
CONDS = list(PERCEPTION)
CONFUSE = {0: (2, 0.03), 2: (0, 0.04), 1: (2, 0.02)}   # true class -> (wrong class, probability)
TTC_CRIT, TTC_SEVERE = 1.5, 1.0
V_MIN = 10 / 3.6
STATE_EDGES = [20, 40, 60, 85]
STATES = ["free flow", "stable", "near capacity", "congested", "stop-and-go"]


def state_of(k):
    return int(np.searchsorted(STATE_EDGES, k))


class Traffic:
    def __init__(self, density, rng):
        self.rng = rng
        self.lanes = []
        nid = 0
        for _ in range(LANES):
            n = int(round(density / (MIX @ PCU) * RING / 1000))
            cls = rng.choice(4, n, p=MIX)
            spacing = RING / n
            pos = np.sort(np.arange(n) * spacing + rng.uniform(0, 0.2 * spacing, n))
            self.lanes.append(dict(pos=pos, vel=np.full(n, 5.0), cls=cls, id=np.arange(nid, nid + n), force=np.zeros(n), ftime=np.zeros(n)))
            nid += n
        self.next_id = nid
        L = self.lanes[1]; e = int(np.argmin(np.abs(L["pos"] - 500)))
        L["cls"][e] = 1
        self.ego = int(L["id"][e])
        self.ego_freeze = 0.0                     # driver reaction delay: the ego keeps its speed until it reacts

    def ego_idx(self):
        return int(np.where(self.lanes[1]["id"] == self.ego)[0][0])

    def step(self):
        for L in self.lanes:
            p, v, c = L["pos"], L["vel"], L["cls"]
            lead = np.roll(np.arange(len(p)), -1)
            gap = np.roll(p, -1) - p - LEN[c[lead]]
            gap[-1] += RING
            gap = np.maximum(gap, 0.1)
            dv = v - v[lead]
            s_star = S0[c] + np.maximum(0, v * TG[c] + v * dv / (2 * np.sqrt(AM[c] * BC[c])))
            acc = AM[c] * (1 - (v / V0[c]) ** 4 - (s_star / gap) ** 2) + self.rng.normal(0, 0.3, len(p))
            forced = L["ftime"] > 0
            acc = np.where(forced, np.minimum(acc, L["force"]), acc)
            if L is self.lanes[1] and self.ego_freeze > 0:
                acc[np.where(L["id"] == self.ego)[0]] = 0.0
                self.ego_freeze = max(0.0, self.ego_freeze - DT)
            L["ftime"] = np.maximum(0, L["ftime"] - DT)
            acc = np.clip(acc, -8.0, AM[c])
            v_new = np.maximum(0.0, v + acc * DT)
            v_new = np.where(gap < 0.5, np.minimum(v_new, v[lead]), v_new)
            L["vel"] = v_new
            L["pos"] = p + v_new * DT

    def event(self):
        """Cut-in in front of the ego (60%) or hard braking by the ego's leader (40%)."""
        L = self.lanes[1]; e = self.ego_idx(); n = len(L["pos"])
        lead = (e + 1) % n
        self.ego_freeze = self.rng.uniform(0.8, 1.5)
        gap = (L["pos"][lead] + (RING if lead == 0 else 0)) - L["pos"][e] - LEN[L["cls"][lead]]
        v_ego = L["vel"][e]
        if self.rng.random() < 0.6 and gap > 9 and v_ego > 2.5:
            side = self.lanes[self.rng.choice([0, 2])]
            rel = (side["pos"] - L["pos"][e]) % RING
            cand = np.where((rel > 2) & (rel < 40))[0]
            cls = int(self.rng.choice(4, p=[0.45, 0.30, 0.20, 0.05]))
            if len(cand):
                k = cand[0]; cls = int(side["cls"][k])
                for key in ["pos", "vel", "cls", "id", "force", "ftime"]:
                    side[key] = np.delete(side[key], k)
            g_new = self.rng.uniform(3.0, min(14.0, gap - 4))
            new_pos = L["pos"][e] + g_new + LEN[cls]
            dv = self.rng.uniform(1.5, min(6.0, v_ego))
            idx = e + 1
            vals = dict(pos=new_pos, vel=max(0.0, v_ego - dv), cls=cls, id=self.next_id, force=0.0, ftime=0.0)
            for key, val in vals.items():
                L[key] = np.insert(L[key], idx, val)
            self.next_id += 1
            # keep unwrapped positions monotonic if the insert happened at the ring seam
            if idx < len(L["pos"]) - 1 and L["pos"][idx] > L["pos"][idx + 1]:
                L["pos"][idx] -= RING
                order = np.argsort(L["pos"])
                for key in ["pos", "vel", "cls", "id", "force", "ftime"]:
                    L[key] = L[key][order]
            return "cut-in"
        L["force"][lead] = -self.rng.uniform(4.0, 6.5); L["ftime"][lead] = self.rng.uniform(1.5, 2.5)
        return "hard brake"

    def scene(self):
        """Objects ahead of the ego within the field of view: lane, distance, class, id, plus the leader gap."""
        e = self.ego_idx(); L1 = self.lanes[1]; pe = L1["pos"][e]
        out = []
        for li, L in enumerate(self.lanes):
            rel = (L["pos"] - pe) % RING
            m = (rel > 0.5) & (rel <= FOV)
            if li == 1:
                m &= L["id"] != self.ego
            o = np.argsort(rel[m])
            out.append(dict(rel=rel[m][o], cls=L["cls"][m][o], id=L["id"][m][o]))
        n = len(L1["pos"]); lead = (e + 1) % n
        gap = (L1["pos"][lead] + (RING if lead == 0 else 0)) - pe - LEN[L1["cls"][lead]]
        return out, dict(gap=gap, v_ego=L1["vel"][e], v_lead=L1["vel"][lead], lead_id=int(L1["id"][lead]), lead_cls=int(L1["cls"][lead]))


class Perceiver:
    """Stand-in for the camera model under one condition. Keeps its own random stream and track memory."""

    def __init__(self, cond, rng, max_id=20000):
        self.c = PERCEPTION[cond]; self.rng = rng; self.cond = cond
        self.last_seen = np.full(max_id, -10 ** 9); self.frame = 0
        self.lead_id = None; self.g = None; self.gd = 0.0; self.n_upd = 0; self.alert_run = 0; self.obj_bias = 0.0; self.f_prev = 0; self.z_prev = 0.0
        self.acc = dict(gt=np.zeros(4), vis=np.zeros(4), tp=np.zeros(4), fp=0, fn=0, idsw=0, gt60=0,
                        ape40=[], dens_true=[], dens_est=[], ttc_true=[], alert=[], frames=0, dropped=0)

    def frame_step(self, scene, lead):
        c, rng, a = self.c, self.rng, self.acc
        self.frame += 1; a["frames"] += 1
        dropped = rng.random() < c["drop"]
        a["dropped"] += dropped
        pcu_true = pcu_est = 0.0
        for li, obj in enumerate(scene):
            rel, cls, ids = obj["rel"], obj["cls"], obj["id"]
            n = len(rel)
            pcu_true += PCU[cls].sum()
            if n == 0:
                continue
            b = BLOCK[cls[:, None], cls[None, :]] * (1.0 if li == 1 else ADJ_LANE_BLOCK)
            b = np.triu(b, 1)                                    # only vehicles in front can hide later ones
            p_vis = np.exp(np.log(1 - b + 1e-12).sum(0))
            vis = rng.random(n) < p_vis
            fade = np.clip((c["d1"] - rel) / (c["d1"] - c["d0"]), 0, 1)
            pd = np.array(c["pd"])[cls] * np.array(c["mul"])[cls] * fade * (c["block_left"] if li == 0 else 1.0)
            det = vis & (rng.random(n) < pd) & (not dropped)
            in60 = rel <= EVAL_R
            np.add.at(a["gt"], cls[in60], 1); np.add.at(a["vis"], cls[in60 & vis], 1); np.add.at(a["tp"], cls[in60 & det], 1)
            a["gt60"] += in60.sum(); a["fn"] += (in60 & ~det).sum()
            meas = rel * (1 + c["bias"] + rng.normal(0, c["sig"], n))
            in40 = det & (rel <= 40)
            a["ape40"].extend(list(np.abs(meas[in40] - rel[in40]) / rel[in40]))
            est_cls = cls.copy()
            for tc, (wc, pw) in CONFUSE.items():
                flip = (cls == tc) & (rng.random(n) < pw)
                est_cls[flip] = wc
            pcu_est += PCU[est_cls[det]].sum()
            idd = ids[det & in60]
            gapf = self.frame - self.last_seen[idd]
            sw = (gapf >= 5) & (gapf < 10 ** 8) & (rng.random(len(idd)) < 0.5)
            a["idsw"] += int(sw.sum())
            self.last_seen[ids[det]] = self.frame
        nfp = rng.poisson(c["fp"]) if not dropped else 0
        a["fp"] += nfp
        pcu_est += nfp * 1.0
        area = FOV / 1000 * LANES
        a["dens_true"].append(pcu_true / area); a["dens_est"].append(pcu_est / area)
        # ---- forward conflict: alpha-beta tracker on the gap to the leader
        if lead["lead_id"] != self.lead_id:
            self.lead_id, self.g, self.gd, self.n_upd = lead["lead_id"], None, 0.0, 0
        lead_rel = lead["gap"]
        fade = np.clip((c["d1"] - lead_rel) / (c["d1"] - c["d0"]), 0, 1)
        pd_lead = c["pd"][lead["lead_cls"]] * c["mul"][lead["lead_cls"]] * fade
        seen = (not dropped) and lead_rel <= FOV and rng.random() < pd_lead
        if self.g is not None:
            self.g += self.gd * DT
        if seen:
            # monocular range error is mostly a per-object bias (size prior) plus small frame-to-frame noise;
            # the bias scales range and range-rate together, so it cancels in TTC = gap / closing speed
            if self.g is None:
                self.obj_bias = rng.normal(0, c["sig"])
            z = lead_rel * (1 + c["bias"] + self.obj_bias) * (1 + rng.normal(0, 0.005 * c["sig"] / 0.03))
            if self.g is None:
                self.g, self.gd, self.n_upd, self.z_prev = z, 0.0, 1, z
            elif self.n_upd == 1:
                self.gd = (z - self.z_prev) / (DT * max(1, self.frame - self.f_prev)); self.g = z; self.n_upd = 2
            else:
                r = z - self.g
                self.g += 0.5 * r; self.gd += 0.2 * r / DT; self.n_upd += 1
            self.f_prev = self.frame
        ttc_est = self.g / -self.gd if (self.g is not None and self.n_upd >= 3 and self.gd < -0.1) else np.inf
        active = lead["v_ego"] >= V_MIN                 # warnings and conflicts are only defined above 10 km/h
        self.alert_run = self.alert_run + 1 if (ttc_est < TTC_CRIT and active) else 0
        closing = lead["v_ego"] - lead["v_lead"]
        a["ttc_true"].append(lead["gap"] / closing if (closing > 0.05 and active) else np.inf)
        a["alert"].append(self.alert_run >= 3)


def episodes(mask, min_len=2, merge=10):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    eps, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge:
            eps.append((s, prev)); s = i
        prev = i
    eps.append((s, prev))
    return [(a, b) for a, b in eps if b - a + 1 >= min_len]


def conflict_metrics(ttc, alert):
    ttc, alert = np.array(ttc), np.array(alert)
    gt = episodes(ttc < TTC_CRIT, min_len=2, merge=3)
    al = episodes(alert, min_len=1, merge=10)
    det, sev, sev_det, lead_times = 0, 0, 0, []
    for s, e in gt:
        hit = alert[max(0, s - 10): e + 1].any()
        det += hit
        is_sev = ttc[s:e + 1].min() < TTC_SEVERE
        sev += is_sev; sev_det += is_sev and hit
        if hit:
            first = max(0, s - 10) + int(np.argmax(alert[max(0, s - 10): e + 1]))
            tmin = s + int(np.argmin(ttc[s:e + 1]))
            lead_times.append((tmin - first) * DT)
    false = nuisance = 0
    for s, e in al:
        if any(s <= ge + 10 and e >= gs - 10 for gs, ge in gt):
            continue
        near = ttc[max(0, s - 10): e + 11].min()
        if near < 3.0:
            nuisance += 1          # alert during a real but not critical closing (true TTC 1.5-3 s)
        else:
            false += 1             # alert while the true TTC never fell below 3 s: a false alert
    return dict(events=len(gt), detected=int(det), severe=int(sev), severe_detected=int(sev_det), false_alerts=false,
                nuisance_alerts=nuisance, lead_times=lead_times)


def run(density, seed):
    rng = np.random.default_rng(seed)
    tr = Traffic(density, rng)
    pers = {c: Perceiver(c, np.random.default_rng(seed * 100 + i)) for i, c in enumerate(CONDS)}
    speeds, flows, next_event, ev_types = [], [], WARM_S + rng.exponential(45), []
    t = 0.0
    while t < WARM_S + SIM_S:
        tr.step(); t += DT
        if t >= WARM_S:
            if t >= next_event:
                ev_types.append(tr.event()); next_event = t + rng.exponential(45)
            scene, lead = tr.scene()
            for p in pers.values():
                p.frame_step(scene, lead)
            if int(round(t / DT)) % 50 == 0:
                v_all = np.concatenate([L["vel"] for L in tr.lanes])
                speeds.append(v_all.mean())
                flows.append(sum(len(L["pos"]) for L in tr.lanes) / LANES / (RING / 1000) * v_all.mean() * 3.6)
    return pers, dict(mean_speed_kmh=float(np.mean(speeds) * 3.6), flow_veh_h_lane=float(np.mean(flows)),
                      speed_series=speeds, events=ev_types)


def summarise(p):
    a = p.acc
    rec_vis = a["tp"] / np.maximum(a["vis"], 1); rec_all = a["tp"] / np.maximum(a["gt"], 1)
    tp = a["tp"].sum()
    w = int(30 / DT)
    dt_, de_ = np.array(a["dens_true"]), np.array(a["dens_est"])
    nwin = len(dt_) // w
    wt = dt_[: nwin * w].reshape(nwin, w).mean(1); we = de_[: nwin * w].reshape(nwin, w).mean(1)
    cm = conflict_metrics(a["ttc_true"], a["alert"])
    return dict(recall_visible=rec_vis.tolist(), recall_all=rec_all.tolist(), recall_visible_total=float(tp / max(a["vis"].sum(), 1)),
                recall_all_total=float(tp / max(a["gt"].sum(), 1)), precision=float(tp / max(tp + a["fp"], 1)),
                mota=float(1 - (a["fn"] + a["fp"] + a["idsw"]) / max(a["gt60"], 1)), idsw_per_1000=float(1000 * a["idsw"] / max(a["gt60"], 1)),
                dist_mape_40=float(100 * np.mean(a["ape40"])) if a["ape40"] else np.nan,
                win_true=wt.tolist(), win_est=we.tolist(), gt_objects=int(a["gt60"]), hours=a["frames"] * DT / 3600, **cm)


def latency_model(mean_objects, throttle=1.0, seconds=600, seed=7):
    """Frame pipeline at 30 fps on one accelerator; newest-frame policy (a waiting frame is replaced by a newer one)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 30); period = 1000 / 30
    objs = rng.poisson(mean_objects, n)
    proc = (3.0 + rng.lognormal(np.log(16.0), 0.15, n) * throttle + 0.15 * objs + 1.0 + 0.25 * objs + 1.5)
    capture = 12.0
    free_at, lat, dropped, waiting = 0.0, [], 0, None
    for i in range(n):
        t_cap = i * period
        start = max(t_cap + capture, free_at)
        if free_at > t_cap + capture + period:          # pipeline is more than a frame behind: drop this frame
            dropped += 1
            continue
        free_at = start + proc[i]
        lat.append(free_at - t_cap)
    lat = np.array(lat)
    return dict(p50=float(np.percentile(lat, 50)), p99=float(np.percentile(lat, 99)), deadline_miss_pct=float(100 * (lat > 100).mean()),
                dropped_pct=float(100 * dropped / n), mean_proc=float(proc.mean())), lat


def boot_ci(values, stat=np.mean, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    v = np.array(values)
    bs = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    t0 = time.time()
    rows, traffic_rows, store = [], [], {}
    for k in DENSITIES:
        for r in range(RUNS):
            pers, tinfo = run(k, SEED + 1000 * k + r)
            traffic_rows.append(dict(density=k, run=r, **{x: y for x, y in tinfo.items() if x not in ("speed_series", "events")},
                                     cut_ins=tinfo["events"].count("cut-in"), hard_brakes=tinfo["events"].count("hard brake")))
            if r == 0:
                store[f"speed_{k}"] = tinfo["speed_series"]
            for c, p in pers.items():
                s = summarise(p)
                store[(k, r, c)] = s
                rows.append(dict(density=k, run=r, condition=c, **{x: y for x, y in s.items() if not isinstance(y, list)},
                                 recall_vis_2w=s["recall_visible"][0], recall_all_2w=s["recall_all"][0]))
            print(f"density {k} run {r} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/pilot_runs.csv", index=False)
    tdf = pd.DataFrame(traffic_rows); tdf.to_csv("results/traffic_runs.csv", index=False)

    # density-correction factor learned on clear-day run 0 of every density (the "training" runs), used everywhere
    tr_t = np.concatenate([store[(k, 0, "clear day")]["win_true"] for k in DENSITIES]); tr_e = np.concatenate([store[(k, 0, "clear day")]["win_est"] for k in DENSITIES])
    corr = float(np.sum(tr_t * tr_e) / np.sum(tr_e ** 2))
    state_rows = []
    for (key, s) in [(kk, v) for kk, v in store.items() if isinstance(kk, tuple)]:
        k, r, c = key
        if r == 0:
            continue                                     # held out: training runs are not scored
        for t_, e_ in zip(s["win_true"], s["win_est"]):
            state_rows.append(dict(density=k, run=r, condition=c, true=t_, est=e_ * corr, s_true=state_of(t_), s_est=state_of(e_ * corr)))
    sdf = pd.DataFrame(state_rows)

    def macro_f1(d):
        f = []
        for s in range(5):
            tp = ((d.s_true == s) & (d.s_est == s)).sum(); fp = ((d.s_true != s) & (d.s_est == s)).sum(); fn = ((d.s_true == s) & (d.s_est != s)).sum()
            if tp + fn == 0:
                continue
            f.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np.mean(f))

    summary = {"runs": RUNS, "sim_minutes_per_run": SIM_S / 60, "densities": DENSITIES, "conditions": CONDS, "density_correction_factor": round(corr, 3),
               "simulated_hours_total": round(len(DENSITIES) * RUNS * SIM_S / 3600, 1), "traffic": tdf.groupby("density")[["mean_speed_kmh", "flow_veh_h_lane", "cut_ins", "hard_brakes"]].mean().round(1).to_dict()}
    cells = []
    for c in CONDS:
        for k in DENSITIES:
            d = df[(df.condition == c) & (df.density == k)]
            cells.append(dict(condition=c, density=k, recall_visible=d.recall_visible_total.mean(), recall_all=d.recall_all_total.mean(),
                              recall_vis_2w=d.recall_vis_2w.mean(), recall_all_2w=d.recall_all_2w.mean(), precision=d.precision.mean(), mota=d.mota.mean(),
                              idsw_per_1000=d.idsw_per_1000.mean(), dist_mape_40=d.dist_mape_40.mean(),
                              density_mape=float(100 * np.mean(np.abs(sdf[(sdf.condition == c) & (sdf.density == k)].est - sdf[(sdf.condition == c) & (sdf.density == k)].true) / sdf[(sdf.condition == c) & (sdf.density == k)].true))))
    cdf = pd.DataFrame(cells); cdf.to_csv("results/pilot_cells.csv", index=False)
    per_cond = {}
    for c in CONDS:
        d = df[df.condition == c]
        ev, det = d.events.sum(), d.detected.sum(); sev, sdet = d.severe.sum(), d.severe_detected.sum()
        hrs = d.hours.sum()
        run_rec = (d.groupby("run").detected.sum() / d.groupby("run").events.sum()).values
        fa_run = (d.groupby("run").false_alerts.sum() / d.groupby("run").hours.sum()).values
        low60 = d[d.density <= 70]
        lead_t = []
        for kk in DENSITIES:
            for rr in range(RUNS):
                lead_t += store[(kk, rr, c)]["lead_times"]
        per_cond[c] = dict(recall_visible=float(d.recall_visible_total.mean()), recall_visible_ci=boot_ci(d.recall_visible_total.values),
                           recall_visible_up_to_70=float(low60.recall_visible_total.mean()), recall_visible_up_to_70_ci=boot_ci(low60.recall_visible_total.values),
                           recall_all=float(d.recall_all_total.mean()), mota_up_to_70=float(low60.mota.mean()), mota_up_to_70_ci=boot_ci(low60.mota.values),
                           dist_mape_40=float(d.dist_mape_40.mean()), dist_mape_40_ci=boot_ci(d.dist_mape_40.values),
                           state_macro_f1=macro_f1(sdf[sdf.condition == c]),
                           conflict_events=int(ev), conflict_recall=float(det / max(ev, 1)), conflict_recall_ci=boot_ci(run_rec),
                           severe_events=int(sev), severe_missed=int(sev - sdet),
                           false_alerts_per_hour=float(d.false_alerts.sum() / hrs), false_alerts_ci=boot_ci(fa_run), hours=float(hrs),
                           nuisance_alerts_per_hour=float(d.nuisance_alerts.sum() / hrs),
                           median_warning_lead_s=float(np.median(lead_t)) if lead_t else np.nan)
    summary["per_condition"] = per_cond
    clear = per_cond["clear day"]
    summary["robustness_ratio"] = {c: dict(detection=round(per_cond[c]["recall_visible"] / clear["recall_visible"], 3),
                                           conflict=round(per_cond[c]["conflict_recall"] / clear["conflict_recall"], 3)) for c in CONDS}
    # confusion matrix of traffic state (all conditions, clear only)
    cl = sdf[sdf.condition == "clear day"]
    summary["state_confusion_clear"] = pd.crosstab(cl.s_true.map(dict(enumerate(STATES))), cl.s_est.map(dict(enumerate(STATES)))).to_dict()
    # latency
    objs = {k: float(np.mean([store[(k, r, 'clear day')]["gt_objects"] for r in range(RUNS)]) / (SIM_S / DT) / (EVAL_R / FOV)) for k in DENSITIES}
    lat_rows, lat_series = [], {}
    for k in DENSITIES:
        for thr, lab in [(1.0, "normal"), (1.3, "thermal throttling (+30%)")]:
            m, lat = latency_model(objs[k] + 0.1, throttle=thr)
            lat_rows.append(dict(density=k, mode=lab, objects_in_view=round(objs[k], 1), **{x: round(y, 2) for x, y in m.items()}))
            lat_series[(k, lab)] = lat
    ldf = pd.DataFrame(lat_rows); ldf.to_csv("results/latency.csv", index=False)
    summary["latency"] = lat_rows
    summary["runtime_s"] = round(time.time() - t0, 1)
    json.dump(summary, open("results/pilot_summary.json", "w"), indent=2, default=float)

    # ------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    g = tdf.groupby("density")
    kk = np.array(DENSITIES); vv = g.mean_speed_kmh.mean().values; qq = g.flow_veh_h_lane.mean().values
    ax[0].plot(kk, vv, "o-"); ax[0].set_xlabel("density (PCU/km/lane)"); ax[0].set_ylabel("mean speed (km/h)"); ax[0].set_title("Speed-density from the simulation"); ax[0].grid(alpha=0.3)
    ax[1].plot(kk, qq, "s-", color="#d62728"); ax[1].set_xlabel("density (PCU/km/lane)"); ax[1].set_ylabel("flow (veh/h/lane)"); ax[1].set_title("Flow-density (fundamental diagram)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/fig6_sim_fundamental_diagram.png", dpi=160); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, col, title in [(axs[0], "recall_visible", "Detection recall, visible objects within 60 m"), (axs[1], "recall_all_2w", "Two-wheeler recall incl. hidden ones")]:
        piv = cdf.pivot(index="condition", columns="density", values=col).reindex(CONDS)
        im = ax.imshow(piv.values * 100, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{100 * piv.values[i, j]:.1f}", ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(len(DENSITIES))); ax.set_xticklabels(DENSITIES); ax.set_yticks(range(len(CONDS))); ax.set_yticklabels(CONDS)
        ax.set_xlabel("density (PCU/km/lane)"); ax.set_title(title, fontsize=10)
    axs[1].tick_params(labelleft=False)
    fig.colorbar(im, ax=axs, label="%", shrink=0.9); fig.savefig("figures/fig7_recall_heatmaps.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (k, lab), lat in lat_series.items():
        if k in (15, 100):
            ax.plot(np.sort(lat), np.linspace(0, 100, len(lat)), label=f"{k} PCU/km/lane, {lab}")
    ax.axvline(100, color="k", ls="--", lw=1); ax.text(101, 5, "100 ms budget", fontsize=8)
    ax.set_xlabel("end-to-end latency (ms)"); ax.set_ylabel("cumulative % of frames"); ax.set_title("Latency distribution from the frame-pipeline model")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("figures/fig8_latency_cdf.png", dpi=160); plt.close(fig)

    # one conflict example from clear day, density 40, run 1
    s = store[(40, 1, "clear day")]
    p_alert = None
    fig, ax = plt.subplots(figsize=(9, 3.4))
    pp = Perceiver("clear day", np.random.default_rng(0))
    ttc = np.array(json.loads(json.dumps(conflict_example(40, SEED + 1000 * 40 + 1))))
    ax.plot(ttc[:, 0], np.minimum(ttc[:, 1], 6), "k-", label="true TTC")
    ax.plot(ttc[:, 0], np.minimum(ttc[:, 2], 6), "b.", ms=3, label="estimated TTC (camera model)")
    al = ttc[:, 3] > 0
    ax.fill_between(ttc[:, 0], 0, 6, where=al, color="red", alpha=0.15, label="alert active")
    ax.axhline(TTC_CRIT, color="r", ls="--", lw=1)
    ax.set_ylim(0, 6); ax.set_xlabel("time (s)"); ax.set_ylabel("TTC (s), capped at 6"); ax.set_title("Example: a cut-in at 40 PCU/km/lane, clear day")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("figures/fig9_conflict_example.png", dpi=160); plt.close(fig)
    print(json.dumps({c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in per_cond[c].items()} for c in CONDS}, indent=1))
    print(cdf.round(3).to_string()); print(ldf.to_string()); print("runtime", summary["runtime_s"])


def conflict_example(density, seed, window=12.0):
    """Re-run one traffic run and return [t, true TTC, estimated TTC, alert] around the first critical cut-in."""
    rng = np.random.default_rng(seed)
    tr = Traffic(density, rng)
    p = Perceiver("clear day", np.random.default_rng(seed * 100))
    rec, t, next_event, last_type, ev_t = [], 0.0, WARM_S + rng.exponential(45), None, []
    while t < WARM_S + SIM_S:
        tr.step(); t += DT
        if t >= WARM_S:
            if t >= next_event:
                last_type = tr.event(); next_event = t + rng.exponential(45)
                if last_type == "cut-in":
                    ev_t.append(t)
            scene, lead = tr.scene()
            p.frame_step(scene, lead)
            ttc_est = p.g / -p.gd if (p.g is not None and p.n_upd >= 3 and p.gd < -0.1) else np.inf
            rec.append([t, p.acc["ttc_true"][-1], ttc_est, float(p.acc["alert"][-1])])
            if ev_t and t > ev_t[0] + window / 2:
                arr = np.array(rec)
                if (arr[:, 1][arr[:, 0] > ev_t[0] - 1] < TTC_CRIT).any():
                    m = (arr[:, 0] > ev_t[0] - window / 2) & (arr[:, 0] <= ev_t[0] + window / 2)
                    out = arr[m]; out[:, 0] -= out[0, 0]
                    return out.tolist()
                ev_t.pop(0)
    arr = np.array(rec[-120:]); arr[:, 0] -= arr[0, 0]
    return arr.tolist()


if __name__ == "__main__":
    main()
