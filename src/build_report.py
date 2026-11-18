"""
Builds the Week 4 simulation and testing strategy (.docx) from the pilot-campaign results and reusable text blocks.
Run from the week folder:  python src/traffic_pilot.py  then  python src/build_report.py
"""
import json
import math
import sys

import pandas as pd

sys.path.insert(0, "src")
from doclib import Report, section  # noqa: E402

OUT = "Week4_Simulation_Testing_Strategy_Traffic_AI.docx"
GH = "https://github.com/himshxrmx/traffic-ai-simulation-testing-strategy"
B = json.load(open("content/base_text.json", encoding="utf-8"))
P = json.load(open("results/pilot_summary.json"))
cells = pd.read_csv("results/pilot_cells.csv"); lat = pd.read_csv("results/latency.csv"); runs = pd.read_csv("results/pilot_runs.csv")
IMG = ["images/fig1_test_framework.png", "images/fig2_simulation_workflow.png", "images/fig3_test_levels.png", "images/fig4_fundamental_diagram.png", "images/fig5_test_matrix.png"]
pc = P["per_condition"]; C = pc["clear day"]; rr = P["robustness_ratio"]; T = P["traffic"]
CONDS = list(pc)
cell = lambda c, k: cells[(cells.condition == c) & (cells.density == k)].iloc[0]  # noqa: E731
pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
lat100 = lat[(lat.density == 100) & (lat["mode"] == "normal")].iloc[0]; lat100t = lat[(lat.density == 100) & (lat["mode"] != "normal")].iloc[0]
# statistics used in the text
n_zero = math.ceil(math.log(0.05) / math.log(0.98))
n_wilson = math.ceil(1.645 ** 2 * 0.99 * 0.01 / (0.99 - 0.98) ** 2)
fa_ub = 3 / C["hours"]


def greenshields(k, vf=60, kj=130):
    v = vf * (1 - k / kj)
    return v, v * k


def geh(m, c):
    return math.sqrt(2 * (m - c) ** 2 / (m + c))


R = Report()
R.title_page("Simulation and Testing Strategy for AI Algorithms in Traffic Analysis",
             "Saarthi Traffic Analytics Model: strategy, metrics with worked examples, a pilot campaign and go/no-go criteria",
             ["Week 4 · Automotive AI Innovation Internship (YuvaIntern)", "Author: Him Sharma, B.Tech CSE, ABES Engineering College",
              "November 2026", f"Pilot simulation code, results and figures: {GH}"])

# ------------------------------------------------------------------ 1
R.h1("1. Executive Summary")
R.p("This report sets out how to simulate and test the Saarthi Traffic Analytics Model, the in-vehicle AI from Week 3 that detects and tracks road users, "
    "estimates the local traffic state and flags conflicts that could lead to a crash, before it is deployed in a 200-vehicle fleet pilot. The strategy "
    "uses SUMO and CARLA co-simulation, a catalogue of ten scenario families under five conditions, five test levels from model-in-the-loop to the road, "
    "and accuracy, latency and robustness metrics analysed with confidence intervals.")
R.p(f"To make the strategy concrete, I ran a small version of it end to end: a pilot campaign written in Python with mixed Indian traffic (45% two-wheelers, "
    f"30% cars, 15% auto-rickshaws, 10% buses and trucks) at four densities, a statistical stand-in for the camera model, five conditions applied to the same "
    f"traffic, and {P['simulated_hours_total']:.0f} hours of simulated driving per condition with {pc['clear day']['conflict_events']} critical conflicts. "
    "Every metric in the strategy was computed on it, with a worked example, and every go/no-go criterion was checked:")
R.bullets([
    ("Met in clear conditions:", f"detection recall on visible objects {pct(C['recall_visible'])} (target 95%), distance error {C['dist_mape_40']:.1f}% (target 5%), "
     f"tracking accuracy (MOTA) {C['mota_up_to_70']:.2f} up to 70 PCU/km/lane, no missed severe conflicts out of {C['severe_events']}, zero false alerts, and 99th-percentile "
     f"latency {lat100.p99:.0f} ms in stop-and-go traffic (budget 100 ms)."),
    ("Not met:", f"traffic-state classification macro-F1 {C['state_macro_f1']:.2f} (target 0.85); conflict recall {pct(C['conflict_recall'])} with a 95% lower bound of "
     f"{pct(C['conflict_recall_ci'][0])} (target 98%); {C['nuisance_alerts_per_hour']:.0f} nuisance alerts an hour during near-misses in congestion; and "
     f"{lat100t.dropped_pct:.1f}% dropped frames when thermal throttling meets stop-and-go traffic."),
    ("Robustness:", f"detection keeps {rr['night']['detection']:.2f} of its clear-day recall at night and {rr['heavy rain']['detection']:.2f} in heavy rain (target 0.75), but only "
     f"{rr['fog / dust']['detection']:.2f} in fog or dust. Conflict detection is nearly unaffected (0.99–1.01), because the vehicle directly ahead is close and rarely hidden."),
    ("Key finding:", "occlusion, not the detector, dominates dense traffic: two-wheeler recall counting hidden ones falls from "
     f"{pct(cell('clear day', 15).recall_all_2w)} at 15 to {pct(cell('clear day', 100).recall_all_2w)} at 100 PCU/km/lane, while recall on visible two-wheelers stays at 95%."),
])
R.p("The resulting decision is a CONDITIONAL GO: the model may enter the fleet pilot in shadow mode, with collision warnings limited to its tested operating "
    "domain, the traffic-state feature kept internal until retrained, alert hysteresis added, and the conflict test set enlarged to at least 300 events per "
    "condition so that the 98% recall target can be confirmed statistically.")

# ------------------------------------------------------------------ 2-4 kept
R.emit(section(B, "2. Introduction"))
R.emit(section(B, "3. Literature Review: Simulation Approaches and the Role of AI"))
R.emit(section(B, "4. Methodology"), images=IMG)

# ------------------------------------------------------------------ 5 metrics + worked examples
R.emit(section(B, "5. Performance Evaluation Metrics"))
R.h2("5.6 Worked Examples of Each Metric")
R.p("Each metric below is shown with a worked example using numbers from the pilot campaign (Section 7), so that its meaning and scale are clear.")
c15 = cell("clear day", 15)
mo_fn = 10000 * (1 - c15.recall_all); mo_tp = 10000 * c15.recall_all; mo_fp = mo_tp * (1 - c15.precision) / c15.precision; mo_sw = c15.idsw_per_1000 * 10
R.table(["Metric", "Formula", "Worked example from the pilot"], [
    ["Detection recall", "TP / (TP + FN)", f"Clear day, 15 PCU/km/lane: of every 1,000 visible road users within 60 m, {1000 * c15.recall_visible:.0f} were detected: recall {pct(c15.recall_visible)}"],
    ["Precision", "TP / (TP + FP)", f"Same cell: false positives appear about 0.02 times per frame, giving precision {pct(c15.precision)}"],
    ["MOTA", "1 − (FN + FP + IDSW) / GT", f"Per 10,000 ground-truth objects: {mo_fn:.0f} missed (including hidden ones), {mo_fp:.0f} false positives and {mo_sw:.0f} identity switches: MOTA = 1 − {mo_fn + mo_fp + mo_sw:.0f} / 10,000 = {1 - (mo_fn + mo_fp + mo_sw) / 10000:.2f}"],
    ["Distance error", "mean |d_est − d_true| / d_true", f"A car at 30.0 m estimated at 30.7 m is a 2.3% error; the clear-day average within 40 m is {C['dist_mape_40']:.2f}%"],
    ["Traffic-state macro-F1", "mean over 5 states of 2TP / (2TP + FP + FN)", f"Clear day: {C['state_macro_f1']:.2f}; errors are between neighbouring states (Table 'Traffic-state confusion')"],
    ["Time to collision (TTC)", "gap / closing speed", "A two-wheeler cuts in 6 m ahead while the car closes at 5 m/s: TTC = 6 / 5 = 1.2 s, below the 1.5 s critical line"],
    ["Conflict recall", "critical conflicts warned / all critical conflicts", f"Clear day: {round(C['conflict_recall'] * C['conflict_events'])} of {C['conflict_events']} warned = {pct(C['conflict_recall'])}"],
    ["False alerts per hour", "alerts when true TTC > 3 s / hours driven", f"0 in {C['hours']:.0f} h; by the rule of three the 95% upper bound is 3 / {C['hours']:.0f} = {fa_ub:.2f} per hour"],
    ["Robustness ratio", "metric in condition / metric in clear conditions", f"Night detection recall {pct(pc['night']['recall_visible'])} / {pct(C['recall_visible'])} = {rr['night']['detection']:.2f}"],
    ["99th-percentile latency", "latency below which 99% of frames fall", f"Stop-and-go traffic, about {lat100.objects_in_view:.0f} objects in view: {lat100.p99:.0f} ms against a 100 ms budget"],
    ["GEH (calibration)", "√(2(M − C)² / (M + C))", f"Simulated flow {T['flow_veh_h_lane']['70']:.0f} vs theoretical {greenshields(70)[1]:.0f} veh/h/lane at 70 PCU/km/lane: GEH = {geh(T['flow_veh_h_lane']['70'], greenshields(70)[1]):.1f} (accept if below 5)"],
], caption="Metrics with worked examples", widths=[3.0, 4.2, 9.3], font=8.5)

# ------------------------------------------------------------------ 6 data analysis + sample size
R.emit(section(B, "6. Data Analysis Methods"))
R.h2("6.6 Sample Size and Confidence: Worked Examples")
R.p("A pass or fail should be decided on the lower bound of a confidence interval, not on a point estimate, and the number of tests must be large enough "
    "for that bound to be meaningful. Three calculations set the size of the full campaign:")
R.bullets([
    ("Zero-failure demonstration:", f"to show conflict recall of at least 98% with 95% confidence when no conflict is missed, n must satisfy 0.98ⁿ ≤ 0.05, so "
     f"n ≥ ln(0.05) / ln(0.98) = {n_zero} critical conflicts per condition with zero misses."),
    ("With some misses expected:", f"if the true recall is 99%, the one-sided 95% lower bound clears 98% only when n ≥ 1.645² × 0.99 × 0.01 / 0.01² ≈ {n_wilson} conflicts. "
     f"The pilot had {C['conflict_events']} per condition, which is why its lower bound ({pct(C['conflict_recall_ci'][0])}) cannot confirm the target."),
    ("Rare false alerts:", f"zero false alerts in T hours gives a 95% upper bound of 3 / T per hour. The pilot's {C['hours']:.0f} hours bound the rate below {fa_ub:.2f} per hour; "
     "showing a rate below 0.1 per hour needs at least 30 hours without a false alert."),
])

# ------------------------------------------------------------------ 7 pilot campaign
R.h1("7. Pilot Campaign: Running the Strategy at Small Scale")
R.h2("7.1 Design")
R.p(f"The pilot (src/traffic_pilot.py) is a scaled-down, fully reproducible version of the SIL campaign. Traffic runs on a three-lane, 1 km ring road with the "
    f"Indian vehicle mix of Section 4.5, driven by the Intelligent Driver Model with class-specific desired speeds, time gaps and braking, plus random "
    f"acceleration noise that creates stop-and-go waves at high density. Four densities were simulated (15, 40, 70 and 100 PCU/km/lane), each with {P['runs']} "
    f"runs of {P['sim_minutes_per_run']:.0f} simulated minutes after a two-minute warm-up. About 80 times an hour a scripted event is injected in front of the ego vehicle: "
    "a cut-in by a vehicle from the next lane (60%) or hard braking by the vehicle ahead (40%), and the ego driver reacts after a 0.8–1.5 s delay, as a "
    "distracted human would.")
R.p("The model under test is a statistical stand-in for the camera model: each road user in the field of view (80 m ahead, three lanes) may be hidden by "
    "vehicles in front of it, with a probability that depends on both classes (a bus hides almost everything behind it; a scooter hides little); visible "
    "objects are detected with a probability that depends on class, range and condition; distances carry a per-object bias and frame noise; classes are "
    "occasionally confused; and false positives appear at random. Traffic state is estimated from the detections with a correction factor learned on "
    f"clear-day training runs (×{P['density_correction_factor']:.2f}, which compensates for hidden vehicles), and conflicts are flagged when a tracked TTC stays below "
    "1.5 s for three frames above 10 km/h. The same traffic is replayed under all five conditions, so differences between conditions are paired and not noise.")
R.h2("7.2 Traffic Validation")
rows = []
for k in [15, 40, 70, 100]:
    v_t, q_t = greenshields(k)
    rows.append([str(k), f"{T['mean_speed_kmh'][str(k)]:.1f}", f"{v_t:.1f}", f"{T['flow_veh_h_lane'][str(k)]:.0f}", f"{q_t:.0f}", f"{geh(T['flow_veh_h_lane'][str(k)], q_t):.1f}"])
R.table(["Density (PCU/km/lane)", "Simulated speed (km/h)", "Greenshields speed (km/h)", "Simulated flow (veh/h/lane)", "Theoretical flow", "GEH"], rows,
        caption="Simulated traffic against the theoretical fundamental diagram (v_f = 60 km/h, k_j = 130)", widths=[2.8, 2.6, 2.8, 2.9, 2.6, 1.4], font=8.5)
R.figure("figures/fig6_sim_fundamental_diagram.png", "Speed-density and flow-density relationships produced by the pilot traffic model", width_cm=14)
R.p(f"The calibration check works as designed: near and above capacity the simulated flows match the theory (GEH below 5), but in light traffic the flow "
    f"fails the test (GEH {geh(T['flow_veh_h_lane']['15'], greenshields(15)[1]):.1f}) because the simulated mean speed is only {T['mean_speed_kmh']['15']:.0f} km/h instead of 53 km/h. The reason is "
    "real and Indian: slow auto-rickshaws hold up the vehicles behind them when overtaking is not modelled. In the full campaign this is exactly the case "
    "where the free-flow speed would be re-fitted to survey data, or SUMO's sublane model used so that two-wheelers can filter past.")
R.h2("7.3 Perception Results")
rows = [[c, str(k), pct(cell(c, k).recall_visible), pct(cell(c, k).recall_all), pct(cell(c, k).recall_vis_2w), pct(cell(c, k).recall_all_2w), f"{cell(c, k).mota:.2f}", f"{cell(c, k).idsw_per_1000:.1f}"]
        for c in CONDS for k in [15, 100]]
R.table(["Condition", "Density", "Recall, visible", "Recall, incl. hidden", "Two-wheeler recall, visible", "Two-wheeler recall, incl. hidden", "MOTA", "ID switches per 1,000"], rows,
        caption="Perception metrics at light (15) and stop-and-go (100 PCU/km/lane) traffic", widths=[2.4, 1.5, 1.9, 2.1, 2.2, 2.3, 1.4, 2.2], font=8)
R.figure("figures/fig7_recall_heatmaps.png", "Recall on visible objects (left) and two-wheeler recall including hidden ones (right), by condition and density")
R.p(f"Two different effects are visible. Recall on visible objects depends on the condition but hardly on density ({pct(cell('clear day', 15).recall_visible)} to "
    f"{pct(cell('clear day', 100).recall_visible)} on a clear day). Recall including hidden objects falls steeply with density because of occlusion: for two-wheelers "
    f"from {pct(cell('clear day', 15).recall_all_2w)} to {pct(cell('clear day', 100).recall_all_2w)} on a clear day and to {pct(cell('fog / dust', 100).recall_all_2w)} in fog. Identity switches rise "
    f"almost five-fold ({cell('clear day', 15).idsw_per_1000:.1f} to {cell('clear day', 100).idsw_per_1000:.1f} per 1,000 objects) as vehicles disappear behind buses and reappear. This is why the "
    "strategy evaluates accuracy against visible objects and analyses occlusion separately (Section 9): a detector cannot be blamed for what no camera could see.")
R.h2("7.4 Traffic-State Estimation")
rows = [[c, f"{pc[c]['state_macro_f1']:.2f}"] for c in CONDS]
R.table(["Condition", "Macro-F1 (5 states)"], rows, caption="Traffic-state classification on held-out runs (target 0.85)", widths=[5, 4])
conf = P["state_confusion_clear"]; states = ["free flow", "stable", "near capacity", "congested", "stop-and-go"]
rows = [[t] + [str(int(conf.get(e, {}).get(t, 0))) for e in states] for t in states]
R.table(["True state (rows) / estimated (columns)"] + states, rows, caption="Traffic-state confusion, clear day (30-second windows)", widths=[4.0, 2.2, 2.0, 2.4, 2.2, 2.4], font=8.5)
R.p(f"The estimator reaches {C['state_macro_f1']:.2f} on clear days, below the 0.85 target, and its errors fall exactly where hypothesis H6 predicted: between neighbouring "
    f"states, such as 'stable' read as 'near capacity' and 'stop-and-go' read as 'congested'. It degrades sharply in fog ({pc['fog / dust']['state_macro_f1']:.2f}) because the occlusion correction was learned "
    "on clear-day runs: when fewer vehicles are detected, density is underestimated. The fix is to give the estimator the condition as an input (or learn one "
    "correction per condition) and to add speed-based features, which do not depend on counting every vehicle.")
R.h2("7.5 Conflict Detection")
rows = [[c, str(pc[c]["conflict_events"]), pct(pc[c]["conflict_recall"]), f"{pct(pc[c]['conflict_recall_ci'][0])} – {pct(pc[c]['conflict_recall_ci'][1])}",
         f"{pc[c]['severe_events'] - pc[c]['severe_missed']}/{pc[c]['severe_events']}", f"{pc[c]['false_alerts_per_hour']:.2f}", f"{pc[c]['nuisance_alerts_per_hour']:.2f}", f"{pc[c]['median_warning_lead_s']:.2f} s"] for c in CONDS]
R.table(["Condition", "Critical conflicts", "Recall", "95% interval", "Severe (TTC < 1 s) warned", "False alerts / h", "Nuisance alerts / h", "Median warning before TTC minimum"], rows,
        caption="Forward-conflict detection by condition (all densities)", widths=[2.4, 1.8, 1.5, 2.6, 2.2, 1.8, 1.9, 2.4], font=8)
R.figure("figures/fig9_conflict_example.png", "One cut-in: true TTC, the model's estimated TTC and the period when the alert was active")
R.p(f"Conflict detection is the model's strongest function and barely changes with the condition, because the vehicle directly ahead is close, large in the "
    f"image and rarely hidden. Every severe conflict (TTC below 1 s) was warned. But two weaknesses appear. First, the warning comes late in short-gap "
    f"cut-ins: a median of only {C['median_warning_lead_s']:.1f} s before the moment of minimum TTC, because a new vehicle must be tracked for a few frames before its "
    "closing speed is known. A warning-only system cannot do much with 0.3 s, which supports the Week 3 conclusion that automatic emergency braking and "
    f"radar are the next steps. Second, about {C['nuisance_alerts_per_hour']:.0f} alerts an hour occur during real but non-critical closings (true TTC 1.5–3 s) in congested "
    "traffic; they are not false, but drivers may find them annoying. Hysteresis (requiring the TTC to keep falling) and speed-dependent thresholds are the planned fixes.")
R.h2("7.6 Latency")
rows = [[str(r.density), r.mode, f"{r.objects_in_view:.1f}", f"{r.p50:.0f}", f"{r.p99:.0f}", f"{r.deadline_miss_pct:.2f}%", f"{r.dropped_pct:.2f}%"] for r in lat.itertuples()]
R.table(["Density", "Mode", "Objects in view", "Median latency (ms)", "p99 latency (ms)", "Over 100 ms", "Frames dropped"], rows,
        caption="End-to-end latency from the frame-pipeline model (30 fps, one accelerator, newest-frame policy)", widths=[1.6, 4.0, 2.0, 2.3, 2.1, 1.9, 2.2], font=8.5)
R.figure("figures/fig8_latency_cdf.png", "Cumulative latency distribution in light and stop-and-go traffic, with and without thermal throttling", width_cm=13)
R.p(f"Latency grows with the number of objects because tracking and prediction run once per object. The 99th percentile stays below the 100 ms budget in every "
    f"case ({lat100t.p99:.0f} ms in the worst one), and no frame finishes late, because the pipeline drops a waiting frame rather than queueing it. The cost appears as "
    f"dropped frames: {lat100t.dropped_pct:.1f}% in stop-and-go traffic with 30% thermal throttling, which lowers the effective tracking rate from 30 to about 27 frames per "
    "second. This links directly to the Week 3 thermal policy, which throttles driver monitoring before forward perception.")
R.h2("7.7 What the Pilot Changed in the Plan")
R.bullets([
    f"The conflict test set grows from about 140 to at least {n_wilson} critical events per condition (Section 6.6).",
    "Occlusion is reported as its own metric (visible versus all objects), and density-binned results are mandatory in every test report.",
    "Free-flow speed in light mixed traffic must be calibrated from survey data, because overtaking behaviour changes it by about 25%.",
    "Two new go/no-go criteria are added: nuisance alerts below 2 per hour, and dropped frames below 1% in the worst thermal case.",
    "The traffic-state estimator needs condition-aware correction before its output is shown to fleets.",
])

# ------------------------------------------------------------------ 8-9
R.emit(section(B, "7. Testing Procedures"), rename={"7. Testing Procedures": "8. Testing Procedures"})
R.emit(section(B, "8. Expected Challenges and Mitigations"), rename={"8. Expected Challenges and Mitigations": "9. Expected Challenges and Mitigations"})
R.h2("9.1 Challenges Observed in the Pilot, Quantified")
R.table(["Challenge", "Evidence from the pilot", "Impact", "Mitigation", "Owner"], [
    ["Occlusion in dense mixed traffic", f"Two-wheeler recall incl. hidden: {pct(cell('clear day', 15).recall_all_2w)} → {pct(cell('clear day', 100).recall_all_2w)} (15 → 100 PCU)", "Under-counted density; missed vulnerable users behind buses", "Visibility-based metrics; radar for fleets; occlusion-aware tracking", "ML lead"],
    ["Identity switches", f"{cell('clear day', 15).idsw_per_1000:.1f} → {cell('clear day', 100).idsw_per_1000:.1f} per 1,000 objects", "Broken tracks; wrong trajectories", "Re-identification features; longer track memory", "ML lead"],
    ["Night and fog", f"Visible recall {pct(pc['night']['recall_visible'])} at night, {pct(pc['fog / dust']['recall_visible'])} in fog", "Fog ratio 0.75, at the limit", "Night-heavy training data; fog mode with reduced features", "Data lead"],
    ["Domain shift in traffic-state estimation", f"Macro-F1 {C['state_macro_f1']:.2f} clear → {pc['fog / dust']['state_macro_f1']:.2f} fog", "Wrong congestion maps", "Condition-aware correction; speed features", "ML lead"],
    ["Late warnings in short cut-ins", f"Median {C['median_warning_lead_s']:.1f} s before TTC minimum", "Little time for the driver to act", "Faster track confirmation; radar; AEB in v2", "Systems lead"],
    ["Nuisance alerts in congestion", f"{C['nuisance_alerts_per_hour']:.0f} per hour (true TTC 1.5–3 s)", "Alert fatigue", "Hysteresis; speed-dependent thresholds", "Product lead"],
    ["Thermal throttling at peak load", f"{lat100t.dropped_pct:.1f}% frames dropped at 100 PCU with throttling", "Lower tracking rate", "Throttle DMS first; thermal design (Week 3)", "Hardware lead"],
    ["Too few critical events", f"{C['conflict_events']} per condition; lower bound {pct(C['conflict_recall_ci'][0])}", "Cannot confirm 98% recall", f"At least {n_wilson} events per condition; importance sampling", "Test lead"],
    ["Calibration of free-flow speed", f"GEH {geh(T['flow_veh_h_lane']['15'], greenshields(15)[1]):.1f} at 15 PCU/km/lane", "Unrealistic light traffic", "Survey data; sublane model for filtering", "Simulation lead"],
], caption="Challenges measured in the pilot campaign", widths=[3.0, 4.2, 3.2, 4.2, 1.9], font=8)

# ------------------------------------------------------------------ 10 expected outcomes + go/no-go
R.h1("10. Expected Outcomes")
R.emit(section(B, "9.1 Hypotheses and Expected Results"), rename={"9.1 Hypotheses and Expected Results": "10.1 Hypotheses and Expected Results"})
R.table(["Hypothesis", "Pilot evidence", "Status"], [
    ["H1 Clear conditions meet accuracy targets", f"Recall {pct(C['recall_visible'])}, distance {C['dist_mape_40']:.1f}%, MOTA {C['mota_up_to_70']:.2f}; but traffic-state F1 {C['state_macro_f1']:.2f}", "Mostly confirmed"],
    ["H2 Tracking is the weakness in dense traffic", f"MOTA {cell('clear day', 15).mota:.2f} → {cell('clear day', 100).mota:.2f}; ID switches ×{cell('clear day', 100).idsw_per_1000 / cell('clear day', 15).idsw_per_1000:.1f}", "Confirmed"],
    ["H3 Night and rain robustness ratio 0.75–0.85", f"Night {rr['night']['detection']:.2f}, rain {rr['heavy rain']['detection']:.2f}, fog {rr['fog / dust']['detection']:.2f}", "Confirmed; fog at the limit"],
    ["H4 Trajectory errors largest for two-wheelers", "Not tested: the pilot has no trajectory predictor", "Open (full campaign)"],
    ["H5 p99 latency below 100 ms", f"{lat100.p99:.0f} ms normal, {lat100t.p99:.0f} ms throttled; {lat100t.dropped_pct:.1f}% frames dropped when throttled", "Confirmed, with a frame-drop caveat"],
    ["H6 State errors between neighbouring states", "All confusions in the clear-day matrix are between adjacent states", "Confirmed"],
], caption="Hypotheses against pilot evidence", widths=[4.8, 8.4, 3.3], font=8.5)
R.h2("10.2 Go/No-Go Criteria Checked Against the Pilot")
R.p("The decision rule uses the lower bound of the 95% confidence interval for every 'at least' criterion and the upper bound for every 'at most' criterion. "
    "Criteria that the pilot cannot test, such as the simulation-to-reality gap, stay open and must be met in the full campaign and shadow-mode road tests.")
gng = [
    ["Detection recall, visible, ≤ 60 m, clear, ≤ 85 PCU", "≥ 95%", f"{pct(C['recall_visible_up_to_70'])} (lower bound {pct(C['recall_visible_up_to_70_ci'][0])})", "PASS", "-"],
    ["Distance error ≤ 40 m, all conditions", "≤ 5%", f"{C['dist_mape_40']:.2f}% clear; worst {max(pc[c]['dist_mape_40'] for c in CONDS):.2f}% (sensor fault)", "PASS", "-"],
    ["Tracking accuracy (MOTA, pilot proxy for HOTA), ≤ 85 PCU", "≥ 0.60", f"{C['mota_up_to_70']:.2f} (lower bound {C['mota_up_to_70_ci'][0]:.2f}) up to 70; {cell('clear day', 100).mota:.2f} at 100", "PASS", "Report HOTA in the full campaign"],
    ["Traffic-state macro-F1", "≥ 0.85", f"{C['state_macro_f1']:.2f}", "FAIL", "Condition-aware correction; keep output internal until fixed"],
    ["Critical conflict recall", "≥ 98% (lower bound)", f"{pct(C['conflict_recall'])} (lower bound {pct(C['conflict_recall_ci'][0])})", "FAIL (not proven)", f"Enlarge to ≥ {n_wilson} events per condition; faster track confirmation"],
    ["Missed severe conflicts (TTC < 1.0 s)", "0", f"0 of {C['severe_events']}", "PASS", "-"],
    ["False alerts (true TTC > 3 s)", "< 1 per hour (upper bound)", f"0 in {C['hours']:.0f} h (upper bound {fa_ub:.2f})", "PASS", "Confirm over ≥ 30 h"],
    ["Nuisance alerts (true TTC 1.5–3 s) [new]", "< 2 per hour", f"{C['nuisance_alerts_per_hour']:.1f} per hour", "FAIL", "Hysteresis; speed-dependent threshold"],
    ["Robustness ratio, night and rain (detection and conflicts)", "≥ 0.75", f"night {rr['night']['detection']:.2f} / {rr['night']['conflict']:.2f}; rain {rr['heavy rain']['detection']:.2f} / {rr['heavy rain']['conflict']:.2f}", "PASS", "Fog (0.75) handled by ODD limit"],
    ["p99 latency on target hardware", "< 100 ms", f"{lat100.p99:.0f} ms ({lat100t.p99:.0f} ms throttled)", "PASS", "Confirm on HIL rigs"],
    ["Frames over the deadline", "< 0.1%", f"{lat100t.deadline_miss_pct:.2f}%", "PASS", "-"],
    ["Dropped frames, worst thermal case [new]", "< 1%", f"{lat100t.dropped_pct:.1f}%", "FAIL", "Throttle DMS first; thermal design"],
    ["Simulation-to-reality gap", "≤ 10%", "Not testable in simulation", "OPEN", "Matched real data in Weeks 14–15"],
    ["Shadow-mode false alerts on public roads", "< 1 per hour", "Not testable in simulation", "OPEN", "Shadow-mode pilot"],
]
R.table(["Criterion", "Go threshold", "Pilot result", "Status", "Action"], gng, caption="Go/no-go criteria with pilot evidence", widths=[4.4, 2.3, 4.2, 1.8, 3.8], font=8)
npass = sum(1 for g in gng if g[3] == "PASS"); nfail = sum(1 for g in gng if g[3].startswith("FAIL")); nopen = sum(1 for g in gng if g[3] == "OPEN")
R.p(f"Of {len(gng)} criteria, {npass} pass, {nfail} fail and {nopen} remain open. The decision logic is:")
R.bullets([
    ("GO:", "every criterion passes, including the open ones after road testing; the model may issue live warnings across its operating domain."),
    ("CONDITIONAL GO:", "all safety-critical criteria pass (no missed severe conflicts, false alerts, latency, night and rain robustness) and each failure has a "
     "bounded fix; the model may run in shadow mode, and live only inside a restricted operating domain."),
    ("NO-GO:", "any safety-critical criterion fails, such as a missed severe conflict or p99 latency above 100 ms; the model returns to development."),
])
R.p("The pilot result is a CONDITIONAL GO. The failures are either non-safety (traffic-state F1), statistical (conflict recall of 97.2% is just below target, "
    "but its 95% interval still includes 98%), comfort-related (nuisance alerts) or a corner case with a known fix (dropped frames under throttling). The model therefore enters the "
    "200-vehicle pilot in shadow mode, with live collision warnings only above 10 km/h and outside dense fog or dust, and the full campaign of about 3,500 runs "
    "re-checks every criterion with the enlarged test set before warnings go live fleet-wide.")
R.emit(section(B, "9.3 Deliverables"), rename={"9.3 Deliverables": "10.3 Deliverables"})
R.emit(section(B, "10. Implementation Timeline and Resources"), rename={"10. Implementation Timeline and Resources": "11. Implementation Timeline and Resources"})

# ------------------------------------------------------------------ 12
R.h1("12. Conclusions")
R.p("A traffic analytics model can only be trusted on Indian roads if it is tested on the traffic it will actually meet: dense, mixed, partly hidden and often "
    "badly lit. The strategy in this report provides that test: a calibrated SUMO and CARLA co-simulation, ten scenario families under five conditions, five "
    "test levels, metrics with clear formulas, statistics that decide on confidence bounds, and explicit go/no-go criteria.")
R.p(f"Running a small version of the strategy showed that it works and that it finds real problems. On {P['simulated_hours_total']:.0f} hours of simulated mixed traffic "
    f"per condition, the stand-in model met its detection, distance, tracking, latency and severe-conflict targets in clear conditions, but missed the traffic-state "
    f"target, could not yet prove 98% conflict recall with {C['conflict_events']} events, and produced nuisance alerts in congestion. It also showed that occlusion, not "
    "detector quality, limits performance in dense traffic, and that warnings in short-gap cut-ins come too late to rely on a driver alone. The resulting "
    "conditional go, with a restricted operating domain and shadow-mode deployment, is the kind of evidence-based decision the full campaign is designed to make.")
R.emit(section(B, "References"))
for e in ["[R1]  Treiber, M., Hennecke, A. and Helbing, D. (2000). Congested traffic states in empirical observations and microscopic simulations. Physical Review E, 62(2), 1805.",
          "[R2]  Bernardin, K. and Stiefelhagen, R. (2008). Evaluating multiple object tracking performance: the CLEAR MOT metrics. EURASIP Journal on Image and Video Processing.",
          "[R3]  Hanley, J. A. and Lippman-Hand, A. (1983). If nothing goes wrong, is everything all right? Interpreting zero numerators. JAMA, 249(13), 1743–1745 (rule of three).",
          "[R4]  Indian Roads Congress (2017). Indo-HCM: Indian Highway Capacity Manual. CSIR-CRRI.",
          f"[R5]  Sharma, H. (2026). Traffic AI pilot campaign: simulation code and results. {GH}"]:
    R.p(e)
R.emit(section(B, "Appendix A: Example SUMO Vehicle Type Configuration"))
R.emit(section(B, "Appendix B: Example Parameterised Scenario (S2 Cut-In)"))
R.emit(section(B, "Appendix C: Glossary"))
R.h1("Appendix D: How to Reproduce the Pilot")
R.code("pip install numpy pandas matplotlib python-docx\npython src/traffic_pilot.py     # about 10 minutes: 24 traffic runs x 5 conditions\npython src/build_report.py      # rebuilds this document")
print("words:", R.save(OUT), "figures:", R.fig_n, "tables:", R.tab_n)
