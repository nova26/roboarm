# v3 predictor — release for hardware evaluation

A selective-spiking SSM ("4SM") that consumes 10 ms windows of DVS events
and predicts where the target will be **50 ms in the future**, in image
coordinates. Built to be the perception/prediction front-end of a
predictive visual servoing loop on a real arm.

This release packages the trained model, the code needed to load and run
it, and everything the evaluation team should know to wire it onto a
**WidowX-200 + DAVIS346** hardware setup.

---

## Contents

| file | purpose |
|---|---|
| [best.pt](best.pt) | the trained v3 checkpoint (~2.6 MB) |
| [model_loader.py](model_loader.py) | `TrackingModel` definition + `build_from_args(...)` helper that reconstructs the network from the checkpoint's `args` |
| [ssm_module.py](ssm_module.py) | underlying `StackedSpikingSSM` core (selective-spiking SSM block stack) imported by `model_loader.py`; copied verbatim from `src/main_spikingjelly_selective_c.py` |
| [event_sim.py](event_sim.py) | Python-native event simulator that matches ESIM's parameters; useful for testing without a DAVIS attached |
| [inference_example.py](inference_example.py) | end-to-end "load + simulate events + predict + compare to GT" sanity script |
| [requirements.txt](requirements.txt) | python deps (torch, numpy, spikingjelly, opencv) |

---

## What the model does

- **Input**: a tensor of binned events `(B, T, H, W, 2)`
  - `B` = batch (use `1` at inference time)
  - `T` = number of 10-ms bins (e.g. `T=20` for a 200 ms history window)
  - `H, W` = `260, 260` (matches the DAVIS346 vertical resolution after centre-crop)
  - Channel 0 = per-pixel count of **ON** events in that bin
  - Channel 1 = per-pixel count of **OFF** events in that bin
- **Output**: `(B, T, 2)` predictions in normalised coords `[-1, +1]² ↦` image
  - The prediction at bin `t` is **where the target will be at `t_end_of_bin + 50 ms`**, in `(x, y)` image-space pixels (after multiplying by `(W/2, H/2)` and shifting by `(W/2, H/2)` — there's a helper `model.norm_to_pix(...)` that does this).
- **Stateful**: the SSM keeps internal state across calls. Call `model.reset_state(B, device=...)` once at the start of each sequence; then either feed the full sequence at once, or feed bins **one at a time** (`T=1`) for a real-time loop — state is preserved.
- **Architecture summary**:
  - Frontend `conv_stem`: 3× strided conv on each `(H, W, 2)` bin → `AdaptiveAvgPool2d(4)` → linear → one 128-D feature per bin.
  - Core: **2-layer stacked selective-spiking SSM** (`d_model=128, state_dim=256`). Mamba-style selective Δt, B, C with spiking gates.
  - Head: residual-from-centroid. The model first computes the centroid of events in the input window and projects it into normalised coords; the SSM head outputs a *correction* on top, so day-zero predictions are already close to the centroid.
- **Parameter count**: ~0.68 M (1.8 MB on disk).
- **Compute**: a single tick (T=1 forward pass) runs in ~5 ms on CPU, ~1 ms on GPU. Fits inside a 10 ms control budget.

---

## Performance (Stage-1 validation)

All numbers are from the original repository's evaluation runs.

| split | RMSE (px) | best classical baseline |
|---|---|---|
| synthetic val (260 × 260) | **9.41** | 40.59 (centroid_now), 43.59 (centroid_linear) |
| synthetic test_iid | **8.84** | 37.76, 38.22 |
| synthetic test_ood | **12.07** | 41.74, 44.13 |
| real DAVIS346 + BAF, 2 s clip | mean `|pred − centroid_now|` 9.81 px; **+3.08 px signed lead in motion direction (predictive, not tracker)** | — |

In MuJoCo closed-loop (1-DOF and 2-DOF visual servoing), the same checkpoint
held the ball within ~5–6 px of image-centre across 4 s of moving-ball
tracking without any sim-domain finetuning.

---

## Quick start: smoke-test the model

```bash
pip install -r requirements.txt
python inference_example.py
```

The script:
1. Loads `best.pt`.
2. Synthesises a 2-second moving-ball scene (uniform black background,
   8-mm-diameter ball, vertical sinusoid).
3. Generates events with the Python ESIM-like simulator in `event_sim.py`.
4. Bins events into 10 ms windows.
5. Runs the model.
6. Prints per-tick predicted `(cx, cy)` next to the GT projection.

Expected output: RMSE ~20–30 px on this *toy* synthetic clip. **Higher than the
9.41 px reported above** because `inference_example.py` uses a simplified
in-memory renderer with brighter pixels than the actual ESIM pipeline, which
generates ~3× more events per second than the training distribution. Don't
read too much into this number — it's a "did this load and run end to end"
smoke check, not a quality benchmark. The real performance numbers are
the synthetic-val (9.41 px) and real-DAVIS-recording (+3.08 px lead) ones
the model was actually validated on.

---

## Minimal loading recipe (drop into your own code)

```python
import torch, sys, os
sys.path.insert(0, "<path/to/v3_predictor_release>")
from model_loader import build_from_args

ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
model = build_from_args(ckpt["args"], state_dict=ckpt["model_state_dict"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Real-time loop:
model.reset_state(1, device=device)
while True:
    bins = read_and_bin_one_tick(...)          # shape (1, 1, 260, 260, 2) float32
    with torch.no_grad():
        pred_norm = model(bins.to(device))     # (1, 1, 2)
        pred_pix  = model.norm_to_pix(pred_norm).cpu().numpy()[0, 0]  # (cx, cy)
    ...
```

---

## How the model was trained

- **Dataset**: 10 000 synthetic 2-second sequences (one per family per seed), generated by ESIM (the official RPG event-camera simulator) at 1 kHz frames → events → 10-ms bins.
- **5 motion families** (each gets 2 000 sequences):
  | family | parameters | speed |
  |---|---|---|
  | `constant_velocity` | speed + angle | 5–50 px/s |
  | `circle` | radius, 1–3 revolutions per 2 s | ω·r |
  | `lissajous` | amplitudes 15 px to half-image, base freq 0.5–1.5 Hz, 7 axis-ratios | up to a few hundred px/s |
  | `piecewise_linear` | 3–6 random waypoints, constant speed per segment | as CV |
  | `bounded_random_walk` | bandwidth 2–8 Hz, max-accel 80–250 px/s², max-speed 25–90 px/s | varies |
- **Image size**: **260 × 260** (matches DAVIS346 vertical resolution after centre-crop). Target ball radius 6 px, sub-pixel rendered with `cv2.circle(..., shift=4, lineType=cv2.LINE_AA)`.
- **No noise** in the training data (background is uniform black). v1 and v2 tried noise injection and lost the predictive lead on real DVS — they're documented as ablations. v3 (this checkpoint) is the clean-training recipe at native resolution.
- **Loss**: masked L2 in normalised coords, prediction horizon Δ = **50 ms**.
- **Train/val/test_iid/test_ood splits**: 6 391 / 798 / 799 / 2 012 (test_ood holds out the highest-difficulty parameter regions of each family).
- **Architecture knobs** for v3 (you can read these directly from `ckpt["args"]` too):
  - `frontend="conv_stem"`, `conv_pool_size=4`, `conv_kernel=3`
  - `d_model=128`, `state_dim=256`, `n_layers=2`, `dropout=0.1`
  - `horizons_ms=[50.0]`, `use_z_gate=False`
- **Optimiser**: AdamW, `lr=3e-4`, `weight_decay=1e-4`, `grad_clip=1.0`, batch size 4, 30 epochs on an A100. Wall time ~3.5 h.

Four post-v3 ablations (more capacity, multi-horizon, two velocity-loss
variants) all failed to beat v3's +3.08 px real-DVS lead. The conclusion
was that the data's slow-motion regime is the predictor's bottleneck,
not the model or the loss.

---

## Goal — how this model fits the hardware pipeline

```
  DAVIS346 events  ──→  10-ms binning + BAF denoise  ──→  (1, 1, 260, 260, 2)
                                                              │
                                                              ▼
                                                          v3 model
                                                              │
                                                              ▼
                                                  (cx_pred, cy_pred)  ← at t + 50 ms
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                pixel error vs image centre        (optional: log to disk)
                                              │
                                              ▼
                       per-joint Δq = -err / J_pre_calibrated
                                              │
                                              ▼
                              Dynamixel position-command write
                                              │
                                              ▼
                                 arm moves → camera moves with it
                                              │
                                              └────────── loop ───────────────┐
                                                                              │
                                                                              ▼
                                                                  next 10-ms tick
```

The **only learned component** is v3. Everything else is a pre-calibrated
linear map (camera-Jacobian) plus the DAVIS346 + Dynamixel SDK clients.

---

## Recommended hardware integration plan

### Phase 5.0 — Hand-eye calibration (half day)

Mount the DAVIS346 rigidly on the EE. Run a standard hand-eye calibration
(ArUco grid or checkerboard at known poses, OpenCV's `calibrateHandEye`)
to determine:
- camera intrinsics (`fx, fy, cx, cy` of the DAVIS346)
- camera-to-EE extrinsic transform

Save the result to a YAML next to the model.

### Phase 5.1 — Open-loop "predict-but-don't-act" (most important)

**This is the single highest-information transfer test. Do it before any closed-loop.**

Set the arm to a known pose with the ball at a known world position
(measured by hand, or with an ArUco marker on the ball). Stream live
DAVIS346 events for 1–2 s. Bin them, run the model, log the predicted
`(cx, cy)`. Compare to the GT pixel position computed by projecting the
known ball world position through the (intrinsic, extrinsic) calibration.

**Pass criterion**: predicted position should be within ~10–20 px of the
GT projection across the workspace, with the **signed lead in the
motion direction positive** (model predicts ahead of the ball).

If this fails, the most likely cause is **lighting/background mismatch
between your lab and the synthetic training data**. The training data
had a uniform black background; if your real workspace has bright
textures or background motion, the model may need BAF denoising tuning
or a small fine-tuning run on a few labelled real-DVS clips. The good
news is that the original `dvs_ball_test.aedat4` recording (which v3
was validated against) was a real DAVIS346 with a moderately cluttered
background, and v3 still delivered a +3.08 px lead — but every lab is
different.

### Phase 5.2 — Open-loop motor test

Independently, verify your Dynamixel control loop. Send position
commands at 100 Hz. Measure:
- settling time per joint (should be <0.5 s for ±5° step)
- max safe slew rate (you'll cap controller output at this rate)

The reference WidowX-200 servo configuration is in
[wx200_summary.md](https://github.com/nova26/roboarm/blob/main/wx200_summary.md):
all 5 main joints use XM430-W350 (4.1 N·m stall, 57 RPM max). For
visual servoing we only use joint0 (waist) and joint1 (shoulder).

### Phase 5.3 — Closed loop, stationary ball

Combine 5.1 + 5.2. Place the ball at a fixed offset from the centre of
the EE camera's FOV; let the loop run.

The control law (both axes are independent):

```python
# At every 10-ms tick:
pred_cx, pred_cy = run_v3(latest_10ms_bin)
err_x = pred_cx - image_centre_x
err_y = pred_cy - image_centre_y

# J = pixel-shift per joint-radian, signed; calibrate at startup
# (rotate joint by ±5°, project a fixed world point, fit slope).
delta_q_waist    = -K_c * err_x / dcx_per_waist
delta_q_shoulder = -K_c * err_y / dcy_per_shoulder

q_waist_target    = current_q_waist    + delta_q_waist
q_shoulder_target = current_q_shoulder + delta_q_shoulder
dynamixel_write_position(WAIST,    q_waist_target)
dynamixel_write_position(SHOULDER, q_shoulder_target)
```

Choose `K_c` so that the per-tick angular change is bounded below the
servo's max safe slew rate (measured in 5.2).

**Sanity check**: with the ball at home, the loop should converge to
`|pred_cy - centre| < 5 px` within a few seconds.

### Phase 5.4 — Closed loop, moving ball

Once 5.3 works, swing the ball through known offsets. Compare to the
MuJoCo step-and-settle benchmarks in the repo (e.g. ~60–420 ms settling
times for ±6 cm setpoints in sim).

---

## Notes on the real-time pipeline

- **Event reader**: a thread that reads from the DAVIS346 SDK and accumulates `(x, y, t_us, polarity)` tuples into a ring buffer. Every 10 ms the main loop pulls events with `t > tick_start` and bins them with the helper in `inference_example.py`.
- **Denoising**: the original real-DVS evaluation in the source repo used a Background-Activity Filter (BAF) before binning — drop events where the most recent neighbour fired more than 2 ms ago. Useful for noisy backgrounds. See `tracking/training/run_real_dvs.py` in the source repo for a reference implementation.
- **Centre-crop**: DAVIS346 is 346 × 260. Centre-crop the events to a 260 × 260 window (drop `x` outside `[43, 303]` and shift origin) before binning.
- **Watchdog**: if a tick's v3 inference takes >50 ms (e.g. GIL contention with the camera thread), freeze the shoulder/waist commands at the previous target — do not extrapolate or accumulate.

---

## What this release is *not*

- It is **not** an ego-motion-aware model. v3 was trained on a static camera. It handled moderate ego motion in MuJoCo simulation, but if your arm slews very fast you may see the predictor degrade.
- It is **not** trained on real-world backgrounds. If transfer fails, the next step is to capture a small real-DVS dataset (10–100 short clips with hand-labelled ball positions) and fine-tune for a few epochs.
- The Dynamixel control + camera streaming code is **not included** here — those are in [github.com/nova26/roboarm](https://github.com/nova26/roboarm) (`widowx_arm.py`, `pid_controller.py`, `widowx_run.py`).

---

## Citation / questions

- Source repository: `selectivespiking` (private), branch `login8`.
- This release was produced from commit referenced in the parent repo's
  Stage-2 history.
- Trained checkpoint identity: `v3_571478` (SLURM job id; 30 epochs on A100).
- For questions, contact the original training team — they have the full
  evaluation logs, ablation results, and the MuJoCo closed-loop scripts
  that produced the 5.58 px RMSE result.
