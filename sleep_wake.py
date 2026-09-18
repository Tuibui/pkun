#!/usr/bin/env python3
"""
sleep_wake.py — ท่านอน (sleep) และท่าลุกขึ้นยืน (wakeup) ของหุ่นสี่ขา

ทำไมต้องมีท่านอน: เซอร์โว analog ไม่มี feedback ต้องออกแรงค้างตลอดเวลาที่ยืน -> ร้อนและกินไฟ
ท่านอนที่ดีคือท่าที่ 'ท้องแตะพื้น' แล้วตัดไฟเซอร์โวได้เลย แรงบิดค้างเหลือศูนย์

sleep : ยืน -> ย่อลง -> พับขาเก็บข้างลำตัว -> ท้องแตะพื้น (ตัด PWM ได้)
wakeup: นอน -> กางขาลงหาพื้นก่อน -> ค่อยดันตัวขึ้น -> ยืน

รัน:  python sleep_wake.py                    ตรวจ + สรุปผลทั้งสองท่า
      python sleep_wake.py --csv-prefix out   บันทึก out_sleep.csv / out_wake.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from leg_kinematics import LEGS, LEG_SIGNS, PARAMS, Params, leg_fk, leg_polyline
from walking_gait import (
    BODY_PAYLOADS,
    Gait,
    Payload,
    Servo,
    Trajectory,
    center_of_gravity,
    export_csv,
    nominal_stance,
    robot_ik,
)

# ---------------------------------------------------------------------------
# 1) ลิมิตข้อต่อ — เซอร์โว 180° หมุนได้ ±90° จากกลาง แต่โครงชนกันก่อน
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointLimits:
    """ช่วงมุมที่ข้อต่อไปได้จริง (องศา) — วัดจากหุ่นจริงแล้วมาแก้."""

    th1: Tuple[float, float] = (-55.0, 55.0)    # abduction ถ่างเข้า-ออก
    th2: Tuple[float, float] = (-90.0, 90.0)    # hip pitch
    th3: Tuple[float, float] = (0.0, 150.0)     # knee พับได้ทางเดียว 0 = เหยียดตรง

    def as_array(self) -> np.ndarray:
        """คืน (2,3): แถวแรก min แถวสอง max หน่วยเรเดียน."""
        return np.radians(np.array([[self.th1[0], self.th2[0], self.th3[0]],
                                    [self.th1[1], self.th2[1], self.th3[1]]]))


LIMITS = JointLimits()


def clamp_to_limits(q_all: Sequence[float], lim: JointLimits = LIMITS) -> np.ndarray:
    """บีบมุมทั้ง 12 ตัวให้อยู่ในลิมิต."""
    lo, hi = lim.as_array()
    q = np.asarray(q_all, dtype=float).reshape(len(LEGS), 3)
    return np.clip(q, lo, hi).ravel()


def limit_violations(q_all: Sequence[float], lim: JointLimits = LIMITS) -> List[str]:
    """คืนรายการข้อต่อที่หลุดลิมิต (ว่าง = ผ่าน)."""
    lo, hi = lim.as_array()
    q = np.asarray(q_all, dtype=float).reshape(len(LEGS), 3)
    out = []
    for i, leg in enumerate(LEGS):
        for j in range(3):
            if q[i, j] < lo[j] - 1e-9 or q[i, j] > hi[j] + 1e-9:
                out.append(f"{leg} θ{j + 1} = {np.degrees(q[i, j]):.1f}° "
                           f"(ลิมิต {np.degrees(lo[j]):.0f}..{np.degrees(hi[j]):.0f})")
    return out


# ---------------------------------------------------------------------------
# 2) หาท่านอน: ให้ลำตัวเตี้ยที่สุดเท่าที่ลิมิตข้อต่อยอม
# ---------------------------------------------------------------------------


def body_clearance(q_leg: Sequence[float], leg: str, p: Params) -> float:
    """ระยะจากระนาบลำตัวถึงจุดต่ำสุดของขา (เข่าหรือเท้าก็ได้) — คือความสูงลำตัวเมื่อวางบนพื้น."""
    return float(-np.min(leg_polyline(leg, q_leg, p)[:, 2]))


def leg_extent(q_leg: Sequence[float], leg: str, p: Params) -> float:
    """ขายื่นไปหน้า/หลังไกลสุดเท่าไร วัดจากศูนย์กลางลำตัว (ใช้ดูว่าท่านอนกินที่แค่ไหน)."""
    return float(np.max(np.abs(leg_polyline(leg, q_leg, p)[:, 0])))


def find_sleep_pose(p: Params = PARAMS, lim: JointLimits = LIMITS,
                    min_clear: float = 14.0, style: str = "tuck",
                    step_deg: float = 2.5) -> np.ndarray:
    """หามุม (θ1,θ2,θ3) ของท่านอนที่ลำตัวเตี้ยที่สุดเท่าที่ลิมิตยอม.

    style = "tuck"  พับเข่าเก็บขาแนบตัว (ค่าตั้งต้น) — สูงกว่านิดหน่อยแต่กะทัดรัดและแรงบิดต่ำ
    style = "splat" ขาเหยียดกางออก — เตี้ยที่สุด แต่ขายื่นไกลและแรงบิดตอนพับสูงมาก
    min_clear = ความหนาครึ่งหนึ่งของลำตัว ลำตัวต้องไม่จมลงไปในพื้น
    """
    knee_min = 90.0 if style == "tuck" else lim.th3[0]
    best_q, best_h = None, np.inf
    for th1 in np.radians(np.arange(0.0, lim.th1[1] + 0.1, step_deg)):
        for th2 in np.radians(np.arange(lim.th2[0], lim.th2[1] + 0.1, step_deg)):
            for th3 in np.radians(np.arange(knee_min, lim.th3[1] + 0.1, step_deg)):
                q = np.array([th1, th2, th3])
                h = body_clearance(q, "FL", p)
                if h < min_clear or h >= best_h:
                    continue
                # ขาต้องไม่พับข้ามไปใต้ท้อง (กันชนกับลำตัวและขาอีกข้าง)
                if leg_fk("FL", q, p)[-1][1, 3] < p.b:
                    continue
                best_q, best_h = q, h
    if best_q is None:
        raise RuntimeError("หาท่านอนไม่ได้ ลองผ่อนลิมิตหรือลด min_clear")
    return best_q


def mirror_pose(q_leg: Sequence[float]) -> Dict[str, np.ndarray]:
    """ขยายมุมของขาซ้ายหน้าไปครบทั้ง 4 ขา (ขาขวากลับเครื่องหมาย θ1)."""
    q = np.asarray(q_leg, dtype=float)
    poses = {}
    for leg in LEGS:
        _, sb, _, _ = LEG_SIGNS[leg]
        poses[leg] = np.array([sb * q[0], q[1], q[2]])
    return poses


def pose_vector(q_leg: Sequence[float]) -> np.ndarray:
    """มุม 12 ตัวจากท่าของขาเดียว."""
    poses = mirror_pose(q_leg)
    return np.concatenate([poses[leg] for leg in LEGS])


# ---------------------------------------------------------------------------
# 3) สร้าง trajectory จาก keyframe
# ---------------------------------------------------------------------------


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """เร่ง-ชะลอนุ่ม ๆ ความเร็วเป็นศูนย์ที่หัวและท้าย (สำคัญมากกับเซอร์โว analog)."""
    return u * u * (3.0 - 2.0 * u)


def stance_pose(p: Params, g: Gait, height: float) -> np.ndarray:
    """มุม 12 ตัวของท่ายืนที่ความสูงลำตัวเท่ากับ height."""
    return robot_ik(nominal_stance(p, replace(g, stand_h=height)), p, g.knee)


def keyframe_trajectory(keys: Sequence[Tuple[np.ndarray, float]], p: Params,
                        payloads: Sequence[Payload] = tuple(BODY_PAYLOADS),
                        hz: float = 50.0) -> Trajectory:
    """ต่อ keyframe (มุม 12 ตัว, เวลาที่ใช้ไปถึงเฟรมนั้น) เป็น trajectory เต็ม ๆ."""
    ts: List[float] = []
    qs: List[np.ndarray] = []
    t0 = 0.0
    for i in range(len(keys) - 1):
        q_a, _ = keys[i]
        q_b, dur = keys[i + 1]
        n = max(1, int(round(dur * hz)))
        for k in range(n):
            u = (k + 1) / n
            s = _smoothstep(np.array([u]))[0]
            qs.append((1.0 - s) * q_a + s * q_b)
            ts.append(t0 + (k + 1) / hz)
        t0 = ts[-1]

    q = np.vstack([keys[0][0], np.array(qs)])
    t = np.concatenate([[0.0], np.array(ts)])

    feet = np.zeros((len(t), 4, 3))
    cog = np.zeros((len(t), 3))
    for k in range(len(t)):
        for i, leg in enumerate(LEGS):
            feet[k, i] = leg_fk(leg, q[k, 3 * i:3 * i + 3], p)[-1][:3, 3]
        cog[k], mass = center_of_gravity(q[k], p, payloads)

    contact = np.ones((len(t), 4), dtype=bool)
    margin = np.full(len(t), np.nan)      # นอน/ลุก ใช้สี่ขายัน ไม่ต้องคิด margin แบบเดิน
    return Trajectory(t, q, feet, contact, cog, margin, mass, 0.0, True)


# ---------------------------------------------------------------------------
# 4) ท่านอน และท่าลุก
# ---------------------------------------------------------------------------

SETTLE_H = 62.0     # ความสูงลำตัวช่วง 'ย่อ' ก่อนพับขา / ช่วงตั้งหลักก่อนดันขึ้น


def sleep_sequence(p: Params = PARAMS, g: Gait = Gait(), sleep_q: np.ndarray | None = None,
                   t_settle: float = 2.5, t_fold: float = 3.0) -> Trajectory:
    """ยืน -> ย่อลงต่ำ -> พับขาเก็บข้างตัว -> ท้องแตะพื้น."""
    sleep_q = find_sleep_pose(p) if sleep_q is None else np.asarray(sleep_q)
    return keyframe_trajectory([
        (stance_pose(p, g, g.stand_h), 0.0),
        (stance_pose(p, g, SETTLE_H), t_settle),
        (pose_vector(sleep_q), t_fold),
    ], p, hz=g.hz)


def wake_sequence(p: Params = PARAMS, g: Gait = Gait(), sleep_q: np.ndarray | None = None,
                  t_unfold: float = 3.0, t_rise: float = 2.5) -> Trajectory:
    """นอน -> กางขาลงหาพื้นก่อน (ยังไม่ยกตัว) -> ดันตัวขึ้นถึงความสูงยืน."""
    sleep_q = find_sleep_pose(p) if sleep_q is None else np.asarray(sleep_q)
    return keyframe_trajectory([
        (pose_vector(sleep_q), 0.0),
        (stance_pose(p, g, SETTLE_H), t_unfold),
        (stance_pose(p, g, g.stand_h), t_rise),
    ], p, hz=g.hz)


# ---------------------------------------------------------------------------
# 5) ตรวจข้อจำกัด
# ---------------------------------------------------------------------------


def check(traj: Trajectory, name: str, p: Params, g: Gait, servo: Servo = Servo(),
          lim: JointLimits = LIMITS, verbose: bool = True) -> Dict[str, float]:
    """ตรวจลิมิตข้อต่อ + ความเร็ว + แรงบิด ของท่า sleep/wake."""
    deg = np.degrees(traj.q)
    rate = np.abs(np.diff(deg, axis=0)) * g.hz
    peak_rate = float(rate.max())

    bad = set()
    for k in range(len(traj.t)):
        for v in limit_violations(traj.q[k], lim):
            bad.add(v.split(" = ")[0])

    # แรงบิดตอนขาพับมาก ๆ คือช่วงอันตรายที่สุด (โมเมนต์ยาว) — คิดแบบสี่ขารับเท่ากัน
    w_n = traj.mass_g * 1e-3 * 9.81 / 4.0
    tau2 = tau3 = 0.0
    for k in range(len(traj.t)):
        for i, leg in enumerate(LEGS):
            chain = leg_fk(leg, traj.q[k, 3 * i:3 * i + 3], p)
            o1, o2, foot = chain[1][:3, 3], chain[2][:3, 3], chain[3][:3, 3]
            tau2 = max(tau2, w_n * abs(foot[0] - o1[0]) * 1e-3)
            tau3 = max(tau3, w_n * abs(foot[0] - o2[0]) * 1e-3)
    to_kgcm = 10.197
    res = {
        "duration_s": float(traj.t[-1]),
        "peak_rate_dps": peak_rate,
        "tau_hip_kgcm": tau2 * to_kgcm,
        "tau_knee_kgcm": tau3 * to_kgcm,
        "budget_kgcm": servo.stall_kgcm * servo.usable_torque,
        "final_clearance_mm": body_clearance(traj.q[-1, :3], "FL", p),
        "start_clearance_mm": body_clearance(traj.q[0, :3], "FL", p),
        "n_limit_violations": len(bad),
    }

    if verbose:
        print(f"--- {name} ---")
        print(f"  ใช้เวลา                {res['duration_s']:6.1f} s")
        print(f"  ความสูงลำตัว           {res['start_clearance_mm']:6.1f} -> "
              f"{res['final_clearance_mm']:.1f} mm")
        print(f"  ความเร็วข้อต่อสูงสุด     {res['peak_rate_dps']:6.0f} °/s "
              f"(เซอร์โวไหว {servo.max_rate_dps:.0f})")
        print(f"  แรงบิด hip / knee     {res['tau_hip_kgcm']:6.2f} / {res['tau_knee_kgcm']:.2f} kg·cm "
              f"(งบ {res['budget_kgcm']:.2f})")
        print("     ^ เป็นค่า upper bound: คิดว่าขารับน้ำหนักเต็มตลอดทาง "
              "ช่วงท้ายที่ท้องแตะพื้นแล้วของจริงจะเบากว่านี้")
        if bad:
            print(f"  ERROR: ข้อต่อหลุดลิมิต {sorted(bad)}")
        if peak_rate > servo.max_rate_dps:
            print("  เตือน: เร็วเกินเซอร์โว -> เพิ่มเวลา t_settle/t_fold")
        if max(res["tau_hip_kgcm"], res["tau_knee_kgcm"]) > res["budget_kgcm"]:
            print("  เตือน: แรงบิดเกินงบระหว่างทาง (ช่วงขาพับมากคือช่วงหนักสุด)")
    return res


def verify(p: Params = PARAMS) -> None:
    """ตรวจว่าท่านอน/ท่าลุกอยู่ในลิมิต ต่อเนื่อง และเชื่อมกับท่ายืนได้จริง."""
    q_sleep = find_sleep_pose(p)
    assert not limit_violations(pose_vector(q_sleep)), "ท่านอนหลุดลิมิต"

    g = Gait()
    s = sleep_sequence(p, g, q_sleep)
    w = wake_sequence(p, g, q_sleep)

    for name, tr in (("sleep", s), ("wake", w)):
        assert np.all(np.isfinite(tr.q)), f"{name}: มุมมี NaN"
        jump = float(np.max(np.abs(np.diff(tr.q, axis=0))))
        assert jump < np.radians(15.0), f"{name}: มุมกระโดด {np.degrees(jump):.1f}° ต่อสเต็ป"

    # นอนแล้วลุก ต้องกลับมาที่ท่ายืนเดิมเป๊ะ และหัว-ท้ายของสองท่าต้องต่อกันสนิท
    assert np.allclose(s.q[0], w.q[-1], atol=1e-9), "ลุกแล้วไม่กลับมาท่ายืนเดิม"
    assert np.allclose(s.q[-1], w.q[0], atol=1e-9), "ท่านอนของ sleep กับ wake ไม่ตรงกัน"

    h_stand = body_clearance(s.q[0, :3], "FL", p)
    h_sleep = body_clearance(s.q[-1, :3], "FL", p)
    assert h_sleep < h_stand, "ท่านอนไม่ได้เตี้ยกว่าท่ายืน"
    print(f"PASS: ท่านอน θ = {np.round(np.degrees(q_sleep), 1)}°  "
          f"ลำตัวลดจาก {h_stand:.1f} -> {h_sleep:.1f} mm, ไม่หลุดลิมิต, ต่อกันสนิท")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-prefix", metavar="NAME", help="บันทึก NAME_sleep.csv / NAME_wake.csv")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    verify(PARAMS)
    if args.verify_only:
        return 0

    g = Gait()
    q_sleep = find_sleep_pose(PARAMS)
    print("=" * 72)
    print("เทียบสไตล์ท่านอน:")
    for style in ("tuck", "splat"):
        q = find_sleep_pose(PARAMS, style=style)
        tr = sleep_sequence(PARAMS, g, q)
        r = check(tr, style, PARAMS, g, verbose=False)
        print(f"  {style:6s} θ={np.round(np.degrees(q), 0)}  ลำตัวสูง {r['final_clearance_mm']:5.1f} mm  "
              f"ขายื่น {leg_extent(q, 'FL', PARAMS):5.1f} mm  "
              f"τhip {r['tau_hip_kgcm']:.2f} kg·cm")
    print("=" * 72)
    s = sleep_sequence(PARAMS, g, q_sleep)
    check(s, "SLEEP  (ยืน -> นอน)", PARAMS, g)
    w = wake_sequence(PARAMS, g, q_sleep)
    check(w, "WAKEUP (นอน -> ยืน)", PARAMS, g)
    print("=" * 72)
    print("  พอถึงท่านอนแล้วตัด PWM ได้เลย ลำตัววางบนพื้น เซอร์โวไม่ต้องออกแรงค้าง")
    print("=" * 72)

    if args.csv_prefix:
        export_csv(s, f"{args.csv_prefix}_sleep.csv")
        export_csv(w, f"{args.csv_prefix}_wake.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
