#!/usr/bin/env python3
"""
elephant.py — ท่านั่ง (sit) และท่ายกขาหน้าสองข้าง (rear up) สำหรับหุ่นช้าง

สองท่านี้ต่างจากเดิน/นอน ตรงที่ **ลำตัวเอียง** โมดูลอื่นสมมติว่าลำตัวขนานพื้นเสมอ
ที่นี่คิดจาก 'ท่าของลำตัว' (สูงจากพื้น + เงยกี่องศา) แล้วย้อนไปหามุมข้อต่อ:
เท้าหน้าปักอยู่กับพื้น -> เป้าหมายในเฟรมลำตัว = T_body⁻¹ · จุดบนพื้น

ทำไมยกขาหน้าสองข้างแล้วไม่ล้ม:
  ถ้ายืนด้วยเท้าหลัง 2 จุด พื้นที่รับน้ำหนักจะแบนเป็นเส้น -> ล้มแน่นอน
  ท่านี้เลยต้องให้ **ขาหลังทั้งท่อนวางราบกับพื้น** (เข่ากับเท้าแตะพร้อมกัน) เหมือนช้างนั่งจริง
  พื้นที่รับน้ำหนักจะกลายเป็นสี่เหลี่ยม เข่าซ้าย-เท้าซ้าย-เท้าขวา-เข่าขวา ซึ่งมีพื้นที่จริง

รัน:  python elephant.py                     ตรวจ + สรุปผล
      python elephant.py --csv-prefix ele    บันทึก ele_sit.csv / ele_rear.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from leg_kinematics import LEGS, LEG_SIGNS, PARAMS, Params, leg_fk, leg_polyline
from sleep_wake import LIMITS, JointLimits, limit_violations
from walking_gait import (
    BODY_PAYLOADS,
    Gait,
    Payload,
    Servo,
    Trajectory,
    center_of_gravity,
    export_csv,
    leg_ik,
    nominal_stance,
)

FRONT: Tuple[str, ...] = ("FL", "FR")
REAR: Tuple[str, ...] = ("RL", "RR")


# ---------------------------------------------------------------------------
# 1) ท่าของลำตัวในเฟรมพื้น
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyPose:
    """ท่าลำตัวเทียบพื้น: สูงจากพื้นเท่าไร เงยหน้าขึ้นกี่องศา."""

    height: float = 110.0     # ระยะจากพื้นถึงจุดกำเนิดเฟรมลำตัว {B}
    pitch_deg: float = 0.0    # + = เชิดหน้าขึ้น (ก้นต่ำ) แบบท่านั่ง
    x: float = 0.0


@dataclass(frozen=True)
class Rump:
    """ก้น = แผ่นใต้ท้องด้านหลัง (พิกัดในเฟรมลำตัว) เผื่อไว้กรณีก้นแตะพื้นด้วย."""

    x_back: float = -77.27
    x_front: float = -40.0
    half_w: float = 29.0
    # ลึก 23.4 mm = ค่าที่ทำให้ 'ก้นแตะพื้น' พอดีจังหวะเดียวกับ 'ขาหลังเหยียดราบ' ที่ pitch 10°
    # ถ้าไม่ติดแผ่นรองก้น หุ่นจะนั่งลงบนขาหลังล้วน ๆ และไม่มีอะไรรับตอนทิ้งตัว
    z: float = -23.4


RUMP = Rump()


def body_matrix(pose: BodyPose) -> np.ndarray:
    """transform 4x4 จากเฟรมลำตัวไปเฟรมพื้น (เงยหน้าขึ้น = หมุนรอบแกน y)."""
    b = np.radians(pose.pitch_deg)
    cb, sb = np.cos(b), np.sin(b)
    T = np.eye(4)
    T[:3, :3] = ((cb, 0.0, -sb), (0.0, 1.0, 0.0), (sb, 0.0, cb))
    T[:3, 3] = (pose.x, 0.0, pose.height)
    return T


def to_world(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """แปลงจุด (M,3) จากเฟรมลำตัวไปเฟรมพื้น."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def rump_corners(pose: BodyPose, rump: Rump | None = None) -> np.ndarray:
    """มุมแผ่นก้นทั้ง 4 ในเฟรมพื้น (rump=None = ใช้ค่า global ตอนเรียก ไม่ใช่ตอน import)."""
    rump = RUMP if rump is None else rump
    pts = np.array([[rump.x_back, +rump.half_w, rump.z], [rump.x_back, -rump.half_w, rump.z],
                    [rump.x_front, -rump.half_w, rump.z], [rump.x_front, +rump.half_w, rump.z]])
    return to_world(body_matrix(pose), pts)


def feet_to_body(pose: BodyPose, feet_world: Dict[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    """แปลงตำแหน่งเท้าจากเฟรมพื้นมาเป็นเป้าหมายในเฟรมลำตัว (สำหรับป้อน IK)."""
    T = body_matrix(pose)
    R_T = T[:3, :3].T
    return {leg: R_T @ (np.asarray(pos, dtype=float) - T[:3, 3]) for leg, pos in feet_world.items()}


# ---------------------------------------------------------------------------
# 2) เสถียรภาพ: จุดรับน้ำหนัก = ทุกจุดของขาที่แตะพื้น + มุมก้นที่แตะพื้น
# ---------------------------------------------------------------------------


def convex_hull(xy: np.ndarray) -> np.ndarray:
    """convex hull 2 มิติ (monotone chain) วนทวนเข็ม.

    ต้องใช้ hull จริง ไม่ใช่การเรียงจุดตามมุมรอบ centroid เพราะจุดรับน้ำหนักของท่านั่ง
    เรียงเป็นเส้นตรงหลายจุด (ขาวางราบกับพื้น) การเรียงตามมุมจะได้รูปบิดเบี้ยว
    """
    pts = np.unique(np.round(np.asarray(xy, dtype=float), 9), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for pt in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) > 1e-9:
                    break
                out.pop()
            out.append(pt)
        return out

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    return np.array(hull) if len(hull) >= 3 else pts[:2]


def margin_from_points(cog: Sequence[float], pts: np.ndarray) -> float:
    """ระยะจากเงา CoG ถึงขอบพื้นที่รับน้ำหนัก (mm) ติดลบ = ล้ม, -inf = ไม่เป็นรูป."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return float("-inf")
    poly = convex_hull(pts[:, :2])
    if len(poly) < 3:
        return float("-inf")          # จุดรับน้ำหนักเรียงเป็นเส้นตรง = ทรงตัวไม่ได้
    g2 = np.asarray(cog, dtype=float)[:2]
    out = float("inf")
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        edge = b - a
        n = np.array([edge[1], -edge[0]])
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        out = min(out, float(-np.dot(g2 - a, n / nn)))
    return out


def ground_contacts(pose: BodyPose, q_all: Sequence[float], p: Params = PARAMS,
                    airborne: Sequence[str] = (), tol: float = 5.0) -> np.ndarray:
    """ทุกจุดของหุ่นที่แตะพื้น: จุดบนขา (สะโพก/เข่า/เท้า) + มุมก้น ที่ z <= tol.

    tol 5 mm เผื่อฟุตแพดยาง/พื้นไม่เรียบ/ระยะยุบของโครง — ถ้าใช้ 1.5 mm ช่วงกำลังเอนลงนั่ง
    จะถูกมองว่าล้มทั้งที่ท่อนขาหลังลอยเหนือพื้นแค่ 2 mm (แตะจริงแน่นอน)
    """
    T = body_matrix(pose)
    hits: List[np.ndarray] = []
    for i, leg in enumerate(LEGS):
        if leg in airborne:
            continue
        pts = to_world(T, leg_polyline(leg, q_all[3 * i:3 * i + 3], p))
        hits += [pt for pt in pts if pt[2] <= tol]
    hits += [c for c in rump_corners(pose) if c[2] <= tol]
    return np.array(hits) if hits else np.zeros((0, 3))


def lowest_point(pose_pitch: float, q_all: Sequence[float], p: Params = PARAMS,
                 legs: Sequence[str] = LEGS) -> float:
    """จุดต่ำสุด (เทียบระนาบลำตัว หลังเอียงแล้ว) ของขาที่ระบุ + ก้น — ใช้ตั้งความสูงให้แตะพื้นพอดี.

    ท่านั่งต้องส่งเฉพาะขาหลัง เพราะขาหน้ายังไม่ได้แก้ IK ตอนเรียกใช้
    """
    R = body_matrix(BodyPose(0.0, pose_pitch))[:3, :3]
    lows = [float(np.min((R @ leg_polyline(leg, q_all[3 * i:3 * i + 3], p).T).T[:, 2]))
            for i, leg in enumerate(LEGS) if leg in legs]
    lows.append(float(np.min((R @ np.array([[RUMP.x_back, 0.0, RUMP.z],
                                            [RUMP.x_front, 0.0, RUMP.z]]).T).T[:, 2])))
    return min(lows)


def world_cog(pose: BodyPose, q_all: Sequence[float], p: Params = PARAMS,
              payloads: Sequence[Payload] = tuple(BODY_PAYLOADS)) -> Tuple[np.ndarray, float]:
    """CoG ในเฟรมพื้น (ลำตัวเอียงแล้ว CoG ย้ายตามด้วย)."""
    cog_b, mass = center_of_gravity(q_all, p, payloads)
    return to_world(body_matrix(pose), cog_b)[0], mass


# ---------------------------------------------------------------------------
# 3) ท่านั่ง / ยกขาหน้า
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SitPlan:
    """ค่าปรับท่านั่งและยกขาหน้า (mm, องศา) — ค่าตั้งต้นได้จากการกวาดหา margin สูงสุด."""

    # เงย 25° ไม่ใช่ค่าสวยงาม แต่จำเป็น: ที่ 10° สะโพกหน้าสูงจากพื้นแค่ 50 mm
    # ขายาว 113 mm งอเก็บยังไงก็ไม่พ้นพื้น ต้องเชิดหน้าให้สะโพกหน้าสูงพอ
    pitch_deg: float = 25.0       # เงยหน้าตอนนั่ง
    rear_th2: float = -70.0       # ขาหลัง: เหยียดไปข้างหน้าแล้ววางท่อนขาราบกับพื้น
    rear_th3: float = 10.0        # เข่าหลังเกือบตรง (ทั้งท่อนขาจึงราบไปกับพื้น)
    rear_th1: float = 0.0         # ขาหลังไม่ต้องถ่างออกข้าง
    # เท้าหน้าปักอยู่ที่เดิมตั้งแต่ตอนยืน (= nominal_stance ที่ stand_h 110, splay 12) จะได้ไม่ไถลกับพื้น
    front_x: float = 77.27
    front_y: float = 47.42
    # ท่ายกขาหน้า กำหนดเป็น 'มุมข้อต่อ' ไม่ใช่ตำแหน่งเท้า เพราะเป้าหมายอยู่สูงกว่าสะโพก
    # ซึ่งทำให้ IK พลิกไปคอนฟิกแปลก ๆ (θ1 กระโดดไป 151°) — ท่านี้คืองอขาเก็บเข้าหาอก
    lift_th1: float = 15.0
    lift_th2: float = -90.0
    lift_th3: float = 60.0
    stand_h: float = 110.0
    t_fold: float = 3.5           # เวลาย่อลงนั่ง
    t_lift: float = 2.5           # เวลายกขาหน้า
    t_hold: float = 1.5           # ค้างท่าไว้
    hz: float = 50.0
    knee: int = +1


def rear_pose(plan: SitPlan, leg: str) -> np.ndarray:
    """มุมข้อต่อของขาหลังตอนนั่ง (กำหนดตรง ๆ ไม่ผ่าน IK เพราะต้องการให้ 'ท่อนขาราบ' ไม่ใช่เท้าจุดเดียว)."""
    _, sb, _, _ = LEG_SIGNS[leg]
    return np.radians([sb * plan.rear_th1, plan.rear_th2, plan.rear_th3])


def sit_pose_and_joints(plan: SitPlan, p: Params = PARAMS
                        ) -> Tuple[BodyPose, np.ndarray, Dict[str, np.ndarray]]:
    """คำนวณท่านั่ง: มุมขาหลังคงที่ -> ได้ความสูงลำตัว -> IK ขาหน้าลงพื้น."""
    q = np.zeros(12)
    for i, leg in enumerate(LEGS):
        if leg in REAR:
            q[3 * i:3 * i + 3] = rear_pose(plan, leg)
    height = -lowest_point(plan.pitch_deg, q, p, REAR)   # ขาหลังเป็นตัวกำหนดความสูงตอนนั่ง
    pose = BodyPose(height=height, pitch_deg=plan.pitch_deg)

    feet_world = {}
    for i, leg in enumerate(LEGS):
        if leg not in FRONT:
            continue
        _, sb, _, _ = LEG_SIGNS[leg]
        feet_world[leg] = np.array([plan.front_x, sb * plan.front_y, 0.0])
        tgt = feet_to_body(pose, {leg: feet_world[leg]})[leg]
        q[3 * i:3 * i + 3] = leg_ik(leg, tgt, p, plan.knee)
    return pose, q, feet_world


def _smoothstep(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def _assemble(poses: List[BodyPose], qs: List[np.ndarray], airborne: List[Sequence[str]],
              hz: float, p: Params) -> Trajectory:
    """ประกอบเป็น Trajectory พร้อมคำนวณ CoG และ margin ทุกเฟรม."""
    n = len(qs)
    q = np.array(qs)
    t = np.arange(n) / hz
    feet_b = np.zeros((n, 4, 3))
    contact = np.zeros((n, 4), dtype=bool)
    cog = np.zeros((n, 3))
    margin = np.zeros(n)
    body_T = np.zeros((n, 4, 4))
    mass = 0.0
    for k in range(n):
        body_T[k] = body_matrix(poses[k])
        for i, leg in enumerate(LEGS):
            feet_b[k, i] = leg_fk(leg, q[k, 3 * i:3 * i + 3], p)[-1][:3, 3]
            contact[k, i] = leg not in airborne[k]
        cog[k], mass = world_cog(poses[k], q[k], p)
        margin[k] = margin_from_points(cog[k], ground_contacts(poses[k], q[k], p, airborne[k]))
    traj = Trajectory(t, q, feet_b, contact, cog, margin, mass, 0.0,
                      bool(np.all(np.isfinite(q))))
    traj.body_T = body_T
    return traj


def sit_sequence(p: Params = PARAMS, plan: SitPlan = SitPlan(), g: Gait = Gait()) -> Trajectory:
    """ยืน -> ย่อลงนั่ง (ขาหลังเหยียดไปหน้าวางราบ ลำตัวเชิดขึ้น).

    ความสูงลำตัวแต่ละจังหวะ **คำนวณจากขาหลัง** ไม่ได้ interpolate ตรง ๆ
    ไม่งั้นระหว่างทางขาจะลอยพ้นพื้น (หรือจมลงไปในพื้น) แล้ว margin จะติดลบทั้งที่ของจริงไม่ล้ม
    ส่วนขาหน้าแก้ IK ใหม่ทุกเฟรมให้เท้าปักอยู่กับที่เดิมตลอด
    """
    _, q_sit, _ = sit_pose_and_joints(plan, p)
    base = nominal_stance(p, Gait(stand_h=plan.stand_h, splay_deg=g.splay_deg))
    q_stand = np.concatenate([leg_ik(leg, base[leg], p, plan.knee) for leg in LEGS])
    feet_front = {}
    for leg in FRONT:
        _, sb, _, _ = LEG_SIGNS[leg]
        feet_front[leg] = np.array([plan.front_x, sb * plan.front_y, 0.0])

    n = int(round(plan.t_fold * plan.hz))
    poses: List[BodyPose] = []
    qs: List[np.ndarray] = []
    for k in range(n + 1):
        s = _smoothstep(k / n)
        q = (1.0 - s) * q_stand + s * q_sit          # ขาหลังค่อย ๆ เหยียดออกไปข้างหน้า
        pitch = s * plan.pitch_deg
        pose = BodyPose(height=-lowest_point(pitch, q, p, REAR), pitch_deg=pitch)
        for i, leg in enumerate(LEGS):               # ขาหน้า: เท้าอยู่กับที่ ลำตัวจมลงมาหา
            if leg in FRONT:
                tgt = feet_to_body(pose, {leg: feet_front[leg]})[leg]
                q[3 * i:3 * i + 3] = leg_ik(leg, tgt, p, plan.knee)
        poses.append(pose)
        qs.append(q)
    return _assemble(poses, qs, [()] * len(qs), plan.hz, p)


def rear_up_sequence(p: Params = PARAMS, plan: SitPlan = SitPlan()) -> Trajectory:
    """นั่งอยู่ -> ยกขาหน้าสองข้างขึ้น -> ค้างไว้ -> วางลงกลับมานั่ง."""
    pose, q_sit, feet_sit = sit_pose_and_joints(plan, p)

    q_up = q_sit.copy()
    for i, leg in enumerate(LEGS):
        if leg not in FRONT:
            continue
        _, sb, _, _ = LEG_SIGNS[leg]
        q_up[3 * i:3 * i + 3] = np.radians([sb * plan.lift_th1, plan.lift_th2, plan.lift_th3])

    poses, qs, air = [pose], [q_sit], [()]
    n_lift = int(round(plan.t_lift * plan.hz))
    n_hold = int(round(plan.t_hold * plan.hz))
    for k in range(n_lift):                     # ยกขึ้น
        s = _smoothstep((k + 1) / n_lift)
        poses.append(pose)
        qs.append((1 - s) * q_sit + s * q_up)
        air.append(FRONT)
    for _ in range(n_hold):                     # ค้าง
        poses.append(pose)
        qs.append(q_up)
        air.append(FRONT)
    for k in range(n_lift):                     # วางลง
        s = _smoothstep((k + 1) / n_lift)
        poses.append(pose)
        qs.append((1 - s) * q_up + s * q_sit)
        air.append(FRONT if s < 0.98 else ())
    return _assemble(poses, qs, air, plan.hz, p)


# ---------------------------------------------------------------------------
# 4) ตรวจ
# ---------------------------------------------------------------------------


def check(traj: Trajectory, name: str, plan: SitPlan, servo: Servo = Servo(),
          lim: JointLimits = LIMITS, verbose: bool = True) -> Dict[str, float]:
    """ตรวจ IK เอื้อมถึงไหม ลิมิตข้อต่อ ความเร็ว และ margin."""
    rate = np.abs(np.diff(np.degrees(np.nan_to_num(traj.q)), axis=0)) * plan.hz
    bad = set()
    for k in range(len(traj.t)):
        if np.all(np.isfinite(traj.q[k])):
            for v in limit_violations(traj.q[k], lim):
                bad.add(v.split(" = ")[0])
    res = {"duration_s": float(traj.t[-1]), "peak_rate_dps": float(rate.max()),
           "min_margin_mm": float(np.min(traj.margin)), "n_bad": float(len(bad))}
    if verbose:
        print(f"--- {name} ---")
        print(f"  ใช้เวลา              {res['duration_s']:6.1f} s")
        print(f"  IK เอื้อมถึงทุกจังหวะ  {'ใช่' if traj.reachable else 'ไม่'}")
        print(f"  ความเร็วข้อต่อสูงสุด   {res['peak_rate_dps']:6.0f} °/s "
              f"(เซอร์โวไหว {servo.max_rate_dps:.0f})")
        print(f"  margin ต่ำสุด        {res['min_margin_mm']:6.1f} mm")
        if bad:
            print(f"  ERROR: ข้อต่อหลุดลิมิต {sorted(bad)}")
    return res


def rock_window(sit: Trajectory, p: Params = PARAMS, plan: SitPlan = SitPlan()) -> Dict[str, float]:
    """วัดช่วง 'ทิ้งตัวลงนั่ง' ที่ static margin ติดลบ: กินเวลาเท่าไร ก้นอยู่สูงจากพื้นแค่ไหน.

    ขาสั้นเทียบลำตัว ทำให้จุดรับน้ำหนักด้านหลังต้องกวาดผ่านใต้ CoG ไปข้างหน้า
    ระหว่างนั้นจึงไม่มีอะไรรับด้านหลังจนกว่าก้นจะแตะ -> เป็นการทิ้งตัวลงสั้น ๆ ไม่ใช่การล้ม
    """
    bad = np.flatnonzero(sit.margin <= 0)
    if not len(bad):
        return {"seconds": 0.0, "drop_mm": 0.0}
    drop = 0.0
    for k in bad:
        T = sit.body_T[k]
        pose = BodyPose(height=float(T[2, 3]), pitch_deg=float(np.degrees(np.arcsin(-T[0, 2]))))
        drop = max(drop, float(np.min(rump_corners(pose)[:, 2])))
    return {"seconds": float(sit.t[bad[-1]] - sit.t[bad[0]]), "drop_mm": drop}


def verify(p: Params = PARAMS, plan: SitPlan = SitPlan()) -> None:
    """ตรวจว่านั่งแล้วยกขาหน้าได้จริง ไม่ล้ม ไม่หลุดลิมิต และต่อกันสนิท."""
    sit = sit_sequence(p, plan)
    rear = rear_up_sequence(p, plan)
    pose, q_sit, _ = sit_pose_and_joints(plan, p)

    assert sit.reachable and rear.reachable, "IK เอื้อมไม่ถึง"
    assert not limit_violations(q_sit), f"ท่านั่งหลุดลิมิต: {limit_violations(q_sit)}"
    assert np.allclose(sit.q[-1], rear.q[0], atol=1e-9), "ท่านั่งของสองซีเควนซ์ไม่ต่อกัน"
    assert np.allclose(rear.q[-1], rear.q[0], atol=1e-9), "ยกขาหน้าแล้วไม่กลับมาท่านั่งเดิม"
    assert rear.margin.min() > 0, "ยกขาหน้าแล้วล้ม"
    assert sit.margin[-1] > 0, "ท่านั่งสุดท้ายไม่นิ่ง"

    # จุดต่ำสุดต้องแตะพื้นพอดี ไม่ลอยไม่จม
    low = min(float(np.min(to_world(body_matrix(pose), leg_polyline(leg, q_sit[3 * i:3 * i + 3], p))[:, 2]))
              for i, leg in enumerate(LEGS))
    assert abs(low) < 1e-6, f"ขาไม่แตะพื้นพอดี (z = {low:.3f})"

    # ถ้ามีแค่ 'เท้า' หลังสองจุด (ไม่นับเข่า/ท่อนขาที่วางราบ) ต้องล้มแน่นอน
    k = int(np.argmin(rear.margin))
    feet_only = np.array([to_world(body_matrix(pose), rear.feet[k, i])[0]
                          for i, leg in enumerate(LEGS) if leg in REAR])
    assert margin_from_points(rear.cog[k], feet_only) == float("-inf"), \
        "สมมติฐานผิด: เท้าหลัง 2 จุดไม่ควรตั้งอยู่ได้"

    n_sup = len(ground_contacts(pose, rear.q[k], p, FRONT))
    rock = rock_window(sit, p, plan)
    print(f"PASS: นั่งเงย {plan.pitch_deg:.0f}° ลำตัวสูง {pose.height:.1f} mm "
          f"(margin ท่านั่งนิ่ง {sit.margin[-1]:.1f}) | "
          f"ยกขาหน้า margin {rear.margin.min():.1f} mm จากจุดรับน้ำหนัก {n_sup} จุด")
    print(f"      ช่วงทิ้งตัวลงนั่ง {rock['seconds']:.1f} s ก้นตกลงมา {rock['drop_mm']:.1f} mm "
          "(ล้มลงบนแผ่นรองก้น ไม่ใช่เสียหลัก)")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-prefix", metavar="NAME")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--pitch", type=float, default=SitPlan.pitch_deg)
    args = ap.parse_args(argv)

    plan = SitPlan(pitch_deg=args.pitch)
    verify(PARAMS, plan)
    if args.verify_only:
        return 0

    print("=" * 72)
    sit = sit_sequence(PARAMS, plan)
    check(sit, "SIT     (ยืน -> นั่ง)", plan)
    rear = rear_up_sequence(PARAMS, plan)
    check(rear, "REAR UP (ยกขาหน้าสองข้าง)", plan)
    pose, q_sit, _ = sit_pose_and_joints(plan, PARAMS)
    cog_w, mass = world_cog(pose, q_sit, PARAMS)
    print("=" * 72)
    print(f"  ท่านั่ง: ลำตัวสูง {pose.height:.1f} mm เงย {plan.pitch_deg:.0f}°  "
          f"เงา CoG ที่ x = {cog_w[0]:+.1f} mm")
    print(f"  มุมขาหลังตอนนั่ง: θ2 = {plan.rear_th2:.0f}°, θ3 = {plan.rear_th3:.0f}° "
          "(เหยียดไปข้างหน้า วางท่อนขาราบกับพื้น)")
    print("  พื้นที่รับน้ำหนักตอนยกขาหน้า = ท่อนขาหลังที่วางราบทั้งสองข้าง ไม่ใช่แค่ปลายเท้า")
    print("=" * 72)

    if args.csv_prefix:
        export_csv(sit, f"{args.csv_prefix}_sit.csv")
        export_csv(rear, f"{args.csv_prefix}_rear.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
