#!/usr/bin/env python3
"""
gestures.py — คลังท่าทาง P-kun (แมปจาก kinematic matrix ที่มีอยู่)

แนวคิด: ท่าทางในเอกสารพูดถึง "หู / หัว / ตา / หาง / LED" ซึ่งหุ่นตัวนี้ (12 DOF ขาล้วน)
ยังไม่มี. โมดูลนี้จึงแยกออกเป็น 2 ชั้นที่เดินคู่กันบนไทม์ไลน์เดียว:

  ชั้นลำตัว (มีจริงแล้ว)  ท่าลำตัว 6 DOF -> IK ขา 12 ตัว โดยเท้า "ปักอยู่กับที่" บนพื้น
                        สูง/เลื่อนหน้า-หลัง-ข้าง/เอียง roll-pitch/หมุน yaw
  ชั้นการแสดงออก (ยังไม่มีฮาร์ดแวร์)  10 ช่องสัญญาณ head_pitch/yaw/roll, ear_l/r, eye,
                        tail, led, led_v, sound -> ส่งออกเป็นคอลัมน์ใน CSV ไว้ต่อทีหลัง

ที่ทำแบบนี้เพราะ "หลักการร่วม" ในเอกสารบอกให้เน้นหู/หัว/สายตามากกว่าขา-ลำตัว.
แปลว่าท่าส่วนใหญ่ต้องอ่านออกแม้ลำตัวแทบไม่ขยับ — ชั้นลำตัวจึงมีหน้าที่แค่ "รับน้ำหนัก
+ ขยับนิด ๆ ให้เข้าจังหวะ" ส่วนความหมายอยู่ที่ชั้นการแสดงออก.

รัน:  python gestures.py --list             ดูคลังทั้งหมด
      python gestures.py --check            ตรวจทุกท่า (ลิมิต/ความเร็ว/แรงบิด/ทรงตัว)
      python gestures.py --csv notice_low   ส่งออกไทม์ไลน์ 50 Hz
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from elephant import convex_hull, margin_from_points, to_world
from leg_kinematics import LEGS, PARAMS, Params, leg_fk, leg_polyline
from walking_gait import (
    BODY_PAYLOADS,
    Gait,
    Payload,
    Servo,
    Trajectory,
    _incenter,
    analyze,
    center_of_gravity,
    leg_ik,
    nominal_stance,
)
from sleep_wake import LIMITS, JointLimits, find_sleep_pose, limit_violations, pose_vector

HZ = 50.0                 # อัตราส่งคำสั่ง = อัตราพัลส์ของเซอร์โว analog
STAND_H = 110.0           # ความสูงลำตัวท่าพัก (rest)
SPLAY = 12.0
KNEE = +1
USER_YAW = 22.0           # ทิศที่ "ผู้ใช้" อยู่ (องศา, + = ทางซ้ายของหุ่น) ใช้ตอนหันเข้าหา


# ---------------------------------------------------------------------------
# 0) ช่องสัญญาณการแสดงออก
# ---------------------------------------------------------------------------

EXPR_COLS: Tuple[str, ...] = (
    "head_pitch",   # องศา + = เงยขึ้น
    "head_yaw",     # องศา + = หันไปทางซ้าย
    "head_roll",    # องศา + = เอียงหัวไปทางซ้าย (ท่า "งง")
    "ear_l",        # -1 = ตกแนบ, 0 = ปกติ, +1 = ตั้งชัน
    "ear_r",
    "eye",          # 0 = ปิดสนิท, 1 = เปิดเต็ม
    "tail",         # องศา กวาดซ้าย-ขวา
    "led",          # ดัชนีสี (ดู LED_NAMES) — เปลี่ยนแบบขั้นบันได ไม่ไล่ผ่านสีอื่น
    "led_v",        # ความสว่าง 0-1
    "sound",        # ความดัง 0-1 (0 = เงียบ)
)
COL = {name: i for i, name in enumerate(EXPR_COLS)}
LED_COL = COL["led"]

LED_NAMES: Tuple[str, ...] = ("off", "rest", "low", "soon", "urgent", "good",
                              "neutral", "warn", "listen", "think", "warm", "power",
                              "focus", "speak")
LED_IDX = {name: i for i, name in enumerate(LED_NAMES)}
LED_HEX = {"off": "#2b2b2b", "rest": "#2196a8", "low": "#4fb3c4", "soon": "#e8a33d",
           "urgent": "#d62728", "good": "#2ca02c", "neutral": "#8f8f8f", "warn": "#c2185b",
           "listen": "#5bc8d8", "think": "#7e57c2", "warm": "#e0805a", "power": "#c9a227",
           "focus": "#3f5f8f", "speak": "#26c6b0"}

# ช่องที่เป็นองศาจริง ๆ ใช้เช็คความเร็วเซอร์โว (หน่วย °/s) — หูใช้สเกล 45° ต่อ 1 หน่วย
# แยกลิมิตหัวกับหางออกจากกัน: หัวแบกกล้อง/ลำโพงมีโมเมนต์ความเฉื่อยสูงกว่าหางมาก
EXPR_HEAD = {"head_pitch": 1.0, "head_yaw": 1.0, "head_roll": 1.0, "ear_l": 45.0, "ear_r": 45.0}
EXPR_TAIL = {"tail": 1.0}
HEAD_RATE_LIMIT = 240.0
TAIL_RATE_LIMIT = 500.0


@dataclass(frozen=True)
class Expr:
    """ค่าช่องการแสดงออกหนึ่งคีย์เฟรม."""

    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    ear_l: float = 0.0
    ear_r: float = 0.0
    eye: float = 1.0
    tail: float = 0.0
    led: str = "rest"
    led_v: float = 0.35
    sound: float = 0.0

    def arr(self) -> np.ndarray:
        return np.array([self.head_pitch, self.head_yaw, self.head_roll, self.ear_l,
                         self.ear_r, self.eye, self.tail, float(LED_IDX[self.led]),
                         self.led_v, self.sound])

    def ears(self, v: float) -> "Expr":
        """ตั้ง/ตกหูพร้อมกันสองข้าง."""
        return replace(self, ear_l=v, ear_r=v)


REST = Expr()


# ---------------------------------------------------------------------------
# 1) ท่าลำตัว 6 DOF -> IK ขา
# ---------------------------------------------------------------------------

POSE_COLS: Tuple[str, ...] = ("h", "x", "y", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class Posture:
    """ท่าลำตัวเทียบพื้น. เท้าปักอยู่กับที่ ลำตัวขยับ -> IK คำนวณมุมขาให้เอง."""

    h: float = STAND_H     # ความสูงจุดกำเนิดเฟรมลำตัวจากพื้น (mm)
    x: float = 0.0         # + = ลำตัวเลื่อนไปข้างหน้า
    y: float = 0.0         # + = ลำตัวเลื่อนไปทางซ้าย
    roll: float = 0.0      # องศา + = ยกฝั่งซ้ายขึ้น
    pitch: float = 0.0     # องศา + = เชิดหน้าขึ้น
    yaw: float = 0.0       # องศา + = หันตัวไปทางซ้าย

    def arr(self) -> np.ndarray:
        return np.array([self.h, self.x, self.y, self.roll, self.pitch, self.yaw])


REST_POSE = Posture()


def posture_T(row: Sequence[float]) -> np.ndarray:
    """transform 4x4 เฟรมลำตัว -> เฟรมพื้น จากเวกเตอร์ท่า 6 ตัว."""
    h, x, y, roll, pitch, yaw = (float(v) for v in row)
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])   # + = เชิดหน้า (เหมือน elephant.py)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = (x, y, h)
    return T


def footprint(p: Params = PARAMS, stand_h: float = STAND_H,
              splay: float = SPLAY) -> Dict[str, np.ndarray]:
    """ตำแหน่งเท้าทั้งสี่บนพื้น (z=0) ของท่ายืนพัก — ทุกท่าใช้รอยเท้าชุดนี้เป็นหลัก."""
    base = nominal_stance(p, Gait(stand_h=stand_h, splay_deg=splay))
    return {leg: np.array([v[0], v[1], 0.0]) for leg, v in base.items()}


FEET0 = footprint()


def posture_ik(T: np.ndarray, feet_world: Dict[str, Sequence[float]],
               p: Params = PARAMS, knee: int = KNEE) -> np.ndarray:
    """มุมข้อต่อ 12 ตัวที่ทำให้เท้าอยู่ตรงตำแหน่งพื้นที่กำหนด ขณะลำตัวอยู่ในท่า T."""
    R_T = T[:3, :3].T
    q = np.zeros(12)
    for i, leg in enumerate(LEGS):
        tgt = R_T @ (np.asarray(feet_world[leg], dtype=float) - T[:3, 3])
        q[3 * i:3 * i + 3] = leg_ik(leg, tgt, p, knee)
    return q


def rump_pts(T: np.ndarray) -> np.ndarray:
    """มุมแผ่นก้น 4 จุดในเฟรมพื้น (ใช้ตอนลำตัวต่ำจนก้นแตะ)."""
    from elephant import RUMP

    pts = np.array([[RUMP.x_back, +RUMP.half_w, RUMP.z], [RUMP.x_back, -RUMP.half_w, RUMP.z],
                    [RUMP.x_front, -RUMP.half_w, RUMP.z], [RUMP.x_front, +RUMP.half_w, RUMP.z]])
    return to_world(T, pts)


def contacts_T(T: np.ndarray, q: Sequence[float], p: Params = PARAMS,
               airborne: Sequence[str] = (), tol: float = 5.0) -> np.ndarray:
    """ทุกจุดของหุ่นที่แตะพื้น (สะโพก/เข่า/เท้าของขาที่ยันอยู่ + มุมก้น)."""
    hits: List[np.ndarray] = []
    for i, leg in enumerate(LEGS):
        if leg in airborne:
            continue
        pts = to_world(T, leg_polyline(leg, q[3 * i:3 * i + 3], p))
        hits += [pt for pt in pts if pt[2] <= tol]
    hits += [c for c in rump_pts(T) if c[2] <= tol]
    return np.array(hits) if hits else np.zeros((0, 3))


def lowest_T(T: np.ndarray, q: Sequence[float], p: Params = PARAMS) -> float:
    """จุดต่ำสุดของหุ่นในเฟรมพื้น (ใช้ยกลำตัวให้แตะพื้นพอดีตอนขาพับ)."""
    lows = [float(np.min(to_world(T, leg_polyline(leg, q[3 * i:3 * i + 3], p))[:, 2]))
            for i, leg in enumerate(LEGS)]
    lows.append(float(np.min(rump_pts(T)[:, 2])))
    return min(lows)


def _chase_cog(pose: Posture, goal_xy: np.ndarray, feet: Dict[str, np.ndarray],
               p: Params, iters: int = 14, damp: float = 0.6) -> Posture | None:
    """ไล่เลื่อนลำตัวจนเงา CoG ไปตกที่ goal_xy. คืน None ถ้าระหว่างทาง IK เอื้อมไม่ถึง.

    ต้องหน่วง (damp<1) เพราะก้าวเต็มระยะครั้งเดียวจะกระโดดข้ามไปนอก workspace
    แล้ว IK คืน NaN ตั้งแต่รอบแรก — ขา 136 mm กับลำตัวยาว 154 mm เหลือระยะเอื้อมไม่มาก
    """
    for _ in range(iters):
        T = posture_T(pose.arr())
        q = posture_ik(T, feet, p)
        if not np.all(np.isfinite(q)):
            return None
        cog = to_world(T, center_of_gravity(q, p)[0])[0]
        err = goal_xy - cog[:2]
        if np.linalg.norm(err) < 0.05:
            break
        pose = replace(pose, x=pose.x + damp * float(err[0]), y=pose.y + damp * float(err[1]))
    q = posture_ik(posture_T(pose.arr()), feet, p)
    if not np.all(np.isfinite(q)) or limit_violations(q):
        return None
    return pose


def solve_shift(pose: Posture, planted: Sequence[str], p: Params = PARAMS,
                feet: Dict[str, np.ndarray] | None = None) -> Posture:
    """เลื่อนลำตัวเข้าหา incenter ของขาที่ยันพื้นอยู่ ให้ได้ไกลที่สุดเท่าที่ IK ยังเอื้อมถึง.

    จำเป็นทุกครั้งที่ยกขาข้างหนึ่ง: ถึงแม้เลย์เอาต์ใหม่ (จอกลางหลัง + แบตซ้าย-ขวาเท่ากัน)
    จะทำให้ CoG อยู่กลางลำตัวแล้ว ฐานเท้าก็ยังกว้างแค่ 2b = 58 mm
    ถ้าไม่เลื่อนลำตัวก่อน พอยกขาหน้าซ้ายขึ้น เงา CoG จะหลุดออกนอกสามเหลี่ยมทันที

    เลื่อนไปถึง incenter เต็ม ๆ **ไม่ได้** กับสัดส่วนตัวนี้ (ต้องถอยหลัง 43 mm + ออกข้าง 27 mm
    ขาหน้าจะเอื้อมไม่ถึง) จึงไล่หาเศษส่วนที่มากที่สุดที่ยังทำได้จริงแทน
    """
    feet = FEET0 if feet is None else feet
    T0 = posture_T(pose.arr())
    q0 = posture_ik(T0, feet, p)
    if not np.all(np.isfinite(q0)):
        return pose
    cog0 = to_world(T0, center_of_gravity(q0, p)[0])[0][:2]
    target = _incenter(np.array([feet[leg][:2] for leg in planted]))
    for s in np.linspace(1.0, 0.0, 21):
        out = _chase_cog(pose, cog0 + s * (target - cog0), feet, p)
        if out is not None:
            return out
    return pose


# ---------------------------------------------------------------------------
# 2) ไทม์ไลน์: คีย์เฟรม + ตัวปรุง (modulator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    """คีย์เฟรมหนึ่งจุด: ใช้เวลา dur วินาทีเดินทางมาถึง แล้วค้างต่ออีก hold วินาที."""

    dur: float = 0.0
    pose: Posture = REST_POSE
    expr: Expr = REST
    hold: float = 0.0


def _smoothstep(u: float) -> float:
    """เร่ง-ชะลอนุ่ม ความเร็วเป็นศูนย์ที่หัวและท้าย — เซอร์โว analog ไม่มี feedback ต้องนุ่ม."""
    return u * u * (3.0 - 2.0 * u)


def sample_keys(keys: Sequence[Key], hz: float = HZ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """กาง keyframe เป็นอาเรย์ต่อเฟรม: (t, ท่าลำตัว (N,6), การแสดงออก (N,10))."""
    P = [keys[0].pose.arr()]
    E = [keys[0].expr.arr()]
    for k in range(int(round(keys[0].hold * hz))):
        P.append(P[0].copy())
        E.append(E[0].copy())
    for prev, cur in zip(keys, keys[1:]):
        a, b = prev.pose.arr(), cur.pose.arr()
        ea, eb = prev.expr.arr(), cur.expr.arr()
        n = max(1, int(round(cur.dur * hz)))
        for i in range(n):
            s = _smoothstep((i + 1) / n)
            P.append((1.0 - s) * a + s * b)
            e = (1.0 - s) * ea + s * eb
            e[LED_COL] = eb[LED_COL]     # สีสลับทันทีที่เริ่มช่วง ไม่ไล่ผ่านสีที่ไม่ได้สั่ง
            E.append(e)
        for _ in range(int(round(cur.hold * hz))):
            P.append(b.copy())
            E.append(eb.copy())
    P_a, E_a = np.array(P), np.array(E)
    return np.arange(len(P_a)) / hz, P_a, E_a


def _noise(n: int, seed: int, rate: float, hz: float = HZ) -> np.ndarray:
    """สัญญาณสุ่มที่กรองความถี่สูงออกแล้ว (ช่วง -1..1 โดยประมาณ) ใช้ทำ micro-motion."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n + 200)
    win = max(3, int(hz / max(rate, 1e-3)))
    ker = np.hanning(win)
    ker /= ker.sum()
    sm = np.convolve(raw, ker, mode="same")[100:100 + n]
    peak = np.max(np.abs(sm))
    return sm / peak if peak > 1e-9 else sm


def breathe(P: np.ndarray, t: np.ndarray, amp: float, period: float,
            period_end: float | None = None) -> None:
    """หายใจ = ลำตัวขึ้นลงตามแนวดิ่ง. period_end != None คือค่อย ๆ ช้าลง (ท่าเคลิ้มหลับ)."""
    if period_end is None:
        phase = t / period
    else:
        # คาบยืดขึ้นเรื่อย ๆ ต้องสะสมเฟสทีละสเต็ป ไม่ใช่ t/period(t) (ไม่งั้นเฟสจะเดินถอยหลัง)
        per = np.linspace(period, period_end, len(t))
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / HZ
        phase = np.cumsum(np.full(len(t), dt) / per)
    P[:, 0] += amp * np.sin(2.0 * np.pi * phase)


def blink(E: np.ndarray, t: np.ndarray, times: Sequence[float], dur: float = 0.16) -> None:
    """กะพริบตา: ตาปิดสนิทแล้วเปิดคืนภายใน dur วินาที."""
    for t0 in times:
        m = (t >= t0) & (t <= t0 + dur)
        if not m.any():
            continue
        u = (t[m] - t0) / dur
        E[m, COL["eye"]] *= 1.0 - np.sin(np.pi * u) ** 2


def wag(E: np.ndarray, t: np.ndarray, t0: float, beats: float, freq: float, amp: float) -> None:
    """กระดิกหาง beats จังหวะ ที่ความถี่ freq Hz แล้วหยุดสนิท (ซองจดหมายค่อย ๆ เบาลง)."""
    dur = beats / freq
    m = (t >= t0) & (t <= t0 + dur)
    if not m.any():
        return
    u = (t[m] - t0) / dur
    E[m, COL["tail"]] += amp * np.sin(2.0 * np.pi * freq * (t[m] - t0)) * np.sin(np.pi * u)


def speak_env(E: np.ndarray, t: np.ndarray, t0: float, t1: float, seed: int = 7) -> None:
    """จังหวะพูด: หัวขยับตามซองเสียงแบบพยางค์ 3-5 Hz ไม่ต้องซิงก์เป๊ะ แค่ให้เสียงมีที่มา."""
    m = (t >= t0) & (t <= t1)
    if not m.any():
        return
    env = np.abs(_noise(int(m.sum()), seed, rate=4.0))
    E[m, COL["head_pitch"]] += 5.0 * (env - env.mean())
    E[m, COL["head_yaw"]] += 3.0 * _noise(int(m.sum()), seed + 1, rate=2.0)
    E[m, COL["ear_l"]] += 0.12 * _noise(int(m.sum()), seed + 2, rate=2.5)
    E[m, COL["ear_r"]] += 0.12 * _noise(int(m.sum()), seed + 3, rate=2.5)
    E[m, COL["sound"]] = np.clip(0.35 + 0.5 * env, 0.0, 1.0)


def led_pulse(E: np.ndarray, t: np.ndarray, t0: float, dur: float, hi: float,
              lo: float | None = None) -> None:
    """พัลส์ความสว่างครั้งเดียว (ขึ้นเร็ว-ลงช้า)."""
    m = (t >= t0) & (t <= t0 + dur)
    if not m.any():
        return
    u = (t[m] - t0) / dur
    base = E[m, COL["led_v"]] if lo is None else lo
    E[m, COL["led_v"]] = base + (hi - base) * np.sin(np.pi * u) ** 0.6


def led_blink(E: np.ndarray, t: np.ndarray, freq: float, lo: float, hi: float,
              t0: float = 0.0) -> None:
    """กะพริบช้าเป็นคาบ (ใช้กับท่าแจ้งด่วน / ท่าขัดข้อง)."""
    m = t >= t0
    s = 0.5 - 0.5 * np.cos(2.0 * np.pi * freq * (t[m] - t0))
    E[m, COL["led_v"]] = lo + (hi - lo) * s


# ---------------------------------------------------------------------------
# 3) ประกอบเป็นคลิป
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    """ท่าทางหนึ่งท่าที่กางเป็นไทม์ไลน์แล้ว พร้อมผลตรวจทรงตัว."""

    gid: str
    group: str
    when: str
    means: str
    traj: Trajectory                 # ชั้นลำตัว (ใช้กับ show_motion.animate ได้เลย)
    expr: np.ndarray                 # (N,10) ชั้นการแสดงออก
    loop_from: float | None = None   # None = ท่าครั้งเดียว, ตัวเลข = วนซ้ำจากวินาทีนี้
    core: bool = False               # อยู่ในชุดแกนที่เอกสารบอกให้ทำก่อน
    notes: List[str] = field(default_factory=list)

    @property
    def t(self) -> np.ndarray:
        return self.traj.t

    @property
    def dur(self) -> float:
        return float(self.traj.t[-1])


def assemble(gid: str, meta: Dict[str, object], t: np.ndarray, P: np.ndarray, E: np.ndarray,
             p: Params = PARAMS, q_track: np.ndarray | None = None,
             feet_fn: Callable[[int, float], Tuple[Dict[str, np.ndarray], Tuple[str, ...]]] | None = None,
             payloads: Sequence[Payload] = tuple(BODY_PAYLOADS)) -> Clip:
    """แปลง (ท่าลำตัว, การแสดงออก) เป็นคลิปเต็ม: IK ขา + CoG + margin ทุกเฟรม.

    q_track != None คือท่าที่ขาพับจนคุมด้วยตำแหน่งเท้าไม่ได้ (นอน/แบตหมด)
    ให้ส่งมุมข้อต่อมาตรง ๆ แล้วโมดูลนี้จะยกลำตัวขึ้นจนจุดต่ำสุดแตะพื้นพอดีเอง
    """
    n = len(t)
    q = np.zeros((n, 12))
    body_T = np.zeros((n, 4, 4))
    feet_b = np.zeros((n, 4, 3))
    contact = np.ones((n, 4), dtype=bool)
    cog = np.zeros((n, 3))
    margin = np.zeros(n)
    mass = 0.0
    reachable = True

    for k in range(n):
        T = posture_T(P[k])
        if q_track is not None:
            q[k] = q_track[k]
            T[2, 3] -= lowest_T(T, q[k], p)          # วางหุ่นลงให้แตะพื้นพอดี ไม่จมไม่ลอย
            air: Tuple[str, ...] = ()
        else:
            feet, air = (feet_fn(k, float(t[k])) if feet_fn else (FEET0, ()))
            q[k] = posture_ik(T, feet, p)
        if not np.all(np.isfinite(q[k])):
            reachable = False
            q[k] = np.nan_to_num(q[k])
        body_T[k] = T
        for i, leg in enumerate(LEGS):
            feet_b[k, i] = leg_fk(leg, q[k, 3 * i:3 * i + 3], p)[-1][:3, 3]
            # ท่าที่ขาพับ ปลายเท้าลอยอยู่กลางอากาศ น้ำหนักลงท้อง/ก้น ไม่ได้ลงขา
            # ต้องแยกให้ออก ไม่งั้นตัวเลขแรงบิดจะสูงเกินจริงตลอดช่วงที่นอนอยู่
            contact[k, i] = (float(to_world(T, feet_b[k, i])[0][2]) <= 5.0 if q_track is not None
                             else leg not in air)
        cog[k], mass = center_of_gravity(q[k], p, payloads)
        cog[k] = to_world(T, cog[k])[0]
        margin[k] = margin_from_points(cog[k], contacts_T(T, q[k], p, air))

    traj = Trajectory(t, q, feet_b, contact, cog, margin, mass, 0.0, reachable)
    traj.body_T = body_T
    return Clip(gid=gid, group=str(meta["group"]), when=str(meta["when"]),
                means=str(meta["means"]), traj=traj, expr=E,
                loop_from=meta.get("loop_from"), core=bool(meta.get("core", False)))  # type: ignore[arg-type]


def _resample(A: np.ndarray, n: int) -> np.ndarray:
    """ยืด/หดอาเรย์ตามแกนเวลาให้ได้ n เฟรม (ใช้ตอนต้องจับชั้นการแสดงออกเข้ากับ q ที่มีอยู่)."""
    src = np.linspace(0.0, 1.0, len(A))
    dst = np.linspace(0.0, 1.0, n)
    out = np.array([np.interp(dst, src, A[:, j]) for j in range(A.shape[1])]).T
    out[:, LED_COL] = np.round(out[:, LED_COL])
    return out


# ---------------------------------------------------------------------------
# 4) คลังท่า
# ---------------------------------------------------------------------------

BUILDERS: Dict[str, Callable[[Params], Clip]] = {}
META: Dict[str, Dict[str, object]] = {}


def gesture(gid: str, group: str, when: str, means: str, loop_from: float | None = None,
            core: bool = False):
    """ทะเบียนท่า: ผูก gesture_id เข้ากับฟังก์ชันสร้างคลิป."""
    def deco(fn: Callable[[Params], Clip]) -> Callable[[Params], Clip]:
        META[gid] = {"group": group, "when": when, "means": means,
                     "loop_from": loop_from, "core": core}
        BUILDERS[gid] = fn
        return fn
    return deco


# ---- 1. วงจรชีวิต -------------------------------------------------------


@gesture("sleep_mode", "lifecycle", "นอกเวลางาน / ไม่มีคนในระยะ / DND",
         "ปิดพักแล้ว ไม่ต้องสนใจ", loop_from=6.5, core=True)
def _sleep_mode(p: Params = PARAMS) -> Clip:
    """ย่อลง -> พับขาเก็บข้างตัว -> ท้องแตะพื้น -> นิ่งสนิท หายใจแผ่วมาก.

    ช่วงพับขาใช้ท่าเดียวกับ sleep_sequence เดิม (θ = 0°, 90°, 90°)
    หลังจากนั้นเป็นช่วงวนซ้ำ: ลำตัวลงไปกองกับพื้นแล้ว ยกไม่ได้
    'ลมหายใจ' จึงไปอยู่ที่ข้อสะโพกแทน — ขยับ ±0.6° ช้ามาก เห็นแค่หางตา
    """
    from sleep_wake import SETTLE_H, stance_pose

    q_sleep = pose_vector(find_sleep_pose(p))
    q_stand = stance_pose(p, Gait(stand_h=STAND_H, splay_deg=SPLAY, hz=HZ), STAND_H)
    q_low = stance_pose(p, Gait(stand_h=STAND_H, splay_deg=SPLAY, hz=HZ), SETTLE_H)

    keys = [
        Key(pose=REST_POSE, expr=REST),                                     # ยืน
        Key(2.5, REST_POSE, replace(REST.ears(-0.3), eye=0.55, led_v=0.2)),  # ย่อ
        Key(4.0, REST_POSE, Expr(head_pitch=-26, ear_l=-1, ear_r=-1, eye=0.0,
                                 led="off", led_v=0.0)),                    # พับขา ตาปิด
        Key(0.0, REST_POSE, Expr(head_pitch=-26, ear_l=-1, ear_r=-1, eye=0.0,
                                 led="off", led_v=0.0), hold=9.0),          # นิ่ง (ช่วงวน)
    ]
    t, P, E = sample_keys(keys, HZ)

    # มุมขา: interpolate จากยืน -> ย่อ -> พับ ตามสัดส่วนเวลาเดียวกับคีย์เฟรม
    n1, n2 = int(2.5 * HZ), int(4.0 * HZ)
    seg1 = np.array([(1 - _smoothstep((i + 1) / n1)) * q_stand + _smoothstep((i + 1) / n1) * q_low
                     for i in range(n1)])
    seg2 = np.array([(1 - _smoothstep((i + 1) / n2)) * q_low + _smoothstep((i + 1) / n2) * q_sleep
                     for i in range(n2)])
    tail = np.tile(q_sleep, (len(t) - 1 - n1 - n2, 1))
    q_track = np.vstack([q_stand, seg1, seg2, tail])

    # ท่านอนวาง θ2 ไว้ที่ลิมิต 90° พอดี ลมหายใจจึงต้องกดลงทางเดียว (ห้ามแกว่งข้ามลิมิต)
    breath = np.zeros(len(t))
    m = t >= 6.5
    breath[m] = np.radians(0.6) * np.abs(np.sin(np.pi * (t[m] - 6.5) / 9.0))  # 1 รอบ 9 วินาที
    for i in range(4):
        q_track[:, 3 * i + 1] -= breath
    return assemble("sleep_mode", META["sleep_mode"], t, P, E, p, q_track=q_track)


@gesture("wake_up", "lifecycle", "เรดาร์เจอคนเข้ามา / เริ่มวันทำงาน",
         "ตื่นแล้ว พร้อมช่วย", core=True)
def _wake_up(p: Params = PARAMS) -> Clip:
    """กางขาลงหาพื้นก่อน -> ดันตัวขึ้น -> หูตั้ง ตาเปิด LED ไล่สว่าง จบที่ท่า Idle.

    ลำดับสำคัญ: ตาเปิดและหูตั้ง 'ตามหลัง' การยืดตัว ไม่ใช่พร้อมกัน
    เพราะถ้าทุกอย่างมาพร้อมกันจะดูเหมือนสวิตช์เปิด ไม่ใช่การตื่น
    """
    from sleep_wake import wake_sequence

    g = Gait(stand_h=STAND_H, splay_deg=SPLAY, hz=HZ)
    base = wake_sequence(p, g, find_sleep_pose(p), t_unfold=3.0, t_rise=2.5)
    n = len(base.t)

    keys = [
        Key(expr=Expr(head_pitch=-26, ear_l=-1, ear_r=-1, eye=0.0, led="off", led_v=0.0)),
        Key(3.0, REST_POSE, Expr(head_pitch=-14, ear_l=-0.5, ear_r=-0.5, eye=0.25,
                                 led="rest", led_v=0.08)),      # กางขา ตาปรือ
        Key(2.0, REST_POSE, Expr(head_pitch=4, ear_l=0.6, ear_r=0.6, eye=1.0,
                                 led="rest", led_v=0.3)),       # ดันตัวขึ้น หูตั้ง
        Key(0.8, REST_POSE, REST, hold=0.6),                    # ลงเป็นท่าพัก
    ]
    t_e, _, E = sample_keys(keys, HZ)
    E = _resample(E, n)
    blink(E, base.t, [3.4, 4.1], 0.18)
    P = np.tile(REST_POSE.arr(), (n, 1))
    return assemble("wake_up", META["wake_up"], base.t, P, E, p, q_track=base.q)


@gesture("idle_breathe", "lifecycle", "ตื่นอยู่แต่ไม่มีงานเข้า",
         "อยู่นะ ยังมีชีวิต แต่ไม่กวน", loop_from=0.0, core=True)
def _idle_breathe(p: Params = PARAMS) -> Clip:
    """ลูป 12 วินาที: หายใจ ±2.5 mm micro-motion ที่หัว หูขยับบางที กะพริบนาน ๆ ครั้ง.

    หัวใจของท่านี้คือ 'ยังไม่ตาย แต่แทบไม่ขยับ' — ทุกช่องจึงตั้งแอมพลิจูดต่ำมาก
    ค่าที่ให้มาคือค่าที่ยังเห็นจากหางตาได้ แต่ไม่ดึงโฟกัส
    """
    keys = [Key(pose=REST_POSE, expr=REST, hold=12.0)]
    t, P, E = sample_keys(keys, HZ)
    breathe(P, t, amp=2.5, period=5.0)
    E[:, COL["head_yaw"]] += 2.2 * _noise(len(t), 11, rate=0.25)
    E[:, COL["head_pitch"]] += 1.6 * _noise(len(t), 12, rate=0.2)
    E[:, COL["ear_l"]] += 0.10 * _noise(len(t), 13, rate=0.5)
    E[:, COL["ear_r"]] += 0.10 * _noise(len(t), 14, rate=0.45)
    E[:, COL["led_v"]] = 0.30 + 0.08 * np.sin(2.0 * np.pi * t / 5.0)      # LED หายใจตามลำตัว
    blink(E, t, [2.6, 7.9], 0.15)
    return assemble("idle_breathe", META["idle_breathe"], t, P, E, p)


@gesture("doze_off", "lifecycle", "ไม่มีกิจกรรมนานพอ ก่อนตัดเข้า Sleep",
         "กำลังจะพักแล้วนะ")
def _doze_off(p: Params = PARAMS) -> Clip:
    """ทางเชื่อมนุ่ม ๆ ไป sleep_mode: หัวต่ำลง หูเริ่มตก จังหวะหายใจช้าลงเรื่อย ๆ ตาปรือ.

    ยังไม่พับขา — แค่ย่อลงจาก 110 เหลือ 96 mm เพื่อให้ต่อกับช่วงย่อของ sleep_mode ได้พอดี
    """
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(4.0, Posture(h=104.0, pitch=-1.5), Expr(head_pitch=-10, ear_l=-0.3, ear_r=-0.3,
                                                    eye=0.7, led="rest", led_v=0.22)),
        Key(4.0, Posture(h=96.0, pitch=-3.0), Expr(head_pitch=-20, ear_l=-0.7, ear_r=-0.7,
                                                   eye=0.35, led="rest", led_v=0.10), hold=1.5),
    ]
    t, P, E = sample_keys(keys, HZ)
    breathe(P, t, amp=2.2, period=5.0, period_end=9.5)      # หายใจช้าลงเรื่อย ๆ
    blink(E, t, [2.2, 5.0, 7.0, 8.4], 0.3)                  # กะพริบถี่ขึ้นและนานขึ้น = ง่วง
    return assemble("doze_off", META["doze_off"], t, P, E, p)


# ---- 2. การแจ้งเตือน & โฟกัส -------------------------------------------


@gesture("notice_low", "attention", "มีเรื่องรออยู่แต่ไม่ด่วน",
         "มีเรื่องรอมึงอยู่ 1 อย่าง ว่างค่อยดู", core=True)
def _notice_low(p: Params = PARAMS) -> Clip:
    """หูตั้ง หันหัวมาทางผู้ใช้หนึ่งครั้ง LED พัลส์สั้น ๆ แล้วค้างสีเบา ๆ รอจนถูกรับรู้.

    ลำตัวขยับแค่ 2 mm ตั้งใจให้เบาที่สุดในกลุ่มแจ้งเตือน — ความหมายอยู่ที่หูกับหัวล้วน ๆ
    ปลายคลิปเป็นท่าค้าง (ไม่กลับ rest เอง) ต้องรอ settle มาเคลียร์
    """
    alert = Expr(head_yaw=USER_YAW, head_pitch=2.0, ear_l=0.9, ear_r=0.9,
                 led="low", led_v=0.30)
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(1.2, Posture(h=STAND_H + 2.0), alert),
        Key(0.0, Posture(h=STAND_H + 2.0), alert, hold=4.0),      # ค้างรอ
    ]
    t, P, E = sample_keys(keys, HZ)
    led_pulse(E, t, 0.35, 0.9, hi=0.85)
    breathe(P, t, amp=1.2, period=5.0)
    blink(E, t, [2.9], 0.14)
    return assemble("notice_low", META["notice_low"], t, P, E, p)


@gesture("notice_soon", "attention", "ประชุม/เดดไลน์ใกล้เข้ามา",
         "ใกล้ถึงเวลาแล้วนะ เตรียมตัว", core=True)
def _notice_soon(p: Params = PARAMS) -> Clip:
    """ลุกขึ้นยืนสูงกว่าปกติ + หันตัวเข้าหาผู้ใช้ชัดขึ้น + LED สีเด่นกว่า + เสียงเบา 1 ครั้ง.

    ตรงนี้ลำตัวเข้ามามีบทบาทจริง: ยืดขึ้น 110 -> 122 mm และหันลำตัว 14°
    (ไม่ใช่แค่หัน 'หัว' แบบ notice_low) — ความต่างของสองท่านี้ต้องอ่านออกจากหางตา
    """
    tall = Posture(h=122.0, yaw=14.0, pitch=1.5)
    up = Expr(head_yaw=USER_YAW - 8.0, head_pitch=6.0, ear_l=1.0, ear_r=1.0,
              led="soon", led_v=0.55)
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(1.6, tall, up),
        Key(0.0, tall, up, hold=3.5),
    ]
    t, P, E = sample_keys(keys, HZ)
    led_pulse(E, t, 1.6, 1.2, hi=0.95)
    m = (t >= 1.7) & (t <= 2.0)
    E[m, COL["sound"]] = 0.25                     # ปี๊บเดียว เบามาก
    breathe(P, t, amp=1.5, period=4.5)
    return assemble("notice_soon", META["notice_soon"], t, P, E, p)


@gesture("notice_urgent", "attention", "เรื่องด่วนจริงที่ต้องดูเดี๋ยวนี้",
         "ต้องดูตอนนี้", core=True)
def _notice_urgent(p: Params = PARAMS) -> Clip:
    """ท่าที่แรงสุดในคลัง: ยืนสูง + ยกขาหน้าซ้ายเขี่ยพื้น 3 ครั้ง + LED กะพริบ + เสียง.

    นี่คือท่าเดียวที่ยกขาลอย จึงต้องเลื่อนลำตัวหา incenter ของสามขาที่เหลือก่อน (solve_shift)
    ไม่งั้นเงา CoG หลุดสามเหลี่ยมทันทีที่ยก — ฐานเท้าแคบ (2b = 58 mm) ต่อให้ CoG อยู่กลางแล้วก็ตาม
    การเขี่ยใช้โปรไฟล์ sin² ความเร็วเป็นศูนย์ทั้งตอนยกและตอนแตะ ไม่กระแทกพื้น
    """
    planted = ("FR", "RL", "RR")
    ready = solve_shift(Posture(h=112.0, yaw=10.0), planted, p)

    t_in, t_tap, n_tap, t_out = 1.5, 0.85, 3, 1.4
    tap_end = t_in + n_tap * t_tap
    alert = Expr(head_yaw=USER_YAW, head_pitch=8.0, ear_l=1.0, ear_r=1.0,
                 led="urgent", led_v=0.9)
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(t_in, ready, alert, hold=n_tap * t_tap),
        Key(t_out, REST_POSE, replace(alert, head_pitch=2.0, led_v=0.6), hold=0.8),
    ]
    t, P, E = sample_keys(keys, HZ)

    base = FEET0["FL"].copy()

    def feet_fn(k: int, tk: float) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...]]:
        if not (t_in <= tk < tap_end):
            return FEET0, ()
        u = ((tk - t_in) % t_tap) / t_tap
        lift = np.sin(np.pi * u) ** 2                      # ขึ้น-ลง จบที่พื้นพอดีทุกครั้ง
        feet = dict(FEET0)
        feet["FL"] = base + np.array([16.0 * lift, 0.0, 34.0 * lift])   # เขี่ยไปข้างหน้า+ขึ้น
        return feet, (("FL",) if lift > 0.06 else ())

    led_blink(E, t, freq=1.1, lo=0.25, hi=1.0, t0=t_in)
    for i in range(n_tap):
        m = (t >= t_in + i * t_tap + 0.30) & (t <= t_in + i * t_tap + 0.45)
        E[m, COL["sound"]] = 0.6
    return assemble("notice_urgent", META["notice_urgent"], t, P, E, p, feet_fn=feet_fn)


@gesture("settle", "attention", "ผู้ใช้ตอบสนอง/รับรู้การเตือนแล้ว",
         "โอเค เคลียร์แล้ว", core=True)
def _settle(p: Params = PARAMS) -> Clip:
    """ผ่อนลง: หูลดจากตั้งเป็นปกติ ลำตัวถอนจากท่าเตือนกลับ rest LED กลับสีพัก.

    เริ่มจากท่าปลายของ notice_soon (ยืนสูง+หันตัว) เพื่อให้ต่อกันได้จริง
    ความรู้สึก 'คลายลง' มาจากการปล่อยลงต่ำกว่า rest นิดหนึ่งก่อนกลับ (h 108 แล้วค่อยขึ้น 110)
    """
    tall = Posture(h=122.0, yaw=14.0, pitch=1.5)
    up = Expr(head_yaw=USER_YAW - 8.0, head_pitch=6.0, ear_l=1.0, ear_r=1.0,
              led="soon", led_v=0.55)
    keys = [
        Key(pose=tall, expr=up),
        Key(1.6, Posture(h=108.0, yaw=4.0), Expr(head_yaw=6.0, head_pitch=-3.0, ear_l=0.15,
                                                 ear_r=0.15, led="rest", led_v=0.30)),
        Key(1.4, REST_POSE, REST, hold=1.0),
    ]
    t, P, E = sample_keys(keys, HZ)
    breathe(P, t, amp=1.8, period=5.0)
    blink(E, t, [1.9], 0.16)
    return assemble("settle", META["settle"], t, P, E, p)


@gesture("focus_with_you", "attention", "ผู้ใช้เข้าโหมดโฟกัสลึก / ห้ามรบกวน",
         "อยู่ข้าง ๆ เงียบ ๆ เป็นเพื่อน (body doubling)", loop_from=2.5)
def _focus_with_you(p: Params = PARAMS) -> Clip:
    """นิ่งกว่า idle อีกขั้น: หันหน้าไปทางเดียวกับผู้ใช้ ย่อลงเล็กน้อย LED หรี่นิ่ง.

    ท่านี้ 'จงใจไม่ทำอะไร' ตัวเลขจึงถูกกดลงหมด: หายใจ 1.2 mm (idle ใช้ 2.5)
    micro-motion หัวเหลือ 0.6° และ LED ไม่หายใจเลย เพราะแสงที่ขยับดึงสายตาได้มากกว่าที่คิด
    """
    calm = Posture(h=104.0, yaw=0.0)
    quiet = Expr(head_pitch=-2.0, head_yaw=0.0, ear_l=0.1, ear_r=0.1, eye=0.85,
                 led="focus", led_v=0.12)
    keys = [Key(pose=REST_POSE, expr=REST), Key(2.5, calm, quiet), Key(0.0, calm, quiet, hold=12.0)]
    t, P, E = sample_keys(keys, HZ)
    breathe(P, t, amp=1.2, period=7.0)
    E[:, COL["head_yaw"]] += 0.6 * _noise(len(t), 21, rate=0.12)
    blink(E, t, [5.5, 11.2], 0.16)
    return assemble("focus_with_you", META["focus_with_you"], t, P, E, p)


# ---- 3. ลูปสนทนา ---------------------------------------------------------


@gesture("listening", "conversation", "ได้ยิน wake word / ผู้ใช้กำลังพูด",
         "ฟังอยู่นะ พูดต่อได้เลย", loop_from=1.2, core=True)
def _listening(p: Params = PARAMS) -> Clip:
    """เอียงหัวเข้าหาผู้ใช้ หูตั้งชี้เข้าหาเสียง แล้วค้างนิ่ง — ความนิ่งคือสิ่งที่บอกว่าตั้งใจฟัง.

    หูสองข้างไม่เท่ากันโดยตั้งใจ (ข้างที่หันเข้าหาเสียงตั้งกว่า) เป็นสัญญาณทิศทาง
    ลำตัวเอนไปข้างหน้า 6 mm — น้อยมากแต่พอให้รู้สึกว่า 'โน้มเข้ามา'
    """
    lean = Posture(h=112.0, x=6.0, yaw=8.0)
    ear_hi, ear_lo = (1.0, 0.55) if USER_YAW >= 0 else (0.55, 1.0)
    on = Expr(head_yaw=USER_YAW, head_roll=6.0, head_pitch=1.0,
              ear_l=ear_hi, ear_r=ear_lo, led="listen", led_v=0.6)
    keys = [Key(pose=REST_POSE, expr=REST), Key(1.2, lean, on), Key(0.0, lean, on, hold=6.0)]
    t, P, E = sample_keys(keys, HZ)
    breathe(P, t, amp=0.8, period=5.0)             # หายใจเบาสุด — ห้ามดูเหมือนกำลังจะขยับ
    blink(E, t, [4.4], 0.14)
    return assemble("listening", META["listening"], t, P, E, p)


@gesture("thinking", "conversation", "ระหว่างรอ STT + ประมวลผลบน cloud",
         "กำลังคิด/ประมวลผลอยู่ (ท่ากลบ latency)", loop_from=1.0, core=True)
def _thinking(p: Params = PARAMS) -> Clip:
    """เอียงหัวแบบสงสัย + ไมโครโมชัน + LED หายใจช้า. รอนานขึ้น = ครุ่นคิดมากขึ้นทีละนิด.

    ออกแบบมาเพื่อกลบ latency โดยเฉพาะ. เคล็ดคือ 'ความเข้ม' ต้องไต่ขึ้น ไม่ใช่ลูปเดิมซ้ำ ๆ
    เพราะลูปที่เท่ากันทุกรอบจะอ่านว่า 'ค้าง' ส่วนที่ไต่ขึ้นจะอ่านว่า 'ยังคิดอยู่'
    คลิปนี้ยาว 9 s ไล่ 3 ระดับ: เอียงหัว 8° -> 13° -> 18°, LED เร็วขึ้น 0.35 -> 0.6 Hz
    """
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(1.0, Posture(h=111.0, x=2.0), Expr(head_roll=8.0, head_pitch=3.0, ear_l=0.5,
                                               ear_r=0.35, led="think", led_v=0.5), hold=2.0),
        Key(1.5, Posture(h=111.5, x=3.0, yaw=-3.0), Expr(head_roll=13.0, head_pitch=1.0,
                                                         head_yaw=-5.0, ear_l=0.4, ear_r=0.3,
                                                         led="think", led_v=0.55), hold=1.5),
        Key(1.5, Posture(h=112.0, x=4.0, yaw=3.0), Expr(head_roll=18.0, head_pitch=-2.0,
                                                        head_yaw=5.0, ear_l=0.3, ear_r=0.2,
                                                        led="think", led_v=0.6), hold=1.5),
    ]
    t, P, E = sample_keys(keys, HZ)
    freq = np.interp(t, [0.0, 9.0], [0.35, 0.6])
    E[:, COL["led_v"]] *= 0.7 + 0.3 * np.sin(2.0 * np.pi * np.cumsum(freq) / HZ)
    E[:, COL["head_yaw"]] += 1.8 * _noise(len(t), 31, rate=0.6)
    E[:, COL["ear_l"]] += 0.08 * _noise(len(t), 32, rate=0.8)
    blink(E, t, [2.4, 6.1], 0.13)
    return assemble("thinking", META["thinking"], t, P, E, p)


@gesture("speaking", "conversation", "ตอบด้วยเสียง / TTS", "กำลังคุยกับมึง", loop_from=0.8)
def _speaking(p: Params = PARAMS) -> Clip:
    """หัว/หูขยับเบา ๆ ตามจังหวะเสียง — ไม่ต้องซิงก์เป๊ะ แค่ให้รู้สึกว่าเสียงมาจากตัวมัน."""
    talk = Posture(h=111.0, yaw=6.0)
    on = Expr(head_yaw=USER_YAW * 0.6, ear_l=0.5, ear_r=0.5, led="speak", led_v=0.5)
    keys = [Key(pose=REST_POSE, expr=REST), Key(0.8, talk, on), Key(0.0, talk, on, hold=5.0),
            Key(0.8, REST_POSE, REST)]
    t, P, E = sample_keys(keys, HZ)
    speak_env(E, t, 0.8, 5.8)
    E[:, COL["led_v"]] = np.clip(E[:, COL["led_v"]] + 0.25 * E[:, COL["sound"]], 0.0, 1.0)
    breathe(P, t, amp=1.0, period=4.0)
    blink(E, t, [3.3], 0.14)
    return assemble("speaking", META["speaking"], t, P, E, p)


@gesture("confirm_done", "conversation", "ทำคำสั่งเสร็จ (เพิ่มงาน/ตั้งเตือนแล้ว)",
         "เรียบร้อย จัดให้แล้ว", core=True)
def _confirm_done(p: Params = PARAMS) -> Clip:
    """พยักหัวสั้น ๆ หนึ่งครั้ง + หางสะบัด + LED แฟลชสีบวก. สั้น เด็ดขาด แล้วกลับ rest.

    ลำตัวยุบตาม 4 mm พร้อมหัวที่ผงก — เพื่อให้ท่านี้ยัง 'อ่านออก' แม้ยังไม่มีเซอร์โวหัว
    ทั้งท่ากินเวลา 1.5 s เท่านั้น ถ้ายาวกว่านี้จะกลายเป็นการเรียกร้องความสนใจ
    """
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(0.25, Posture(h=STAND_H + 3.0), Expr(head_pitch=9.0, ear_l=0.6, ear_r=0.6,
                                                 led="good", led_v=0.5)),
        Key(0.35, Posture(h=STAND_H - 4.0), Expr(head_pitch=-13.0, ear_l=0.35, ear_r=0.35,
                                                 led="good", led_v=1.0)),
        Key(0.55, REST_POSE, Expr(ear_l=0.2, ear_r=0.2, led="good", led_v=0.5)),
        Key(0.35, REST_POSE, REST),
    ]
    t, P, E = sample_keys(keys, HZ)
    wag(E, t, 0.35, beats=2, freq=3.0, amp=22.0)
    return assemble("confirm_done", META["confirm_done"], t, P, E, p)


@gesture("didnt_get_it", "conversation", "STT ไม่ชัด / ไม่เข้าใจคำสั่ง",
         "อ๊ะ ไม่เข้าใจ พูดใหม่ได้ไหม")
def _didnt_get_it(p: Params = PARAMS) -> Clip:
    """ส่ายหัวเบา ๆ หนึ่งครั้ง แล้วเอียงหัวค้างแบบงง หูตกนิด — ขอใหม่ ไม่ใช่ตำหนิ.

    ห้ามใช้ LED สีเตือน: ท่านี้ต้องไม่ทำให้ผู้ใช้รู้สึกว่าทำผิด จึงใช้สีกลาง
    """
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(0.45, REST_POSE, Expr(head_yaw=-11.0, ear_l=-0.2, ear_r=-0.2,
                                  led="neutral", led_v=0.45)),
        Key(0.6, REST_POSE, Expr(head_yaw=11.0, ear_l=-0.25, ear_r=-0.25,
                                 led="neutral", led_v=0.45)),
        Key(0.6, Posture(h=109.0, roll=2.0), Expr(head_yaw=3.0, head_roll=16.0, head_pitch=-3.0,
                                                  ear_l=-0.35, ear_r=-0.1, led="neutral",
                                                  led_v=0.4), hold=1.6),
        Key(0.8, REST_POSE, REST),
    ]
    t, P, E = sample_keys(keys, HZ)
    blink(E, t, [1.9], 0.2)
    return assemble("didnt_get_it", META["didnt_get_it"], t, P, E, p)


# ---- 4. บุคลิก & สังคม ---------------------------------------------------


@gesture("greet", "social", "เจอผู้ใช้ครั้งแรกของวัน / กลับมานั่งโต๊ะ", "เฮ้ ดีจ้า")
def _greet(p: Params = PARAMS) -> Clip:
    """หูตั้ง หันมามอง หางกระดิกสั้น ๆ หนึ่ง-สองที แล้วหยุด. อบอุ่นแต่ไม่โอเวอร์."""
    hello = Posture(h=115.0, yaw=10.0, x=4.0)
    on = Expr(head_yaw=USER_YAW, head_pitch=7.0, ear_l=1.0, ear_r=1.0, led="rest", led_v=0.7)
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(0.9, hello, on, hold=1.4),
        Key(1.0, REST_POSE, replace(REST, ear_l=0.3, ear_r=0.3), hold=0.5),
    ]
    t, P, E = sample_keys(keys, HZ)
    wag(E, t, 0.9, beats=3, freq=2.2, amp=26.0)
    led_pulse(E, t, 0.9, 0.8, hi=0.95)
    blink(E, t, [2.6], 0.15)
    return assemble("greet", META["greet"], t, P, E, p)


@gesture("celebrate", "social", "งานเสร็จ / ปิดเดดไลน์ / ทำ milestone ได้", "เก่งมาก! เย้")
def _celebrate(p: Params = PARAMS) -> Clip:
    """ท่าที่มีชีวิตชีวาที่สุดในคลัง: เด้งตัวสองครั้ง หัวเงย หางกระดิกเร็ว LED สีสด.

    ทั้งหมดเป็นการเด้ง 'แนวดิ่ง' ล้วน เท้าทั้งสี่ไม่ยกเลย — margin จึงเท่ากับท่ายืนตลอด
    ไม่มีจังหวะไหนเสี่ยงล้ม แม้จะเป็นท่าที่เร็วที่สุด (ตรวจแล้วในหัวข้อ check)
    จบใน 2.1 s ตามข้อกำหนด 'ต้องสั้นและจบเร็ว เพื่อไม่ให้กลายเป็นตัวป่วน'
    """
    hi = Expr(head_pitch=16.0, ear_l=1.0, ear_r=1.0, led="good", led_v=1.0)
    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(0.40, Posture(h=126.0, pitch=4.0), hi),
        Key(0.40, Posture(h=104.0, pitch=-1.0), replace(hi, head_pitch=4.0, led_v=0.6)),
        Key(0.40, Posture(h=122.0, pitch=3.0), replace(hi, head_pitch=13.0, led_v=1.0)),
        Key(0.40, Posture(h=108.0), replace(hi, head_pitch=2.0, led_v=0.7)),
        Key(0.45, REST_POSE, replace(REST, ear_l=0.5, ear_r=0.5, led="good", led_v=0.5), hold=0.4),
    ]
    t, P, E = sample_keys(keys, HZ)
    wag(E, t, 0.30, beats=6, freq=3.0, amp=25.0)
    return assemble("celebrate", META["celebrate"], t, P, E, p)


@gesture("comfort", "social", "จับสัญญาณผู้ใช้เครียด/หัวร้อน (ใช้อย่างระวัง)",
         "ใจเย็น ๆ อยู่ด้วยนะ", loop_from=2.0)
def _comfort(p: Params = PARAMS) -> Clip:
    """หายใจช้ามากให้ผู้ใช้หายใจตาม (5.5 ครั้ง/นาที = รอบละ 11 s) LED สีอุ่น ไม่พูด ไม่เตือน.

    แอมพลิจูด 6 mm มากกว่า idle เกือบสามเท่า — ตั้งใจให้ 'มองแล้วหายใจตามได้'
    เข้า 4.5 s ออก 6.5 s (หายใจออกยาวกว่าเข้า คือรูปแบบที่กระตุ้นระบบพาราซิมพาเทติก)
    """
    warm = Posture(h=106.0)
    soft = Expr(head_pitch=-4.0, ear_l=-0.1, ear_r=-0.1, eye=0.8, led="warm", led_v=0.3)
    keys = [Key(pose=REST_POSE, expr=REST), Key(2.0, warm, soft), Key(0.0, warm, soft, hold=22.0)]
    t, P, E = sample_keys(keys, HZ)
    # หายใจอสมมาตร: เข้า 4.5 s ออก 6.5 s สร้างเป็นเฟสที่วิ่งไม่เท่ากันสองครึ่ง
    per_in, per_out = 4.5, 6.5
    u = np.mod(t - 2.0 + per_in / 2.0, per_in + per_out)   # เริ่มที่กลางคาบ = ออฟเซ็ต 0
    s = np.where(u < per_in, 0.5 - 0.5 * np.cos(np.pi * u / per_in),
                 0.5 + 0.5 * np.cos(np.pi * (u - per_in) / per_out))
    m = t >= 2.0
    P[m, 0] += 6.0 * (s[m] - 0.5)
    E[m, COL["led_v"]] = 0.18 + 0.22 * s[m]
    blink(E, t, [6.0, 14.5, 21.0], 0.2)
    return assemble("comfort", META["comfort"], t, P, E, p)


# ---- 5. ระบบ -------------------------------------------------------------


@gesture("error_state", "system", "เน็ตหลุด / เซอร์โวค้าง / STT ล่ม", "มีอะไรผิดปกติ",
         loop_from=2.0)
def _error_state(p: Params = PARAMS) -> Clip:
    """หัวตกค้าง + หูตก + ลำตัวเอียงค้าง 6° + LED สีเตือนกะพริบ.

    ต้องแยกจาก 'ง่วง' ให้ออก จุดที่ทำให้แยกออกคือ: ลำตัวเอียง (ท่าง่วงทุกท่าสมมาตร)
    และ **ตายังเปิด** — ของที่เสียไม่ได้หลับตา. ไม่หายใจด้วย: นิ่งค้างแบบผิดธรรมชาติ
    """
    broken = Posture(h=88.0, roll=6.0, pitch=-4.0)
    dead = Expr(head_pitch=-28.0, head_roll=-9.0, ear_l=-0.9, ear_r=-0.9, eye=0.9,
                led="warn", led_v=0.8)
    keys = [Key(pose=REST_POSE, expr=REST), Key(2.0, broken, dead), Key(0.0, broken, dead, hold=8.0)]
    t, P, E = sample_keys(keys, HZ)
    led_blink(E, t, freq=0.55, lo=0.15, hi=0.9, t0=2.0)
    return assemble("error_state", META["error_state"], t, P, E, p)


@gesture("low_power", "system", "แบตใกล้หมด / กำลังชาร์จ", "หิวไฟ / กำลังกินไฟ",
         loop_from=5.0)
def _low_power(p: Params = PARAMS) -> Clip:
    """ท่าพักคล้าย sleep (พับขาลงกองกับพื้น) แต่ตาปรือครึ่งเดียว และ LED เต้นเป็นลายพลังงาน.

    พับขาเหมือนกันโดยตั้งใจ: แบตต่ำ = ต้องหยุดจ่ายไฟให้เซอร์โว ท่าที่ 'กองกับพื้น'
    คือท่าเดียวที่ปล่อยแรงบิดได้หมดจริง ๆ ความต่างจาก sleep ไปอยู่ที่ตากับ LED
    """
    from sleep_wake import SETTLE_H, stance_pose

    q_sleep = pose_vector(find_sleep_pose(p))
    q_stand = stance_pose(p, Gait(stand_h=STAND_H, splay_deg=SPLAY, hz=HZ), STAND_H)
    q_low = stance_pose(p, Gait(stand_h=STAND_H, splay_deg=SPLAY, hz=HZ), SETTLE_H)

    keys = [
        Key(pose=REST_POSE, expr=REST),
        Key(2.0, REST_POSE, Expr(head_pitch=-12, ear_l=-0.5, ear_r=-0.5, eye=0.6,
                                 led="power", led_v=0.4)),
        Key(3.0, REST_POSE, Expr(head_pitch=-22, ear_l=-0.85, ear_r=-0.85, eye=0.35,
                                 led="power", led_v=0.5), hold=8.0),
    ]
    t, P, E = sample_keys(keys, HZ)
    n1, n2 = int(2.0 * HZ), int(3.0 * HZ)
    seg1 = np.array([(1 - _smoothstep((i + 1) / n1)) * q_stand + _smoothstep((i + 1) / n1) * q_low
                     for i in range(n1)])
    seg2 = np.array([(1 - _smoothstep((i + 1) / n2)) * q_low + _smoothstep((i + 1) / n2) * q_sleep
                     for i in range(n2)])
    q_track = np.vstack([q_stand, seg1, seg2, np.tile(q_sleep, (len(t) - 1 - n1 - n2, 1))])

    # ลายพลังงาน: ไล่สว่างขึ้นแล้วดับวูบ ซ้ำทุก 2.5 s — อ่านว่า 'กำลังเติม/กำลังจะหมด'
    m = t >= 5.0
    u = np.mod(t[m] - 5.0, 2.5) / 2.5
    E[m, COL["led_v"]] = np.where(u < 0.8, 0.12 + 0.75 * (u / 0.8), 0.05)
    blink(E, t, [7.0, 11.5], 0.4)
    return assemble("low_power", META["low_power"], t, P, E, p, q_track=q_track)


@gesture("home_reset", "system", "เริ่มระบบ / คาลิเบรต / จบทุกซีเควนซ์",
         "(ภายใน) กลับจุดตั้งต้น", core=False)
def _home_reset(p: Params = PARAMS) -> Clip:
    """ทุกข้อกลับ neutral ช้า ๆ ปลอดภัย — pose อ้างอิงที่ทุกท่าเริ่มและจบ.

    ตั้งใจให้ช้า (3 s) และไม่มี LED/เสียงอะไรเลย เพราะท่านี้ผู้ใช้ไม่ควรต้องอ่านความหมาย
    """
    keys = [
        Key(pose=Posture(h=98.0, roll=3.0, pitch=-2.0, yaw=4.0),
            expr=Expr(head_pitch=-10.0, head_roll=6.0, ear_l=-0.4, ear_r=0.2, eye=0.6,
                      led="off", led_v=0.0)),
        Key(3.0, REST_POSE, REST, hold=1.0),
    ]
    t, P, E = sample_keys(keys, HZ)
    return assemble("home_reset", META["home_reset"], t, P, E, p)


# ---------------------------------------------------------------------------
# 5) เรียกใช้ + ตรวจ
# ---------------------------------------------------------------------------

ORDER: Tuple[str, ...] = (
    "sleep_mode", "wake_up", "idle_breathe", "doze_off",
    "notice_low", "notice_soon", "notice_urgent", "settle", "focus_with_you",
    "listening", "thinking", "speaking", "confirm_done", "didnt_get_it",
    "greet", "celebrate", "comfort",
    "error_state", "low_power", "home_reset",
)


def build(gid: str, p: Params = PARAMS) -> Clip:
    """สร้างคลิปของ gesture_id หนึ่งท่า."""
    if gid not in BUILDERS:
        raise KeyError(f"ไม่มีท่าชื่อ {gid!r} — มีให้เลือก: {', '.join(ORDER)}")
    return BUILDERS[gid](p)


def build_all(p: Params = PARAMS) -> Dict[str, Clip]:
    return {gid: build(gid, p) for gid in ORDER}


def check(clip: Clip, servo: Servo = Servo(), lim: JointLimits = LIMITS) -> Dict[str, float]:
    """ตรวจคลิปหนึ่งท่ากับข้อจำกัดจริง: ลิมิตข้อต่อ ความเร็ว แรงบิด ทรงตัว และช่วงค่าของช่องแสดงออก."""
    g = Gait(hz=HZ, cycle_s=max(clip.dur, 1e-3), stand_h=STAND_H, splay_deg=SPLAY)
    res = analyze(clip.traj, PARAMS, g, servo)

    bad = set()
    for k in range(len(clip.t)):
        if np.all(np.isfinite(clip.traj.q[k])):
            for v in limit_violations(clip.traj.q[k], lim):
                bad.add(v.split(" = ")[0])

    def peak(group: Dict[str, float]) -> float:
        out = 0.0
        for name, scale in group.items():
            d = np.abs(np.diff(clip.expr[:, COL[name]] * scale)) * HZ
            out = max(out, float(d.max()) if len(d) else 0.0)
        return out

    e_rate, tail_rate = peak(EXPR_HEAD), peak(EXPR_TAIL)

    out = {
        "dur_s": clip.dur,
        "peak_rate_dps": res["peak_rate_dps"],
        "tau_max_kgcm": max(res["tau_abduction_kgcm"], res["tau_hip_kgcm"], res["tau_knee_kgcm"]),
        "torque_budget_kgcm": res["torque_budget_kgcm"],
        "min_margin_mm": res["min_margin_mm"],
        "head_rate_dps": e_rate,
        "tail_rate_dps": tail_rate,
        "n_bad_joint": float(len(bad)),
        "reachable": float(clip.traj.reachable),
    }
    if bad:
        clip.notes.append(f"ERROR: ข้อต่อหลุดลิมิต {sorted(bad)}")
    if out["min_margin_mm"] <= 0:
        clip.notes.append(f"ERROR: margin {out['min_margin_mm']:.1f} mm -> ล้ม")
    elif out["min_margin_mm"] < 8.0:
        clip.notes.append(f"เตือน: margin {out['min_margin_mm']:.1f} mm บางไป")
    if out["peak_rate_dps"] > servo.max_rate_dps:
        clip.notes.append(f"เตือน: ขาต้องการ {out['peak_rate_dps']:.0f}°/s "
                          f"> เซอร์โวไหว {servo.max_rate_dps:.0f}°/s")
    if e_rate > HEAD_RATE_LIMIT:
        clip.notes.append(f"เตือน: หัว/หูต้องการ {e_rate:.0f}°/s > {HEAD_RATE_LIMIT:.0f}°/s")
    if tail_rate > TAIL_RATE_LIMIT:
        clip.notes.append(f"เตือน: หางต้องการ {tail_rate:.0f}°/s > {TAIL_RATE_LIMIT:.0f}°/s")
    if not clip.traj.reachable:
        clip.notes.append("ERROR: IK เอื้อมไม่ถึง")
    return out


def report(p: Params = PARAMS, servo: Servo = Servo()) -> Dict[str, Dict[str, float]]:
    """ตรวจทุกท่าในคลังแล้วพิมพ์เป็นตาราง."""
    print(f"เซอร์โว: stall {servo.stall_kgcm} kg·cm, ใช้ได้ {servo.usable_torque:.0%} "
          f"= งบ {servo.stall_kgcm * servo.usable_torque:.2f} kg·cm, "
          f"ไหว {servo.max_rate_dps:.0f}°/s\n")
    head = (f"{'gesture_id':<16}{'กลุ่ม':<13}{'วินาที':>7}{'วน':>5}"
            f"{'°/s ขา':>9}{'°/s หัว':>9}{'°/s หาง':>9}{'τ kg·cm':>10}{'margin':>9}  หมายเหตุ")
    print(head)
    print("-" * len(head))
    out: Dict[str, Dict[str, float]] = {}
    for gid in ORDER:
        clip = build(gid, p)
        res = check(clip, servo)
        out[gid] = res
        loop = "ใช่" if clip.loop_from is not None else "-"
        marg = ("ไม่ต้อง" if not np.isfinite(res["min_margin_mm"])
                else f"{res['min_margin_mm']:8.1f}")
        note = "; ".join(clip.notes) if clip.notes else ("แกน" if clip.core else "")
        print(f"{gid:<16}{clip.group:<13}{res['dur_s']:7.1f}{loop:>5}"
              f"{res['peak_rate_dps']:9.0f}{res['head_rate_dps']:9.0f}"
              f"{res['tail_rate_dps']:9.0f}"
              f"{res['tau_max_kgcm']:10.2f}{marg:>9}  {note}")
    return out


def verify(p: Params = PARAMS) -> None:
    """ตรวจความถูกต้องของตัวเอนจิน (ไม่ใช่แค่ของท่า)."""
    T = posture_T(REST_POSE.arr())
    q = posture_ik(T, FEET0, p)
    for i, leg in enumerate(LEGS):
        foot = to_world(T, leg_fk(leg, q[3 * i:3 * i + 3], p)[-1][:3, 3])[0]
        assert np.allclose(foot, FEET0[leg], atol=1e-6), f"IK ท่าพักคลาดที่ {leg}: {foot}"

    # ลำตัวเอียง/หมุน/เลื่อน แล้วเท้ายังต้องปักอยู่ที่เดิมเป๊ะ
    odd = Posture(h=104.0, x=8.0, y=-5.0, roll=4.0, pitch=-6.0, yaw=9.0)
    T2 = posture_T(odd.arr())
    q2 = posture_ik(T2, FEET0, p)
    for i, leg in enumerate(LEGS):
        foot = to_world(T2, leg_fk(leg, q2[3 * i:3 * i + 3], p)[-1][:3, 3])[0]
        assert np.allclose(foot, FEET0[leg], atol=1e-6), f"IK ท่าเอียงคลาดที่ {leg}: {foot}"

    assert np.linalg.det(T2[:3, :3]) > 0.999, "เมทริกซ์หมุนไม่เป็น orthonormal"

    # สีต้องสลับแบบขั้นบันได ไม่ไล่ผ่านดัชนีสีที่ไม่ได้สั่ง
    _, _, E = sample_keys([Key(expr=Expr(led="off")), Key(1.0, expr=Expr(led="urgent"))], HZ)
    assert set(np.unique(E[:, LED_COL])) <= {float(LED_IDX["off"]), float(LED_IDX["urgent"])}, \
        "LED ไล่ผ่านสีกลางทาง"

    # เลื่อนลำตัวหา incenter แล้วเงา CoG ต้องตกในสามเหลี่ยมสามขาจริง
    planted = ("FR", "RL", "RR")
    pose = solve_shift(Posture(h=118.0), planted, p)
    Ts = posture_T(pose.arr())
    qs = posture_ik(Ts, FEET0, p)
    cog = to_world(Ts, center_of_gravity(qs, p)[0])[0]
    tri = np.array([FEET0[leg][:2] for leg in planted])
    assert margin_from_points(cog, np.c_[tri, np.zeros(3)]) > 15.0, "เลื่อนลำตัวแล้วยังไม่พ้นขอบ"

    # ทุกท่าต้องเริ่มหรือจบที่ท่าอ้างอิง (rest หรือท่าค้างของตัวเอง) และต่อเนื่องไม่กระโดด
    for gid in ORDER:
        clip = build(gid, p)
        jump = np.max(np.abs(np.diff(np.degrees(clip.traj.q), axis=0))) * HZ
        assert jump < 900.0, f"{gid}: มุมกระโดด {jump:.0f}°/s"
        assert clip.expr[:, COL["eye"]].min() >= -1e-9, f"{gid}: ค่า eye ติดลบ"
        assert np.all(np.isin(clip.expr[:, LED_COL], np.arange(len(LED_NAMES)))), \
            f"{gid}: ดัชนีสี LED ไม่ถูกต้อง"
    print("verify: ผ่านทุกข้อ")


def peak_frame(clip: Clip) -> int:
    """เฟรมที่ท่า 'อ่านออกชัดที่สุด' = ห่างจากเฟรมแรกของท่านั้นเองมากที่สุด.

    ต้องวัดจากเฟรมแรกของตัวเอง ไม่ใช่จากท่าพักกลาง เพราะท่าเปลี่ยนสถานะอย่าง wake_up
    เริ่มจากท่านอน ถ้าวัดจากท่าพักกลางจะได้เฟรมแรก (ยังหลับอยู่) ซึ่งอ่านผิดความหมาย
    ไม่นับช่อง eye เพราะการกะพริบเป็นจังหวะสั้น ๆ ที่ทุกท่ามีเหมือนกัน ถ้านับจะชนะทุกครั้ง
    """
    d = clip.expr - clip.expr[0]
    score = (np.abs(d[:, COL["head_pitch"]]) + np.abs(d[:, COL["head_yaw"]])
             + np.abs(d[:, COL["head_roll"]]) + 30.0 * np.abs(d[:, COL["ear_l"]])
             + 0.6 * np.abs(d[:, COL["tail"]]))
    return int(np.argmax(score))


LED_BONUS = 0.9      # สี LED ต่างกัน = แยกออกได้ทันทีจากหางตา ให้ค่าระยะห่างเพิ่มก้อนหนึ่ง


def signature(clip: Clip) -> Tuple[np.ndarray, str]:
    """ลายเซ็นของท่า ณ เฟรมเด่น (นอร์มัลไลซ์แล้ว) + สี LED — ใช้วัดว่าท่าไหนซ้ำกับท่าไหน."""
    k = peak_frame(clip)
    e = clip.expr[k]
    h = clip.traj.body_T[:, 2, 3]
    vec = np.array([
        e[COL["head_pitch"]] / 35.0, e[COL["head_yaw"]] / 60.0, e[COL["head_roll"]] / 25.0,
        e[COL["ear_l"]], e[COL["ear_r"]], e[COL["eye"]],
        (h[k] - STAND_H) / 30.0,                      # ความสูงลำตัวตอนนั้น
        float(h.max() - h.min()) / 20.0,              # ลำตัวขยับมากแค่ไหนตลอดท่า
    ])
    return vec, LED_NAMES[int(round(e[COL["led"]]))]


def distinct(p: Params = PARAMS, thresh: float = 0.85, top: int = 10) -> List[Tuple[str, str, float]]:
    """หา 'คู่ท่าที่เสี่ยงอ่านสับสน' ตามกติกาในเอกสาร: ต้องอ่านออกจากหางตาในเสี้ยววินาที.

    วัดที่เฟรมเด่นของแต่ละท่าเท่านั้น = จำลองการเหลือบมองครั้งเดียว ไม่ได้ดูทั้งคลิป
    ท่าที่ระยะห่างต่ำแปลว่า 'ภาพนิ่งแยกไม่ออก' ต้องพึ่งจังหวะเวลาหรือสีเพิ่ม
    """
    sigs = {gid: signature(build(gid, p)) for gid in ORDER}
    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            va, la = sigs[a]
            vb, lb = sigs[b]
            d = float(np.linalg.norm(va - vb)) + (0.0 if la == lb else LED_BONUS)
            pairs.append((a, b, d))
    pairs.sort(key=lambda r: r[2])
    print(f"คู่ท่าที่ 'ภาพนิ่ง' ใกล้กันที่สุด (ต่ำกว่า {thresh:.2f} = เสี่ยงอ่านสับสน)\n")
    for a, b, d in pairs[:top]:
        flag = "  <-- เสี่ยง" if d < thresh else ""
        print(f"  {d:5.2f}  {a:<16} vs {b:<16}{flag}")
    return pairs[:top]


def export_csv(clip: Clip, path: str) -> None:
    """ส่งออกไทม์ไลน์ 50 Hz: มุมข้อต่อ 12 ตัว (องศา) + 10 ช่องการแสดงออก + สถานะทรงตัว."""
    names = [f"{leg}_{j}" for leg in LEGS for j in ("th1", "th2", "th3")]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", *names, *EXPR_COLS, "led_name", "margin_mm", "contact"])
        deg = np.degrees(clip.traj.q)
        for k, tk in enumerate(clip.t):
            led = LED_NAMES[int(round(clip.expr[k, LED_COL]))]
            con = "".join("1" if c else "0" for c in clip.traj.contact[k])
            w.writerow([f"{tk:.3f}", *[f"{v:.3f}" for v in deg[k]],
                        *[f"{v:.4f}" for v in clip.expr[k]], led,
                        f"{clip.traj.margin[k]:.2f}", con])
    print(f"เขียน {path} ({len(clip.t)} เฟรม @ {HZ:.0f} Hz, {clip.dur:.1f} s)")


def show_catalog() -> None:
    """พิมพ์คลังท่าแบบอ่านง่าย."""
    groups = {"lifecycle": "1. วงจรชีวิต", "attention": "2. การแจ้งเตือน & โฟกัส",
              "conversation": "3. ลูปสนทนา", "social": "4. บุคลิก & สังคม", "system": "5. ระบบ"}
    seen = None
    for gid in ORDER:
        m = META[gid]
        if m["group"] != seen:
            seen = m["group"]
            print(f"\n=== {groups[str(seen)]} ===")
        tag = " [แกน]" if m.get("core") else ""
        loop = f"  (ลูปจาก {m['loop_from']:.1f}s)" if m["loop_from"] is not None else ""
        print(f"  {gid:<16}{m['means']}{tag}{loop}")
        print(f"  {'':<16}เมื่อไร: {m['when']}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="แสดงคลังท่าทั้งหมด")
    ap.add_argument("--check", action="store_true", help="ตรวจทุกท่ากับข้อจำกัดฮาร์ดแวร์")
    ap.add_argument("--verify", action="store_true", help="ตรวจความถูกต้องของเอนจิน")
    ap.add_argument("--distinct", action="store_true",
                    help="หาคู่ท่าที่ภาพนิ่งอ่านสับสนกันได้")
    ap.add_argument("--csv", metavar="GID", help="ส่งออกไทม์ไลน์ของท่านี้เป็น CSV")
    ap.add_argument("--out", metavar="PATH", help="ชื่อไฟล์ CSV (ค่าตั้งต้น = <gid>.csv)")
    ap.add_argument("--servo", default="ES08MA II (analog)", help="รุ่นเซอร์โวที่ใช้ตรวจ")
    args = ap.parse_args(argv)

    from walking_gait import SERVO_MODELS

    servo = SERVO_MODELS.get(args.servo, Servo())
    if args.verify:
        verify()
    if args.list:
        show_catalog()
    if args.csv:
        clip = build(args.csv)
        check(clip, servo)
        export_csv(clip, args.out or f"{args.csv}.csv")
    if args.distinct:
        distinct()
    if args.check or not (args.list or args.csv or args.verify or args.distinct):
        report(servo=servo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
