#!/usr/bin/env python3
"""
walking_gait.py — สร้าง walking motion ของหุ่นสี่ขา 12 DOF บนข้อจำกัดจริงของฮาร์ดแวร์

ข้อจำกัดที่คิดให้แล้ว:
  * เซอร์โว EMAX analog   -> อัปเดตได้ 50 Hz, มี deadband, ไม่มี feedback, แรงบิดจำกัด
                             => ใช้ crawl gait ช้า ๆ ทางเดินเท้าเรียบ ไม่มีการกระชาก
  * Raspberry Pi 4 + UPS   -> แขวนใต้ท้อง มวลอยู่ต่ำ ดึง CoG ลง (ดีต่อการทรงตัว)
  * จอ 7" กลางหลัง         -> สมมาตรตามแกน y แล้ว ไม่ต้อง bias ถาวรชดเชยข้างอีก
  * แบต 21700 x4           -> ฝั่งละ 2 ก้อน สมมาตร แต่หนัก มวลรวมพุ่งขึ้น
                             => แรงบิดเซอร์โวคือคอขวดหลัก ฐานเท้ายังกว้างแค่ 2b
                             ต้องชดเชยด้วยการถ่างขา (splay) + แกว่งลำตัว (sway)

รัน:  python walking_gait.py                 ตรวจ + สรุปผล + เปิดรูป
      python walking_gait.py --save out.png  บันทึกรูปแทนการเปิดหน้าต่าง
      python walking_gait.py --csv gait.csv  บันทึกมุมเซอร์โวไปรันบน Pi
      python walking_gait.py --verify-only   ตรวจอย่างเดียว
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

if {"--save", "--csv", "--verify-only"} & set(sys.argv[1:]):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from leg_kinematics import (  # noqa: E402
    LEGS,
    LEG_SIGNS,
    PARAMS,
    Params,
    body_to_frame0,
    leg_fk,
    leg_polyline,
)

# ---------------------------------------------------------------------------
# 1) ฮาร์ดแวร์: เซอร์โว + มวลที่แขวนอยู่บนลำตัว
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Servo:
    """สเปกเซอร์โว analog (ค่าตั้งต้นอิงตระกูล EMAX ES08MA II — แก้ตามรุ่นที่ใช้จริง)."""

    max_rate_dps: float = 300.0   # ความเร็วที่ใช้วางแผนจริง (spec 0.12s/60° = 500°/s ตอนไม่มีโหลด)
    deadband_deg: float = 1.5     # โซนตาย: สั่งละเอียดกว่านี้เซอร์โวไม่ขยับ
    update_hz: float = 50.0       # analog servo กินพัลส์ 50 Hz เท่านั้น
    stall_kgcm: float = 1.8       # แรงบิดสูงสุด @6V
    usable_torque: float = 0.35   # ใช้ได้จริงกี่ % ของ stall (analog ร้อนง่าย อย่ารีดเกินนี้)


# สเปกอ้างอิงจากแคตตาล็อกทั่วไป @6V — ของจริงแต่ละล็อตต่างกัน ควรเช็คสเปกรุ่นที่ซื้อจริง
# usable_torque: digital ถือตำแหน่งนิ่งกว่าและร้อนน้อยกว่าที่โหลดเท่ากัน เลยรีดได้มากกว่านิดหน่อย
SERVO_MODELS: Dict[str, "Servo"] = {}


def _register_servos() -> None:
    """ลงทะเบียนรุ่นเซอร์โวไว้เทียบกัน (เรียกหลังนิยาม Servo)."""
    SERVO_MODELS.update({
        "ES08MA II (analog)":  Servo(max_rate_dps=300.0, deadband_deg=1.5, update_hz=50.0,
                                     stall_kgcm=1.8, usable_torque=0.35),
        "ES08MD II (digital)": Servo(max_rate_dps=420.0, deadband_deg=0.4, update_hz=333.0,
                                     stall_kgcm=2.0, usable_torque=0.45),
        "ES3352 MD (digital)": Servo(max_rate_dps=500.0, deadband_deg=0.3, update_hz=333.0,
                                     stall_kgcm=3.6, usable_torque=0.45),
    })


@dataclass(frozen=True)
class Payload:
    """ก้อนมวลที่ยึดติดกับลำตัว: มวล (กรัม) + ตำแหน่งเทียบเฟรมลำตัว {B} (mm)."""

    name: str
    mass_g: float
    pos: Tuple[float, float, float]


# ตำแหน่งก้อนแบต 21700 — ยึดข้างลำตัวทั้งสองฝั่ง ฝั่งละ 2 ก้อน (สมมาตรกัน)
CELL_Y = 44.0   # กึ่งกลางก้อนแบต ห่างจากแกนกลางลำตัว (b = 29 -> ยื่นพ้นสะโพกออกไปเล็กน้อย)
CELL_Z = 4.0    # อยู่ระดับแผ่นฐาน ไม่ยกสูง

# มวลทั้งหมดเป็น "ค่าประมาณ" — ชั่งของจริงแล้วมาแก้ตัวเลขตรงนี้
#
# เลย์เอาต์ปี 2026: จอย้ายมาไว้ "กลางหลัง", แบต 21700 แบ่งซ้าย-ขวาฝั่งละ 2 ก้อน,
# Pi 4 + UPS ย้ายลงไปแขวน "ใต้ท้อง" ผลรวมคือ CoG สมมาตรตามแกน y และต่ำลง
# แต่มวลรวมเพิ่มขึ้นเยอะ (แบต 4 ก้อนหนักกว่าแบต 2S เดิมมาก) -> แรงบิดคือปัญหาหลัก
BODY_PAYLOADS: List[Payload] = [
    Payload("โครง/แผ่นฐาน", 120.0, (0.0, 0.0, 0.0)),
    Payload("จอ 7 นิ้ว + กรอบ", 277.0, (0.0, 0.0, 22.0)),                  # กลางหลัง ไม่เยื้องข้างแล้ว
    Payload("Raspberry Pi 4 + UPS", 115.0, (-5.0, 0.0, -20.0)),            # แขวนใต้ท้อง -> ถ่วง CoG ลง
    Payload("แบต 21700 x2 (ซ้าย)", 155.0, (0.0, +CELL_Y, CELL_Z)),
    Payload("แบต 21700 x2 (ขวา)", 155.0, (0.0, -CELL_Y, CELL_Z)),
    Payload("เซอร์โว abduction x4", 48.0, (0.0, 0.0, -5.0)),               # ติดกับลำตัว ไม่ขยับ
]

_register_servos()

SERVO_HIP_G = 12.0    # เซอร์โว hip pitch ติดอยู่ที่ต้นขา ขยับตามขา
SERVO_KNEE_G = 12.0   # เซอร์โว knee ติดอยู่ที่เข่า
SHANK_G = 6.0         # ชิ้นแข้ง + ฟุตแพด

# ---------------------------------------------------------------------------
# 2) inverse kinematics
# ---------------------------------------------------------------------------


def leg_ik(leg: str, foot: Sequence[float], p: Params = PARAMS, knee: int = +1) -> np.ndarray:
    """หามุมข้อต่อ (θ1,θ2,θ3) ที่ทำให้ปลายเท้าไปอยู่ที่ foot (เทียบเฟรมลำตัว, mm).

    knee = +1/-1 เลือกว่าจะให้เข่าพับไปทางไหน (สองคำตอบของ 2-link)
    คืน NaN ถ้าจุดนั้นเอื้อมไม่ถึง
    """
    _, _, sD1, sD2 = LEG_SIGNS[leg]
    hip = body_to_frame0(leg, p)[:3, 3]
    rx, ry, rz = np.asarray(foot, dtype=float) - hip

    d_net = sD1 * p.D1 - sD2 * p.D2          # offset ด้านข้างสุทธิ อยู่ตามแกน z1 เสมอ
    R = float(np.hypot(ry, rz))
    if R * R < d_net * d_net:
        return np.full(3, np.nan)            # ใกล้แกน abduction เกินไป offset ด้านข้างพาไปไม่ถึง

    reach = float(np.sqrt(R * R - d_net * d_net))   # ระยะในระนาบขา นับจากแกน abduction
    lam = reach - p.h                                # ส่วนที่เหลือให้ 2-link รับผิดชอบ (ตามแนวขา)
    th1 = np.arctan2(ry, -rz) - np.arctan2(d_net, reach)

    # 2-link ในระนาบขา: แกนแรกคือแนว "ลง" ของขา (lam), อีกแกนคือแกน x ของลำตัว
    m = -rx                                   # +θ2 ทำให้ปลายเท้าถอยไปทาง -x (วัดจาก FK)
    rho2 = lam * lam + m * m
    cos3 = (rho2 - p.L1 ** 2 - p.L2 ** 2) / (2.0 * p.L1 * p.L2)
    if abs(cos3) > 1.0:
        return np.full(3, np.nan)             # เอื้อมไม่ถึง หรือหดเกินพับ
    th3 = knee * float(np.arccos(cos3))
    th2 = np.arctan2(m, lam) - np.arctan2(p.L2 * np.sin(th3), p.L1 + p.L2 * np.cos(th3))
    return np.array([th1, th2, th3])


def robot_ik(feet: Dict[str, Sequence[float]], p: Params = PARAMS, knee: int = +1) -> np.ndarray:
    """IK ทั้ง 4 ขา คืนมุม 12 ตัวเรียงตาม LEGS."""
    return np.concatenate([leg_ik(leg, feet[leg], p, knee) for leg in LEGS])


# ---------------------------------------------------------------------------
# 3) จุดศูนย์ถ่วง (CoG) — รวมมวลลำตัวกับมวลขาที่ขยับตามท่า
# ---------------------------------------------------------------------------


def center_of_gravity(q_all: Sequence[float], p: Params = PARAMS,
                      payloads: Sequence[Payload] = tuple(BODY_PAYLOADS)) -> Tuple[np.ndarray, float]:
    """คืน (ตำแหน่ง CoG เทียบเฟรมลำตัว mm, มวลรวม กรัม)."""
    moment = np.zeros(3)
    total = 0.0
    for pay in payloads:
        moment += pay.mass_g * np.asarray(pay.pos, dtype=float)
        total += pay.mass_g

    q = np.asarray(q_all, dtype=float).reshape(len(LEGS), 3)
    for i, leg in enumerate(LEGS):
        chain = leg_fk(leg, q[i], p)
        o1, o2, o3 = chain[1][:3, 3], chain[2][:3, 3], chain[3][:3, 3]
        for mass, pos in ((SERVO_HIP_G, o1), (SERVO_KNEE_G, o2), (SHANK_G, (o2 + o3) / 2.0)):
            moment += mass * pos
            total += mass
    return moment / total, total


# ---------------------------------------------------------------------------
# 4) พารามิเตอร์ท่าเดิน
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gait:
    """ค่าปรับท่าเดิน — ทุกระยะเป็น mm ทุกมุมเป็นองศา."""

    # ค่าตั้งต้นนี้มาจากการกวาดพารามิเตอร์ (ดูหัวข้อ "ผลการจูน" ท้ายไฟล์)
    stand_h: float = 110.0     # ยืนสูง = ขาตรงขึ้น = โมเมนต์สั้นลง = แรงบิดต่ำลง (สูงสุด 136)
    splay_deg: float = 12.0    # ถ่างขาออกข้าง เพิ่มความกว้างฐาน — แลกกับแรงบิด abduction
    stride: float = 30.0       # ระยะก้าวต่อรอบ (ปลายเท้าเคลื่อนไปข้างหน้าเทียบลำตัว)
    heading_deg: float = 0.0   # ทิศเดิน: 0 = หน้า, 90 = ซ้าย, -90 = ขวา, 180 = ถอยหลัง
    yaw_deg: float = 0.0       # เลี้ยวกี่องศาต่อรอบ: + = เลี้ยวซ้าย, - = เลี้ยวขวา
    clearance: float = 14.0    # ยกเท้าสูงเท่าไรตอน swing
    cycle_s: float = 6.0       # เวลา 1 รอบ — ช้าไว้ เซอร์โว analog ไม่มี feedback ตามไม่ทันจะ overshoot
    duty: float = 0.85         # สัดส่วนเวลาที่เท้าแตะพื้น ต้อง > 0.75 ไม่งั้นไม่เหลือเวลาย้ายลำตัว
    sway_gain: float = 0.6     # แกน y: 0 = ไม่แกว่งลำตัว, 1 = ย้าย CoG ไป incenter เต็มที่
    # แกน x แยกต่างหาก เพราะมันคือตัวสร้าง margin เกือบทั้งหมด แต่ก็ทำให้ลำตัวโคลงหน้า-หลัง
    # 0.0 -> margin 1 mm (ล้ม), 0.5 -> 12.8 mm โคลง 43 mm, 1.0 -> 23.6 mm แต่โคลง 87 mm
    sway_gain_x: float = 0.5
    bias_y: float | None = None  # เลื่อนลำตัวถาวรชดเชยจอ (None = คำนวณให้อัตโนมัติ)
    x_center: float = 0.0      # เลื่อนจุดกึ่งกลางก้าวไปหน้า/หลัง (ปรับ trim ตอน CoG เยื้องหน้าหลัง)
    hz: float = 50.0           # อัตราส่งคำสั่ง = อัตราพัลส์ของเซอร์โว analog
    knee: int = +1             # ทิศพับเข่า


# ลำดับยกขาแบบ crawl ที่เสถียรที่สุด: ทแยงสลับหน้า-หลัง
SWING_ORDER: Tuple[str, ...] = ("FL", "RR", "FR", "RL")

# ชุดค่าสำเร็จรูปของแต่ละทิศ — เดินข้างต้องใช้คนละชุดกับเดินหน้า เพราะฐานเท้าแคบทางด้านข้าง
# (ได้จากการกวาดพารามิเตอร์ ดู margin/แรงบิดในตารางท้ายไฟล์)
MOVES: Dict[str, Dict[str, float]] = {
    "forward":  dict(heading_deg=0.0),
    "back":     dict(heading_deg=180.0, stride=20.0, sway_gain=1.0, sway_gain_x=0.7),
    "left":     dict(heading_deg=90.0, stride=20.0, sway_gain=1.0, sway_gain_x=0.8),
    "right":    dict(heading_deg=-90.0, stride=20.0, sway_gain=1.0, sway_gain_x=0.8),
    "turn_left":  dict(stride=0.0, yaw_deg=20.0, sway_gain=0.8, sway_gain_x=0.8),
    "turn_right": dict(stride=0.0, yaw_deg=-20.0, sway_gain=0.8, sway_gain_x=0.8),
}


def gait_for(move: str, **override) -> Gait:
    """คืน Gait ที่จูนไว้แล้วสำหรับทิศที่ต้องการ: forward/back/left/right/turn_left/turn_right."""
    if move not in MOVES:
        raise KeyError(f"ไม่รู้จักท่า {move!r} มีให้เลือก: {sorted(MOVES)}")
    return Gait(**{**MOVES[move], **override})


def nominal_stance(p: Params, g: Gait) -> Dict[str, np.ndarray]:
    """ตำแหน่งเท้ากลางท่ายืน (ก่อนบวกการก้าว/แกว่ง) — ถ่างออกตาม splay."""
    stance = {}
    for leg in LEGS:
        sa, sb, _, _ = LEG_SIGNS[leg]
        hip = body_to_frame0(leg, p)[:3, 3]
        # ถ่างขา: หมุนรอบแกน abduction ทำให้เท้าออกข้างและยกสูงขึ้นเล็กน้อย
        gamma = np.radians(g.splay_deg)
        reach = g.stand_h - p.h
        stance[leg] = np.array([hip[0] + g.x_center,
                                hip[1] + sb * reach * np.tan(gamma),
                                hip[2] - g.stand_h])
    return stance


def _incenter(pts: np.ndarray) -> np.ndarray:
    """จุดในสุดของสามเหลี่ยม (incenter) = จุดที่ห่างจากขอบทุกด้านมากที่สุดเท่าที่เป็นไปได้."""
    a = np.linalg.norm(pts[1] - pts[2])
    b = np.linalg.norm(pts[2] - pts[0])
    c = np.linalg.norm(pts[0] - pts[1])
    return (a * pts[0] + b * pts[1] + c * pts[2]) / (a + b + c)


def sway_schedule(p: Params, g: Gait,
                  payloads: Sequence[Payload] = tuple(BODY_PAYLOADS)) -> np.ndarray:
    """เป้าหมายการเลื่อนลำตัว (x,y) ของแต่ละจังหวะยกขา — คืน array (4,2) เรียงตาม SWING_ORDER.

    หลักการ: ก่อนยกขาไหน ต้องย้ายลำตัวให้เงา CoG ไปอยู่ 'incenter' ของสามเหลี่ยมสามขาที่เหลือ
    จุดนั้นคือจุดที่ห่างขอบมากที่สุด -> margin สูงสุด และชดเชยน้ำหนักจอที่เยื้องข้างไปในตัว
    """
    base = nominal_stance(p, g)
    cog0, _ = center_of_gravity(robot_ik(base, p, g.knee), p, payloads)
    targets = np.zeros((len(SWING_ORDER), 2))
    for k, swing_leg in enumerate(SWING_ORDER):
        pts = np.array([base[leg][:2] for leg in LEGS if leg != swing_leg])
        want = _incenter(pts) - cog0[:2]
        targets[k] = (g.sway_gain_x * want[0], g.sway_gain * want[1])
    return targets


def body_offset(t: float, g: Gait, targets: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """การเลื่อนลำตัวเทียบพื้น ณ เวลา t: ค้างที่เป้าหมายตอนขาลอย แล้วค่อย ๆ ย้ายตอนสี่ขาแตะพื้น."""
    n = len(SWING_ORDER)
    phase = (t / g.cycle_s) % 1.0
    k = min(int(phase * n), n - 1)
    local = phase * n - k                       # 0..1 ภายในหนึ่งจังหวะ
    swing_frac = (1.0 - g.duty) * n             # ส่วนของจังหวะที่ขาลอย (duty>0.75 -> <1)

    if local < swing_frac:
        shift = targets[k]                      # ขาลอยอยู่: ห้ามขยับลำตัว ค้างไว้ที่จุดปลอดภัย
    else:
        u = (local - swing_frac) / (1.0 - swing_frac)
        s = u * u * (3.0 - 2.0 * u)             # smoothstep: ออกตัวและหยุดแบบไม่กระชาก
        shift = (1.0 - s) * targets[k] + s * targets[(k + 1) % n]
    return np.array([shift[0] + bias[0], shift[1] + bias[1], 0.0])


def swing_window(leg: str, g: Gait) -> Tuple[float, float]:
    """ช่วงเฟส [เริ่ม, จบ) ที่ขานี้ลอย."""
    k = SWING_ORDER.index(leg)
    start = k / len(SWING_ORDER)
    return start, start + (1.0 - g.duty)


def foot_targets(t: float, p: Params, g: Gait, targets: np.ndarray,
                 bias: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    """ตำแหน่งเป้าหมายของเท้าทั้ง 4 ณ เวลา t (เทียบเฟรมลำตัว) + สถานะแตะพื้น."""
    base = nominal_stance(p, g)
    shift = body_offset(t, g, targets, bias)
    phase = (t / g.cycle_s) % 1.0

    # เวกเตอร์ก้าว: ชี้ไปทางที่อยากเดิน ส่วน yaw คือการหมุนรอบแกนตั้งของลำตัว
    head = np.radians(g.heading_deg)
    stride_vec = g.stride * np.array([np.cos(head), np.sin(head)])
    psi = np.radians(g.yaw_deg)

    feet, contact = {}, {}
    for leg in LEGS:
        s0, s1 = swing_window(leg, g)
        in_swing = s0 <= phase < s1
        pos = base[leg].copy()

        if in_swing:
            u = (phase - s0) / (s1 - s0)                       # 0..1 ตลอดช่วงลอย
            # cycloid: ความเร็วเป็นศูนย์ทั้งตอนยกและตอนวาง -> เซอร์โว analog ตามทัน ไม่กระแทก
            prof = u - np.sin(2.0 * np.pi * u) / (2.0 * np.pi)
            disp = -stride_vec / 2.0 + stride_vec * prof
            ang = -psi / 2.0 + psi * prof
            pos[2] += g.clearance * (1.0 - np.cos(2.0 * np.pi * u)) / 2.0
        else:
            # ช่วงแตะพื้น: เท้าเคลื่อนสวนทางการเดินด้วยความเร็วคงที่ = ลำตัวเคลื่อนไปข้างหน้า
            v = (phase - s1) % 1.0 / g.duty                    # 0..1 ตลอดช่วงแตะพื้น
            disp = stride_vec / 2.0 - stride_vec * v
            ang = psi / 2.0 - psi * v

        ca, sa = np.cos(ang), np.sin(ang)                      # เลี้ยว = หมุนจุดวางเท้ารอบแกนตั้ง
        pos[:2] = (ca * pos[0] - sa * pos[1], sa * pos[0] + ca * pos[1]) + disp

        feet[leg] = pos - shift          # ลำตัวขยับ +y -> เท้าในเฟรมลำตัวเลื่อน -y
        contact[leg] = not in_swing
    return feet, contact


# ---------------------------------------------------------------------------
# 5) เสถียรภาพเชิงสถิต
# ---------------------------------------------------------------------------


def stability_margin(cog: np.ndarray, feet: Dict[str, np.ndarray],
                     contact: Dict[str, bool]) -> float:
    """ระยะจากเงา CoG ถึงขอบสามเหลี่ยมรับน้ำหนักที่ใกล้ที่สุด (mm) ติดลบ = ล้ม."""
    pts = np.array([feet[leg][:2] for leg in LEGS if contact[leg]])
    if len(pts) < 3:
        return float("-inf")
    center = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    poly = pts[order]

    g2 = cog[:2]
    margin = float("inf")
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        edge = b - a
        n = np.array([edge[1], -edge[0]])          # normal ชี้ออกนอกรูป (วนทวนเข็ม)
        n /= np.linalg.norm(n)
        margin = min(margin, float(-np.dot(g2 - a, n)))
    return margin


# ---------------------------------------------------------------------------
# 6) สร้าง trajectory + ตรวจทุกข้อจำกัด
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    t: np.ndarray
    q: np.ndarray                      # (N,12) เรเดียน
    feet: np.ndarray                   # (N,4,3) mm
    contact: np.ndarray                # (N,4) bool
    cog: np.ndarray                    # (N,3) mm
    margin: np.ndarray                 # (N,) mm
    mass_g: float
    bias: float
    reachable: bool = True
    notes: List[str] = field(default_factory=list)
    # transform ลำตัว->พื้น (N,4,4) ใช้เฉพาะท่าที่ลำตัวเอียง เช่น นั่ง/ยกขาหน้า
    body_T: np.ndarray | None = None


def build_trajectory(p: Params = PARAMS, g: Gait = Gait(),
                     payloads: Sequence[Payload] = tuple(BODY_PAYLOADS)) -> Trajectory:
    """สร้างท่าเดิน 1 รอบเต็ม พร้อมคำนวณ CoG และ margin ทุก timestep."""
    # การชดเชยน้ำหนักจออยู่ใน sway_schedule แล้ว (incenter คิดจาก CoG จริง) bias_y ไว้ trim มือ
    targets = sway_schedule(p, g, payloads)
    bias = np.array([0.0, 0.0 if g.bias_y is None else float(g.bias_y)])

    n = int(round(g.cycle_s * g.hz))
    t = np.arange(n + 1) / g.hz
    q = np.zeros((n + 1, 12))
    feet = np.zeros((n + 1, 4, 3))
    contact = np.zeros((n + 1, 4), dtype=bool)
    cog = np.zeros((n + 1, 3))
    margin = np.zeros(n + 1)
    reachable = True

    for k, tk in enumerate(t):
        tgt, con = foot_targets(tk, p, g, targets, bias)
        q[k] = robot_ik(tgt, p, g.knee)
        if not np.all(np.isfinite(q[k])):
            reachable = False
            q[k] = np.nan_to_num(q[k])
        feet[k] = np.array([tgt[leg] for leg in LEGS])
        contact[k] = [con[leg] for leg in LEGS]
        cog[k], mass = center_of_gravity(q[k], p, payloads)
        margin[k] = stability_margin(cog[k], tgt, con)

    # ระยะเลื่อนลำตัวสูงสุดจากแนวกลาง ใช้รายงานว่าโครงต้องเผื่อ workspace เท่าไร
    max_shift = float(np.max(np.abs(targets)))
    return Trajectory(t, q, feet, contact, cog, margin, mass, max_shift, reachable)


def analyze(traj: Trajectory, p: Params, g: Gait, servo: Servo = Servo()) -> Dict[str, float]:
    """เช็คทุกข้อจำกัดฮาร์ดแวร์ แล้วคืนตัวเลขสรุป (พร้อมเก็บคำเตือนใส่ traj.notes)."""
    deg = np.degrees(traj.q)
    rate = np.abs(np.diff(deg, axis=0)) * g.hz          # °/s ต่อข้อต่อ
    peak_rate = float(rate.max())
    span = float(deg.max() - deg.min())

    # แรงบิด: กระจายน้ำหนักเท่า ๆ กันในขาที่แตะพื้น แล้วคิดโมเมนต์รอบแต่ละแกน
    w_n = traj.mass_g * 1e-3 * 9.81
    tau1 = tau2 = tau3 = 0.0
    for k in range(len(traj.t)):
        n_stance = max(1, int(traj.contact[k].sum()))
        f = w_n / n_stance
        for i, leg in enumerate(LEGS):
            if not traj.contact[k, i]:
                continue
            hip = body_to_frame0(leg, p)[:3, 3]
            chain = leg_fk(leg, traj.q[k, 3 * i:3 * i + 3], p)
            o1, o2 = chain[1][:3, 3], chain[2][:3, 3]
            foot = traj.feet[k, i]
            tau1 = max(tau1, f * abs(foot[1] - hip[1]) * 1e-3)     # abduction: โมเมนต์ด้านข้าง
            tau2 = max(tau2, f * abs(foot[0] - o1[0]) * 1e-3)      # hip pitch: โมเมนต์หน้า-หลัง
            tau3 = max(tau3, f * abs(foot[0] - o2[0]) * 1e-3)      # knee
    to_kgcm = 10.197
    budget = servo.stall_kgcm * servo.usable_torque

    res = {
        "peak_rate_dps": peak_rate,
        "rate_limit_dps": servo.max_rate_dps,
        "cmd_per_step_deg": float(rate.max() / g.hz),
        "deadband_deg": servo.deadband_deg,
        "angle_span_deg": span,
        "min_margin_mm": float(traj.margin.min()),
        "mass_g": traj.mass_g,
        "shift_mm": traj.bias,
        "tau_abduction_kgcm": tau1 * to_kgcm,
        "tau_hip_kgcm": tau2 * to_kgcm,
        "tau_knee_kgcm": tau3 * to_kgcm,
        "torque_budget_kgcm": budget,
        "stall_needed_kgcm": max(tau1, tau2, tau3) * to_kgcm / servo.usable_torque,
        "stall_used_pct": 100.0 * max(tau1, tau2, tau3) * to_kgcm / servo.stall_kgcm,
        "base_width_mm": float(traj.feet[0, :, 1].max() - traj.feet[0, :, 1].min()),
        "speed_mm_s": g.stride / g.cycle_s,
    }

    notes = traj.notes
    if not traj.reachable:
        notes.append("ERROR: มีจังหวะที่ IK เอื้อมไม่ถึง — ลด stride/ลด stand_h/ลด splay")
    if res["min_margin_mm"] <= 0:
        notes.append(f"ERROR: margin ต่ำสุด {res['min_margin_mm']:.1f} mm -> ล้มแน่นอน")
    elif res["min_margin_mm"] < 8.0:
        notes.append(f"เตือน: margin ต่ำสุด {res['min_margin_mm']:.1f} mm บางเกินไป "
                     "(ควร >8 mm) เพิ่ม splay_deg หรือ sway_y")
    if peak_rate > servo.max_rate_dps:
        notes.append(f"เตือน: ต้องการ {peak_rate:.0f}°/s > เซอร์โวทำได้ {servo.max_rate_dps:.0f}°/s "
                     "-> เพิ่ม cycle_s")
    if res["cmd_per_step_deg"] < servo.deadband_deg * 0.5:
        notes.append(f"หมายเหตุ: ก้าวคำสั่งเล็กสุด {res['cmd_per_step_deg']:.2f}° เทียบ deadband "
                     f"{servo.deadband_deg}° -> การเคลื่อนที่จะเป็นขั้นบันได ไม่ลื่น")
    for name, key in (("abduction", "tau_abduction_kgcm"), ("hip pitch", "tau_hip_kgcm"),
                      ("knee", "tau_knee_kgcm")):
        if res[key] > budget:
            notes.append(f"เตือน: แรงบิด {name} {res[key]:.2f} kg·cm > งบ {budget:.2f} kg·cm "
                         "-> เซอร์โวจะสั่น/ร้อน ลดความสูงหรือลด splay")
    return res


# ---------------------------------------------------------------------------
# 7) ตรวจความถูกต้อง
# ---------------------------------------------------------------------------


def verify(p: Params = PARAMS, tol: float = 1e-6, verbose: bool = True) -> None:
    """ตรวจว่า IK เป็นอินเวอร์สของ FK จริง และ margin คำนวณถูก."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for leg in LEGS:
        for _ in range(300):
            q = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-1.0, 1.0), rng.uniform(0.1, 2.0)])
            foot = leg_fk(leg, q, p)[-1][:3, 3]
            knee = +1 if q[2] >= 0 else -1
            back = leg_ik(leg, foot, p, knee)
            assert np.all(np.isfinite(back)), f"{leg}: IK แก้ไม่ได้ทั้งที่มาจาก FK"
            foot2 = leg_fk(leg, back, p)[-1][:3, 3]
            worst = max(worst, float(np.max(np.abs(foot - foot2))))
    assert worst < tol, f"IK/FK ไม่ตรงกัน (worst = {worst:g})"

    # margin: CoG กลางสี่เหลี่ยมจัตุรัสด้าน 100 -> margin ต้องเท่ากับ 50
    feet = {"FL": np.array([50.0, 50.0, -95.0]), "FR": np.array([50.0, -50.0, -95.0]),
            "RL": np.array([-50.0, 50.0, -95.0]), "RR": np.array([-50.0, -50.0, -95.0])}
    con = {leg: True for leg in LEGS}
    assert abs(stability_margin(np.zeros(3), feet, con) - 50.0) < tol, "margin ผิด (กรณีกลางรูป)"
    # ยก FL -> เหลือสามเหลี่ยม FR/RL/RR ด้านตรงข้ามมุมฉากคือเส้น x+y=0 ผ่านจุดกำเนิดพอดี
    con["FL"] = False
    assert abs(stability_margin(np.zeros(3), feet, con)) < tol, "CoG บนขอบพอดีต้องได้ 0"
    m3 = stability_margin(np.array([-20.0, -20.0, 0.0]), feet, con)
    assert 0.0 < m3 < 50.0, "margin สามเหลี่ยมควรบวกแต่น้อยกว่ากรณีสี่ขา"
    assert stability_margin(np.array([200.0, 0.0, 0.0]), feet, con) < 0.0, "CoG นอกรูปต้องติดลบ"

    if verbose:
        print(f"PASS: IK<->FK ตรงกันทุกขา (คลาดเคลื่อนสูงสุด {worst:.2e} mm) + margin ถูกต้อง")


# ---------------------------------------------------------------------------
# 8) รายงาน + รูป
# ---------------------------------------------------------------------------

TEAL, ROW_A, ROW_B, INK = "#2196a8", "#dceef4", "#eef7fa", "#5a5a5a"
LEG_COLORS = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#c47f17"}


def print_report(traj: Trajectory, res: Dict[str, float], g: Gait, servo: Servo) -> None:
    """สรุปผลเป็นข้อความ."""
    line = "=" * 72
    print(line)
    print("WALKING GAIT — crawl gait 1 รอบ")
    print(line)
    print(f"  มวลรวม                {res['mass_g']:8.0f} g")
    print(f"  ความเร็วเดิน           {res['speed_mm_s']:8.1f} mm/s  "
          f"({g.stride:.0f} mm ต่อ {g.cycle_s:.1f} s)")
    print(f"  ความกว้างฐานเท้า        {res['base_width_mm']:8.1f} mm  (splay {g.splay_deg:.0f}°)")
    print(f"  ระยะย้ายลำตัวสูงสุด      {res['shift_mm']:8.1f} mm")
    print(f"  margin ต่ำสุด          {res['min_margin_mm']:8.1f} mm")
    print()
    print(f"  ความเร็วข้อต่อสูงสุด     {res['peak_rate_dps']:8.0f} °/s  "
          f"(เซอร์โวไหว {servo.max_rate_dps:.0f})")
    print(f"  ก้าวคำสั่งต่อ 1 พัลส์    {res['cmd_per_step_deg']:8.2f} °   "
          f"(deadband {servo.deadband_deg})")
    print(f"  แรงบิด abduction      {res['tau_abduction_kgcm']:8.2f} kg·cm")
    print(f"  แรงบิด hip pitch      {res['tau_hip_kgcm']:8.2f} kg·cm")
    print(f"  แรงบิด knee           {res['tau_knee_kgcm']:8.2f} kg·cm")
    print(f"  งบแรงบิดที่ยอมให้       {res['torque_budget_kgcm']:8.2f} kg·cm  "
          f"({servo.stall_kgcm:.1f} x {servo.usable_torque * 100:.0f}%)")
    print(f"  เซอร์โวที่ควรใช้         {res['stall_needed_kgcm']:8.1f} kg·cm  "
          f"(ตัวปัจจุบันจะโดนรีดถึง {res['stall_used_pct']:.0f}% ของ stall)")
    print(line)
    for note in traj.notes:
        print("  " + note)
    if not traj.notes:
        print("  ผ่านทุกข้อจำกัด")
    print(line)


def plot(traj: Trajectory, res: Dict[str, float], p: Params, g: Gait,
         save: str | None = None) -> None:
    """รูปสรุป 4 ช่อง: มองจากบน / margin / มุมข้อต่อ / ตารางสรุป."""
    fig = plt.figure(figsize=(14.0, 8.2), facecolor="white")
    fig.text(0.035, 0.955, "Walking Gait — Crawl, Static Stability",
             fontsize=22, color="#6b6b6b", fontweight="bold", va="center")

    # --- (1) มองจากบน: รอยเท้า + เงา CoG + สามเหลี่ยมรับน้ำหนัก ---
    ax = fig.add_axes([0.05, 0.50, 0.40, 0.38])
    for i, leg in enumerate(LEGS):
        ax.plot(traj.feet[:, i, 0], traj.feet[:, i, 1], "-", color=LEG_COLORS[leg],
                linewidth=1.4, label=leg)
        st = traj.contact[:, i]
        ax.plot(traj.feet[st, i, 0], traj.feet[st, i, 1], ".", color=LEG_COLORS[leg], markersize=2)
    k = int(np.argmin(traj.margin))
    pts = np.array([traj.feet[k, i, :2] for i in range(4) if traj.contact[k, i]])
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    poly = np.vstack([pts[order], pts[order][:1]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color="#888888", linewidth=1.2)
    ax.fill(poly[:, 0], poly[:, 1], color="#888888", alpha=0.10)
    ax.plot(traj.cog[:, 0], traj.cog[:, 1], "-", color="#111111", linewidth=2.0, label="CoG")
    ax.plot(traj.cog[k, 0], traj.cog[k, 1], "o", color="#111111", markersize=6)
    lbl = f"margin ต่ำสุด {traj.margin[k]:.0f} mm"
    ax.annotate(lbl, traj.cog[k, :2], textcoords="offset points", xytext=(8, -12),
                fontsize=8.5, color="#111111", **_fk(lbl))
    t1 = "มองจากบน: รอยเท้า + เส้นทาง CoG" if _THAI else "top view: feet + CoG"
    xl = "x (mm) ไปข้างหน้า" if _THAI else "x (mm) forward"
    yl = "y (mm) ไปทางซ้าย" if _THAI else "y (mm) left"
    ax.set_title(t1, fontsize=10, color=INK, **_fk(t1))
    ax.set_xlabel(xl, fontsize=8.5, **_fk(xl))
    ax.set_ylabel(yl, fontsize=8.5, **_fk(yl))
    ax.axhline(0, color="#dddddd", linewidth=0.8)
    ax.axvline(0, color="#dddddd", linewidth=0.8)
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", frameon=False)
    ax.tick_params(labelsize=7.5)

    # --- (2) margin เทียบเวลา + แถบขาที่ลอย ---
    ax2 = fig.add_axes([0.05, 0.10, 0.40, 0.29])
    ax2.axhspan(-100, 0, color="#d62728", alpha=0.10)
    ax2.axhspan(0, 8, color="#c47f17", alpha=0.10)
    ax2.plot(traj.t, traj.margin, "-", color=TEAL, linewidth=2.0)
    ax2.axhline(0, color="#d62728", linewidth=1.0)
    ax2.axhline(8, color="#c47f17", linewidth=0.8, linestyle="--")
    for i, leg in enumerate(LEGS):
        sw = ~traj.contact[:, i]
        ax2.fill_between(traj.t, -6 - 4 * i, -2 - 4 * i, where=sw,
                         color=LEG_COLORS[leg], alpha=0.75, linewidth=0)
        ax2.text(traj.t[-1] * 1.01, -4.5 - 4 * i, leg, fontsize=7,
                 color=LEG_COLORS[leg], va="center")
    ax2.set_ylim(-20, max(30.0, traj.margin.max() * 1.15))
    ax2.set_xlim(0, traj.t[-1])
    ax2.set_xlabel("t (s)", fontsize=8.5)
    ax2.set_ylabel("stability margin (mm)", fontsize=8.5)
    ax2.tick_params(labelsize=7.5)
    ax2.grid(alpha=0.25)

    # --- (3) มุมข้อต่อ ---
    ax3 = fig.add_axes([0.55, 0.50, 0.41, 0.38])
    styles = ("-", "--", ":")
    for i, leg in enumerate(LEGS):
        for j in range(3):
            ax3.plot(traj.t, np.degrees(traj.q[:, 3 * i + j]), styles[j],
                     color=LEG_COLORS[leg], linewidth=1.3,
                     label=f"{leg} θ{j + 1}" if i == 0 else None)
    ax3.set_xlim(0, traj.t[-1])
    ax3.set_xlabel("t (s)", fontsize=8.5)
    ax3.set_ylabel("joint angle (deg)", fontsize=8.5)
    t3 = ("คำสั่งเซอร์โว 12 ตัว (ทึบ/ประ/จุด = ข้อที่ 1/2/3)" if _THAI
          else "12 servo commands (solid/dash/dot = joint 1/2/3)")
    ax3.set_title(t3, fontsize=10, color=INK, **_fk(t3))
    ax3.legend(fontsize=7, ncol=3, frameon=False, loc="upper right")
    ax3.tick_params(labelsize=7.5)
    ax3.grid(alpha=0.25)

    # --- (4) ตารางสรุป ---
    ax4 = fig.add_axes([0.55, 0.08, 0.41, 0.34])
    ax4.set_axis_off()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    rows = [
        ("มวลรวม", f"{res['mass_g']:.0f} g", ""),
        ("ความเร็วเดิน", f"{res['speed_mm_s']:.1f} mm/s", f"{g.stride:.0f} mm / {g.cycle_s:.1f} s"),
        ("ฐานเท้ากว้าง", f"{res['base_width_mm']:.0f} mm", f"splay {g.splay_deg:.0f}°"),
        ("ระยะย้ายลำตัวสูงสุด", f"{res['shift_mm']:.1f} mm", "auto"),
        ("margin ต่ำสุด", f"{res['min_margin_mm']:.1f} mm", "ควร > 8"),
        ("ความเร็วข้อต่อสูงสุด", f"{res['peak_rate_dps']:.0f} °/s", f"servo {res['rate_limit_dps']:.0f}"),
        ("ก้าว/พัลส์", f"{res['cmd_per_step_deg']:.2f} °", f"deadband {res['deadband_deg']}"),
        ("τ abduction", f"{res['tau_abduction_kgcm']:.2f} kg·cm", f"งบ {res['torque_budget_kgcm']:.2f}"),
        ("τ hip pitch", f"{res['tau_hip_kgcm']:.2f} kg·cm", ""),
        ("τ knee", f"{res['tau_knee_kgcm']:.2f} kg·cm", ""),
        ("เซอร์โวที่ควรใช้", f"{res['stall_needed_kgcm']:.1f} kg·cm",
         f"ตัวนี้โดน {res['stall_used_pct']:.0f}% ของ stall"),
    ]
    hr = 0.078
    ax4.add_patch(plt.Rectangle((0, 1 - hr), 1, hr, facecolor=TEAL, edgecolor="none"))
    t4 = "สรุปผลตรวจข้อจำกัดฮาร์ดแวร์" if _THAI else "hardware check"
    ax4.text(0.5, 1 - hr / 2, t4, color="white", fontsize=10, fontweight="bold",
             ha="center", va="center", **_fk(t4))
    for r, (name, val, note) in enumerate(rows):
        y = 1 - (r + 2) * hr
        ax4.add_patch(plt.Rectangle((0, y), 1, hr,
                                    facecolor=ROW_A if r % 2 == 0 else ROW_B, edgecolor="none"))
        ax4.text(0.03, y + hr / 2, name, fontsize=9, color="#20404a", va="center", **_fk(name))
        ax4.text(0.62, y + hr / 2, val, fontsize=9, color="#20404a", va="center", ha="right")
        ax4.text(0.66, y + hr / 2, note, fontsize=8, color="#6b8b95", va="center", **_fk(note))

    if save:
        fig.savefig(save, dpi=130, facecolor="white")
        print(f"saved figure -> {save}")
    else:
        plt.show()


def _pick_thai_font() -> str | None:
    from matplotlib.font_manager import get_font_names

    have = set(get_font_names())
    return next((f for f in ("Garuda", "Loma", "Norasi", "Kinnari") if f in have), None)


_THAI = _pick_thai_font()


def _fk(text: str) -> dict:
    """เลือกฟอนต์ตามเนื้อความ: ไทยใช้ Garuda, ที่เหลือใช้ฟอนต์เดิม (Garuda ไม่มี θ τ → ←)."""
    has_thai = any("฀" <= ch <= "๿" for ch in text)
    return {"family": _THAI} if (has_thai and _THAI) else {}


def export_csv(traj: Trajectory, path: str) -> None:
    """บันทึกมุมเป็นองศา พร้อมคอลัมน์เวลา เอาไปป้อน PCA9685 บน Pi ได้ตรง ๆ."""
    header = "t_s," + ",".join(f"{leg}_th{j + 1}_deg" for leg in LEGS for j in range(3))
    data = np.column_stack([traj.t, np.degrees(traj.q)])
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.4f")
    print(f"saved csv -> {path}  ({data.shape[0]} แถว @ {1 / (traj.t[1] - traj.t[0]):.0f} Hz)")


# ---------------------------------------------------------------------------
# 9) main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--save", metavar="PNG")
    ap.add_argument("--csv", metavar="CSV")
    ap.add_argument("--cycle", type=float, default=Gait.cycle_s, help="เวลา 1 รอบ (วินาที)")
    ap.add_argument("--stride", type=float, default=Gait.stride, help="ระยะก้าว (mm)")
    ap.add_argument("--splay", type=float, default=Gait.splay_deg, help="องศาถ่างขา")
    ap.add_argument("--height", type=float, default=Gait.stand_h, help="ความสูงลำตัว (mm)")
    args = ap.parse_args(argv)

    verify(PARAMS)
    if args.verify_only:
        return 0

    g = Gait(cycle_s=args.cycle, stride=args.stride, splay_deg=args.splay, stand_h=args.height)
    servo = Servo()
    traj = build_trajectory(PARAMS, g)
    res = analyze(traj, PARAMS, g, servo)
    print_report(traj, res, g, servo)

    if args.csv:
        export_csv(traj, args.csv)
    plot(traj, res, PARAMS, g, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
