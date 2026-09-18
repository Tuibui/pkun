#!/usr/bin/env python3
"""
servo_map.py — จุดนัดพบระหว่าง "มุมในโมเดล" กับ "มุมบนเซอร์โวจริง 0-180"

ปัญหาที่ไฟล์นี้แก้: โมเดล (leg_kinematics.py) คิดมุมแบบ signed รอบศูนย์
  θ1 abduction -55..+55, θ2 hip pitch -90..+90, θ3 knee 0..150
แต่เซอร์โว 180° คิดเป็น 0..180 โดยกลางอยู่ที่ 90 คนละระบบกันคนละที่มา
ถ้าไม่ตกลงกันให้ชัดก่อน เราจะพูดคนละภาษาตลอดเวลาที่คุยเรื่อง "องศา"

สมการเดียวที่ใช้ทั้งไฟล์:

    servo_deg = home_deg + direction * (model_deg + trim_deg)
    model_deg = direction * (servo_deg - home_deg) - trim_deg

  home_deg   มุมบนหน้าปัดเซอร์โวที่ตรงกับ "โมเดล = 0" (ท่า zero pose)
  direction  +1 = โมเดลบวก -> เลขเซอร์โวเพิ่ม, -1 = สวนทาง (ขาขวาติดกลับข้าง)
  trim_deg   ชดเชยร่องเฟืองฮอร์นที่ใส่ไม่ลงตัว (ค่าเดียวกับ set_trim ใน ROS)

ทำไม home ของเข่าไม่ใช่ 90:
  เข่าพับทางเดียว 0..150° ถ้าวางศูนย์ไว้ที่ 90 จะต้องใช้ถึง 240° เกินตัวเซอร์โว
  ต้องใส่ฮอร์นให้ช่วง 0..150 ไปนอนอยู่กลางราง คือ 15..165 (เหลือกันชนปลายละ 15°)
  ขาซ้าย (dir +1) -> home 15 | ขาขวา (dir -1) -> home 165 เพราะนับกลับทาง
  ถ้าใช้ home เดียวกันทั้งสองฝั่ง ขาขวาจะวิ่งไป -135° ซึ่งไม่มีอยู่จริง
  -> home คำนวณจาก home_for() ห้ามฮาร์ดโค้ด

รัน:  python3 servo_map.py --table              ตารางสัญลักษณ์ + ช่วงมุมทั้ง 12 ข้อ
      python3 servo_map.py --check              ตรวจว่าทุกลิมิตตกอยู่ใน 0..180 จริง
      python3 servo_map.py --diagram ref.png    วาดแผนภาพอ้างอิง (0 อยู่ไหน 180 อยู่ไหน)
      python3 servo_map.py --to-servo FL 0 0 0  แปลงมุมโมเดล -> มุมเซอร์โว
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from leg_kinematics import LEGS, PARAMS, Params, leg_polyline

# ---------------------------------------------------------------------------
# 1) สเปกพัลส์ — ต้องตรงกับ servos.yaml และ pca9685.cpp เป๊ะ ๆ
# ---------------------------------------------------------------------------

PULSE_MIN_US = 500.0    # เซอร์โวที่ 0°
PULSE_MAX_US = 2500.0   # เซอร์โวที่ 180°
SERVO_SPAN_DEG = 180.0
US_PER_DEG = (PULSE_MAX_US - PULSE_MIN_US) / SERVO_SPAN_DEG   # = 11.111 ตรงกับ servos.yaml
CENTER_US = (PULSE_MIN_US + PULSE_MAX_US) / 2.0               # = 1500 ที่เซอร์โว 90°

PWM_HZ = 50.0           # analog servo รับได้แค่นี้
EDGE_GUARD_DEG = 5.0    # กันชนปลายราง: ไม่สั่งเข้าใกล้ 0/180 กว่านี้

# ---------------------------------------------------------------------------
# 2) นิยามข้อต่อ — สัญลักษณ์ ทิศบวก ลิมิต และตำแหน่งบนบอร์ด
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointMap:
    """ข้อต่อหนึ่งข้อ: ผูกมุมโมเดลเข้ากับช่องเซอร์โวหนึ่งช่อง."""

    name: str            # ชื่อเดียวกับ JointCommand เช่น "fl_knee"
    leg: str             # FL / FR / RL / RR
    symbol: str          # θ1 / θ2 / θ3
    joint: str           # abduction / hip_pitch / knee
    home_deg: float      # เซอร์โวกี่องศา เมื่อโมเดล = 0
    direction: int       # +1 หรือ -1
    model_min: float     # ลิมิตโมเดล (องศา) — มาจาก JointLimits ใน sleep_wake.py
    model_max: float
    board: int           # 0 = 0x40, 1 = 0x41
    channel: int         # 0..15
    trim_deg: float = 0.0

    # -- แปลงไป-กลับ ------------------------------------------------------

    def to_servo(self, model_deg: float) -> float:
        """มุมโมเดล (องศา) -> มุมบนหน้าปัดเซอร์โว 0..180."""
        return self.home_deg + self.direction * (model_deg + self.trim_deg)

    def to_model(self, servo_deg: float) -> float:
        """มุมบนหน้าปัดเซอร์โว 0..180 -> มุมโมเดล (องศา)."""
        return self.direction * (servo_deg - self.home_deg) - self.trim_deg

    def to_pulse_us(self, servo_deg: float) -> float:
        """มุมเซอร์โว -> ความกว้างพัลส์ (ไมโครวินาที)."""
        return PULSE_MIN_US + servo_deg * US_PER_DEG

    # -- ขอบเขต -----------------------------------------------------------

    def servo_range(self) -> Tuple[float, float]:
        """ช่วงมุมเซอร์โวที่ลิมิตโมเดลกินไปถึง (เรียงน้อย->มากแล้ว)."""
        lo, hi = self.to_servo(self.model_min), self.to_servo(self.model_max)
        return (lo, hi) if lo <= hi else (hi, lo)

    def fits(self) -> bool:
        """ลิมิตโมเดลตกอยู่ในราง 0..180 ของเซอร์โวจริงไหม (ข้อบังคับ)."""
        lo, hi = self.servo_range()
        return lo >= -1e-9 and hi <= SERVO_SPAN_DEG + 1e-9

    def guard_deg(self) -> float:
        """เหลือกันชนถึงปลายรางกี่องศา (น้อยสุดของสองปลาย) — ติดลบ = ล้นราง."""
        lo, hi = self.servo_range()
        return min(lo, SERVO_SPAN_DEG - hi)

    def clamp_model(self, model_deg: float) -> float:
        """บีบมุมโมเดลให้อยู่ในลิมิตข้อต่อ."""
        return float(np.clip(model_deg, self.model_min, self.model_max))


# ทิศบวกของแต่ละขา: ขาขวาติดเซอร์โวกลับข้าง ตรงกับ invert ใน servos.yaml
LEG_DIRECTION: Dict[str, int] = {"FL": +1, "FR": -1, "RL": +1, "RR": -1}

# นิยามกลางของ 3 ข้อต่อ — ใช้ร่วมกันทั้ง 4 ขา ต่างกันแค่ direction กับ channel
#   (symbol, joint, model_min, model_max)   ส่วน home_deg คำนวณจาก home_for()
JOINT_SPEC: Tuple[Tuple[str, str, float, float], ...] = (
    ("θ1", "abduction", -55.0,  55.0),   # สมมาตรรอบศูนย์ -> home 90 ทั้งสองฝั่ง
    ("θ2", "hip_pitch", -90.0,  90.0),   # สมมาตร แต่กินเต็มราง 0..180 พอดี ไม่เหลือกันชน
    ("θ3", "knee",        0.0, 150.0),   # พับทางเดียว -> home เยื้องไปคนละปลายตามฝั่งขา
)


def home_for(model_min: float, model_max: float, direction: int) -> float:
    """มุมเซอร์โวที่ตรงกับโมเดล 0 — วางให้ช่วงที่ใช้จริงอยู่กลางราง 0..180 พอดี.

    ต้องคิดจาก direction ด้วย ไม่งั้นข้อต่อที่ช่วงไม่สมมาตร (เข่า 0..150) จะพัง:
    ฝั่งซ้าย dir=+1 ได้ home 15 (ใช้ 15..165) ส่วนฝั่งขวา dir=-1 ต้องเป็น 165
    ถึงจะได้ 165..15 ซึ่งเป็นช่วงเดียวกันแค่นับกลับทาง ถ้าใช้ 15 เหมือนกันทั้งสองฝั่ง
    ขาขวาจะวิ่งไป -135° ซึ่งไม่มีอยู่จริง
    """
    mid = (model_min + model_max) / 2.0
    return SERVO_SPAN_DEG / 2.0 - direction * mid

# ลำดับช่องบนบอร์ด 0 — ตรงกับ servos.yaml
_LEG_ORDER: Tuple[str, ...] = ("FL", "FR", "RL", "RR")


def _build_map() -> Dict[str, JointMap]:
    """สร้างตารางข้อต่อทั้ง 12 ข้อ เรียงช่องเหมือน servos.yaml."""
    out: Dict[str, JointMap] = {}
    for li, leg in enumerate(_LEG_ORDER):
        for ji, (symbol, joint, lo, hi) in enumerate(JOINT_SPEC):
            name = f"{leg.lower()}_{joint}"
            direction = LEG_DIRECTION[leg]
            out[name] = JointMap(
                name=name, leg=leg, symbol=symbol, joint=joint,
                home_deg=home_for(lo, hi, direction), direction=direction,
                model_min=lo, model_max=hi,
                board=0, channel=li * 3 + ji,
            )
    return out


SERVO_MAP: Dict[str, JointMap] = _build_map()

# ชื่อข้อต่อเรียงตรงกับเวกเตอร์มุม 12 ตัวที่โมเดลใช้ (LEGS x 3)
JOINT_ORDER: Tuple[str, ...] = tuple(
    f"{leg.lower()}_{joint}" for leg in LEGS for _, joint, _, _ in JOINT_SPEC
)


# ---------------------------------------------------------------------------
# 3) แปลงทั้งตัว
# ---------------------------------------------------------------------------


def model_to_servo(q_all_deg: Sequence[float]) -> np.ndarray:
    """มุมโมเดล 12 ตัว (องศา เรียงตาม LEGS) -> มุมเซอร์โว 12 ตัว 0..180."""
    q = np.asarray(q_all_deg, dtype=float).ravel()
    if q.size != len(JOINT_ORDER):
        raise ValueError(f"ต้องการ {len(JOINT_ORDER)} มุม แต่ได้ {q.size}")
    return np.array([SERVO_MAP[n].to_servo(v) for n, v in zip(JOINT_ORDER, q)])


def servo_to_model(servo_deg: Sequence[float]) -> np.ndarray:
    """มุมเซอร์โว 12 ตัว -> มุมโมเดล 12 ตัว (องศา)."""
    s = np.asarray(servo_deg, dtype=float).ravel()
    if s.size != len(JOINT_ORDER):
        raise ValueError(f"ต้องการ {len(JOINT_ORDER)} มุม แต่ได้ {s.size}")
    return np.array([SERVO_MAP[n].to_model(v) for n, v in zip(JOINT_ORDER, s)])


# ---------------------------------------------------------------------------
# 4) ตารางสัญลักษณ์
# ---------------------------------------------------------------------------

SYMBOLS: Tuple[Tuple[str, str, str], ...] = (
    ("{B}", "body frame",  "จุดกำเนิดกลางลำตัว: x หน้า, y ซ้าย, z ขึ้น (mm)"),
    ("θ1",  "abduction",   "หมุนรอบแกน x | + = เท้าไปทาง +y (ซ้ายของหุ่น) ทุกขา"),
    ("",    "",            "  -> ขาซ้าย + = ออกนอกตัว | ขาขวา + = เข้าใต้ท้อง (ไม่ mirror!)"),
    ("θ2",  "hip pitch",   "หมุนรอบแกน y | + = เท้าถอยไปทางท้าย (-x) เหมือนกันทุกขา"),
    ("θ3",  "knee",        "พับทางเดียว   | + = พับเก็บ, 0 = เหยียดตรง เหมือนกันทุกขา"),
    ("a",   "77.27 mm",    "ครึ่งระยะสะโพก หน้า-หลัง"),
    ("b",   "29.00 mm",    "ครึ่งระยะสะโพก ซ้าย-ขวา"),
    ("h",   "23.38 mm",    "แกน abduction ลงมาถึงแกน hip pitch"),
    ("L1",  "56.39 mm",    "ต้นขา: แกน hip pitch -> แกนเข่า"),
    ("L2",  "56.58 mm",    "แข้ง: แกนเข่า -> ปลายเท้า"),
    ("D1",  "12.75 mm",    "offset ด้านข้าง ไปตำแหน่งเซอร์โว hip pitch จริง"),
    ("D2",  "12.75 mm",    "offset ที่เข่า เยื้องกลับเข้าหาลำตัว"),
)


def print_table() -> None:
    """พิมพ์ตารางสัญลักษณ์ + ตารางแปลงมุมทั้ง 12 ข้อ."""
    line = "=" * 86
    print(line)
    print("P-KUN — ตารางสัญลักษณ์อ้างอิง")
    print(line)
    for sym, short, desc in SYMBOLS:
        print(f"  {sym:<5} {short:<14} {desc}")

    print()
    print(line)
    print("ท่า ZERO POSE (โมเดล θ1=θ2=θ3=0) = ขาเหยียดตรงดิ่งลงพื้นทั้ง 4 ขา")
    print("ปลายเท้า FL อยู่ที่ (x=+77.27, y=+29.00, z=-136.35) mm เทียบเฟรมลำตัว")
    print(line)
    print("ข้อควรระวังที่สุด: θ1 ในโมเดล 'ไม่ mirror' ซ้าย-ขวา")
    print("  ท่ายืนถ่างขา 12° -> FL/RL ได้ θ1 = +9.50°  แต่ FR/RR ได้ -9.50°")
    print("  แต่บนหน้าปัดเซอร์โว ทั้งสี่ขาอ่านได้ 99.50° เท่ากันหมด เพราะ dir=-1 หักล้างไปแล้ว")
    print("  => ท่าที่สมมาตร เลขเซอร์โวของสี่ขาต้องเท่ากัน ใช้เป็นตัวเช็คตอนคาลิเบรตได้เลย")
    print(line)
    print()

    print(line)
    print("การแปลงมุม:  servo = home + dir x (model + trim)")
    print(line)
    hdr = (f"{'joint':<16}{'sym':<5}{'dir':>4}{'home':>6}"
           f"{'model min..max':>17}{'servo min..max':>17}{'pulse us':>16}{'guard':>7}")
    print(hdr)
    print("-" * 86)
    for name in JOINT_ORDER:
        j = SERVO_MAP[name]
        slo, shi = j.servo_range()
        print(f"{j.name:<16}{j.symbol:<5}{j.direction:>+4}{j.home_deg:>6.0f}"
              f"{f'{j.model_min:+.0f} .. {j.model_max:+.0f}':>17}"
              f"{f'{slo:.0f} .. {shi:.0f}':>17}"
              f"{f'{j.to_pulse_us(slo):.0f} .. {j.to_pulse_us(shi):.0f}':>16}"
              f"{j.guard_deg():>6.0f}°" + ("" if j.guard_deg() >= EDGE_GUARD_DEG else "  <<"))
    print(line)
    print(f"  พัลส์: {PULSE_MIN_US:.0f} us = เซอร์โว 0°, {CENTER_US:.0f} us = 90°, "
          f"{PULSE_MAX_US:.0f} us = 180°  ({US_PER_DEG:.3f} us/deg @ {PWM_HZ:.0f} Hz)")
    print(line)


# ---------------------------------------------------------------------------
# 5) ตรวจความถูกต้อง
# ---------------------------------------------------------------------------


def verify(verbose: bool = True) -> None:
    """ตรวจว่าการแปลงไป-กลับตรงกัน และทุกลิมิตอยู่ในราง 0..180 จริง."""
    worst = 0.0
    for name in JOINT_ORDER:
        j = SERVO_MAP[name]
        for model in np.linspace(j.model_min, j.model_max, 41):
            back = j.to_model(j.to_servo(model))
            worst = max(worst, abs(back - model))
        assert j.fits(), (
            f"{name}: ลิมิตโมเดล {j.model_min:+.0f}..{j.model_max:+.0f} "
            f"ตกนอกราง 0..180 (ได้ {j.servo_range()[0]:.1f}..{j.servo_range()[1]:.1f}) "
            f"-> ต้องขยับ home_deg หรือหุบลิมิต")
    assert worst < 1e-9, f"แปลงไป-กลับไม่ตรง คลาดเคลื่อน {worst:g}"

    # zero pose ต้องได้มุมเซอร์โวเท่ากับ home ทุกข้อ
    servo0 = model_to_servo(np.zeros(12))
    home = np.array([SERVO_MAP[n].home_deg for n in JOINT_ORDER])
    assert np.allclose(servo0, home), "zero pose ไม่ตรงกับ home_deg"

    # ความชันต้องเท่ากับ 1 องศาโมเดล = 1 องศาเซอร์โว (ขนาดเท่ากัน แค่กลับทิศ)
    for name in JOINT_ORDER:
        j = SERVO_MAP[name]
        assert abs(abs(j.to_servo(10.0) - j.to_servo(0.0)) - 10.0) < 1e-9, name

    thin = [n for n in JOINT_ORDER if SERVO_MAP[n].guard_deg() < EDGE_GUARD_DEG]
    if verbose:
        print(f"PASS: แปลงไป-กลับตรงกันทุกข้อ (คลาดเคลื่อนสูงสุด {worst:.2e} deg)")
        print("PASS: ลิมิตโมเดลทั้ง 12 ข้อตกอยู่ในราง 0..180")
        if thin:
            names = ", ".join(sorted({SERVO_MAP[n].joint for n in thin}))
            print(f"เตือน: {names} กินเต็มรางพอดี ไม่เหลือกันชน "
                  f"(ควรเหลือ >{EDGE_GUARD_DEG:.0f}°) -> วัดมุมชนโครงจริงแล้วหุบลิมิตลง")
        print("PASS: zero pose -> เซอร์โว " + ", ".join(
            f"{s:.0f}" for s in servo0[:3]) + " (θ1 θ2 θ3 ของ FL)")


# ---------------------------------------------------------------------------
# 6) แผนภาพอ้างอิง — "0 ของกู อยู่ตรงนี้ 180 อยู่ตรงนี้"
# ---------------------------------------------------------------------------

TEAL = "#2196a8"
INK = "#3a3a3a"
WARM = "#e8a33d"
RED = "#d62728"
GREEN = "#2ca02c"
MUTED = "#9aa0a3"


def _fan(ax, origin: np.ndarray, radius: float, a0: float, a1: float,
         color: str, alpha: float = 0.16) -> None:
    """ระบายพัดลมแสดงช่วงมุมที่ข้อต่อไปได้ (มุมเป็นเรเดียน วัดจากแกนนอน)."""
    th = np.linspace(a0, a1, 60)
    pts = np.vstack([origin,
                     origin + radius * np.column_stack([np.cos(th), np.sin(th)])])
    ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=alpha, zorder=1, linewidth=0)


def _leg_2d(ax, q_deg: Sequence[float], plane: str, color: str,
            lw: float = 2.4, alpha: float = 1.0, zorder: int = 3) -> np.ndarray:
    """วาดขา FL ลงบนระนาบ 2 มิติ: plane='front' = (y,z), 'side' = (x,z)."""
    pts = leg_polyline("FL", np.radians(q_deg), PARAMS)
    xy = pts[:, [1, 2]] if plane == "front" else pts[:, [0, 2]]
    ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, linewidth=lw, markersize=3.4,
            alpha=alpha, zorder=zorder, solid_capstyle="round")
    return xy


def _dial(ax, j: JointMap) -> None:
    """หน้าปัดเซอร์โวหนึ่งตัว: ราง 0..180, home, และช่วงที่โมเดลใช้จริง."""
    ax.set_aspect("equal")
    ax.axis("off")

    # ราง 0..180 อ่านซ้าย -> ขวา เหมือนหน้าปัด: servo 0 อยู่ซ้าย, 180 อยู่ขวา
    def ang(servo_deg: float) -> float:
        return np.radians(SERVO_SPAN_DEG - servo_deg)

    th = np.linspace(0.0, np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=MUTED, linewidth=6.0,
            solid_capstyle="butt", alpha=0.30, zorder=1)

    lo, hi = j.servo_range()
    th_use = np.linspace(ang(lo), ang(hi), 120)
    ax.plot(np.cos(th_use), np.sin(th_use), color=TEAL, linewidth=6.0,
            solid_capstyle="butt", zorder=2)

    # ขีดบอกตำแหน่ง 0 / 90 / 180 — ป้ายต้องเท่ากับค่าที่ตำแหน่งนั้นเสมอ
    for tick in (0.0, 90.0, 180.0):
        a = ang(tick)
        ax.plot([0.86 * np.cos(a), 1.14 * np.cos(a)],
                [0.86 * np.sin(a), 1.14 * np.sin(a)],
                color=INK, linewidth=1.0, zorder=4)
        ax.text(1.32 * np.cos(a), 1.32 * np.sin(a), f"{tick:.0f}", ha="center",
                va="center", fontsize=7.5, color=INK)

    # เข็มชี้ home = โมเดล 0
    a_home = ang(j.home_deg)
    ax.annotate("", xy=(0.92 * np.cos(a_home), 0.92 * np.sin(a_home)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.9,
                                mutation_scale=11), zorder=5)
    ax.plot([0], [0], "o", color=INK, markersize=4.0, zorder=6)

    ax.text(0, -0.40, f"{j.symbol}  {j.joint}  [{j.leg}]", ha="center", va="center",
            fontsize=9.0, color=INK, fontweight="bold")
    ax.text(0, -0.66, f"model 0  =  servo {j.home_deg:.0f}", ha="center", va="center",
            fontsize=8.0, color=RED)
    ax.text(0, -0.88, f"uses servo {lo:.0f}..{hi:.0f}", ha="center", va="center",
            fontsize=8.0, color=TEAL)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.02, 1.55)


def draw_reference(save: str | None = None) -> None:
    """แผนภาพอ้างอิงหน้าเดียว: ท่า 0 อยู่ไหน ทิศบวกไปทางไหน เซอร์โวอ่านได้เท่าไร."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15.0, 9.6), facecolor="white")
    # 12 คอลัมน์ เพื่อให้แถวบน (3 ช่อง) กับแถวหน้าปัด (4 ช่อง) ลงตัวพร้อมกัน
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.22, 0.58, 1.10],
                  hspace=0.34, wspace=0.55,
                  left=0.045, right=0.975, top=0.895, bottom=0.058)

    fig.text(0.045, 0.955, "P-kun — Angle Reference  (leg FL)", fontsize=17,
             color=INK, fontweight="bold")
    fig.text(0.045, 0.925,
             "model angle is signed about zero pose  |  servo dial is 0..180 with "
             "1 model degree = 1 servo degree",
             fontsize=10, color=MUTED)
    # หมายเหตุสำคัญที่สุด วางไว้ช่องว่างระหว่างหน้าปัดกับตาราง
    fig.text(0.5, 0.335,
             "θ1 is NOT mirrored in the model: + always rotates toward +y (the robot's left), "
             "so + is outward on FL/RL but inward on FR/RR.\n"
             "The servo dial IS mirrored (dir −1), and the two cancel: a symmetric stance "
             "reads the SAME servo number on all four legs.\n"
             "12° splay  →  model θ1 = +9.50° (left) / −9.50° (right)  →  servo 99.50° on "
             "all four. Use that as your calibration check.",
             fontsize=9.2, color=RED, ha="center", va="center", linespacing=1.7,
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#fdf2f2",
                       edgecolor="#f0c8c8", linewidth=0.9))

    # ---- A: มองจากหน้า (ระนาบ y-z) — θ1 abduction ------------------------
    axA = fig.add_subplot(gs[0, 0:4])
    for v, alpha in ((-55.0, 0.30), (55.0, 0.30)):
        _leg_2d(axA, (v, 0, 0), "front", MUTED, lw=1.7, alpha=alpha, zorder=2)
    _leg_2d(axA, (0, 0, 0), "front", TEAL, lw=2.8)

    hip = leg_polyline("FL", np.zeros(3), PARAMS)[0][[1, 2]]
    _fan(axA, hip, 132.0, np.radians(-90 - 55), np.radians(-90 + 55), TEAL)
    axA.annotate("", xy=(hip[0] + 96, hip[1] - 104), xytext=(hip[0] + 30, hip[1] - 128),
                 arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.7,
                                 connectionstyle="arc3,rad=0.30", mutation_scale=12))
    axA.text(hip[0] + 88, hip[1] - 106, "θ1 +", color=RED, fontsize=11, fontweight="bold")
    axA.text(hip[0] - 96, hip[1] - 66, "−55°", color=MUTED, fontsize=8.5, ha="center")
    axA.text(hip[0] + 96, hip[1] - 66, "+55°", color=MUTED, fontsize=8.5, ha="center")
    axA.set_title("FRONT view  (y–z)   θ1 abduction", fontsize=10.5, color=INK, pad=7)
    axA.set_xlabel("y  ← right   left →  (mm)", fontsize=8.5, color=MUTED)
    axA.set_ylabel("z  up (mm)", fontsize=8.5, color=MUTED)

    # ---- B: มองจากข้าง (ระนาบ x-z) — θ2 hip pitch ------------------------
    axB = fig.add_subplot(gs[0, 4:8])
    for v, alpha in ((-90.0, 0.28), (90.0, 0.28)):
        _leg_2d(axB, (0, v, 0), "side", MUTED, lw=1.7, alpha=alpha, zorder=2)
    _leg_2d(axB, (0, 0, 0), "side", TEAL, lw=2.8)

    hipS = leg_polyline("FL", np.zeros(3), PARAMS)[0][[0, 2]]
    _fan(axB, hipS, 128.0, np.radians(-180.0), np.radians(0.0), WARM)
    axB.annotate("", xy=(hipS[0] - 104, hipS[1] - 92), xytext=(hipS[0] - 30, hipS[1] - 126),
                 arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.7,
                                 connectionstyle="arc3,rad=-0.30", mutation_scale=12))
    axB.text(hipS[0] - 124, hipS[1] - 84, "θ2 +", color=RED, fontsize=11, fontweight="bold")
    axB.text(hipS[0] - 104, hipS[1] - 46, "+90°\n(foot back)", color=MUTED,
             fontsize=8.0, ha="center", va="top")
    axB.text(hipS[0] + 104, hipS[1] - 46, "−90°\n(foot fwd)", color=MUTED,
             fontsize=8.0, ha="center", va="top")
    axB.set_title("SIDE view  (x–z)   θ2 hip pitch", fontsize=10.5, color=INK, pad=7)
    axB.set_xlabel("x  ← back   front →  (mm)", fontsize=8.5, color=MUTED)

    # ---- C: มองจากข้าง — θ3 knee ----------------------------------------
    axC = fig.add_subplot(gs[0, 8:12])
    for v, alpha in ((75.0, 0.28), (150.0, 0.28)):
        _leg_2d(axC, (0, 0, v), "side", MUTED, lw=1.7, alpha=alpha, zorder=2)
    _leg_2d(axC, (0, 0, 0), "side", TEAL, lw=2.8)

    knee = leg_polyline("FL", np.zeros(3), PARAMS)[-2][[0, 2]]
    _fan(axC, knee, 68.0, np.radians(-90.0), np.radians(60.0), GREEN)
    axC.annotate("", xy=(knee[0] - 52, knee[1] - 30), xytext=(knee[0] - 8, knee[1] - 62),
                 arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.7,
                                 connectionstyle="arc3,rad=-0.32", mutation_scale=12))
    axC.text(knee[0] - 66, knee[1] - 22, "θ3 +", color=RED, fontsize=11, fontweight="bold")
    axC.text(knee[0] + 14, knee[1] - 48, "0° = straight", color=TEAL, fontsize=8.5)
    axC.text(knee[0] - 40, knee[1] + 30, "+150°\n(folded)", color=MUTED, fontsize=8.0,
             ha="center", va="bottom")
    axC.set_title("SIDE view  (x–z)   θ3 knee  (one way only)", fontsize=10.5,
                  color=INK, pad=7)
    axC.set_xlabel("x  ← back   front →  (mm)", fontsize=8.5, color=MUTED)

    for ax in (axA, axB, axC):
        ax.set_aspect("equal")
        ax.grid(True, color="#e6e9ea", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d4d8da")
        ax.tick_params(labelsize=7.5, colors=MUTED)

    # ---- D: หน้าปัดเซอร์โว 3 ตัว ----------------------------------------
    # เข่าโชว์ทั้งซ้ายและขวา เพราะ home ของมันสลับข้างกัน (15 vs 165) — จุดที่พลาดง่ายที่สุด
    for k, name in enumerate(("fl_abduction", "fl_hip_pitch", "fl_knee", "fr_knee")):
        _dial(fig.add_subplot(gs[1, 3 * k:3 * k + 3]), SERVO_MAP[name])

    # ---- E: ตารางสรุป ---------------------------------------------------
    axT = fig.add_subplot(gs[2, 0:12])
    axT.axis("off")
    cols = ("joint", "sym", "dir", "model 0 =\nservo", "model\nmin..max",
            "servo\nmin..max", "pulse us\nmin..max")
    rows = []
    for name in JOINT_ORDER:
        j = SERVO_MAP[name]
        lo, hi = j.servo_range()
        rows.append((j.name, j.symbol, f"{j.direction:+d}", f"{j.home_deg:.0f}",
                     f"{j.model_min:+.0f} .. {j.model_max:+.0f}",
                     f"{lo:.0f} .. {hi:.0f}",
                     f"{j.to_pulse_us(lo):.0f} .. {j.to_pulse_us(hi):.0f}"))
    tab = axT.table(cellText=rows, colLabels=cols, loc="center",
                    cellLoc="center", colWidths=[0.16, 0.06, 0.06, 0.11, 0.14, 0.13, 0.15])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.2)
    tab.scale(1.0, 0.96)
    for (r, _c), cell in tab.get_celld().items():
        cell.set_edgecolor("#dfe3e5")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(TEAL)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.108)
        elif r % 2 == 0:
            cell.set_facecolor("#f6f8f9")

    fig.text(0.045, 0.012,
             "dir −1 = the servo counts the other way (right-side horns are mirrored, "
             "matching invert: true in servos.yaml).    "
             "knee home is 15 (left legs) / 165 (right legs) because 0..150 will not "
             "fit around a 90 centre.",
             fontsize=8.5, color=MUTED)

    if save:
        fig.savefig(save, dpi=150, facecolor="white")
        print(f"บันทึกแผนภาพแล้ว: {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 7) CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="แปลงมุมโมเดล <-> มุมเซอร์โว 0-180")
    ap.add_argument("--table", action="store_true", help="ตารางสัญลักษณ์ + ช่วงมุม")
    ap.add_argument("--check", action="store_true", help="ตรวจการแปลงไป-กลับ")
    ap.add_argument("--diagram", nargs="?", const="", metavar="PNG",
                    help="วาดแผนภาพอ้างอิง (ใส่ชื่อไฟล์เพื่อบันทึก)")
    ap.add_argument("--to-servo", nargs=4, metavar=("LEG", "TH1", "TH2", "TH3"),
                    help="แปลงมุมโมเดลของขาหนึ่ง -> มุมเซอร์โว")
    ap.add_argument("--to-model", nargs=4, metavar=("LEG", "S1", "S2", "S3"),
                    help="แปลงมุมเซอร์โวของขาหนึ่ง -> มุมโมเดล")
    args = ap.parse_args(argv)

    did = False
    if args.table:
        print_table()
        did = True
    if args.check:
        verify()
        did = True

    for flag, fwd in ((args.to_servo, True), (args.to_model, False)):
        if not flag:
            continue
        leg = flag[0].upper()
        if leg not in LEGS:
            print(f"ไม่รู้จักขา {leg!r} — มี {', '.join(LEGS)}", file=sys.stderr)
            return 2
        vals = [float(v) for v in flag[1:]]
        for (symbol, joint, *_), v in zip(JOINT_SPEC, vals):
            j = SERVO_MAP[f"{leg.lower()}_{joint}"]
            if fwd:
                s = j.to_servo(v)
                flag_txt = "" if j.model_min <= v <= j.model_max else "  << นอกลิมิต"
                print(f"  {j.symbol} {joint:<10} model {v:+8.2f}°  ->  servo "
                      f"{s:7.2f}°  ({j.to_pulse_us(s):7.1f} us){flag_txt}")
            else:
                m = j.to_model(v)
                flag_txt = "" if j.model_min <= m <= j.model_max else "  << นอกลิมิต"
                print(f"  {j.symbol} {joint:<10} servo {v:7.2f}°  ->  model "
                      f"{m:+8.2f}°{flag_txt}")
        did = True

    if args.diagram is not None:
        if args.diagram == "":
            draw_reference(None)
        else:
            import matplotlib
            matplotlib.use("Agg")
            draw_reference(args.diagram)
        did = True

    if not did:
        print_table()
        print()
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
