#!/usr/bin/env python3
"""
leg_kinematics.py — kinematic ของหุ่นสี่ขา 12 DOF (4 ขา x 3 ข้อต่อ) ด้วย standard DH (Spong)

เฟรมลำตัว {B}: x ไปข้างหน้า, y ไปทางซ้าย, z ขึ้นบน  (หน่วยทุกอย่างเป็น mm)

รันได้เลย:      python leg_kinematics.py
ตรวจอย่างเดียว:  python leg_kinematics.py --verify-only
บันทึกรูป:       python leg_kinematics.py --save out.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

if {"--save", "--verify-only"} & set(sys.argv[1:]):
    matplotlib.use("Agg")  # ไม่ต้องมีหน้าจอเวลาสั่งบันทึกรูป/ตรวจอย่างเดียว

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, RadioButtons, Slider  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (ลงทะเบียน projection="3d")

# ---------------------------------------------------------------------------
# 1) พารามิเตอร์ — แก้ที่นี่ที่เดียว ห้าม hardcode ตัวเลขซ้ำที่อื่น
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """ขนาดโครงสร้างหุ่น หน่วย mm ทั้งหมด."""

    a: float = 77.27   # ครึ่งระยะสะโพกหน้า-หลัง (ตามแกน x ของลำตัว)
    b: float = 29.0    # ครึ่งระยะสะโพกซ้าย-ขวา (ตามแกน y ของลำตัว)
    c: float = 0.0     # ระยะดิ่งจากศูนย์กลางลำตัวลงมาถึงแกน abduction (z0)
    h: float = 23.38   # ระยะดิ่งจากแกน abduction ลงมาถึงแกน hip pitch (z1)
    D1: float = 12.75  # offset ด้านข้าง จากแกน hip pitch ไปตำแหน่งเซอร์โวจริง
    D2: float = 12.75  # offset ที่เข่า เยื้องกลับเข้าหาลำตัว
    L1: float = 56.39  # ความยาวต้นขา (hip pitch -> knee)
    L2: float = 56.58  # ความยาวแข้ง (knee -> ปลายเท้า)


PARAMS = Params()

# เครื่องหมายของแต่ละขา: (sign_a, sign_b, sign_D1, sign_D2)
LEG_SIGNS: Dict[str, Tuple[float, float, float, float]] = {
    "FL": (+1.0, +1.0, +1.0, +1.0),  # หน้า-ซ้าย
    "FR": (+1.0, -1.0, -1.0, -1.0),  # หน้า-ขวา
    "RL": (-1.0, +1.0, +1.0, +1.0),  # หลัง-ซ้าย
    "RR": (-1.0, -1.0, -1.0, -1.0),  # หลัง-ขวา
}
LEGS: Tuple[str, ...] = ("FL", "FR", "RL", "RR")
JOINT_NAMES: Tuple[str, ...] = ("θ1 abduction", "θ2 hip pitch", "θ3 knee")

# ---------------------------------------------------------------------------
# 2) พีชคณิตพื้นฐาน
# ---------------------------------------------------------------------------


def trans(x: float, y: float, z: float) -> np.ndarray:
    """homogeneous transform ของการเลื่อนล้วน."""
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def rot_y(angle: float) -> np.ndarray:
    """homogeneous transform ของการหมุนรอบแกน y."""
    ca, sa = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:3, :3] = ((ca, 0.0, sa), (0.0, 1.0, 0.0), (-sa, 0.0, ca))
    return T


def rot_z(angle: float) -> np.ndarray:
    """homogeneous transform ของการหมุนรอบแกน z."""
    ca, sa = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:3, :3] = ((ca, -sa, 0.0), (sa, ca, 0.0), (0.0, 0.0, 1.0))
    return T


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """standard DH: T = Rot_z(theta) @ Trans_z(d) @ Trans_x(a) @ Rot_x(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# ---------------------------------------------------------------------------
# 3) ตาราง DH ของขาหนึ่งข้าง
# ---------------------------------------------------------------------------


def dh_table(leg_name: str, q: Sequence[float], p: Params = PARAMS) -> List[Tuple[float, float, float, float]]:
    """คืนแถว DH (theta, d, a, alpha) ทั้ง 3 แถวของขาที่ระบุ (มุมเป็นเรเดียน)."""
    _, _, sD1, sD2 = LEG_SIGNS[leg_name]
    return [
        (q[0], 0.0, p.h, -np.pi / 2),      # 1: abduction
        (q[1], sD1 * p.D1, p.L1, 0.0),     # 2: hip pitch
        (q[2], -sD2 * p.D2, p.L2, 0.0),    # 3: knee
    ]


def body_to_frame0(leg_name: str, p: Params = PARAMS) -> np.ndarray:
    """transform คงที่จากลำตัวไปเฟรม 0 (ไม่ใช่แถว DH): Trans(±a, ±b, -c) @ Roty(90°)."""
    sa, sb, _, _ = LEG_SIGNS[leg_name]
    return trans(sa * p.a, sb * p.b, -p.c) @ rot_y(np.pi / 2)


def leg_fk(leg_name: str, q: Sequence[float], p: Params = PARAMS) -> List[np.ndarray]:
    """FK ของขาหนึ่งข้าง คืน transform สะสมเทียบเฟรมลำตัว [T_B0, T_B1, T_B2, T_B3]."""
    T = body_to_frame0(leg_name, p)
    chain = [T]
    for theta, d, a, alpha in dh_table(leg_name, q, p):
        T = T @ dh_transform(theta, d, a, alpha)
        chain.append(T)
    return chain


def robot_fk(q_all: Sequence[float], p: Params = PARAMS) -> Dict[str, np.ndarray]:
    """รับมุม 12 ตัว (เรียงตาม LEGS ขาละ 3) คืน dict ชื่อขา -> ตำแหน่งปลายเท้าเทียบเฟรมลำตัว."""
    q = np.asarray(q_all, dtype=float).reshape(len(LEGS), 3)
    return {leg: leg_fk(leg, q[i], p)[-1][:3, 3] for i, leg in enumerate(LEGS)}


def leg_polyline(leg_name: str, q: Sequence[float], p: Params = PARAMS) -> np.ndarray:
    """จุดต่อเนื่องของ link รวม 'จุดหักมุม' ที่เกิดจาก offset d ของแต่ละแถว DH."""
    T = body_to_frame0(leg_name, p)
    pts = [T[:3, 3]]
    for theta, d, a, alpha in dh_table(leg_name, q, p):
        T_kink = T @ rot_z(theta) @ trans(0.0, 0.0, d)  # ปลายทางของส่วน d ก่อนเดินตาม a
        if abs(d) > 1e-12:
            pts.append(T_kink[:3, 3])
        T = T @ dh_transform(theta, d, a, alpha)
        pts.append(T[:3, 3])
    return np.asarray(pts)


def hip_points(p: Params = PARAMS) -> np.ndarray:
    """ตำแหน่งสะโพก (origin ของเฟรม 0) ทั้ง 4 ขา เรียงเป็นวงรอบกล่องลำตัว."""
    order = ("FL", "FR", "RR", "RL")
    return np.asarray([body_to_frame0(leg, p)[:3, 3] for leg in order])


# ---------------------------------------------------------------------------
# 4) โหมดตรวจสอบ
# ---------------------------------------------------------------------------


def expected_zero_foot(leg_name: str, p: Params = PARAMS) -> np.ndarray:
    """ตำแหน่งปลายเท้าที่ท่า zero pose ตามสูตรคำนวณมือ."""
    sa, sb, sD1, sD2 = LEG_SIGNS[leg_name]
    return np.array(
        [
            sa * p.a,
            sb * p.b + sD1 * p.D1 - sD2 * p.D2,
            -(p.c + p.h + p.L1 + p.L2),
        ]
    )


def verify(p: Params = PARAMS, tol: float = 1e-6, verbose: bool = True) -> None:
    """ตรวจ FK เทียบสูตรมือ + คุณสมบัติเชิงเรขาคณิต ถ้าไม่ผ่านจะ assert error."""
    zero = np.zeros(3)

    if verbose:
        print("=" * 68)
        print("ZERO POSE — FK vs สูตรคำนวณมือ   (tolerance = %g)" % tol)
        print("=" * 68)
        print(f"{'leg':<5}{'x_fk':>10}{'y_fk':>10}{'z_fk':>10}   "
              f"{'x_exp':>10}{'y_exp':>10}{'z_exp':>10}{'err':>12}")

    # (1) ปลายเท้าที่ zero pose ต้องตรงสูตร ทั้ง 4 ขา
    for leg in LEGS:
        got = leg_fk(leg, zero, p)[-1][:3, 3]
        exp = expected_zero_foot(leg, p)
        err = float(np.max(np.abs(got - exp)))
        if verbose:
            print(f"{leg:<5}{got[0]:>10.3f}{got[1]:>10.3f}{got[2]:>10.3f}   "
                  f"{exp[0]:>10.3f}{exp[1]:>10.3f}{exp[2]:>10.3f}{err:>12.2e}")
        assert err < tol, f"{leg}: FK ไม่ตรงสูตร zero pose (err={err:g}) — ตาราง DH หรือเครื่องหมายผิด"

    # (2) หมุน θ1 อย่างเดียว: ปลายเท้าเป็นวงกลมในระนาบ y-z, x คงที่, ระยะจากแกน z0 คงที่
    for leg in LEGS:
        hip = body_to_frame0(leg, p)[:3, 3]
        radii, xs = [], []
        for th1 in np.linspace(-1.0, 1.0, 21):
            foot = leg_fk(leg, (th1, 0.0, 0.0), p)[-1][:3, 3]
            xs.append(foot[0])
            radii.append(np.hypot(foot[1] - hip[1], foot[2] - hip[2]))
        assert np.ptp(xs) < tol, f"{leg}: หมุน θ1 แล้ว x ของปลายเท้าไม่คงที่"
        assert np.ptp(radii) < tol, f"{leg}: หมุน θ1 แล้วระยะจากแกน z0 ไม่คงที่"

    # (3) หมุน θ2 หรือ θ3 อย่างเดียว: y ของปลายเท้าต้องไม่เปลี่ยน
    for leg in LEGS:
        for j in (1, 2):
            ys = []
            for th in np.linspace(-1.0, 1.0, 21):
                q = [0.0, 0.0, 0.0]
                q[j] = th
                ys.append(leg_fk(leg, q, p)[-1][1, 3])
            assert np.ptp(ys) < tol, f"{leg}: หมุน θ{j + 1} แล้ว y ของปลายเท้าเปลี่ยน"

    # (4) ถ้า D1 == D2 ปลายเท้าต้องอยู่ระนาบเดียวกับแกน z0 คือ y = ±b
    p_eq = replace(p, D2=p.D1)
    for leg in LEGS:
        _, sb, _, _ = LEG_SIGNS[leg]
        y = leg_fk(leg, zero, p_eq)[-1][1, 3]
        assert abs(y - sb * p_eq.b) < tol, f"{leg}: D1==D2 แล้ว y_foot != ±b"

    # (5) กรณีขอบ h = 0 และ c = 0 ต้องยังถูกต้อง
    for edge in (replace(p, h=0.0), replace(p, c=0.0), replace(p, h=0.0, c=0.0)):
        for leg in LEGS:
            got = leg_fk(leg, zero, edge)[-1][:3, 3]
            assert np.allclose(got, expected_zero_foot(leg, edge), atol=tol), \
                f"{leg}: กรณี h/c = 0 ผลไม่ตรงสูตร"

    if verbose:
        print("-" * 68)
        print("PASS: zero pose / วงกลม θ1 / y คงที่เมื่อหมุน θ2,θ3 / D1==D2 / h=0,c=0")
        print()


# ---------------------------------------------------------------------------
# 5) การวาด — สไตล์สไลด์ DH Parameters
# ---------------------------------------------------------------------------

SHOW_FRAMES = True      # เปิด/ปิดลูกศรแกน x y z ของทุกเฟรม
SHOW_DIMS = True        # เปิด/ปิดเส้นบอกระยะ a b c h D1 D2 L1 L2
AXIS_LEN = 45.0         # ความยาวลูกศรแกน (mm)
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # x=แดง, y=เขียว, z=น้ำเงิน
DIM_COLOR = "#c47f17"   # สีเส้นบอกระยะ

def _thai_font() -> str | None:
    """หาฟอนต์ที่มีทั้งอักษรไทยและละติน (matplotlib 3.6 ไม่ fallback ราย glyph ให้)."""
    from matplotlib.font_manager import get_font_names

    available = set(get_font_names())
    return next((f for f in ("Garuda", "Loma", "Norasi", "Kinnari") if f in available), None)


THAI_FONT = _thai_font()  # ถ้าเครื่องไม่มีฟอนต์ไทย จะสลับไปใช้ข้อความอังกฤษแทน

TEAL = "#2196a8"        # สีหัวตารางแบบสไลด์อ้างอิง
ROW_A, ROW_B = "#dceef4", "#eef7fa"
INK = "#5a5a5a"


def _draw_frame(ax, T: np.ndarray, idx: int, s: float = 1.0) -> None:
    """วาดลูกศร x y z ของเฟรมหนึ่ง พร้อม label xN yN zN และชื่อ origin O_N (s = สเกลรูป)."""
    o = T[:3, 3]
    for k, color in enumerate(AXIS_COLORS):
        v = T[:3, k] * AXIS_LEN * s
        ax.quiver(*o, *v, color=color, linewidth=1.6, arrow_length_ratio=0.22)
        tip = o + v * 1.12
        ax.text(*tip, f"{'xyz'[k]}{idx}", color=color, fontsize=9, fontweight="bold")
    ax.text(o[0], o[1], o[2] + 9.0 * s, f"$O_{idx}$", color="#222222", fontsize=10)


def _draw_leg(ax, leg: str, q: Sequence[float], p: Params, highlight: bool, s: float = 1.0) -> None:
    """วาด link + ข้อต่อ + ปลายเท้า ของขาหนึ่งข้าง."""
    pts = leg_polyline(leg, q, p)
    chain = leg_fk(leg, q, p)
    color = "#222222" if highlight else "#9aa5ad"
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=color,
            linewidth=3.0 if highlight else 1.8, solid_capstyle="round", zorder=3)

    joints = np.asarray([T[:3, 3] for T in chain[:3]])
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=42 if highlight else 22,
               c="white", edgecolors=color, linewidths=1.6, depthshade=False, zorder=4)
    foot = chain[-1][:3, 3]
    ax.scatter(*foot, marker="X", s=90 if highlight else 45,
               c="#d62728" if highlight else "#c8a0a0", depthshade=False, zorder=5)

    if highlight:
        ax.text(foot[0], foot[1], foot[2] - 22.0 * s, f"foot {leg}", fontsize=9, color="#d62728")

        if SHOW_DIMS:
            _draw_dimensions(ax, leg, q, p, s)
        if SHOW_FRAMES:
            for idx, T in enumerate(chain):
                _draw_frame(ax, T, idx, s)


def _dim(ax, p0: np.ndarray, p1: np.ndarray, off: np.ndarray, label: str, value: float) -> None:
    """เส้นบอกระยะหนึ่งเส้น: เลื่อนออกจากชิ้นงานตาม off มีหัวลูกศรสองข้าง + เส้นต่อบาง ๆ."""
    p0, p1, off = np.asarray(p0, float), np.asarray(p1, float), np.asarray(off, float)
    if np.linalg.norm(p1 - p0) < 1e-9:
        return  # ระยะเป็นศูนย์ (เช่น h = 0) ไม่ต้องวาด
    q0, q1 = p0 + off, p1 + off

    for s, e in ((p0, q0), (p1, q1)):  # เส้นต่อจากชิ้นงานออกมาที่เส้นบอกระยะ
        seg = np.vstack([s, e])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], ":", color="#b0b0b0", linewidth=0.9, zorder=1)
    seg = np.vstack([q0, q1])
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], "-", color=DIM_COLOR, linewidth=1.0, zorder=6)
    for tail, head in ((q0, q1), (q1, q0)):  # หัวลูกศรสองข้าง
        v = (head - tail) * 0.18
        ax.quiver(*tail, *v, color=DIM_COLOR, linewidth=1.0, arrow_length_ratio=0.55, zorder=6)

    mid = (q0 + q1) / 2.0 + off * 0.12
    ax.text(*mid, f"{label} = {value:g}", fontsize=8.5, color=DIM_COLOR, zorder=7,
            ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0))


def _draw_dimensions(ax, leg: str, q: Sequence[float], p: Params, s: float = 1.0) -> None:
    """วาดเส้นบอกระยะของพารามิเตอร์ทั้ง 8 ตัวบนขาที่เลือก เพื่อดูว่าค่าไหนคือระยะไหน."""
    sa, sb, _, _ = LEG_SIGNS[leg]
    chain = leg_fk(leg, q, p)
    rows = dh_table(leg, q, p)
    o0, o1, o2, o3 = [T[:3, 3] for T in chain]
    kink1 = (chain[1] @ rot_z(rows[1][0]) @ trans(0.0, 0.0, rows[1][1]))[:3, 3]
    kink2 = (chain[2] @ rot_z(rows[2][0]) @ trans(0.0, 0.0, rows[2][1]))[:3, 3]

    # ฝั่งลำตัว: a, b วัดในระนาบ z = -c, c วัดดิ่งจากศูนย์กลางลำตัว
    corner = np.array([sa * p.a, 0.0, -p.c])
    _dim(ax, np.array([0.0, 0.0, -p.c]), corner, s * np.array([0.0, -sb * 34.0, 16.0]), "a", p.a)
    _dim(ax, corner, np.array([sa * p.a, sb * p.b, -p.c]), s * np.array([0.0, 0.0, 40.0]), "b", p.b)
    _dim(ax, np.zeros(3), np.array([0.0, 0.0, -p.c]), s * np.array([-34.0, 0.0, 0.0]), "c", p.c)

    # ฝั่งขา: เลื่อนเส้นบอกระยะออกมาทาง +x ของลำตัว จะได้ไม่ทับเส้น link
    out = s * np.array([50.0, 0.0, 0.0])
    _dim(ax, o0, o1, s * np.array([0.0, -sb * 62.0, 0.0]), "h", p.h)  # h แยกไปฝั่งในลำตัว
    _dim(ax, o1, kink1, out, "D1", p.D1)
    _dim(ax, kink1, o2, out, "L1", p.L1)
    _dim(ax, o2, kink2, out, "D2", p.D2)
    _dim(ax, kink2, o3, out, "L2", p.L2)


def _equal_aspect(ax, pts: np.ndarray) -> None:
    """ตั้ง aspect ให้เท่ากันทุกแกน ไม่งั้นรูปบิด."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid, span = (lo + hi) / 2.0, float(np.max(hi - lo)) * 0.58 + 1e-9
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _draw_dh_table(ax, leg: str, q: Sequence[float], p: Params) -> None:
    """วาดตาราง DH สไตล์หัวตารางสีเขียวน้ำทะเลแบบสไลด์อ้างอิง."""
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _, _, sD1, sD2 = LEG_SIGNS[leg]
    sgn = lambda s, name: ("+" if s > 0 else "−") + name  # noqa: E731
    rows = [
        ("1", "h", "−pi/2", "0", "th1"),
        ("2", "L1", "0", sgn(sD1, "D1"), "th2"),
        ("3", "L2", "0", sgn(-sD2, "D2"), "th3"),
    ]
    header = ("link", "ai", "alpha", "di", "thetai")
    xs = (0.08, 0.28, 0.46, 0.66, 0.87)
    h_row = 0.19

    ax.add_patch(plt.Rectangle((0.0, 0.80), 1.0, h_row, facecolor=TEAL, edgecolor="none"))
    for x, txt in zip(xs, header):
        ax.text(x, 0.80 + h_row / 2, txt, color="white", fontsize=11,
                fontweight="bold", ha="center", va="center")
    for r, row in enumerate(rows):
        y = 0.80 - (r + 1) * h_row
        ax.add_patch(plt.Rectangle((0.0, y), 1.0, h_row,
                                   facecolor=ROW_A if r % 2 == 0 else ROW_B, edgecolor="none"))
        for x, txt in zip(xs, row):
            ax.text(x, y + h_row / 2, txt, color="#20404a", fontsize=11, ha="center", va="center")

    ax.text(0.0, 1.06, f"leg {leg}   (standard DH, Spong)", fontsize=10, color=INK,
            va="bottom", clip_on=False)
    ax.text(0.0, 0.06, "T_B0 = Trans(%+.1f, %+.1f, %+.1f) @ Roty(90°)"
            % (LEG_SIGNS[leg][0] * p.a, LEG_SIGNS[leg][1] * p.b, -p.c),
            fontsize=9.5, color=INK, family="monospace")


def _draw_params(ax, p: Params) -> None:
    """แผงค่าพารามิเตอร์ทั้ง 8 ตัว (mm) — ตรงกับเส้นบอกระยะสีส้มในรูป."""
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cells = (("a", p.a), ("b", p.b), ("c", p.c), ("h", p.h),
             ("D1", p.D1), ("D2", p.D2), ("L1", p.L1), ("L2", p.L2))
    h_row = 0.30
    ax.add_patch(plt.Rectangle((0.0, 0.70), 1.0, h_row, facecolor=TEAL, edgecolor="none"))
    title = ("dimension parameters  (mm)  —  แก้ค่าที่ class Params" if THAI_FONT
             else "dimension parameters  (mm)  —  edit class Params")
    ax.text(0.5, 0.70 + h_row / 2, title, color="white", fontsize=10, fontweight="bold",
            ha="center", va="center", **({"family": THAI_FONT} if THAI_FONT else {}))
    for r in range(2):
        y = 0.70 - (r + 1) * h_row
        ax.add_patch(plt.Rectangle((0.0, y), 1.0, h_row,
                                   facecolor=ROW_A if r % 2 == 0 else ROW_B, edgecolor="none"))
        for c, (name, val) in enumerate(cells[4 * r:4 * r + 4]):
            ax.text(0.125 + c * 0.25, y + h_row / 2, f"{name} = {val:g}",
                    color="#20404a", fontsize=10.5, ha="center", va="center")


def _draw_matrix(ax, leg: str, q: Sequence[float], p: Params) -> None:
    """แสดง T_B3 เป็นตัวเลข พร้อมกรอบแดงล้อมคอลัมน์ตำแหน่ง แบบในสไลด์."""
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    T = leg_fk(leg, q, p)[-1]
    ax.text(0.0, 0.93, "T_B3 =", fontsize=12, family="monospace", color="#111111", va="top")
    for r in range(4):
        y = 0.70 - r * 0.19
        cells = "".join(f"{T[r, ccol]:>9.3f}," for ccol in range(3))
        ax.text(0.02, y, f"[{cells}", fontsize=11.5, family="monospace", color="#111111", va="center")
        ax.text(0.86, y, f"{T[r, 3]:>9.3f}]", fontsize=11.5, family="monospace",
                color="#111111", va="center", ha="right")
    ax.add_patch(plt.Rectangle((0.655, 0.04), 0.235, 0.79, fill=False,
                               edgecolor="#d62728", linewidth=1.8))
    ax.text(0.89, 0.93, "position", fontsize=9, color="#d62728", ha="right", va="top")


def _redraw(ax3d, leg: str, q_all: np.ndarray, p: Params) -> None:
    """วาดหุ่นทั้งตัวใหม่ทั้งหมด (เรียกทุกครั้งที่สไลเดอร์ขยับ)."""
    ax3d.clear()
    ax3d.set_facecolor("white")

    hips = hip_points(p)
    polys = [leg_polyline(name, q_all[3 * i:3 * i + 3], p) for i, name in enumerate(LEGS)]
    all_pts = np.vstack([hips, *polys])

    # สเกลรูป: ลูกศรแกนกับเส้นบอกระยะต้องยืด/หดตามขนาดหุ่น ไม่งั้นหุ่นเล็กจะโดนลูกศรกลบ
    extent = float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)))
    s = float(np.clip(extent / 300.0, 0.25, 2.0))

    loop = np.vstack([hips, hips[:1]])
    ax3d.plot(loop[:, 0], loop[:, 1], loop[:, 2], "-", color="#2196a8", linewidth=2.6, zorder=2)
    ax3d.scatter([0.0], [0.0], [0.0], s=30, c="#2196a8", depthshade=False)
    ax3d.text(0.0, 0.0, 14.0 * s, "{B}", color="#2196a8", fontsize=10)

    for i, name in enumerate(LEGS):
        _draw_leg(ax3d, name, q_all[3 * i:3 * i + 3], p, highlight=(name == leg), s=s)
    _equal_aspect(ax3d, all_pts)

    ax3d.set_xlabel("x  (mm)", fontsize=9)
    ax3d.set_ylabel("y  (mm)", fontsize=9)
    ax3d.set_zlabel("z  (mm)", fontsize=9)
    ax3d.tick_params(labelsize=7, colors="#888888")
    ax3d.grid(True, alpha=0.25)
    ax3d.view_init(elev=18.0, azim=-58.0)
    for pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#dddddd")


def show(p: Params = PARAMS, save: str | None = None) -> None:
    """หน้าต่างหลัก: ไดอะแกรม 3D + ตาราง DH + เมทริกซ์ + สไลเดอร์ + เลือกขา."""
    fig = plt.figure(figsize=(13.5, 7.6), facecolor="white")
    fig.text(0.035, 0.945, "DH Parameters — Quadruped Leg (12 DOF)",
             fontsize=23, color="#6b6b6b", fontweight="bold", va="center")

    ax_tab = fig.add_axes([0.035, 0.615, 0.36, 0.215])
    ax_par = fig.add_axes([0.035, 0.495, 0.36, 0.10])
    ax_mat = fig.add_axes([0.035, 0.265, 0.36, 0.18])
    ax3d = fig.add_axes([0.44, 0.10, 0.55, 0.80], projection="3d")

    state = {"leg": LEGS[0], "q": np.zeros(3 * len(LEGS))}

    def refresh(_=None) -> None:
        leg = state["leg"]
        i = LEGS.index(leg)
        q = np.array([s.val for s in sliders]) * np.pi / 180.0
        state["q"][3 * i:3 * i + 3] = q
        _redraw(ax3d, leg, state["q"], p)
        _draw_dh_table(ax_tab, leg, q, p)
        _draw_params(ax_par, p)
        _draw_matrix(ax_mat, leg, q, p)
        fig.canvas.draw_idle()

    sliders = []
    for k, label in enumerate(JOINT_NAMES):
        ax_s = fig.add_axes([0.10, 0.20 - k * 0.055, 0.26, 0.03], facecolor=ROW_B)
        s = Slider(ax_s, label, -180.0, 180.0, valinit=0.0, valstep=0.5, color=TEAL)
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)
        s.on_changed(refresh)
        sliders.append(s)

    ax_radio = fig.add_axes([0.905, 0.76, 0.075, 0.15], facecolor=ROW_B)
    ax_radio.set_title("leg", fontsize=9, color=INK)
    radio = RadioButtons(ax_radio, LEGS, active=0)
    for lbl in radio.labels:
        lbl.set_fontsize(9)

    def on_leg(name: str) -> None:
        state["leg"] = name
        i = LEGS.index(name)
        for k, s in enumerate(sliders):
            s.eventson = False
            s.set_val(np.degrees(state["q"][3 * i + k]))
            s.eventson = True
        refresh()

    radio.on_clicked(on_leg)

    ax_reset = fig.add_axes([0.30, 0.02, 0.07, 0.035])
    btn = Button(ax_reset, "zero", color=ROW_A, hovercolor=ROW_B)

    def on_reset(_) -> None:
        for s in sliders:
            s.reset()
        refresh()

    btn.on_clicked(on_reset)

    refresh()
    fig._keep_alive = (sliders, radio, btn)  # กัน widget ถูก garbage-collect
    if save:
        fig.savefig(save, dpi=130, facecolor="white")
        print(f"saved figure -> {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 6) main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """รัน verify() ก่อนเสมอ แล้วค่อยเปิดหน้าต่างวาด."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true", help="รันแค่โหมดตรวจสอบ ไม่เปิดหน้าต่าง")
    ap.add_argument("--save", metavar="PNG", help="บันทึกรูปเป็นไฟล์แทนการเปิดหน้าต่าง")
    ap.add_argument("--no-frames", action="store_true", help="ปิดลูกศรแกนของทุกเฟรม")
    ap.add_argument("--no-dims", action="store_true", help="ปิดเส้นบอกระยะ a b c h D1 D2 L1 L2")
    args = ap.parse_args(argv)

    global SHOW_FRAMES, SHOW_DIMS
    if args.no_frames:
        SHOW_FRAMES = False
    if args.no_dims:
        SHOW_DIMS = False

    verify(PARAMS)

    feet = robot_fk(np.zeros(3 * len(LEGS)))
    print("robot_fk(zero pose) — ตำแหน่งปลายเท้าเทียบเฟรมลำตัว {B}:")
    for leg, pos in feet.items():
        print(f"  {leg}: [{pos[0]:9.3f} {pos[1]:9.3f} {pos[2]:9.3f}]")

    if not args.verify_only:
        show(PARAMS, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
