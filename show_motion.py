#!/usr/bin/env python3
"""
show_motion.py — ดูท่าเดินเป็นภาพเคลื่อนไหว (แปลงจากเฟรมลำตัวเป็นเฟรมพื้นให้แล้ว)

ซ้าย  = มุมมอง 3 มิติ หุ่นเดินหน้าไปเรื่อย ๆ รอยเท้าค้างอยู่กับพื้น
ขวา   = มองจากบน สามเหลี่ยมรับน้ำหนัก + เงา CoG (จุดดำ) ต้องอยู่ในสามเหลี่ยมตลอด

รัน:  python show_motion.py                  เปิดหน้าต่างเล่นวนไปเรื่อย ๆ
      python show_motion.py --gif walk.gif   บันทึกเป็น GIF
      python show_motion.py --cycles 2       เดินกี่รอบ
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence, Tuple

import matplotlib
import numpy as np

if "--gif" in sys.argv[1:]:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from leg_kinematics import LEGS, PARAMS, Params, hip_points, leg_polyline  # noqa: E402
from walking_gait import MOVES, Gait, Trajectory, build_trajectory, gait_for  # noqa: E402

from sleep_wake import find_sleep_pose, sleep_sequence, wake_sequence  # noqa: E402

LEG_COLORS = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#c47f17"}
TEAL = "#2196a8"


def _pick_thai_font() -> str | None:
    from matplotlib.font_manager import get_font_names

    have = set(get_font_names())
    return next((f for f in ("Garuda", "Loma", "Norasi", "Kinnari") if f in have), None)


_THAI = _pick_thai_font()


def _rot(a: float) -> np.ndarray:
    """เมทริกซ์หมุน 2 มิติ."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def body_pose(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
    """ท่าของลำตัวในเฟรมพื้น: คืน (ตำแหน่ง (N,3), มุมหัน (N,)).

    หลักการ: เท้าที่แตะพื้นต้องอยู่นิ่งในเฟรมพื้น หา transform (หมุน+เลื่อน) ระหว่างสองเฟรม
    ที่ทำให้เท้าชุดเดิมทับกันพอดี (Procrustes) แล้วสะสมไปเรื่อย ๆ — ใช้ได้ทั้งเดินตรงและเลี้ยว
    """
    n = len(traj.t)
    pos = np.zeros((n, 3))
    yaw = np.zeros(n)
    for k in range(1, n):
        idx = [i for i in range(4) if traj.contact[k - 1, i] and traj.contact[k, i]]
        A = traj.feet[k][idx][:, :2]          # เท้า (เฟรมลำตัว) ที่เวลา k
        B = traj.feet[k - 1][idx][:, :2]      # เท้าชุดเดียวกันที่เวลา k-1
        ca, cb = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - ca, B - cb
        d_ang = float(np.arctan2(np.sum(A0[:, 0] * B0[:, 1] - A0[:, 1] * B0[:, 0]),
                                 np.sum(A0[:, 0] * B0[:, 0] + A0[:, 1] * B0[:, 1])))
        dR = _rot(d_ang)
        dt = cb - dR @ ca
        R_prev = _rot(yaw[k - 1])
        yaw[k] = yaw[k - 1] + d_ang
        pos[k, :2] = R_prev @ dt + pos[k - 1, :2]
    return pos, yaw


def repeat_trajectory(traj: Trajectory, cycles: int) -> Trajectory:
    """ต่อ trajectory ซ้ำหลายรอบ (ตัดเฟรมสุดท้ายที่ซ้ำกับเฟรมแรกออก)."""
    if cycles <= 1:
        return traj
    dt = traj.t[1] - traj.t[0]
    q = np.vstack([np.tile(traj.q[:-1], (cycles, 1)), traj.q[-1:]])
    feet = np.vstack([np.tile(traj.feet[:-1], (cycles, 1, 1)), traj.feet[-1:]])
    con = np.vstack([np.tile(traj.contact[:-1], (cycles, 1)), traj.contact[-1:]])
    cog = np.vstack([np.tile(traj.cog[:-1], (cycles, 1)), traj.cog[-1:]])
    margin = np.concatenate([np.tile(traj.margin[:-1], cycles), traj.margin[-1:]])
    t = np.arange(len(q)) * dt
    return Trajectory(t, q, feet, con, cog, margin, traj.mass_g, traj.bias, traj.reachable)


def drop_path(traj: Trajectory, p: Params) -> np.ndarray:
    """สำหรับท่านอน/ท่าลุก: ลำตัวไม่เดินหน้า แต่ยุบลง-ดันขึ้นตามจุดต่ำสุดของขาที่ยันพื้นอยู่."""
    pos = np.zeros((len(traj.t), 3))
    for k in range(len(traj.t)):
        low = min(float(np.min(leg_polyline(leg, traj.q[k, 3 * i:3 * i + 3], p)[:, 2]))
                  for i, leg in enumerate(LEGS))
        pos[k, 2] = -low          # ยกลำตัวขึ้นจนจุดต่ำสุดของขาแตะพื้นพอดี
    return pos


def _hull_closed(pts: np.ndarray) -> np.ndarray:
    """convex hull ของจุดที่แตะพื้น (ปิดรูปแล้ว) — ใช้ตอนลำตัวเอียงและขาวางราบ."""
    from elephant import convex_hull

    hull = convex_hull(pts[:, :2])
    return np.vstack([hull, hull[:1]]) if len(hull) >= 3 else pts[:, :2]


def _support_polygon(feet_w: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """คืนจุดมุมสามเหลี่ยม/สี่เหลี่ยมรับน้ำหนัก เรียงวนรอบ (ปิดรูปแล้ว)."""
    pts = feet_w[contact][:, :2]
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return np.vstack([pts[order], pts[order][:1]])


def animate(traj: Trajectory, p: Params, g: Gait, cycles: int = 1,
            gif: str | None = None, fps: int = 25, mode: str = "walk",
            follow: bool = False, title: str = "Walking Motion — crawl gait") -> None:
    """เล่นภาพเคลื่อนไหว หรือบันทึกเป็น GIF.

    mode   "walk" = ลำตัวเดินหน้าตามเท้าที่แตะพื้น, "drop" = อยู่กับที่แต่ยุบ/ดันตัวขึ้น
    follow True   = กล้องวิ่งตามหุ่น (ดูขาชัดแต่ดูไม่ออกว่าเคลื่อนที่), False = กล้องนิ่ง
    """
    walking = mode == "walk"
    if walking and cycles > 1:
        traj = repeat_trajectory(traj, cycles)
    if getattr(traj, "body_T", None) is not None:
        # ท่าที่ลำตัวเอียง (นั่ง/ยกขาหน้า) ส่ง transform เต็ม ๆ มาให้แล้ว ไม่ต้องเดาจากเท้า
        body = traj.body_T[:, :3, 3].copy()
        rots = traj.body_T[:, :3, :3].copy()
        walking = False
    elif walking:
        body, yaw = body_pose(traj)
        rots = np.array([np.eye(3) for _ in yaw])
        for k, a in enumerate(yaw):
            rots[k, :2, :2] = _rot(a)
    else:
        body = drop_path(traj, p)
        rots = np.array([np.eye(3)] * len(traj.t))
    step = max(1, int(round(g.hz / fps)))
    frames = np.arange(0, len(traj.t) - 1, step)

    ground = -g.stand_h if walking else 0.0
    # กรอบภาพ: กว้างพอใส่ลำตัว (2a) กับเท้าที่ถ่างออก แล้วเลื่อนตามลำตัวไปเรื่อย ๆ
    reach = max(float(np.max(np.abs(traj.feet[:, :, :2]))) + 35.0, p.a + 45.0)
    z_top = ground + (45.0 if walking else g.stand_h + 45.0)
    span = np.abs(body[:, :2]).max() + reach     # กล้องนิ่ง: กรอบต้องครอบเส้นทางทั้งหมด
    x_fixed = (min(-reach, body[:, 0].min() - reach), max(reach, body[:, 0].max() + reach))
    y_half = max(reach, np.abs(body[:, 1]).max() + reach)
    travel = float(np.hypot(body[-1, 0], body[-1, 1]))

    fig = plt.figure(figsize=(12.5, 5.4), facecolor="white")
    has_thai = any("฀" <= ch <= "๿" for ch in title)
    fig.suptitle(title, fontsize=15, color="#6b6b6b", fontweight="bold", x=0.06, ha="left",
                 y=0.96, **({"family": _THAI} if has_thai and _THAI else {}))

    ax = fig.add_axes([0.01, 0.02, 0.56, 0.86], projection="3d")
    ax2 = fig.add_axes([0.63, 0.10, 0.34, 0.74])

    feet_world = np.einsum("kij,knj->kni", rots, traj.feet) + body[:, None, :]

    def to_world(pts: np.ndarray, k: int) -> np.ndarray:
        """แปลงจุดจากเฟรมลำตัวไปเฟรมพื้น (หมุนเต็ม 3 มิติ รองรับลำตัวเอียง)."""
        out = np.atleast_2d(np.asarray(pts, dtype=float))
        return out @ rots[k].T + body[k]

    def draw(fk: int) -> None:
        k = int(fk)
        off = body[k]

        # ---------------- 3D ----------------
        ax.clear()
        ax.set_facecolor("white")
        hips = to_world(hip_points(p), k)
        loop = np.vstack([hips, hips[:1]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], "-", color=TEAL, linewidth=2.6)

        touching: List[np.ndarray] = []
        for i, leg in enumerate(LEGS):
            pts = to_world(leg_polyline(leg, traj.q[k, 3 * i:3 * i + 3], p), k)
            touching += [pt for pt in pts if pt[2] <= ground + 5.0]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=LEG_COLORS[leg],
                    linewidth=2.4, solid_capstyle="round")
            foot = to_world(traj.feet[k, i], k)[0]
            down = traj.contact[k, i]
            ax.scatter(*foot, marker="o" if down else "^", s=45 if down else 55,
                       c=LEG_COLORS[leg] if down else "white",
                       edgecolors=LEG_COLORS[leg], linewidths=1.5, depthshade=False)

        # รอยเท้าที่เคยเหยียบ ค้างไว้บนพื้น
        for i, leg in enumerate(LEGS):
            j = np.flatnonzero(traj.contact[:k + 1, i])[::3]
            if len(j):
                ax.scatter(feet_world[j, i, 0], feet_world[j, i, 1], np.full(len(j), ground),
                           s=2, c=LEG_COLORS[leg], alpha=0.35, depthshade=False)

        cog = to_world(traj.cog[k], k)[0]
        ax.plot([cog[0], cog[0]], [cog[1], cog[1]], [cog[2], ground], ":", color="#111111",
                linewidth=1.0)
        ax.scatter(cog[0], cog[1], cog[2], s=55, c="#111111", depthshade=False)
        ax.scatter(cog[0], cog[1], ground, s=45, marker="x", c="#111111", depthshade=False)

        x0, x1 = (off[0] - reach, off[0] + reach) if follow else x_fixed
        ax.set_xlim(x0, x1)
        ax.set_ylim(-y_half, y_half)
        ax.set_zlim(ground - 10.0, z_top)
        # box_aspect ต้องเท่ากับสัดส่วนช่วงข้อมูลจริง ไม่งั้นหุ่นจะยืดผิดส่วน
        ax.set_box_aspect((x1 - x0, 2 * y_half, z_top - ground + 10.0))
        ax.view_init(elev=20.0, azim=-62.0)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
        ax.tick_params(labelsize=6.5, colors="#999999")
        ax.set_zticks([])
        ax.grid(True, alpha=0.2)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white")
            pane.pane.set_edgecolor("#e6e6e6")

        # ---------------- มองจากบน ----------------
        ax2.clear()
        feet_w = feet_world[k]
        # ท่านั่ง/ยกขาหน้า รับน้ำหนักด้วย 'ท่อนขา' ที่วางราบ ไม่ใช่ปลายเท้า จึงต้องใช้ทุกจุดที่แตะพื้น
        poly = (_hull_closed(np.array(touching)) if len(touching) >= 3
                else _support_polygon(feet_w, traj.contact[k]))
        inside = not np.isfinite(traj.margin[k]) or traj.margin[k] > 0
        ax2.fill(poly[:, 0], poly[:, 1], color=TEAL if inside else "#d62728", alpha=0.14)
        ax2.plot(poly[:, 0], poly[:, 1], "-", color=TEAL if inside else "#d62728", linewidth=1.6)
        for i, leg in enumerate(LEGS):
            down = traj.contact[k, i]
            ax2.scatter(feet_w[i, 0], feet_w[i, 1], s=60 if down else 70,
                        marker="o" if down else "^",
                        c=LEG_COLORS[leg] if down else "white",
                        edgecolors=LEG_COLORS[leg], linewidths=1.6, zorder=4)
            ax2.annotate(leg, feet_w[i, :2], textcoords="offset points", xytext=(6, 5),
                         fontsize=7.5, color=LEG_COLORS[leg])
        ax2.plot(hips[:, 0], hips[:, 1], "-", color="#bbbbbb", linewidth=1.0)
        ax2.scatter(cog[0], cog[1], s=70, c="#111111", zorder=5)
        ax2.set_xlim(x0, x1)
        ax2.set_ylim(-y_half, y_half)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x (mm)", fontsize=8)
        ax2.set_ylabel("y (mm)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(alpha=0.2)
        if walking:
            sub = f"t = {traj.t[k]:4.2f} s     margin = {traj.margin[k]:5.1f} mm     " \
                  f"เดินไปแล้ว {np.hypot(off[0], off[1]):5.0f} mm"
        else:
            sub = f"t = {traj.t[k]:4.2f} s     ความสูงลำตัว {off[2]:5.1f} mm"
        ax2.set_title(sub, fontsize=9.5, color="#111111" if inside else "#d62728",
                      **({"family": _THAI} if _THAI else {}))

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, repeat=True)

    if gif:
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=80)
        print(f"saved gif -> {gif}  ({len(frames)} เฟรม @ {fps} fps, "
              f"เคลื่อนที่ {travel:.0f} mm)")
    else:
        plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", choices=("walk", "sleep", "wake", "sit", "rear", *MOVES),
                    default="walk",
                    help="ท่าที่จะดู (walk = เดินหน้า, left/right = เดินข้าง, turn_left/turn_right)")
    ap.add_argument("--gif", metavar="PATH", help="บันทึกเป็น GIF แทนการเปิดหน้าต่าง")
    ap.add_argument("--cycles", type=int, default=1, help="เดินกี่รอบ (เฉพาะ walk)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--cycle", type=float, default=Gait.cycle_s, help="เวลา 1 รอบ (วินาที)")
    ap.add_argument("--stride", type=float, default=Gait.stride, help="ระยะก้าว (mm)")
    ap.add_argument("--follow", action="store_true",
                    help="ให้กล้องวิ่งตามหุ่น (ค่าตั้งต้นคือกล้องนิ่ง จะได้เห็นว่าเดินไปจริง)")
    args = ap.parse_args(argv)

    if args.motion in MOVES or args.motion == "walk":
        move = "forward" if args.motion == "walk" else args.motion
        g = gait_for(move, cycle_s=args.cycle)
        if args.stride != Gait.stride:
            g = gait_for(move, cycle_s=args.cycle, stride=args.stride)
        traj = build_trajectory(PARAMS, g)
        if not traj.reachable:
            print("เตือน: มีจังหวะที่ IK เอื้อมไม่ถึง ภาพอาจเพี้ยน")
        titles = {"forward": "Walk forward — เดินหน้า", "back": "Walk back — ถอยหลัง",
                  "left": "Walk left — เดินข้างไปทางซ้าย",
                  "right": "Walk right — เดินข้างไปทางขวา",
                  "turn_left": "Turn left — เลี้ยวซ้ายอยู่กับที่",
                  "turn_right": "Turn right — เลี้ยวขวาอยู่กับที่"}
        animate(traj, PARAMS, g, cycles=args.cycles, gif=args.gif, fps=args.fps,
                mode="walk", follow=args.follow, title=titles[move])
    elif args.motion in ("sit", "rear"):
        from elephant import SitPlan, rear_up_sequence, sit_sequence
        g = Gait(cycle_s=args.cycle)
        plan = SitPlan()
        traj = sit_sequence(PARAMS, plan) if args.motion == "sit" else rear_up_sequence(PARAMS, plan)
        animate(traj, PARAMS, g, cycles=1, gif=args.gif, fps=args.fps, mode="drop", follow=False,
                title=("Sit — ยืน ไป นั่ง" if args.motion == "sit"
                       else "Rear up — ยกขาหน้าสองข้าง"))
    else:
        g = Gait(cycle_s=args.cycle)
        q_sleep = find_sleep_pose(PARAMS)
        traj = (sleep_sequence(PARAMS, g, q_sleep) if args.motion == "sleep"
                else wake_sequence(PARAMS, g, q_sleep))
        animate(traj, PARAMS, g, cycles=1, gif=args.gif, fps=args.fps, mode="drop",
                follow=False,
                title=("Sleep — ยืน ไป นอน" if args.motion == "sleep"
                       else "Wakeup — นอน ไป ยืน"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
