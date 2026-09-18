#!/usr/bin/env python3
"""
show_gesture.py — ดูคลังท่าทาง P-kun เป็นภาพ

ซ้าย  = ลำตัวจริง (12 DOF ที่ IK คำนวณให้) มองแบบ 3 มิติ กล้องนิ่ง
ขวาบน = ชั้นการแสดงออก: หัว/หู/ตา/หาง/LED ที่ยังไม่มีฮาร์ดแวร์ วาดให้เห็นว่าท่าจะ "อ่านออก" ไหม
ขวาล่าง = แถบค่าของทุกช่องสัญญาณ ณ เฟรมนั้น

รัน:  python3 show_gesture.py --gesture notice_urgent
      python3 show_gesture.py --gesture thinking --gif thinking.gif
      python3 show_gesture.py --sheet catalog.png       ภาพรวมทุกท่าในหน้าเดียว
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence, Tuple

import matplotlib
import numpy as np

if "--gif" in sys.argv[1:] or "--sheet" in sys.argv[1:]:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from elephant import to_world  # noqa: E402
from leg_kinematics import LEGS, PARAMS, Params, hip_points, leg_polyline  # noqa: E402

import gestures as G  # noqa: E402

LEG_COLORS = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#2ca02c", "RR": "#c47f17"}
TEAL = "#2196a8"
INK = "#3a3a3a"


def _thai() -> str | None:
    from matplotlib.font_manager import get_font_names

    have = set(get_font_names())
    return next((f for f in ("Garuda", "Loma", "Norasi", "Kinnari") if f in have), None)


THAI = _thai()


def _tf(size: float, **kw) -> dict:
    """ฟอนต์ที่รองรับภาษาไทย (matplotlib 3.6 ไม่มี per-glyph fallback ต้องบังคับทั้งสตริง)."""
    out = {"fontsize": size, **kw}
    if THAI:
        out["family"] = THAI
    return out


# ---------------------------------------------------------------------------
# หน้า/หู/ตา/หาง — วาดชั้นการแสดงออก
# ---------------------------------------------------------------------------


def _rot2(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class FaceView:
    """หัว/หู/ตา/งวง/หาง/LED — สร้าง artist ครั้งเดียวแล้วอัปเดตค่า.

    แผงนี้ถูกวาดทุกเฟรมเหมือนกับหุ่น ถ้าสร้าง patch ใหม่ทุกครั้งจะกิน ~20 ms/เฟรม
    ซึ่งพอรวมกับแผงแถบค่าแล้วทำให้ GUI ตกเหลือ 11 fps จนรู้สึกกระตุก
    """

    N_SOUND = 3

    def __init__(self, ax, small: bool = False, show_led_name: bool = True) -> None:
        self.ax = ax
        self.small = small
        lw = 2.4 if small else 3.4
        self.skull = Polygon(np.zeros((3, 2)), closed=True, fc="#f2f4f5", ec=INK,
                             lw=1.6, zorder=2)
        ax.add_patch(self.skull)
        self.ears = [ax.plot([], [], "-", color=INK, lw=lw, solid_capstyle="round",
                             zorder=3)[0] for _ in range(2)]
        self.eyes = [Ellipse((0, 0), 3.0, 3.4, fc=INK, ec=INK, lw=1.4, zorder=4)
                     for _ in range(2)]
        for e in self.eyes:
            ax.add_patch(e)
        self.trunk, = ax.plot([], [], "-", color=INK, lw=3.0 if not small else 2.2,
                              solid_capstyle="round", zorder=3)
        self.tail, = ax.plot([], [], "-", color="#9a9a9a", lw=2.4,
                             solid_capstyle="round", zorder=2)
        self.led = Circle((-12.5, -9.0), 2.2, fc="#2b2b2b", ec=INK, lw=0.8, zorder=4)
        ax.add_patch(self.led)
        self.led_txt = (ax.text(-12.5, -12.8, "", ha="center", va="top", fontsize=7.5)
                        if show_led_name else None)
        arc = np.linspace(-0.7, 0.7, 24)
        self.sound = []
        for i in range(self.N_SOUND):
            r = 3.0 + 2.0 * i
            ln, = ax.plot(12.0 + r * np.cos(arc), 8.0 + r * np.sin(arc), "-",
                          color="#e8a33d", lw=1.4, zorder=2)
            ln.set_visible(False)
            self.sound.append(ln)
        self.label = ax.text(0, -13.6, "", ha="center", va="top",
                             **_tf(8.5 if small else 10, color=INK))
        ax.set_xlim(-17, 17)
        ax.set_ylim(-14, 15)
        ax.set_aspect("equal")
        ax.axis("off")

    def update(self, e: np.ndarray, label: str = "") -> None:
        hp, hy, hr = e[G.COL["head_pitch"]], e[G.COL["head_yaw"]], e[G.COL["head_roll"]]
        el, er = e[G.COL["ear_l"]], e[G.COL["ear_r"]]
        eye, tail = e[G.COL["eye"]], e[G.COL["tail"]]
        led = G.LED_NAMES[int(round(e[G.COL["led"]]))]
        led_v, snd = e[G.COL["led_v"]], e[G.COL["sound"]]

        cx, cy = 0.30 * hy, 0.9 + 0.30 * hp        # หันหัว = หน้าเลื่อนไปทางนั้น
        R = _rot2(hr)

        def put(pts) -> np.ndarray:
            return (R @ np.asarray(pts, dtype=float).T).T + (cx, cy)

        th = np.linspace(0, 2 * np.pi, 60)
        self.skull.set_xy(put(np.c_[9.5 * np.cos(th),
                                    8.0 * np.sin(th) * (1.0 - 0.10 * hp / 30.0)]))

        # หู: วัดมุมจากแนวดิ่ง +1 = ตั้งชี้ขึ้น (25°), 0 = ปกติ (65°), -1 = ตกลู่ลงข้างหัว (105°)
        for art, side, val in zip(self.ears, (+1, -1), (el, er)):
            a = np.radians(25.0 + 40.0 * (1.0 - float(val)))
            d = np.array([side * np.sin(a), np.cos(a)])
            root = np.array([side * 5.5, 6.0])
            bow = np.array([side * 1.6, 0.0])      # โก่งออกข้างเล็กน้อยให้ดูเป็นใบหู
            pts = put(np.array([root, root + 5.0 * d + bow, root + 10.0 * d]))
            art.set_data(pts[:, 0], pts[:, 1])

        # ตา: ความสูงของวงรี = ระดับการลืมตา (0 = ปิดสนิท เหลือเป็นขีด)
        for art, side in zip(self.eyes, (+1, -1)):
            art.set_center(tuple(put(np.array([[side * 3.6, 0.8]]))[0]))
            art.height = max(0.35, 3.4 * eye)
            art.angle = hr
            art.set_facecolor(INK if eye > 0.06 else "none")

        trunk = put(np.c_[np.linspace(0.0, 1.4, 5), np.linspace(-3.4, -11.5, 5)])
        self.trunk.set_data(trunk[:, 0], trunk[:, 1])

        tp = np.array([12.5, -9.0])                # หาง มองจากด้านหลัง วางมุมขวาล่าง
        tt = tp + _rot2(tail) @ np.array([2.0, -6.5])
        self.tail.set_data([tp[0], tt[0]], [tp[1], tt[1]])

        self.led.set_facecolor(G.LED_HEX[led])
        self.led.set_alpha(float(np.clip(0.12 + 0.88 * led_v, 0.0, 1.0)))
        if self.led_txt is not None:
            self.led_txt.set_text(led)
            self.led_txt.set_color(G.LED_HEX[led])
        for i, ln in enumerate(self.sound):
            ln.set_visible(snd > (i + 1) / 4.0)
        self.label.set_text(label)


def draw_face(ax, e: np.ndarray, label: str = "", small: bool = False) -> FaceView:
    """วาดหน้าลงแกนเปล่าครั้งเดียว (ใช้กับภาพนิ่ง) — คืน view ไว้อัปเดตต่อได้."""
    view = FaceView(ax, small=small, show_led_name=not small)
    view.update(e, label)
    return view


CHANNEL_ROWS: Tuple[Tuple[str, float, float], ...] = (
    ("head_pitch", -35.0, 35.0), ("head_yaw", -60.0, 60.0), ("head_roll", -25.0, 25.0),
    ("ear_l", -1.0, 1.0), ("ear_r", -1.0, 1.0), ("eye", 0.0, 1.0),
    ("tail", -35.0, 35.0), ("led_v", 0.0, 1.0), ("sound", 0.0, 1.0),
)


class ChannelView:
    """แถบค่าของทุกช่องสัญญาณ — พื้นหลังกับชื่อสร้างครั้งเดียว อัปเดตเฉพาะแท่งกับตัวเลข."""

    def __init__(self, ax) -> None:
        self.ax = ax
        self.bars: list = []
        self.vals: list = []
        n = len(CHANNEL_ROWS)
        for i, (name, lo, hi) in enumerate(CHANNEL_ROWS):
            y = n - 1 - i
            zero = (0.0 - lo) / (hi - lo)
            ax.add_patch(Rectangle((0, y - 0.28), 1.0, 0.56, fc="#eef1f2", ec="none"))
            ax.plot([zero, zero], [y - 0.32, y + 0.32], "-", color="#c9cfd2", lw=0.8)
            bar = Rectangle((zero, y - 0.22), 0.0, 0.44, fc=TEAL, ec="none")
            ax.add_patch(bar)
            self.bars.append(bar)
            ax.text(-0.02, y, name, ha="right", va="center", fontsize=7.5, color="#666666")
            self.vals.append(ax.text(1.02, y, "", ha="left", va="center", fontsize=7,
                                     color="#999999"))
        ax.set_xlim(-0.32, 1.18)
        ax.set_ylim(-0.6, n - 0.4)
        ax.axis("off")

    def update(self, e: np.ndarray) -> None:
        for bar, val, (name, lo, hi) in zip(self.bars, self.vals, CHANNEL_ROWS):
            v = float(e[G.COL[name]])
            u = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
            zero = (0.0 - lo) / (hi - lo)
            bar.set_x(min(u, zero))
            bar.set_width(abs(u - zero) + 1e-3)
            val.set_text(f"{v:+.2f}")


def draw_channels(ax, e: np.ndarray) -> ChannelView:
    """วาดแถบค่าลงแกนเปล่าครั้งเดียว — คืน view ไว้อัปเดตต่อได้."""
    view = ChannelView(ax)
    view.update(e)
    return view


# ---------------------------------------------------------------------------
# ภาพเคลื่อนไหว
# ---------------------------------------------------------------------------


def view_range(clip: "G.Clip") -> Tuple[float, float]:
    """ขอบเขตกล้องของท่านั้น — คงที่ตลอดคลิป จะได้ไม่รู้สึกว่าหุ่นซูมเข้าออกเอง."""
    return float(np.max(np.abs(clip.traj.feet[:, :, :2]))) + 40.0, 165.0


class RobotView:
    """วาดหุ่นแบบ 'สร้างเส้นครั้งเดียวแล้วอัปเดตพิกัด' ไม่ใช่ ax.clear() ทุกเฟรม.

    การ clear แกน 3 มิติแล้วตั้งค่าใหม่ (แกน/กริด/แผ่นพื้นหลัง/box_aspect) กินเวลาราว 60 ms
    ต่อเฟรม = เพดาน 14 fps ซึ่งดูแล้วรู้สึกกระตุก การอัปเดตเฉพาะข้อมูลของเส้นเดิมเร็วกว่าหลายเท่า
    ใช้ ax.plot ที่ใส่ marker แทน ax.scatter เพราะ Line3D อัปเดตด้วย set_data_3d ได้ตรง ๆ
    """

    def __init__(self, ax, p: Params = PARAMS) -> None:
        self.ax = ax
        self.p = p
        ax.set_facecolor("white")
        self.body, = ax.plot([], [], [], "-", color=TEAL, lw=2.6)
        self.legs = {leg: ax.plot([], [], [], "-", color=LEG_COLORS[leg], lw=2.4,
                                  solid_capstyle="round")[0] for leg in LEGS}
        self.feet = {leg: ax.plot([], [], [], ls="none", marker="o", ms=6.5,
                                  mfc=LEG_COLORS[leg], mec=LEG_COLORS[leg], mew=1.5)[0]
                     for leg in LEGS}
        self.cog_stem, = ax.plot([], [], [], ":", color="#111111", lw=1.0)
        self.cog, = ax.plot([], [], [], ls="none", marker="o", ms=7, color="#111111")
        self.shadow, = ax.plot([], [], [], ls="none", marker="x", ms=7, color="#111111")
        ax.view_init(elev=18.0, azim=-62.0)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
        ax.tick_params(labelsize=6.5, colors="#999999")
        ax.set_zticks([])
        ax.grid(True, alpha=0.2)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white")
            pane.pane.set_edgecolor("#e6e6e6")

    def set_scale(self, reach: float, z_top: float = 165.0) -> None:
        """ตั้งกรอบกล้องครั้งเดียวต่อหนึ่งท่า (ไม่ใช่ทุกเฟรม) จะได้ไม่ซูมเข้าออกเอง."""
        self.ax.set_xlim(-reach, reach)
        self.ax.set_ylim(-reach, reach)
        self.ax.set_zlim(-5.0, z_top)
        self.ax.set_box_aspect((2 * reach, 2 * reach, z_top + 5.0))

    def update(self, clip: "G.Clip", k: int) -> None:
        traj = clip.traj
        T = traj.body_T[k]
        hips = to_world(T, hip_points(self.p))
        loop = np.vstack([hips, hips[:1]])
        self.body.set_data_3d(loop[:, 0], loop[:, 1], loop[:, 2])
        for i, leg in enumerate(LEGS):
            pts = to_world(T, leg_polyline(leg, traj.q[k, 3 * i:3 * i + 3], self.p))
            self.legs[leg].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            foot, down = pts[-1], bool(traj.contact[k, i])
            art = self.feet[leg]
            art.set_data_3d([foot[0]], [foot[1]], [foot[2]])
            art.set_marker("o" if down else "^")           # ขาลอย = สามเหลี่ยมกลวง
            art.set_markerfacecolor(LEG_COLORS[leg] if down else "white")
        c = traj.cog[k]
        self.cog_stem.set_data_3d([c[0], c[0]], [c[1], c[1]], [c[2], 0.0])
        self.cog.set_data_3d([c[0]], [c[1]], [c[2]])
        self.shadow.set_data_3d([c[0]], [c[1]], [0.0])


def draw_robot(ax, clip: "G.Clip", k: int, p: Params = PARAMS,
               reach: float | None = None, z_top: float = 165.0) -> RobotView:
    """วาดหุ่นครั้งเดียวลงแกนเปล่า (ทางสะดวกสำหรับภาพนิ่ง) — คืน view ไว้อัปเดตต่อได้."""
    view = RobotView(ax, p)
    view.set_scale(view_range(clip)[0] if reach is None else reach, z_top)
    view.update(clip, k)
    return view


def frame_title(clip: "G.Clip", k: int) -> Tuple[str, bool]:
    """ข้อความเวลา + ธงว่าเฟรมนี้อยู่ในช่วงที่ท่าวนซ้ำหรือยัง."""
    looping = clip.loop_from is not None and clip.traj.t[k] >= clip.loop_from
    return (f"t = {clip.traj.t[k]:4.2f} / {clip.dur:.1f} s"
            + ("   [ช่วงวนซ้ำ]" if looping else ""), looping)


def animate(gid: str, p: Params = PARAMS, gif: str | None = None, fps: int = 25,
            cycles: int = 1) -> None:
    """เล่นท่าหนึ่งท่า: ลำตัวจริงทางซ้าย ชั้นการแสดงออกทางขวา."""
    clip = G.build(gid, p)
    res = G.check(clip)
    traj = clip.traj
    T = traj.body_T
    step = max(1, int(round(G.HZ / fps)))
    frames = np.tile(np.arange(0, len(traj.t), step), max(1, cycles))

    fig = plt.figure(figsize=(13.0, 5.6), facecolor="white")
    fig.suptitle(f"{gid} — {clip.means}", x=0.045, ha="left", y=0.975,
                 **_tf(15, color="#6b6b6b", fontweight="bold"))
    fig.text(0.045, 0.905, f"เมื่อไร: {clip.when}", va="top", **_tf(9.5, color="#9a9a9a"))

    ax = fig.add_axes([0.00, 0.02, 0.50, 0.84], projection="3d")
    axf = fig.add_axes([0.52, 0.20, 0.26, 0.62])
    axc = fig.add_axes([0.80, 0.16, 0.18, 0.68])

    reach, z_top = view_range(clip)
    view = RobotView(ax, p)
    view.set_scale(reach, z_top)
    face = FaceView(axf)
    chan = ChannelView(axc)

    def draw(fk: int) -> None:
        k = int(fk) % len(traj.t)
        view.update(clip, k)
        face.update(clip.expr[k])
        title, looping = frame_title(clip, k)
        axf.set_title(title, **_tf(9.5, color=TEAL if looping else "#111111"))
        chan.update(clip.expr[k])

    fig.text(0.52, 0.10,
             f"ลำตัว: สูง {T[:, 2, 3].min():.0f}–{T[:, 2, 3].max():.0f} mm    "
             f"ขาเร็วสุด {res['peak_rate_dps']:.0f}°/s    "
             f"tau สูงสุด {res['tau_max_kgcm']:.2f} kg·cm (งบ {res['torque_budget_kgcm']:.2f})    "
             f"margin ต่ำสุด {res['min_margin_mm']:.1f} mm", **_tf(8.5, color="#9a9a9a"))
    if clip.notes:
        fig.text(0.52, 0.06, " | ".join(clip.notes), **_tf(8.5, color="#d62728"))

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, repeat=True)
    if gif:
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=80)
        print(f"saved -> {gif}  ({len(frames)} เฟรม @ {fps} fps)")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# ภาพรวมทั้งคลัง
# ---------------------------------------------------------------------------


peak_frame = G.peak_frame     # ใช้ตัวเดียวกับ gestures.py ไม่ให้ตรรกะแตกเป็นสองที่


GROUP_COLOR = {"lifecycle": TEAL, "attention": "#d62728", "conversation": "#7e57c2",
               "social": "#2ca02c", "system": "#c47f17"}
GROUP_NAME = {"lifecycle": "วงจรชีวิต", "attention": "แจ้งเตือน & โฟกัส",
              "conversation": "ลูปสนทนา", "social": "บุคลิก & สังคม", "system": "ระบบ"}


def sheet(path: str, p: Params = PARAMS) -> None:
    """ภาพรวมทุกท่าในหน้าเดียว: หน้าตาที่เฟรมเด่นสุด + ความสูงลำตัว + ผลตรวจ.

    ข้อความทุกชิ้นวางด้วยพิกัดของ figure ไม่ใช่พิกัดข้อมูลในแกน
    เพราะแกนของหน้าถูกบังคับ aspect เท่ากัน ขอบบน-ล่างจริงจึงไม่ตรงกับที่ตั้ง ylim ไว้
    """
    clips = [(gid, G.build(gid, p)) for gid in G.ORDER]
    ncol = 5
    fig = plt.figure(figsize=(15.5, 13.6), facecolor="white")
    fig.suptitle("P-kun — คลังท่าทาง (gesture catalog)", x=0.035, ha="left", y=0.985,
                 **_tf(19, color="#6b6b6b", fontweight="bold"))
    fig.text(0.035, 0.950, va="top", s="ชั้นการแสดงออก (หัว/หู/ตา/งวง/หาง/LED) ที่จังหวะซึ่งท่าอ่านออกชัดที่สุด "
                             "— ตัวเลขใต้ภาพคือความยาวท่า ความสูงลำตัวจริงจาก IK และแรงบิดสูงสุด",
             **_tf(9.5, color="#a5a5a5"))

    col_w, row_h, y0 = 0.192, 0.232, 0.918
    for i, (gid, clip) in enumerate(clips):
        r, c = divmod(i, ncol)
        x = 0.035 + c * col_w
        top = y0 - r * row_h
        xc = x + 0.078
        k = peak_frame(clip)
        res = G.check(clip)
        color = GROUP_COLOR[clip.group]

        fig.text(xc, top, gid, ha="center", va="top", **_tf(11.5, color=color,
                                                            fontweight="bold"))
        fig.text(xc, top - 0.018, GROUP_NAME[clip.group], ha="center", va="top",
                 **_tf(7.5, color="#c2c2c2"))
        ax = fig.add_axes([x, top - 0.155, 0.156, 0.112])
        draw_face(ax, clip.expr[k], small=True)
        fig.text(xc, top - 0.163, clip.means, ha="center", va="top", **_tf(8.5, color="#7a7a7a"))
        bad = any(n.startswith("ERROR") for n in clip.notes)
        loop = " · loop" if clip.loop_from is not None else ""
        fig.text(xc, top - 0.190,
                 f"{clip.dur:.1f}s · สูง {clip.traj.body_T[k, 2, 3]:.0f}mm · "
                 f"tau {res['tau_max_kgcm']:.2f}{loop}"
                 + (" · !" if clip.notes else ""), ha="center", va="top",
                 **_tf(7.5, color="#d62728" if bad else ("#e8a33d" if clip.notes else "#b5b5b5")))
    fig.savefig(path, dpi=110, facecolor="white")
    print(f"saved -> {path}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gesture", default="notice_low", choices=G.ORDER, help="ท่าที่จะดู")
    ap.add_argument("--gif", metavar="PATH", help="บันทึกเป็น GIF แทนการเปิดหน้าต่าง")
    ap.add_argument("--sheet", metavar="PATH", help="ออกภาพรวมทั้งคลังเป็น PNG")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--cycles", type=int, default=1, help="เล่นซ้ำกี่รอบ (ท่าลูป)")
    args = ap.parse_args(argv)

    if args.sheet:
        sheet(args.sheet)
        return 0
    animate(args.gesture, gif=args.gif, fps=args.fps, cycles=args.cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
