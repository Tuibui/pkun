#!/usr/bin/env python3
"""
gesture_gui.py — หน้าต่างคลิกเลือกท่าทางของ P-kun แล้วดูการเคลื่อนไหวทันที

ซ้าย  = ปุ่มท่าทั้ง 20 ท่า จัดกลุ่มตามเอกสาร (สีหัวข้อ = กลุ่ม)
ขวา   = หุ่นจริงจาก IK + หน้า/หู/ตา/หาง/LED + แถบค่าทุกช่อง
ล่าง  = เล่น/หยุด, เลื่อนดูทีละเฟรม, วนซ้ำ, ปรับความเร็ว

รัน:  python3 gesture_gui.py
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Sequence

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from leg_kinematics import PARAMS  # noqa: E402

import gestures as G  # noqa: E402
import show_gesture as S  # noqa: E402

BG = "#ffffff"
PANEL = "#f4f6f7"
INK = "#3a3a3a"
MUTED = "#8f8f8f"

GROUPS: Sequence[tuple[str, str]] = (
    ("lifecycle", "1. วงจรชีวิต"),
    ("attention", "2. แจ้งเตือน & โฟกัส"),
    ("conversation", "3. ลูปสนทนา"),
    ("social", "4. บุคลิก & สังคม"),
    ("system", "5. ระบบ"),
)


def _font(size: int, weight: str = "normal") -> tuple:
    """ฟอนต์ที่มีอักษรไทย — ถ้าไม่มีตัวไหนเลยให้ Tk เลือกเองตามปกติ."""
    fam = S.THAI or "TkDefaultFont"
    return (fam, size, weight)


class GestureGUI:
    """ตัวเล่นท่าทางแบบคลิกเลือก.

    เฟรมถูกเดินด้วย root.after ไม่ใช่ FuncAnimation เพราะต้องหยุด/กระโดดตามสไลเดอร์ได้
    และ 'สร้างคลิปครั้งเดียวแล้วเก็บไว้' (แคช) เพราะการสร้างท่าหนึ่งต้องแก้ IK ทุกเฟรม
    ท่ายาว 24 วินาที = 1200 เฟรม กดปุ่มซ้ำแล้วสร้างใหม่ทุกครั้งจะหน่วงจนรู้สึกได้
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.cache: Dict[str, G.Clip] = {}
        self.clip: G.Clip | None = None
        self.k = 0
        self.playing = False
        self.fps = 20
        self.speed = 1.0
        self.after_id: str | None = None
        self.current = ""

        root.title("P-kun — คลังท่าทาง")
        root.configure(bg=BG)

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        self._build_canvas(body)
        self._build_controls(root)
        self._build_keys(root)

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.select("idle_breathe")

    def _build_keys(self, root: tk.Tk) -> None:
        """คีย์ลัด: เว้นวรรค = เล่น/หยุด, ซ้าย-ขวา = เดินทีละเฟรม, ขึ้น-ลง = เปลี่ยนท่า."""
        root.bind("<space>", lambda _e: self.toggle())
        root.bind("<Left>", lambda _e: self.nudge(-1))
        root.bind("<Right>", lambda _e: self.nudge(+1))
        root.bind("<Up>", lambda _e: self.hop(-1))
        root.bind("<Down>", lambda _e: self.hop(+1))
        root.bind("<Escape>", lambda _e: self.close())

    def nudge(self, d: int) -> None:
        """เดินทีละเฟรม (หยุดเล่นก่อน) — ใช้ตรวจจังหวะที่สงสัยว่าขากระตุกไหม."""
        if self.clip is None:
            return
        self.playing = False
        self.play_btn.configure(text="▶ เล่น")
        self.slider.set(int(np.clip(self.k + d, 0, len(self.clip.t) - 1)))

    def hop(self, d: int) -> None:
        """ข้ามไปท่าถัดไป/ก่อนหน้าในลำดับเดียวกับเอกสาร."""
        gids = list(G.ORDER)
        cur = gids.index(self.current) if self.current in gids else 0
        self.select(gids[(cur + d) % len(gids)])

    def close(self) -> None:
        """ปิดให้เรียบร้อย: ต้องยกเลิกทั้งจังหวะเล่นของเราเองและคิววาดค้างของ matplotlib.

        ถ้าไม่ยกเลิก draw_idle ที่ค้างอยู่ Tk จะพยายามเรียกฟังก์ชันวาดหลังหน้าต่างถูกทำลายแล้ว
        แล้วพ่น 'invalid command name ...idle_draw' ออกมาตอนปิดโปรแกรม
        """
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        idle = getattr(self.canvas, "_idle_draw_id", None)
        if idle:
            try:
                self.canvas.get_tk_widget().after_cancel(idle)
            except tk.TclError:
                pass
            self.canvas._idle_draw_id = None
        self.root.destroy()

    # ---------------- โครงหน้าต่าง ----------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        """แถบเลือกท่า — ต้องเลื่อนได้ เพราะ 20 ปุ่ม + 5 หัวข้อ ยาวเกินความสูงหน้าต่างปกติ."""
        outer = tk.Frame(parent, bg=PANEL, width=222)
        outer.pack(side="left", fill="y")
        outer.pack_propagate(False)

        scroller = tk.Canvas(outer, bg=PANEL, highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=scroller.yview)
        side = tk.Frame(scroller, bg=PANEL)
        scroller.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        scroller.pack(side="left", fill="both", expand=True)
        scroller.create_window((0, 0), window=side, anchor="nw", width=204)
        side.bind("<Configure>",
                  lambda _e: scroller.configure(scrollregion=scroller.bbox("all")))

        def wheel(ev: tk.Event) -> None:
            scroller.yview_scroll(-1 if ev.num == 4 or getattr(ev, "delta", 0) > 0 else 1,
                                  "units")

        # ผูกล้อเมาส์เฉพาะตอนเคอร์เซอร์อยู่เหนือแถบ ไม่งั้นจะไปแย่งล้อของแผงกราฟ
        outer.bind("<Enter>", lambda _e: [scroller.bind_all(s, wheel)
                                          for s in ("<Button-4>", "<Button-5>",
                                                    "<MouseWheel>")])
        outer.bind("<Leave>", lambda _e: [scroller.unbind_all(s)
                                          for s in ("<Button-4>", "<Button-5>",
                                                    "<MouseWheel>")])

        tk.Label(side, text="เลือกท่า", bg=PANEL, fg=INK, font=_font(13, "bold"),
                 anchor="w").pack(fill="x", padx=12, pady=(12, 2))
        tk.Label(side, text="[แกน] = ชุดที่เอกสารให้ทำก่อน · ↑↓ เปลี่ยนท่า", bg=PANEL,
                 fg=MUTED, font=_font(8), anchor="w").pack(fill="x", padx=12, pady=(0, 8))

        self.buttons: Dict[str, tk.Button] = {}
        for group, title in GROUPS:
            color = S.GROUP_COLOR[group]
            tk.Label(side, text=title, bg=PANEL, fg=color, font=_font(9, "bold"),
                     anchor="w").pack(fill="x", padx=12, pady=(8, 2))
            for gid in G.ORDER:
                if G.META[gid]["group"] != group:
                    continue
                label = gid + ("  [แกน]" if G.META[gid].get("core") else "")
                btn = tk.Button(side, text=label, anchor="w", relief="flat", bd=0,
                                bg=PANEL, fg=INK, activebackground="#e3e8ea",
                                font=_font(9), padx=10, pady=3,
                                command=lambda g=gid: self.select(g))
                btn.pack(fill="x", padx=8)
                self.buttons[gid] = btn

    def _build_canvas(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.fig = plt.figure(figsize=(11.4, 5.0), facecolor=BG)
        self.ax = self.fig.add_axes([0.00, 0.03, 0.50, 0.80], projection="3d")
        self.axf = self.fig.add_axes([0.52, 0.16, 0.26, 0.66])
        self.axc = self.fig.add_axes([0.80, 0.12, 0.18, 0.74])
        self.title = self.fig.text(0.02, 0.975, "", va="top", ha="left",
                                   **S._tf(14, color="#6b6b6b", fontweight="bold"))
        self.subtitle = self.fig.text(0.02, 0.925, "", va="top", ha="left",
                                      **S._tf(9, color="#9a9a9a"))
        self.stats = self.fig.text(0.52, 0.075, "", va="top", ha="left",
                                   **S._tf(8.5, color="#9a9a9a"))
        self.warn = self.fig.text(0.52, 0.030, "", va="top", ha="left",
                                  **S._tf(8.5, color="#d62728"))

        self.view = S.RobotView(self.ax, PARAMS)
        self.face = S.FaceView(self.axf)
        self.chan = S.ChannelView(self.axc)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_controls(self, parent: tk.Tk) -> None:
        bar = tk.Frame(parent, bg=PANEL, height=44)
        bar.pack(fill="x", side="bottom")

        self.play_btn = tk.Button(bar, text="■ หยุด", width=9, relief="flat", bd=0,
                                  bg="#2196a8", fg="white", activebackground="#1b7f8e",
                                  font=_font(10, "bold"), command=self.toggle)
        self.play_btn.pack(side="left", padx=(12, 8), pady=8)

        tk.Button(bar, text="⟲ เริ่มใหม่", width=10, relief="flat", bd=0, bg=PANEL,
                  fg=INK, font=_font(10), command=self.restart).pack(side="left", padx=4)

        self.loop_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="วนซ้ำ", variable=self.loop_var, bg=PANEL, fg=INK,
                       activebackground=PANEL, selectcolor=BG, font=_font(10),
                       bd=0, highlightthickness=0).pack(side="left", padx=10)

        tk.Label(bar, text="ความเร็ว", bg=PANEL, fg=MUTED,
                 font=_font(9)).pack(side="left", padx=(12, 4))
        self.speed_box = ttk.Combobox(bar, width=6, state="readonly", font=_font(9),
                                      values=("0.25x", "0.5x", "1x", "2x"))
        self.speed_box.set("1x")
        self.speed_box.bind("<<ComboboxSelected>>", self._on_speed)
        self.speed_box.pack(side="left")

        self.slider = tk.Scale(bar, from_=0, to=1, orient="horizontal", showvalue=False,
                               bg=PANEL, fg=INK, troughcolor="#dfe4e6", bd=0,
                               highlightthickness=0, sliderrelief="flat", length=340,
                               command=self._on_slider)
        self.slider.pack(side="left", padx=14, fill="x", expand=True)

        self.time_lbl = tk.Label(bar, text="0.00 s", bg=PANEL, fg=MUTED, width=9,
                                 font=_font(9))
        self.time_lbl.pack(side="left", padx=(0, 12))

    # ---------------- การทำงาน ----------------

    def select(self, gid: str) -> None:
        """โหลดท่า (สร้างครั้งแรกครั้งเดียว) แล้วเริ่มเล่นตั้งแต่ต้น."""
        for g, btn in self.buttons.items():
            btn.configure(bg="#dbe6e9" if g == gid else PANEL,
                          fg=S.GROUP_COLOR[str(G.META[g]["group"])] if g == gid else INK)
        if gid not in self.cache:
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            clip = G.build(gid, PARAMS)
            G.check(clip)                      # เติม clip.notes ให้ครบก่อนเอาไปโชว์
            self.cache[gid] = clip
            self.root.config(cursor="")
        self.clip = self.cache[gid]
        self.current = gid
        self.reach, self.z_top = S.view_range(self.clip)
        self.view.set_scale(self.reach, self.z_top)
        self.res = G.check(self.clip)

        self.title.set_text(f"{gid} — {self.clip.means}")
        self.subtitle.set_text(f"เมื่อไร: {self.clip.when}")
        T = self.clip.traj.body_T
        self.stats.set_text(
            f"ลำตัว: สูง {T[:, 2, 3].min():.0f}–{T[:, 2, 3].max():.0f} mm    "
            f"ขาเร็วสุด {self.res['peak_rate_dps']:.0f}°/s    "
            f"tau สูงสุด {self.res['tau_max_kgcm']:.2f} kg·cm "
            f"(งบ {self.res['torque_budget_kgcm']:.2f})    "
            f"margin ต่ำสุด {self.res['min_margin_mm']:.1f} mm")
        self.warn.set_text(" | ".join(self.clip.notes))

        self.slider.configure(to=len(self.clip.t) - 1)
        self.k = 0
        self.playing = True
        self.play_btn.configure(text="■ หยุด")
        self.render()
        self._schedule()

    def render(self) -> None:
        clip = self.clip
        if clip is None:
            return
        self.view.update(clip, self.k)
        self.face.update(clip.expr[self.k])
        text, looping = S.frame_title(clip, self.k)
        self.axf.set_title(text, **S._tf(9.5, color=S.TEAL if looping else "#111111"))
        self.chan.update(clip.expr[self.k])
        self.canvas.draw_idle()
        self.time_lbl.configure(text=f"{clip.t[self.k]:.2f} s")

    def _step(self) -> int:
        """ข้ามกี่เฟรมต่อการวาดหนึ่งครั้ง — คิดจากอัตราคำสั่งจริง 50 Hz หารด้วย fps ที่วาดไหว."""
        return max(1, int(round(G.HZ * self.speed / self.fps)))

    def tick(self) -> None:
        if not self.playing or self.clip is None:
            return
        n = len(self.clip.t)
        nxt = self.k + self._step()
        if nxt >= n:
            if not self.loop_var.get():
                self.playing = False
                self.play_btn.configure(text="▶ เล่น")
                self.k = n - 1
                self.render()
                return
            # ท่าที่มี loop_from ให้วนเฉพาะช่วงท้าย ไม่ต้องเล่นช่วง 'เข้าท่า' ซ้ำทุกรอบ
            start = 0
            if self.clip.loop_from is not None:
                start = int(round(self.clip.loop_from * G.HZ))
            nxt = min(start, n - 1)
        self.k = nxt
        self.slider.set(self.k)             # จะไปเรียก _on_slider ซึ่งวาดให้เอง
        self._schedule()

    def _schedule(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(int(1000 / self.fps), self.tick)

    def toggle(self) -> None:
        self.playing = not self.playing
        self.play_btn.configure(text="■ หยุด" if self.playing else "▶ เล่น")
        if self.playing:
            self._schedule()

    def restart(self) -> None:
        self.k = 0
        self.slider.set(0)
        self.playing = True
        self.play_btn.configure(text="■ หยุด")
        self._schedule()

    def _on_slider(self, val: str) -> None:
        if self.clip is None:
            return
        k = int(float(val))
        if k != self.k:
            self.k = k          # ผู้ใช้ลากเอง: ไม่หยุดเล่น แค่กระโดดไปเฟรมนั้น
        self.render()

    def _on_speed(self, _event=None) -> None:
        self.speed = float(self.speed_box.get().rstrip("x"))


def main(argv: Sequence[str] | None = None) -> int:
    root = tk.Tk()
    root.geometry("1360x740")
    root.minsize(1060, 560)
    GestureGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
