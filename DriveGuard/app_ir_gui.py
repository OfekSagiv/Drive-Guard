#!/usr/bin/env python3
"""
DriveGuard IR — Desktop GUI  (CustomTkinter)
Modern dark-theme wrapper around the infer_live_ir pipeline.
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

from infer_live_ir import (
    build_models, run_live,
    DEFAULT_RTSP, PROC_WIDTH, CLASSES,
    _ensure_downloaded,
)

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('dark-blue')

# Per-class accent colours (hex RGB)
_CLS_COLOR = {
    0: '#FF8C00',   # Drink  — orange
    1: '#FF3B3B',   # Phone  — red
    2: '#00C864',   # Safe   — green
}
_CLS_BG = {
    0: '#3a2000',
    1: '#3a0000',
    2: '#003a1a',
}

# ──────────────────────────────────────────────────────────────────────────────

class ClassPanel(ctk.CTkFrame):
    """Right-side panel: big class name + three animated probability bars."""

    def __init__(self, master, **kwargs):
        super().__init__(master, corner_radius=12, fg_color='#1c1c1e', **kwargs)

        self._cls_lbl = ctk.CTkLabel(
            self, text='—', font=ctk.CTkFont(size=44, weight='bold'),
            text_color='#444444',
        )
        self._cls_lbl.pack(pady=(28, 0))

        self._conf_lbl = ctk.CTkLabel(
            self, text='', font=ctk.CTkFont(size=16),
            text_color='#666666',
        )
        self._conf_lbl.pack(pady=(4, 24))

        # Per-class progress bars
        self._bars: dict[str, ctk.CTkProgressBar] = {}
        self._pct_lbls: dict[str, ctk.CTkLabel] = {}

        for i, cls in enumerate(CLASSES):
            color = _CLS_COLOR[i]

            row = ctk.CTkFrame(self, fg_color='transparent')
            row.pack(fill='x', padx=18, pady=5)

            ctk.CTkLabel(
                row, text=cls, font=ctk.CTkFont(size=12, weight='bold'),
                text_color='#aaaaaa', width=52, anchor='w',
            ).pack(side='left')

            bar = ctk.CTkProgressBar(
                row, height=10, corner_radius=5,
                fg_color='#2a2a2a', progress_color=color,
            )
            bar.set(0)
            bar.pack(side='left', fill='x', expand=True, padx=(6, 8))

            pct = ctk.CTkLabel(
                row, text='0%', font=ctk.CTkFont(size=11),
                text_color='#666666', width=36, anchor='e',
            )
            pct.pack(side='left')

            self._bars[cls]     = bar
            self._pct_lbls[cls] = pct

        # Divider
        ctk.CTkFrame(self, height=1, fg_color='#2a2a2a').pack(
            fill='x', padx=18, pady=(18, 12))

        # Elapsed time
        elapsed_row = ctk.CTkFrame(self, fg_color='transparent')
        elapsed_row.pack(fill='x', padx=18, pady=(0, 4))
        ctk.CTkLabel(elapsed_row, text='Elapsed', font=ctk.CTkFont(size=10),
                     text_color='#555555', anchor='w').pack(side='left')
        self._elapsed_lbl = ctk.CTkLabel(elapsed_row, text='—',
                                         font=ctk.CTkFont(size=10, weight='bold'),
                                         text_color='#888888', anchor='e')
        self._elapsed_lbl.pack(side='right')

        # Session window counts per class
        self._count_lbls: dict[str, ctk.CTkLabel] = {}
        for i, cls in enumerate(CLASSES):
            row = ctk.CTkFrame(self, fg_color='transparent')
            row.pack(fill='x', padx=18, pady=1)
            ctk.CTkLabel(row, text=cls, font=ctk.CTkFont(size=10),
                         text_color='#444444', width=52, anchor='w').pack(side='left')
            lbl = ctk.CTkLabel(row, text='0 windows',
                               font=ctk.CTkFont(size=10), text_color='#444444', anchor='e')
            lbl.pack(side='right')
            self._count_lbls[cls] = lbl

        # Fusion detail rows (shown only when YOLOE prior is available)
        ctk.CTkFrame(self, height=1, fg_color='#2a2a2a').pack(
            fill='x', padx=18, pady=(8, 8))

        self._fusion_frame = ctk.CTkFrame(self, fg_color='transparent')
        self._fusion_frame.pack(fill='x', padx=18, pady=(0, 8))

        self._fusion_rows: dict[str, ctk.CTkLabel] = {}
        for row_name in ('Temporal', 'OD Prior', 'Fused'):
            lbl = ctk.CTkLabel(
                self._fusion_frame,
                text=f'{row_name}: —',
                font=ctk.CTkFont(family='Courier', size=10),
                text_color='#444444',
                anchor='w',
            )
            lbl.pack(fill='x', pady=1)
            self._fusion_rows[row_name] = lbl

    def update(self, cls_idx: int, probs: list, counts: dict = None,
               temporal_probs: list = None, prior: list = None,
               fused_probs: list = None, initialized: bool = True):
        # During init phase only update fusion rows to show "warming up…"
        if not initialized:
            for row_name in ('Temporal', 'OD Prior', 'Fused'):
                self._fusion_rows[row_name].configure(
                    text=f'{row_name}: warming up…', text_color='#444444')
            return

        cls_name = CLASSES[cls_idx]
        color    = _CLS_COLOR[cls_idx]
        conf     = probs[cls_idx]

        self._cls_lbl.configure(text=cls_name, text_color=color)
        self._conf_lbl.configure(text=f'{conf * 100:.1f}%', text_color=color)

        for i, cls in enumerate(CLASSES):
            self._bars[cls].set(probs[i])
            self._pct_lbls[cls].configure(text=f'{probs[i] * 100:.0f}%')

        if counts:
            total = sum(counts.values()) or 1
            for cls, n in counts.items():
                self._count_lbls[cls].configure(
                    text=f'{n}  ({n/total*100:.0f}%)',
                    text_color=_CLS_COLOR[CLASSES.index(cls)] if n > 0 else '#444444',
                )

        # Fusion detail rows
        def _fmt(p: list | None) -> str:
            if p is None:
                return '—'
            parts = [f'{CLASSES[i][0]} {p[i]*100:.0f}%' for i in range(len(CLASSES))]
            return '  '.join(parts)

        if temporal_probs is not None:
            self._fusion_rows['Temporal'].configure(
                text=f'Temporal: {_fmt(temporal_probs)}', text_color='#666666')
            self._fusion_rows['OD Prior'].configure(
                text=f'OD Prior: {_fmt(prior)}',
                text_color='#888888' if prior is not None else '#444444')
            self._fusion_rows['Fused'].configure(
                text=f'Fused:    {_fmt(fused_probs)}',
                text_color='#aaaaaa' if fused_probs is not None else '#444444')

    def set_elapsed(self, text: str):
        self._elapsed_lbl.configure(text=text)

    def reset(self):
        self._cls_lbl.configure(text='—', text_color='#444444')
        self._conf_lbl.configure(text='')
        for cls in CLASSES:
            self._bars[cls].set(0)
            self._pct_lbls[cls].configure(text='0%')
            self._count_lbls[cls].configure(text='0 windows', text_color='#444444')
        self._elapsed_lbl.configure(text='—')
        for row_name, lbl in self._fusion_rows.items():
            lbl.configure(text=f'{row_name}: —', text_color='#444444')


# ──────────────────────────────────────────────────────────────────────────────

class DriveGuardApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('DriveGuard IR')
        self.geometry('1100x680')
        self.minsize(780, 500)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        # Inference state
        self._running = False
        self._models  = None
        self._worker  = None

        # Thread-shared buffers
        self._lock           = threading.Lock()
        self._frame          = None
        self._probs          = None
        self._cls_idx        = None
        self._initialized    = False
        self._window_counts  = {cls: 0 for cls in CLASSES}
        self._temporal_probs: list | None = None
        self._prior:          list | None = None
        self._fused_probs:    list | None = None

        # Session timing (main-thread only)
        self._session_start: float | None = None

        # Alert state (main-thread only)
        self._alert_cls_start: float | None = None
        self._alert_last_cls:  int | None   = None
        self._alert_sounding   = False
        self._flash_on         = False

        self._photo = None   # keep ImageTk reference

        self._build_ui()
        self._poll()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, corner_radius=0, fg_color='#161618', height=56)
        top.pack(side='top', fill='x')
        top.pack_propagate(False)

        ctk.CTkLabel(
            top, text='DriveGuard', font=ctk.CTkFont(size=15, weight='bold'),
            text_color='#eeeeee',
        ).pack(side='left', padx=(16, 24), pady=14)

        ctk.CTkLabel(
            top, text='Source', font=ctk.CTkFont(size=11),
            text_color='#666666',
        ).pack(side='left')

        self._src_var = ctk.StringVar(value=DEFAULT_RTSP)
        self._src_entry = ctk.CTkEntry(
            top, textvariable=self._src_var, width=360,
            font=ctk.CTkFont(family='Courier', size=10),
            fg_color='#252527', border_color='#333333', corner_radius=8,
        )
        self._src_entry.pack(side='left', padx=(6, 16))

        ctk.CTkLabel(
            top, text='Width', font=ctk.CTkFont(size=11),
            text_color='#666666',
        ).pack(side='left')

        self._width_var = ctk.StringVar(value=str(PROC_WIDTH))
        ctk.CTkEntry(
            top, textvariable=self._width_var, width=56,
            font=ctk.CTkFont(size=11),
            fg_color='#252527', border_color='#333333', corner_radius=8,
        ).pack(side='left', padx=(6, 16))

        ctk.CTkLabel(
            top, text='Alert after', font=ctk.CTkFont(size=11),
            text_color='#666666',
        ).pack(side='left')

        self._alert_var = ctk.StringVar(value='3')
        ctk.CTkEntry(
            top, textvariable=self._alert_var, width=40,
            font=ctk.CTkFont(size=11),
            fg_color='#252527', border_color='#333333', corner_radius=8,
        ).pack(side='left', padx=(6, 2))

        ctk.CTkLabel(
            top, text='s', font=ctk.CTkFont(size=11),
            text_color='#666666',
        ).pack(side='left', padx=(0, 16))

        self._start_btn = ctk.CTkButton(
            top, text='▶  Start', width=100, height=32,
            corner_radius=8, font=ctk.CTkFont(size=12, weight='bold'),
            fg_color='#1a7a3a', hover_color='#156030',
            command=self._start,
        )
        self._start_btn.pack(side='left', padx=(0, 8))

        self._stop_btn = ctk.CTkButton(
            top, text='■  Stop', width=100, height=32,
            corner_radius=8, font=ctk.CTkFont(size=12, weight='bold'),
            fg_color='#3a3a3a', hover_color='#2a2a2a', state='disabled',
            command=self._stop,
        )
        self._stop_btn.pack(side='left')

        ctk.CTkLabel(
            top, text='Object Det.', font=ctk.CTkFont(size=11),
            text_color='#666666',
        ).pack(side='right', padx=(0, 4))

        self._fusion_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            top,
            text='',
            variable=self._fusion_var,
            onvalue=True,
            offvalue=False,
            width=40,
            height=20,
        ).pack(side='right', padx=(16, 4))

        self._status_lbl = ctk.CTkLabel(
            top, text='Idle', font=ctk.CTkFont(size=11),
            text_color='#555555',
        )
        self._status_lbl.pack(side='left', padx=16)

        # ── Main content ──────────────────────────────────────────────────────
        content = ctk.CTkFrame(self, corner_radius=0, fg_color='#0d0d0f')
        content.pack(side='top', fill='both', expand=True, padx=0, pady=0)

        # Video area
        self._video_lbl = ctk.CTkLabel(content, text='', fg_color='#0d0d0f')
        self._video_lbl.pack(side='left', fill='both', expand=True, padx=(12, 6), pady=12)

        # Right panel
        self._panel = ClassPanel(content, width=220)
        self._panel.pack(side='right', fill='y', padx=(6, 12), pady=12)

    # ── Controls ──────────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return
        self._running = True
        self._session_start = time.monotonic()
        with self._lock:
            self._window_counts = {cls: 0 for cls in CLASSES}
        self._start_btn.configure(state='disabled')
        self._stop_btn.configure(state='normal', fg_color='#7a1a1a', hover_color='#601515')
        self._src_entry.configure(state='disabled')
        self._set_status('Loading models…', '#ffaa00')
        fusion_on = self._fusion_var.get()
        self._worker = threading.Thread(target=self._worker_fn, args=(fusion_on,), daemon=True)
        self._worker.start()

    def _stop(self):
        self._running = False
        self._set_status('Stopping…', '#555555')
        def _join():
            if self._worker:
                self._worker.join(timeout=6)
            self.after(0, self._on_stopped)
        threading.Thread(target=_join, daemon=True).start()

    def _on_stopped(self):
        self._start_btn.configure(state='normal')
        self._stop_btn.configure(state='disabled', fg_color='#3a3a3a', hover_color='#2a2a2a')
        self._src_entry.configure(state='normal')
        self._set_status('Stopped', '#555555')
        self._session_start    = None
        self._alert_cls_start  = None
        self._alert_last_cls   = None
        self._alert_sounding   = False
        self._flash_on         = False
        self._panel.configure(fg_color='#1c1c1e')
        with self._lock:
            self._initialized = False
        self._panel.reset()

    def _play_alert(self):
        import subprocess
        subprocess.Popen(['afplay', '/System/Library/Sounds/Funk.aiff'])

    def _on_close(self):
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3)
        self.destroy()

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_fn(self, fusion_on: bool):
        try:
            source = self._src_var.get().strip()
            try:
                proc_w = int(self._width_var.get().strip())
            except ValueError:
                proc_w = PROC_WIDTH

            if self._models is None:
                sw = _ensure_downloaded('vit_spatial_model_v1.pth')
                tw = _ensure_downloaded('temporal_head_model.pth')
                self._models = build_models(sw, tw)

            run_live(
                source         = source,
                proc_width     = proc_w,
                models         = self._models,
                headless       = True,
                should_run     = lambda: self._running,
                on_frame       = self._stash_frame,
                on_prediction  = self._stash_pred,
                on_status      = self._set_status,
                enable_fusion  = fusion_on,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_status(f'Error: {e}', '#ff4444')
            self._running = False
        finally:
            self.after(0, self._on_stopped)

    # ── Worker callbacks (NO Tk/CTk calls here) ────────────────────────────────

    def _stash_frame(self, bgr: np.ndarray):
        with self._lock:
            self._frame = bgr

    def _stash_pred(self, cls_idx, temporal_probs, prior, fused_probs, initialized: bool):
        def _to_list(t):
            if t is None:
                return None
            return t.tolist() if hasattr(t, 'tolist') else list(t)
        with self._lock:
            self._initialized = initialized
            if cls_idx is None:
                # Warmup tick: propagate initialized=False without updating prediction state
                return
            self._cls_idx        = cls_idx
            self._probs          = _to_list(fused_probs if fused_probs is not None else temporal_probs)
            self._temporal_probs = _to_list(temporal_probs)
            self._prior          = _to_list(prior)
            self._fused_probs    = _to_list(fused_probs)
            if initialized:
                self._window_counts[CLASSES[cls_idx]] += 1

    def _set_status(self, text: str, color: str = '#555555'):
        self.after(0, lambda t=text, c=color: self._status_lbl.configure(
            text=t, text_color=c))

    # ── 30 fps main-thread refresh ────────────────────────────────────────────

    def _poll(self):
        with self._lock:
            frame         = self._frame
            probs         = self._probs
            ci            = self._cls_idx
            init          = self._initialized
            counts        = dict(self._window_counts)
            temporal_probs = self._temporal_probs
            prior          = self._prior
            fused_probs    = self._fused_probs

        if frame is not None:
            self._render_video(frame)
        if probs is not None:
            self._panel.update(ci, probs, counts,
                               temporal_probs=temporal_probs,
                               prior=prior,
                               fused_probs=fused_probs,
                               initialized=init)
        elif not init and self._running:
            self._panel.update(None, None, initialized=False)

        if self._session_start is not None:
            elapsed = time.monotonic() - self._session_start
            m = int(elapsed) // 60
            s = int(elapsed) % 60
            self._panel.set_elapsed(f'{m:02d}:{s:02d}')

        # Alert threshold — flash panel + sound when distraction held > N seconds
        now = time.monotonic()
        is_distraction = init and ci is not None and CLASSES[ci] != 'Safe'
        if is_distraction:
            if ci != self._alert_last_cls:
                self._alert_cls_start = now
                self._alert_last_cls  = ci
                self._alert_sounding  = False
            try:
                threshold = float(self._alert_var.get())
            except ValueError:
                threshold = 3.0
            held = now - (self._alert_cls_start or now)
            if held >= threshold:
                self._flash_on = not self._flash_on
                self._panel.configure(
                    fg_color='#3a0505' if self._flash_on else '#1c1c1e')
                if not self._alert_sounding:
                    self._alert_sounding = True
                    threading.Thread(target=self._play_alert, daemon=True).start()
            else:
                self._panel.configure(fg_color='#1c1c1e')
        else:
            if self._alert_last_cls is not None:
                self._alert_last_cls  = None
                self._alert_cls_start = None
                self._alert_sounding  = False
                self._flash_on        = False
                self._panel.configure(fg_color='#1c1c1e')

        self.after(33, self._poll)

    def _render_video(self, bgr: np.ndarray):
        lw = self._video_lbl.winfo_width()
        lh = self._video_lbl.winfo_height()
        h, w = bgr.shape[:2]

        if lw > 1 and lh > 1 and w > 0 and h > 0:
            scale = min(lw / w, lh / h)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            img = Image.fromarray(bgr[:, :, ::-1]).resize((nw, nh), Image.BILINEAR)
        else:
            img = Image.fromarray(bgr[:, :, ::-1])

        self._photo = ImageTk.PhotoImage(img)
        self._video_lbl.configure(image=self._photo)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = DriveGuardApp()
    app.mainloop()


if __name__ == '__main__':
    main()
