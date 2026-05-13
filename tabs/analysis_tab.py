"""
analysis_tab.py  –  Tab 3: Calculus Analysis (Gradient, Directional Derivative, Extrema).

Overlays gradient vector fields, directional derivative arrows, and
critical-point markers on a contour plot of f(x,y).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math_engine import (parse_function, parse_implicit_function, evaluate_safe,
                         numerical_gradient, numerical_gradient_scalar,
                         directional_derivative, find_critical_points, parse_domain_value)


# ─────────────────────────────────────────────────────────────────────────────
class AnalysisTab(ttk.Frame):
    """Calculus analysis tab: gradient field, directional derivative, critical points."""

    COLORMAPS = ['viridis', 'coolwarm', 'RdYlBu', 'Spectral', 'plasma']

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._show_field   = tk.BooleanVar(value=True)
        self._show_grad_pt = tk.BooleanVar(value=True)
        self._show_crit    = tk.BooleanVar(value=True)
        self._build_styles()
        self._build_ui()

    def _build_styles(self):
        s = ttk.Style()
        s.configure('An.TLabelframe', font=('Segoe UI', 10, 'bold'))
        s.configure('An.TLabel', font=('Segoe UI', 10))
        s.configure('An.TButton', font=('Segoe UI', 10))

    # ── Layout ───────────────────────────────────────────────────────────
    def _build_ui(self):
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # Left controls
        ctrl_outer = ttk.Frame(pw, width=340)

        # Fixed Analyze button at the top
        btn_bar = ttk.Frame(ctrl_outer)
        btn_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)
        ttk.Button(btn_bar, text='⟳  Analyze', command=self._do_analyze,
                   style='An.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_bar, text='Reset', command=self._reset_tab,
                   style='An.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Scrollable area
        scroll_frame = ttk.Frame(ctrl_outer)
        scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_ctrl = tk.Canvas(scroll_frame, width=320, highlightthickness=0)
        sb = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=canvas_ctrl.yview)
        self._ctrl = ttk.Frame(canvas_ctrl)
        self._ctrl.bind('<Configure>',
            lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox('all')))
        canvas_ctrl.create_window((0, 0), window=self._ctrl, anchor='nw')
        canvas_ctrl.configure(yscrollcommand=sb.set)
        canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._build_controls(self._ctrl)
        pw.add(ctrl_outer, weight=0)

        # Right: figure + info panel
        right = ttk.Frame(pw)
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Info text area
        self._info = tk.Text(right, height=6, font=('Consolas', 9),
                             bg='#1e1e1e', fg='#dcdcdc', wrap=tk.WORD,
                             state=tk.DISABLED, relief=tk.SUNKEN)
        self._info.pack(fill=tk.X, side=tk.BOTTOM, padx=2, pady=2)

        pw.add(right, weight=1)

        # Connect hover
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)

    def _build_controls(self, parent):
        row = 0
        # -- Expression --
        grp = ttk.LabelFrame(parent, text='Function  f(x, y)', style='An.TLabelframe')
        grp.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        ttk.Label(grp, text='f(x,y) =', style='An.TLabel').grid(row=0, column=0, sticky='w', padx=4)
        self._expr_var = tk.StringVar(value='x**2 - y**2')
        ttk.Entry(grp, textvariable=self._expr_var, width=28, font=('Consolas', 10)).grid(
            row=0, column=1, padx=4, pady=3, sticky='ew')

        row += 1
        # -- Point --
        pt = ttk.LabelFrame(parent, text='Evaluation Point', style='An.TLabelframe')
        pt.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._x0_var = tk.StringVar(value='1')
        self._y0_var = tk.StringVar(value='1')
        ttk.Label(pt, text='x₀', style='An.TLabel').grid(row=0, column=0, padx=4, sticky='w')
        ttk.Entry(pt, textvariable=self._x0_var, width=8, font=('Consolas', 10)).grid(
            row=0, column=1, padx=4, pady=1, sticky='w')
        ttk.Label(pt, text='y₀', style='An.TLabel').grid(row=1, column=0, padx=4, sticky='w')
        ttk.Entry(pt, textvariable=self._y0_var, width=8, font=('Consolas', 10)).grid(
            row=1, column=1, padx=4, pady=1, sticky='w')

        row += 1
        # -- Direction --
        dr = ttk.LabelFrame(parent, text='Direction (dx, dy)', style='An.TLabelframe')
        dr.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._dx_var = tk.StringVar(value='1')
        self._dy_var = tk.StringVar(value='0')
        ttk.Label(dr, text='dx', style='An.TLabel').grid(row=0, column=0, padx=4, sticky='w')
        ttk.Entry(dr, textvariable=self._dx_var, width=8, font=('Consolas', 10)).grid(
            row=0, column=1, padx=4, pady=1, sticky='w')
        ttk.Label(dr, text='dy', style='An.TLabel').grid(row=1, column=0, padx=4, sticky='w')
        ttk.Entry(dr, textvariable=self._dy_var, width=8, font=('Consolas', 10)).grid(
            row=1, column=1, padx=4, pady=1, sticky='w')

        ttk.Label(dr, text='or angle θ°', style='An.TLabel').grid(row=2, column=0, padx=4, sticky='w')
        self._angle_var = tk.StringVar(value='')
        ttk.Entry(dr, textvariable=self._angle_var, width=8, font=('Consolas', 10)).grid(
            row=2, column=1, padx=4, pady=1, sticky='w')

        row += 1
        # -- Ranges --
        rng = ttk.LabelFrame(parent, text='Domain', style='An.TLabelframe')
        rng.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._xmin = tk.StringVar(value='-5')
        self._xmax = tk.StringVar(value='5')
        self._ymin = tk.StringVar(value='-5')
        self._ymax = tk.StringVar(value='5')
        for i, (lbl, v) in enumerate([('x min', self._xmin), ('x max', self._xmax),
                                       ('y min', self._ymin), ('y max', self._ymax)]):
            ttk.Label(rng, text=lbl, style='An.TLabel').grid(row=i, column=0, sticky='w', padx=4)
            ttk.Entry(rng, textvariable=v, width=8, font=('Consolas', 10)).grid(
                row=i, column=1, padx=4, pady=1, sticky='w')

        row += 1
        # -- Display toggles --
        tog = ttk.LabelFrame(parent, text='Display Options', style='An.TLabelframe')
        tog.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        ttk.Checkbutton(tog, text='Gradient Field', variable=self._show_field).pack(
            anchor='w', padx=4)
        ttk.Checkbutton(tog, text='Gradient at Point', variable=self._show_grad_pt).pack(
            anchor='w', padx=4)
        ttk.Checkbutton(tog, text='Critical Points', variable=self._show_crit).pack(
            anchor='w', padx=4)

        row += 1
        # -- Colormap --
        cm = ttk.LabelFrame(parent, text='Colormap', style='An.TLabelframe')
        cm.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._cmap_var = tk.StringVar(value='coolwarm')
        ttk.Combobox(cm, textvariable=self._cmap_var, values=self.COLORMAPS,
                     state='readonly').pack(fill=tk.X, padx=4, pady=3)

        # (Analyze button is in the fixed bar above the scroll area)

    def _reset_tab(self):
        self._expr_var.set('x**2 - y**2')
        self._x0_var.set('1')
        self._y0_var.set('1')
        self._dx_var.set('1')
        self._dy_var.set('0')
        self._angle_var.set('')
        self._xmin.set('-5')
        self._xmax.set('5')
        self._ymin.set('-5')
        self._ymax.set('5')
        self._show_field.set(True)
        self._show_grad_pt.set(True)
        self._show_crit.set(True)
        self._cmap_var.set('coolwarm')
        self.fig.clear()
        self.canvas.draw_idle()
        self._info.configure(state=tk.NORMAL)
        self._info.delete('1.0', tk.END)
        self._info.configure(state=tk.DISABLED)

    # ── Analysis ─────────────────────────────────────────────────────────
    def _do_analyze(self, *_):
        expr = self._expr_var.get().strip()
        if not expr:
            return
        try:
            func, is_implicit = parse_implicit_function(expr, ('x', 'y'))
        except Exception as exc:
            messagebox.showerror('Parse Error', str(exc))
            return

        try:
            xmin = parse_domain_value(self._xmin.get());  xmax = parse_domain_value(self._xmax.get())
            ymin = parse_domain_value(self._ymin.get());  ymax = parse_domain_value(self._ymax.get())
            x0 = parse_domain_value(self._x0_var.get());  y0 = parse_domain_value(self._y0_var.get())
            dx = parse_domain_value(self._dx_var.get());  dy = parse_domain_value(self._dy_var.get())
        except Exception:
            messagebox.showerror('Input Error', 'Enter valid numeric expressions.')
            return

        # Override direction from angle if provided
        angle_str = self._angle_var.get().strip()
        if angle_str:
            try:
                theta = parse_domain_value(angle_str) * np.pi / 180
                dx, dy = np.cos(theta), np.sin(theta)
            except Exception:
                pass

        res = 80
        x = np.linspace(xmin, xmax, res)
        y = np.linspace(ymin, ymax, res)
        X, Y = np.meshgrid(x, y)
        Z = evaluate_safe(func, X, Y)

        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)

        # Background contour
        try:
            cf = ax.contourf(X, Y, Z, levels=30, cmap=self._cmap_var.get(), alpha=0.7)
            self.fig.colorbar(cf, ax=ax, shrink=0.8)
            ax.contour(X, Y, Z, levels=15, colors='k', alpha=0.3, linewidths=0.5)
        except Exception:
            pass

        # Implicit curve overlay
        if is_implicit:
            try:
                ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2.5)
            except Exception:
                pass

        # Gradient field
        if self._show_field.get():
            skip = max(1, res // 16)
            Xs = X[::skip, ::skip]
            Ys = Y[::skip, ::skip]
            dFdx, dFdy = numerical_gradient(func, Xs, Ys)
            mag = np.sqrt(dFdx**2 + dFdy**2) + 1e-30
            ax.quiver(Xs, Ys, dFdx / mag, dFdy / mag, mag,
                      cmap='autumn', alpha=0.7, scale=30, width=0.004)

        # Gradient at point
        grad = numerical_gradient_scalar(func, x0, y0)
        if self._show_grad_pt.get():
            scale = (xmax - xmin) / 10
            ax.annotate('', xy=(x0 + grad[0] * scale / (np.linalg.norm(grad) + 1e-30),
                                y0 + grad[1] * scale / (np.linalg.norm(grad) + 1e-30)),
                        xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='lime', lw=2.5))
            ax.plot(x0, y0, 'o', color='lime', ms=7, zorder=5)
            ax.annotate(f'∇f = ({grad[0]:.3f}, {grad[1]:.3f})',
                        xy=(x0, y0), xytext=(8, 8), textcoords='offset points',
                        fontsize=8, color='white',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7))

        # Directional derivative arrow
        u = np.array([dx, dy], dtype=float)
        u_hat = u / (np.linalg.norm(u) + 1e-30)
        dd_val = directional_derivative(func, x0, y0, (dx, dy))
        dd_scale = (xmax - xmin) / 12
        ax.annotate('', xy=(x0 + u_hat[0] * dd_scale, y0 + u_hat[1] * dd_scale),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=2, ls='--'))

        # Critical points
        crit_info = []
        self._crit_annotations = []
        self._hover_threshold = np.hypot(xmax - xmin, ymax - ymin) * 0.03

        if self._show_crit.get():
            cps = find_critical_points(func, (xmin, xmax), (ymin, ymax), n_starts=8)
            markers = {'local_min': ('o', '#00ff00', 'Min'),
                       'local_max': ('^', '#ff3333', 'Max'),
                       'saddle':    ('D', '#ffcc00', 'Saddle'),
                       'inconclusive': ('s', '#aaaaaa', '?')}
            for cp in cps:
                mk, color, label = markers.get(cp['type'], ('s', 'gray', '?'))
                ax.plot(cp['x'], cp['y'], mk, color=color, ms=10, markeredgecolor='k',
                        markeredgewidth=1.5, zorder=6)
                ann = ax.annotate(f'{label}\n({cp["x"]:.2f}, {cp["y"]:.2f})\nf={cp["z"]:.3f}',
                                  xy=(cp['x'], cp['y']), xytext=(12, 12),
                                  textcoords='offset points', fontsize=7, color='white',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7),
                                  zorder=10)
                ann.set_visible(False)
                self._crit_annotations.append((cp['x'], cp['y'], ann))
                crit_info.append(cp)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Analysis of  f(x,y) = {expr}')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        # Update info panel
        fval = evaluate_safe(func, np.array([x0]), np.array([y0]))[0]
        info_lines = [
            f'f({x0}, {y0}) = {fval:.6f}',
            f'∇f({x0}, {y0}) = ({grad[0]:.6f}, {grad[1]:.6f})',
            f'|∇f| = {np.linalg.norm(grad):.6f}',
            f'Direction u = ({dx:.4f}, {dy:.4f})',
            f'D_u f({x0}, {y0}) = {dd_val:.6f}',
        ]
        if crit_info:
            info_lines.append('')
            info_lines.append('── Critical Points ──')
            for cp in crit_info:
                info_lines.append(
                    f'  ({cp["x"]:.4f}, {cp["y"]:.4f})  f = {cp["z"]:.4f}  [{cp["type"]}]')

        self._info.configure(state=tk.NORMAL)
        self._info.delete('1.0', tk.END)
        self._info.insert('1.0', '\n'.join(info_lines))
        self._info.configure(state=tk.DISABLED)

    # ── Hover ────────────────────────────────────────────────────────────
    def _on_hover(self, event):
        if event.inaxes is None or not hasattr(self, '_crit_annotations'):
            return
        
        xm, ym = event.xdata, event.ydata
        if xm is None or ym is None:
            return
            
        redraw = False
        for cx, cy, ann in self._crit_annotations:
            dist = np.hypot(cx - xm, cy - ym)
            is_near = dist < self._hover_threshold
            if ann.get_visible() != is_near:
                ann.set_visible(is_near)
                redraw = True
                
        if redraw:
            self.canvas.draw_idle()
