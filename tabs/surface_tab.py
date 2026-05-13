"""
surface_tab.py  –  Tab 1: f(x,y) Surface Plot / Level Curves / Contour Plots.

Supports:
  • Multiple simultaneous functions
  • Surface, level-curve, contour, and combined views
  • Interactive x/y sliders for cross-section slices
  • Hover readout of function value
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math_engine import parse_function, parse_implicit_function, evaluate_safe, parse_domain_value


# ─────────────────────────────────────────────────────────────────────────────
class SurfaceTab(ttk.Frame):
    """f(x,y) visualizer tab with surface plots, level curves, and contours."""

    COLORMAPS = ['viridis', 'plasma', 'inferno', 'coolwarm', 'RdYlBu', 'Spectral']

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.functions = []          # list of dicts: {name, expr, func}
        self._cmap_idx = 0
        self._hover_data = []        # initialise for hover handler
        self._plot_type = tk.StringVar(value='surface')
        self._slice_axis = tk.StringVar(value='none')
        self._build_styles()
        self._build_ui()

    # ── Styles ───────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style()
        s.configure('Ctrl.TLabelframe', font=('Segoe UI', 10, 'bold'))
        s.configure('Ctrl.TLabel', font=('Segoe UI', 10))
        s.configure('Ctrl.TButton', font=('Segoe UI', 10))
        s.configure('Ctrl.TRadiobutton', font=('Segoe UI', 10))

    # ── Main layout ──────────────────────────────────────────────────────
    def _build_ui(self):
        # Split: left controls | right canvas
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # ---- Left panel: scrollable controls + fixed Plot button ----
        ctrl_outer = ttk.Frame(pw, width=340)

        # Fixed Plot button at the top — always visible
        btn_bar = ttk.Frame(ctrl_outer)
        btn_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)
        ttk.Button(btn_bar, text='⟳  Plot', command=self._do_plot,
                   style='Ctrl.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_bar, text='Reset', command=self._reset_tab,
                   style='Ctrl.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Scrollable area for the rest of the controls
        scroll_frame = ttk.Frame(ctrl_outer)
        scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_ctrl = tk.Canvas(scroll_frame, width=320, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=canvas_ctrl.yview)
        self._ctrl_frame = ttk.Frame(canvas_ctrl)

        self._ctrl_frame.bind('<Configure>',
            lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox('all')))
        canvas_ctrl.create_window((0, 0), window=self._ctrl_frame, anchor='nw')
        canvas_ctrl.configure(yscrollcommand=scrollbar.set)

        canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            canvas_ctrl.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas_ctrl.bind_all('<MouseWheel>', _on_mousewheel)

        self._build_controls(self._ctrl_frame)
        pw.add(ctrl_outer, weight=0)

        # ---- Right: matplotlib canvas ----
        fig_frame = ttk.Frame(pw)
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar for hover info
        self._status_var = tk.StringVar(value='')
        status_bar = ttk.Label(fig_frame, textvariable=self._status_var,
                               relief=tk.SUNKEN, anchor=tk.W, font=('Consolas', 9))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        pw.add(fig_frame, weight=1)

        # Connect hover
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)

    # ── Controls ─────────────────────────────────────────────────────────
    def _build_controls(self, parent):
        row = 0
        # -- Function entry --
        grp = ttk.LabelFrame(parent, text='Function  f(x, y)', style='Ctrl.TLabelframe')
        grp.grid(row=row, column=0, sticky='ew', padx=5, pady=4)

        ttk.Label(grp, text='f(x,y) =', style='Ctrl.TLabel').grid(row=0, column=0, sticky='w', padx=4)
        self._expr_var = tk.StringVar(value='x**2 + y**2')
        ttk.Entry(grp, textvariable=self._expr_var, width=30, font=('Consolas', 10)).grid(
            row=0, column=1, padx=4, pady=3, sticky='ew')

        btn_frame = ttk.Frame(grp)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Button(btn_frame, text='Add', command=self._add_function, style='Ctrl.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text='Remove', command=self._remove_function, style='Ctrl.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text='Update', command=self._update_function, style='Ctrl.TButton').pack(
            side=tk.LEFT, padx=3)

        row += 1
        # -- Function list --
        lf = ttk.LabelFrame(parent, text='Function List', style='Ctrl.TLabelframe')
        lf.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._func_listbox = tk.Listbox(lf, height=5, font=('Consolas', 9),
                                        selectmode=tk.SINGLE)
        self._func_listbox.pack(fill=tk.X, padx=4, pady=3)
        self._func_listbox.bind('<<ListboxSelect>>', self._on_func_select)

        row += 1
        # -- Ranges --
        rng = ttk.LabelFrame(parent, text='Domain', style='Ctrl.TLabelframe')
        rng.grid(row=row, column=0, sticky='ew', padx=5, pady=4)

        self._xmin_var = tk.StringVar(value='-5')
        self._xmax_var = tk.StringVar(value='5')
        self._ymin_var = tk.StringVar(value='-5')
        self._ymax_var = tk.StringVar(value='5')

        for i, (lbl, var) in enumerate([
            ('x min', self._xmin_var), ('x max', self._xmax_var),
            ('y min', self._ymin_var), ('y max', self._ymax_var),
        ]):
            ttk.Label(rng, text=lbl, style='Ctrl.TLabel').grid(row=i, column=0, sticky='w', padx=4)
            ttk.Entry(rng, textvariable=var, width=8, font=('Consolas', 10)).grid(
                row=i, column=1, padx=4, pady=1, sticky='w')

        row += 1
        # -- Resolution --
        res_f = ttk.LabelFrame(parent, text='Resolution', style='Ctrl.TLabelframe')
        res_f.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._res_var = tk.IntVar(value=80)
        ttk.Scale(res_f, from_=20, to=200, variable=self._res_var,
                  orient=tk.HORIZONTAL, command=lambda _: self._res_label.config(
                      text=str(self._res_var.get()))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._res_label = ttk.Label(res_f, text='80', style='Ctrl.TLabel', width=4)
        self._res_label.pack(side=tk.RIGHT, padx=4)

        row += 1
        # -- Plot type --
        pt = ttk.LabelFrame(parent, text='Plot Type', style='Ctrl.TLabelframe')
        pt.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        for val, lbl in [('surface', 'Surface'),
                         ('contourf', 'Level Curves (Filled)'),
                         ('contour', 'Contour Lines'),
                         ('combined', 'All Three')]:
            ttk.Radiobutton(pt, text=lbl, variable=self._plot_type, value=val,
                            style='Ctrl.TRadiobutton').pack(anchor='w', padx=4)

        row += 1
        # -- Slice controls --
        sl = ttk.LabelFrame(parent, text='Slice Controls', style='Ctrl.TLabelframe')
        sl.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        for val, lbl in [('none', 'None'), ('x', 'Slice along x'),
                         ('y', 'Slice along y'), ('vec', 'Slice along û')]:
            ttk.Radiobutton(sl, text=lbl, variable=self._slice_axis, value=val,
                            style='Ctrl.TRadiobutton').pack(anchor='w', padx=4)

        # x₀ slider + entry
        x_row = ttk.Frame(sl)
        x_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(x_row, text='x₀ :', style='Ctrl.TLabel').pack(side=tk.LEFT)
        self._x_entry_var = tk.StringVar(value='0.00')
        x_entry = ttk.Entry(x_row, textvariable=self._x_entry_var, width=7,
                            font=('Consolas', 9))
        x_entry.pack(side=tk.RIGHT, padx=2)
        x_entry.bind('<Return>', self._on_x_entry)
        x_entry.bind('<FocusOut>', self._on_x_entry)
        self._x_slider = ttk.Scale(sl, from_=-5, to=5, orient=tk.HORIZONTAL,
                                    command=self._on_x_slider)
        self._x_slider.set(0)
        self._x_slider.pack(fill=tk.X, padx=4, pady=1)

        # y₀ slider + entry
        y_row = ttk.Frame(sl)
        y_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(y_row, text='y₀ :', style='Ctrl.TLabel').pack(side=tk.LEFT)
        self._y_entry_var = tk.StringVar(value='0.00')
        y_entry = ttk.Entry(y_row, textvariable=self._y_entry_var, width=7,
                            font=('Consolas', 9))
        y_entry.pack(side=tk.RIGHT, padx=2)
        y_entry.bind('<Return>', self._on_y_entry)
        y_entry.bind('<FocusOut>', self._on_y_entry)
        self._y_slider = ttk.Scale(sl, from_=-5, to=5, orient=tk.HORIZONTAL,
                                    command=self._on_y_slider)
        self._y_slider.set(0)
        self._y_slider.pack(fill=tk.X, padx=4, pady=1)

        # Vector direction and origin
        vec_row = ttk.Frame(sl)
        vec_row.pack(fill=tk.X, padx=4, pady=3)
        ttk.Label(vec_row, text='û (a,b):', style='Ctrl.TLabel').pack(side=tk.LEFT)
        self._vec_b_var = tk.StringVar(value='1')
        ttk.Entry(vec_row, textvariable=self._vec_b_var, width=4, font=('Consolas', 9)).pack(side=tk.RIGHT)
        ttk.Label(vec_row, text=' b:', style='Ctrl.TLabel').pack(side=tk.RIGHT)
        self._vec_a_var = tk.StringVar(value='1')
        ttk.Entry(vec_row, textvariable=self._vec_a_var, width=4, font=('Consolas', 9)).pack(side=tk.RIGHT)
        ttk.Label(vec_row, text=' a:', style='Ctrl.TLabel').pack(side=tk.RIGHT)

        orig_row = ttk.Frame(sl)
        orig_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(orig_row, text='Origin:', style='Ctrl.TLabel').pack(side=tk.LEFT)
        self._orig_p2_var = tk.StringVar(value='0.0')
        ttk.Entry(orig_row, textvariable=self._orig_p2_var, width=4, font=('Consolas', 9)).pack(side=tk.RIGHT)
        ttk.Label(orig_row, text=' p₂:', style='Ctrl.TLabel').pack(side=tk.RIGHT)
        self._orig_p1_var = tk.StringVar(value='0.0')
        ttk.Entry(orig_row, textvariable=self._orig_p1_var, width=4, font=('Consolas', 9)).pack(side=tk.RIGHT)
        ttk.Label(orig_row, text=' p₁:', style='Ctrl.TLabel').pack(side=tk.RIGHT)

        row += 1
        # -- Colormap --
        cm = ttk.LabelFrame(parent, text='Colormap', style='Ctrl.TLabelframe')
        cm.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._cmap_var = tk.StringVar(value=self.COLORMAPS[0])
        cmap_combo = ttk.Combobox(cm, textvariable=self._cmap_var,
                                  values=self.COLORMAPS, state='readonly')
        cmap_combo.pack(fill=tk.X, padx=4, pady=3)

        # (Plot button is in the fixed bar above the scroll area)

    # ── Function list management ─────────────────────────────────────────
    def _add_function(self):
        expr = self._expr_var.get().strip()
        if not expr:
            return
        try:
            func, implicit = parse_implicit_function(expr, ('x', 'y'))
            # quick test eval
            evaluate_safe(func, np.array([0.0]), np.array([0.0]))
        except Exception as exc:
            messagebox.showerror('Parse Error', str(exc))
            return
        self.functions.append({'name': expr, 'expr': expr, 'func': func, 'implicit': implicit})
        self._refresh_listbox()
        self._do_plot()  # auto-plot on add

    def _remove_function(self):
        sel = self._func_listbox.curselection()
        if sel:
            self.functions.pop(sel[0])
            self._refresh_listbox()

    def _update_function(self):
        sel = self._func_listbox.curselection()
        if not sel:
            return
        expr = self._expr_var.get().strip()
        try:
            func, implicit = parse_implicit_function(expr, ('x', 'y'))
            evaluate_safe(func, np.array([0.0]), np.array([0.0]))
        except Exception as exc:
            messagebox.showerror('Parse Error', str(exc))
            return
        self.functions[sel[0]] = {'name': expr, 'expr': expr, 'func': func, 'implicit': implicit}
        self._refresh_listbox()

    def _refresh_listbox(self):
        self._func_listbox.delete(0, tk.END)
        for i, fd in enumerate(self.functions):
            self._func_listbox.insert(tk.END, f'{i+1}. {fd["name"]}')

    def _on_func_select(self, event):
        sel = self._func_listbox.curselection()
        if sel:
            self._expr_var.set(self.functions[sel[0]]['expr'])

    # ── Plotting ─────────────────────────────────────────────────────────
    def _parse_ranges(self):
        xmin = parse_domain_value(self._xmin_var.get())
        xmax = parse_domain_value(self._xmax_var.get())
        ymin = parse_domain_value(self._ymin_var.get())
        ymax = parse_domain_value(self._ymax_var.get())
        return xmin, xmax, ymin, ymax

    def _on_x_slider(self, *_):
        self._x_entry_var.set(f'{float(self._x_slider.get()):.2f}')
        if self._slice_axis.get() != 'none' and self.functions:
            self._do_plot()

    def _on_y_slider(self, *_):
        self._y_entry_var.set(f'{float(self._y_slider.get()):.2f}')
        if self._slice_axis.get() != 'none' and self.functions:
            self._do_plot()

    def _on_x_entry(self, *_):
        try:
            val = parse_domain_value(self._x_entry_var.get())
            self._x_slider.set(val)
        except Exception:
            pass

    def _on_y_entry(self, *_):
        try:
            val = parse_domain_value(self._y_entry_var.get())
            self._y_slider.set(val)
        except Exception:
            pass

    def _reset_tab(self):
        self.functions = []
        self._refresh_listbox()
        self._expr_var.set('x**2 + y**2')
        self._xmin_var.set('-5')
        self._xmax_var.set('5')
        self._ymin_var.set('-5')
        self._ymax_var.set('5')
        self._res_var.set(80)
        self._res_label.config(text='80')
        self._plot_type.set('surface')
        self._slice_axis.set('none')
        self._x_entry_var.set('0.00')
        self._x_slider.set(0)
        self._y_entry_var.set('0.00')
        self._y_slider.set(0)
        self._vec_a_var.set('1')
        self._vec_b_var.set('1')
        self._orig_p1_var.set('0.0')
        self._orig_p2_var.set('0.0')
        self._cmap_var.set(self.COLORMAPS[0])
        self.fig.clear()
        self.canvas.draw_idle()

    def _do_plot(self, *_):
        if not self.functions:
            # If nothing in the list yet, auto-add the current expression
            self._add_function()
            if not self.functions:
                return

        try:
            xmin, xmax, ymin, ymax = self._parse_ranges()
        except Exception:
            messagebox.showerror('Range Error', 'Enter valid expressions for x/y ranges.')
            return

        # Update slider ranges
        self._x_slider.configure(from_=xmin, to=xmax)
        self._y_slider.configure(from_=ymin, to=ymax)

        res = self._res_var.get()
        x = np.linspace(xmin, xmax, res)
        y = np.linspace(ymin, ymax, res)
        X, Y = np.meshgrid(x, y)
        cmap = self._cmap_var.get()

        self.fig.clear()
        plot_type = self._plot_type.get()
        slice_axis = self._slice_axis.get()
        show_slice = slice_axis != 'none'

        if plot_type == 'combined':
            nrows, ncols = 2, 2
            if show_slice:
                axes = [self.fig.add_subplot(2, 2, 1, projection='3d'),
                        self.fig.add_subplot(2, 2, 2),
                        self.fig.add_subplot(2, 2, 3),
                        self.fig.add_subplot(2, 2, 4)]
            else:
                axes = [self.fig.add_subplot(2, 2, 1, projection='3d'),
                        self.fig.add_subplot(2, 2, 2),
                        self.fig.add_subplot(2, 2, 3),
                        self.fig.add_subplot(2, 2, 4)]
        elif show_slice:
            if plot_type == 'surface':
                axes = [self.fig.add_subplot(1, 2, 1, projection='3d'),
                        self.fig.add_subplot(1, 2, 2)]
            else:
                axes = [self.fig.add_subplot(1, 2, 1),
                        self.fig.add_subplot(1, 2, 2)]
        else:
            if plot_type == 'surface':
                axes = [self.fig.add_subplot(1, 1, 1, projection='3d')]
            else:
                axes = [self.fig.add_subplot(1, 1, 1)]

        # Store data for hover
        self._hover_data = []

        for fd in self.functions:
            func = fd['func']
            Z = evaluate_safe(func, X, Y)
            is_implicit = fd.get('implicit', False)
            self._hover_data.append((X, Y, Z, fd['name']))

            if is_implicit:
                # Determine which axes are 2D and draw the implicit curve on all of them
                if plot_type == 'combined':
                    implicit_axes = [axes[1], axes[2]]  # filled contour + contour lines
                else:
                    implicit_axes = [axes[0]]

                for ax in implicit_axes:
                    if hasattr(ax, 'get_zlim'):
                        continue  # skip 3D axes
                    try:
                        cs = ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)
                        ax.clabel(cs, inline=True, fontsize=8, fmt='0')
                    except Exception:
                        pass
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title(f'Implicit Curve: {fd["name"]}')
                continue

            if plot_type == 'surface' or (plot_type == 'combined'):
                ax = axes[0]
                ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, edgecolor='none')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('f(x,y)')
                ax.set_title('Surface Plot')

            if plot_type == 'contourf' or plot_type == 'combined':
                ax = axes[1] if plot_type == 'combined' else axes[0]
                lvls = 25
                try:
                    cf = ax.contourf(X, Y, Z, levels=lvls, cmap=cmap)
                    self.fig.colorbar(cf, ax=ax, shrink=0.8)
                except Exception:
                    pass
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Filled Contour (Level Curves)')

            if plot_type == 'contour' or plot_type == 'combined':
                ax = axes[2] if plot_type == 'combined' else axes[0]
                try:
                    cs = ax.contour(X, Y, Z, levels=20, cmap=cmap)
                    ax.clabel(cs, inline=True, fontsize=7)
                except Exception:
                    pass
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Contour Lines')

        # Slice view
        if show_slice:
            ax_slice = axes[-1] if plot_type != 'combined' else axes[3]
            for fd in self.functions:
                if fd.get('implicit', False):
                    continue  # skip implicit functions in slice view
                func = fd['func']
                if slice_axis == 'x':
                    x0 = float(self._x_slider.get())
                    yy = np.linspace(ymin, ymax, res)
                    zz = evaluate_safe(func, np.full_like(yy, x0), yy)
                    ax_slice.plot(yy, zz, label=f'{fd["name"]}  (x={x0:.2f})')
                    ax_slice.set_xlabel('y')
                elif slice_axis == 'y':
                    y0 = float(self._y_slider.get())
                    xx = np.linspace(xmin, xmax, res)
                    zz = evaluate_safe(func, xx, np.full_like(xx, y0))
                    ax_slice.plot(xx, zz, label=f'{fd["name"]}  (y={y0:.2f})')
                    ax_slice.set_xlabel('x')
                elif slice_axis == 'vec':
                    try:
                        x0 = parse_domain_value(self._orig_p1_var.get())
                        y0 = parse_domain_value(self._orig_p2_var.get())
                        a = parse_domain_value(self._vec_a_var.get())
                        b = parse_domain_value(self._vec_b_var.get())
                    except Exception:
                        continue
                    
                    n = np.array([a, b], dtype=float)
                    norm_n = np.linalg.norm(n)
                    if norm_n < 1e-12:
                        continue
                    ux, uy = n / norm_n
                    
                    span = np.hypot(xmax - xmin, ymax - ymin)
                    t_vals = np.linspace(-span, span, res * 2)
                    
                    xx = x0 + t_vals * ux
                    yy = y0 + t_vals * uy
                    valid = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
                    # Filter to only the valid internal domain to prevent weird squashing
                    t_vals = t_vals[valid]
                    xx = xx[valid]
                    yy = yy[valid]
                    
                    if len(t_vals) > 0:
                        zz = evaluate_safe(func, xx, yy)
                        ax_slice.plot(t_vals, zz, label=f'{fd["name"]}  (û=({a:.1f},{b:.1f}))')
                    ax_slice.set_xlabel('t (distance along û)')
            ax_slice.set_ylabel('f')
            ax_slice.set_title('Cross-Section Slice')
            ax_slice.legend(fontsize=8)
            ax_slice.grid(True, alpha=0.3)

        # Empty plot for "combined" mode 4th panel if no slice
        if plot_type == 'combined' and not show_slice:
            axes[3].set_visible(False)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── Hover ────────────────────────────────────────────────────────────
    def _on_hover(self, event):
        if event.inaxes is None or not self._hover_data:
            self._status_var.set('')
            return
        # Only for 2D axes
        ax = event.inaxes
        if hasattr(ax, 'get_zlim'):
            return  # 3D axis, skip
        xm, ym = event.xdata, event.ydata
        if xm is None or ym is None:
            return
        parts = []
        for X, Y, Z, name in self._hover_data:
            # Find nearest grid point
            ix = np.searchsorted(X[0, :], xm)
            iy = np.searchsorted(Y[:, 0], ym)
            ix = np.clip(ix, 0, Z.shape[1] - 1)
            iy = np.clip(iy, 0, Z.shape[0] - 1)
            val = Z[iy, ix]
            parts.append(f'{name}: f({xm:.3f}, {ym:.3f}) = {val:.4f}')
        self._status_var.set('  |  '.join(parts))
