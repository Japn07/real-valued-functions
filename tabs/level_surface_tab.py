"""
level_surface_tab.py  –  Tab 2: f(x,y,z) Level-Surface Visualizer.

Renders isosurfaces  f(x,y,z) = c  using marching cubes,
with optional axis-aligned cross-section slicing.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math_engine import parse_function, parse_implicit_function, evaluate_safe, parse_domain_value

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ─────────────────────────────────────────────────────────────────────────────
class LevelSurfaceTab(ttk.Frame):
    """f(x,y,z) level-surface visualizer tab."""

    COLORMAPS = ['viridis', 'coolwarm', 'plasma', 'Spectral', 'RdYlBu']

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.functions = []
        self._slice_axis = tk.StringVar(value='none')
        self._build_styles()
        self._build_ui()

    def _build_styles(self):
        s = ttk.Style()
        s.configure('LS.TLabelframe', font=('Segoe UI', 10, 'bold'))
        s.configure('LS.TLabel', font=('Segoe UI', 10))
        s.configure('LS.TButton', font=('Segoe UI', 10))

    # ── Layout ───────────────────────────────────────────────────────────
    def _build_ui(self):
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # Left controls
        ctrl_outer = ttk.Frame(pw, width=340)

        # Fixed Plot button at the top
        btn_bar = ttk.Frame(ctrl_outer)
        btn_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)
        ttk.Button(btn_bar, text='⟳  Plot', command=self._do_plot,
                   style='LS.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_bar, text='Reset', command=self._reset_tab,
                   style='LS.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

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

        # Right canvas
        fig_frame = ttk.Frame(pw)
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        pw.add(fig_frame, weight=1)

    def _build_controls(self, parent):
        row = 0
        # -- Expression & Level --
        grp = ttk.LabelFrame(parent, text='Function  f(x, y, z)', style='LS.TLabelframe')
        grp.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        
        ttk.Label(grp, text='f(x,y,z) =', style='LS.TLabel').grid(row=0, column=0, sticky='w', padx=4)
        self._expr_var = tk.StringVar(value='x**2 + y**2 + z**2')
        ttk.Entry(grp, textvariable=self._expr_var, width=28, font=('Consolas', 10)).grid(
            row=0, column=1, padx=4, pady=3, sticky='ew')

        ttk.Label(grp, text='Level (c) =', style='LS.TLabel').grid(row=1, column=0, sticky='w', padx=4)
        self._level_var = tk.StringVar(value='1')
        ttk.Entry(grp, textvariable=self._level_var, width=10, font=('Consolas', 10)).grid(
            row=1, column=1, padx=4, pady=3, sticky='w')

        btn_frame = ttk.Frame(grp)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Button(btn_frame, text='Add', command=self._add_function, style='LS.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text='Remove', command=self._remove_function, style='LS.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text='Update', command=self._update_function, style='LS.TButton').pack(
            side=tk.LEFT, padx=3)

        row += 1
        # -- Function list --
        lf = ttk.LabelFrame(parent, text='Function List', style='LS.TLabelframe')
        lf.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._func_listbox = tk.Listbox(lf, height=4, font=('Consolas', 9),
                                        selectmode=tk.SINGLE)
        self._func_listbox.pack(fill=tk.X, padx=4, pady=3)
        self._func_listbox.bind('<<ListboxSelect>>', self._on_func_select)

        row += 1
        # -- Ranges --
        rng = ttk.LabelFrame(parent, text='Domain', style='LS.TLabelframe')
        rng.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._range_vars = {}
        for i, (lbl, dv) in enumerate([
            ('x min', '-3'), ('x max', '3'),
            ('y min', '-3'), ('y max', '3'),
            ('z min', '-3'), ('z max', '3'),
        ]):
            ttk.Label(rng, text=lbl, style='LS.TLabel').grid(row=i, column=0, sticky='w', padx=4)
            v = tk.StringVar(value=dv)
            self._range_vars[lbl] = v
            ttk.Entry(rng, textvariable=v, width=8, font=('Consolas', 10)).grid(
                row=i, column=1, padx=4, pady=1, sticky='w')

        row += 1
        # -- Resolution --
        res_f = ttk.LabelFrame(parent, text='Resolution', style='LS.TLabelframe')
        res_f.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._res_var = tk.IntVar(value=40)
        ttk.Scale(res_f, from_=15, to=80, variable=self._res_var,
                  orient=tk.HORIZONTAL, command=lambda _: self._res_lbl.config(
                      text=str(self._res_var.get()))).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._res_lbl = ttk.Label(res_f, text='40', style='LS.TLabel', width=4)
        self._res_lbl.pack(side=tk.RIGHT, padx=4)

        row += 1
        # -- Slice --
        sl = ttk.LabelFrame(parent, text='Slice Axis', style='LS.TLabelframe')
        sl.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        for val, lbl in [('none', 'None'), ('x', 'Slice x'),
                         ('y', 'Slice y'), ('z', 'Slice z'),
                         ('vec', 'Slice along û')]:
            ttk.Radiobutton(sl, text=lbl, variable=self._slice_axis,
                            value=val).pack(anchor='w', padx=4)

        pos_row = ttk.Frame(sl)
        pos_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(pos_row, text='Slice position:', style='LS.TLabel').pack(side=tk.LEFT)
        self._slice_entry_var = tk.StringVar(value='0.00')
        slice_entry = ttk.Entry(pos_row, textvariable=self._slice_entry_var, width=7,
                                font=('Consolas', 9))
        slice_entry.pack(side=tk.RIGHT, padx=2)
        slice_entry.bind('<Return>', self._on_slice_entry)
        slice_entry.bind('<FocusOut>', self._on_slice_entry)
        self._slice_pos = tk.DoubleVar(value=0)
        self._slice_slider = ttk.Scale(sl, from_=-3, to=3, variable=self._slice_pos,
                                        orient=tk.HORIZONTAL,
                                        command=self._on_slice_slider)
        self._slice_slider.pack(fill=tk.X, padx=4, pady=2)

        # -- Unit-vector slice controls --
        uv = ttk.LabelFrame(parent, text='Unit Vector Slice', style='LS.TLabelframe')
        uv.grid(row=row + 1, column=0, sticky='ew', padx=5, pady=4)
        ttk.Label(uv, text='Direction û (a, b, c):', style='LS.TLabel').grid(
            row=0, column=0, columnspan=6, sticky='w', padx=4)
        self._uv_a = tk.StringVar(value='1')
        self._uv_b = tk.StringVar(value='1')
        self._uv_c = tk.StringVar(value='1')
        for i, (lbl, var) in enumerate([('a', self._uv_a), ('b', self._uv_b), ('c', self._uv_c)]):
            ttk.Label(uv, text=lbl, style='LS.TLabel').grid(row=1, column=i*2, padx=2, sticky='e')
            ttk.Entry(uv, textvariable=var, width=5, font=('Consolas', 9)).grid(
                row=1, column=i*2+1, padx=2, pady=1, sticky='w')

        ttk.Label(uv, text='Origin (p₁, p₂, p₃):', style='LS.TLabel').grid(
            row=2, column=0, columnspan=6, sticky='w', padx=4)
        self._uv_p1 = tk.StringVar(value='0')
        self._uv_p2 = tk.StringVar(value='0')
        self._uv_p3 = tk.StringVar(value='0')
        for i, (lbl, var) in enumerate([('p₁', self._uv_p1), ('p₂', self._uv_p2), ('p₃', self._uv_p3)]):
            ttk.Label(uv, text=lbl, style='LS.TLabel').grid(row=3, column=i*2, padx=2, sticky='e')
            ttk.Entry(uv, textvariable=var, width=5, font=('Consolas', 9)).grid(
                row=3, column=i*2+1, padx=2, pady=1, sticky='w')
        row += 1  # account for unit-vector frame

        row += 1
        # -- Colormap --
        cm = ttk.LabelFrame(parent, text='Colormap', style='LS.TLabelframe')
        cm.grid(row=row, column=0, sticky='ew', padx=5, pady=4)
        self._cmap_var = tk.StringVar(value='viridis')
        ttk.Combobox(cm, textvariable=self._cmap_var, values=self.COLORMAPS,
                     state='readonly').pack(fill=tk.X, padx=4, pady=3)

        # (Plot button is in the fixed bar above the scroll area)

    # ── Function list management ─────────────────────────────────────────
    def _add_function(self):
        expr = self._expr_var.get().strip()
        lvl_str = self._level_var.get().strip()
        if not expr: return
        try:
            func, is_implicit = parse_implicit_function(expr, ('x', 'y', 'z'))
            if is_implicit:
                level = 0.0
            else:
                level = parse_domain_value(lvl_str)
        except Exception as exc:
            messagebox.showerror('Parse Error', str(exc))
            return
        self.functions.append({'expr': expr, 'level': level, 'func': func, 'implicit': is_implicit})
        self._refresh_listbox()
        self._do_plot()

    def _remove_function(self):
        sel = self._func_listbox.curselection()
        if sel:
            self.functions.pop(sel[0])
            self._refresh_listbox()

    def _update_function(self):
        sel = self._func_listbox.curselection()
        if not sel: return
        expr = self._expr_var.get().strip()
        lvl_str = self._level_var.get().strip()
        try:
            func, is_implicit = parse_implicit_function(expr, ('x', 'y', 'z'))
            if is_implicit:
                level = 0.0
            else:
                level = parse_domain_value(lvl_str)
        except Exception as exc:
            messagebox.showerror('Parse Error', str(exc))
            return
        self.functions[sel[0]] = {'expr': expr, 'level': level, 'func': func, 'implicit': is_implicit}
        self._refresh_listbox()

    def _refresh_listbox(self):
        self._func_listbox.delete(0, tk.END)
        for i, fd in enumerate(self.functions):
            lbl = f"{fd['expr']}"
            if not fd['implicit']: lbl += f"  (c={fd['level']})"
            self._func_listbox.insert(tk.END, f"{i+1}. {lbl}")

    def _on_func_select(self, event):
        sel = self._func_listbox.curselection()
        if sel:
            fd = self.functions[sel[0]]
            self._expr_var.set(fd['expr'])
            self._level_var.set(str(fd['level']))

    def _reset_tab(self):
        self.functions = []
        self._refresh_listbox()
        self._expr_var.set('x**2 + y**2 + z**2')
        self._level_var.set('1')
        for k, dv in [('x min', '-3'), ('x max', '3'), ('y min', '-3'), ('y max', '3'), ('z min', '-3'), ('z max', '3')]:
            self._range_vars[k].set(dv)
        self._res_var.set(40)
        self._res_lbl.config(text='40')
        self._slice_axis.set('none')
        self._slice_entry_var.set('0.00')
        self._slice_pos.set(0)
        self._uv_a.set('1'); self._uv_b.set('1'); self._uv_c.set('1')
        self._uv_p1.set('0'); self._uv_p2.set('0'); self._uv_p3.set('0')
        self._cmap_var.set('viridis')
        self.fig.clear()
        self.canvas.draw_idle()

    # ── Slice sync ─────────────────────────────────────────────────────────
    def _on_slice_slider(self, *_):
        self._slice_entry_var.set(f'{float(self._slice_pos.get()):.2f}')

    def _on_slice_entry(self, *_):
        try:
            val = parse_domain_value(self._slice_entry_var.get())
            self._slice_pos.set(val)
        except Exception:
            pass

    # ── Plotting ─────────────────────────────────────────────────────────
    def _do_plot(self, *_):
        if not self.functions:
            self._add_function()
            if not self.functions:
                return

        try:
            xmin = parse_domain_value(self._range_vars['x min'].get())
            xmax = parse_domain_value(self._range_vars['x max'].get())
            ymin = parse_domain_value(self._range_vars['y min'].get())
            ymax = parse_domain_value(self._range_vars['y max'].get())
            zmin = parse_domain_value(self._range_vars['z min'].get())
            zmax = parse_domain_value(self._range_vars['z max'].get())
        except Exception:
            messagebox.showerror('Input Error', 'Enter valid numeric expressions.')
            return


        self._slice_slider.configure(from_=min(xmin, ymin, zmin),
                                      to=max(xmax, ymax, zmax))

        res = self._res_var.get()
        x = np.linspace(xmin, xmax, res)
        y = np.linspace(ymin, ymax, res)
        z = np.linspace(zmin, zmax, res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        self.fig.clear()
        slice_axis = self._slice_axis.get()
        show_slice = slice_axis != 'none' and slice_axis != 'vec'
        show_vec_slice = slice_axis == 'vec'

        if show_slice or show_vec_slice:
            ax3d = self.fig.add_subplot(1, 2, 1, projection='3d')
            ax2d = self.fig.add_subplot(1, 2, 2)
        else:
            ax3d = self.fig.add_subplot(1, 1, 1, projection='3d')
            ax2d = None

        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('z')
        ax3d.set_title('Level Surfaces')

        cmap = self._cmap_var.get()

        for fd in self.functions:
            func = fd['func']
            level = fd['level']
            F = evaluate_safe(func, X, Y, Z)

            # Isosurface via marching cubes
            if HAS_SKIMAGE:
                try:
                    spacing = ((xmax - xmin) / (res - 1),
                               (ymax - ymin) / (res - 1),
                               (zmax - zmin) / (res - 1))
                    verts, faces, _, _ = marching_cubes(F, level=level, spacing=spacing)
                    verts[:, 0] += xmin
                    verts[:, 1] += ymin
                    verts[:, 2] += zmin
                    ax3d.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                      cmap=cmap, alpha=0.7, edgecolor='none')
                except Exception:
                    pass
            
            if show_slice and ax2d:
                pos = float(self._slice_pos.get())
                if slice_axis == 'x':
                    yy = np.linspace(ymin, ymax, res)
                    zz = np.linspace(zmin, zmax, res)
                    YY, ZZ = np.meshgrid(yy, zz)
                    XX = np.full_like(YY, pos)
                    S = evaluate_safe(func, XX, YY, ZZ)
                    try:
                        cs = ax2d.contour(YY, ZZ, S, levels=[level], colors='red')
                        ax2d.clabel(cs, inline=True, fontsize=8)
                    except Exception:
                        pass
                    ax2d.contourf(YY, ZZ, S, levels=25, cmap=cmap, alpha=0.5)
                    ax2d.set_xlabel('y'); ax2d.set_ylabel('z'); ax2d.set_title(f'Slice at x = {pos:.2f}')
                elif slice_axis == 'y':
                    xx = np.linspace(xmin, xmax, res)
                    zz = np.linspace(zmin, zmax, res)
                    XX, ZZ = np.meshgrid(xx, zz)
                    YY = np.full_like(XX, pos)
                    S = evaluate_safe(func, XX, YY, ZZ)
                    try:
                        cs = ax2d.contour(XX, ZZ, S, levels=[level], colors='red')
                        ax2d.clabel(cs, inline=True, fontsize=8)
                    except Exception:
                        pass
                    ax2d.contourf(XX, ZZ, S, levels=25, cmap=cmap, alpha=0.5)
                    ax2d.set_xlabel('x'); ax2d.set_ylabel('z'); ax2d.set_title(f'Slice at y = {pos:.2f}')
                else:  # z
                    xx = np.linspace(xmin, xmax, res)
                    yy = np.linspace(ymin, ymax, res)
                    XX, YY = np.meshgrid(xx, yy)
                    ZZ = np.full_like(XX, pos)
                    S = evaluate_safe(func, XX, YY, ZZ)
                    try:
                        cs = ax2d.contour(XX, YY, S, levels=[level], colors='red')
                        ax2d.clabel(cs, inline=True, fontsize=8)
                    except Exception:
                        pass
                    ax2d.contourf(XX, YY, S, levels=25, cmap=cmap, alpha=0.5)
                    ax2d.set_xlabel('x'); ax2d.set_ylabel('y'); ax2d.set_title(f'Slice at z = {pos:.2f}')
            
            if show_vec_slice and ax2d:
                try:
                    a = parse_domain_value(self._uv_a.get())
                    b = parse_domain_value(self._uv_b.get())
                    c = parse_domain_value(self._uv_c.get())
                    p1 = parse_domain_value(self._uv_p1.get())
                    p2 = parse_domain_value(self._uv_p2.get())
                    p3 = parse_domain_value(self._uv_p3.get())
                except Exception:
                    continue
                
                n = np.array([a, b, c], dtype=float)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12: continue
                n = n / norm_n
                origin = np.array([p1, p2, p3], dtype=float)

                if abs(n[0]) < 0.9:
                    t = np.array([1, 0, 0], dtype=float)
                else:
                    t = np.array([0, 1, 0], dtype=float)
                e1 = t - np.dot(t, n) * n
                e1 = e1 / np.linalg.norm(e1)
                e2 = np.cross(n, e1)

                span = np.hypot(xmax - xmin, np.hypot(ymax - ymin, zmax - zmin))
                s_vals = np.linspace(-span, span, res)
                t_vals = np.linspace(-span, span, res)
                SS, TT = np.meshgrid(s_vals, t_vals)

                Xp = origin[0] + SS * e1[0] + TT * e2[0]
                Yp = origin[1] + SS * e1[1] + TT * e2[1]
                Zp = origin[2] + SS * e1[2] + TT * e2[2]

                Sp = evaluate_safe(func, Xp, Yp, Zp)
                try:
                    cs = ax2d.contour(SS, TT, Sp, levels=[level], colors='red', linewidths=2)
                    ax2d.clabel(cs, inline=True, fontsize=8)
                except Exception:
                    pass
                ax2d.contourf(SS, TT, Sp, levels=25, cmap=cmap, alpha=0.5)
                ax2d.set_xlabel('e₁')
                ax2d.set_ylabel('e₂')
                ax2d.set_title(f'Slice along û=({a:.1f},{b:.1f},{c:.1f})\nthrough ({p1:.1f},{p2:.1f},{p3:.1f})')

        if not HAS_SKIMAGE:
            ax3d.text(0.5, 0.5, 0.5, 'scikit-image not available\nfor marching cubes',
                      transform=ax3d.transAxes, ha='center', fontsize=10)

        if ax2d:
            ax2d.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw_idle()
