#!/usr/bin/env python3
"""
Electromagnetic Field & Equations — Interactive Learning Application (v2.0)
Requires: PySide6, numpy
"""

import sys
import math
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QScrollArea,
    QComboBox, QSlider, QDoubleSpinBox, QCheckBox,
    QGroupBox, QSizePolicy, QGridLayout, QSpacerItem, QStatusBar
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QLinearGradient, QRadialGradient, QImage
)

# ═══════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════
K_COULOMB = 8.9875e9
MU_0 = 4 * math.pi * 1e-7
EPSILON_0 = 8.854e-12
C_LIGHT = 2.998e8
H_PLANCK = 6.626e-34
K_BOLTZ = 1.381e-23
WIEN_B = 2.898e-3

# ═══════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════
STYLESHEET = """
QMainWindow { background-color: #0d1117; }
QWidget { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Arial; font-size: 13px; }
QPushButton { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 10px 16px; color: #c9d1d9; font-size: 13px; }
QPushButton:hover { background-color: #1f2937; border-color: #58a6ff; }
QPushButton:checked { background-color: #1f3a5f; border-color: #58a6ff; color: #58a6ff; }
QPushButton:pressed { background-color: #264f78; }
QLabel { color: #c9d1d9; background: transparent; }
QComboBox { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;
            padding: 6px 12px; color: #c9d1d9; min-width: 120px; }
QComboBox:hover { border-color: #58a6ff; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background-color: #161b22; color: #c9d1d9;
                              selection-background-color: #1f3a5f; border: 1px solid #30363d; }
QDoubleSpinBox, QLineEdit { background-color: #161b22; border: 1px solid #30363d;
                             border-radius: 6px; padding: 6px; color: #c9d1d9; }
QDoubleSpinBox:focus, QLineEdit:focus { border-color: #58a6ff; }
QSlider::groove:horizontal { background: #21262d; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #58a6ff; width: 18px; height: 18px;
                              margin: -6px 0; border-radius: 9px; }
QSlider::sub-page:horizontal { background: #1f3a5f; border-radius: 3px; }
QCheckBox { color: #c9d1d9; spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px;
                        border: 2px solid #30363d; background: #161b22; }
QCheckBox::indicator:checked { background: #58a6ff; border-color: #58a6ff; }
QGroupBox { border: 1px solid #30363d; border-radius: 10px; margin-top: 14px;
            padding: 18px 12px 12px 12px; color: #58a6ff; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #0d1117; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background-color: #0a0e14; color: #484f58; border-top: 1px solid #21262d; font-size: 12px; }
"""


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def draw_arrow(painter, x1, y1, x2, y2, head_size=6, pen_width=1.5):
    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    angle = math.atan2(y2 - y1, x2 - x1)
    a1 = angle + math.radians(150)
    a2 = angle - math.radians(150)
    painter.drawLine(QPointF(x2, y2),
                     QPointF(x2 + head_size * math.cos(a1), y2 + head_size * math.sin(a1)))
    painter.drawLine(QPointF(x2, y2),
                     QPointF(x2 + head_size * math.cos(a2), y2 + head_size * math.sin(a2)))


def wavelength_to_rgb(wl_nm):
    if wl_nm < 380: r, g, b = 0.4, 0.0, 0.6
    elif wl_nm < 440: r, g, b = -(wl_nm - 440) / (440 - 380), 0.0, 1.0
    elif wl_nm < 490: r, g, b = 0.0, (wl_nm - 440) / (490 - 440), 1.0
    elif wl_nm < 510: r, g, b = 0.0, 1.0, -(wl_nm - 510) / (510 - 490)
    elif wl_nm < 580: r, g, b = (wl_nm - 510) / (580 - 510), 1.0, 0.0
    elif wl_nm < 645: r, g, b = 1.0, -(wl_nm - 645) / (645 - 580), 0.0
    elif wl_nm <= 780: r, g, b = 1.0, 0.0, 0.0
    else: r, g, b = 0.5, 0.0, 0.0
    if 380 <= wl_nm < 420: factor = 0.3 + 0.7 * (wl_nm - 380) / (420 - 380)
    elif 645 < wl_nm <= 780: factor = 0.3 + 0.7 * (780 - wl_nm) / (780 - 645)
    elif 420 <= wl_nm <= 645: factor = 1.0
    else: factor = 0.4
    return QColor(int(r * factor * 255), int(g * factor * 255), int(b * factor * 255))


class NavButton(QPushButton):
    def __init__(self, text, icon_text, parent=None):
        super().__init__(parent)
        self.setText(f"  {icon_text}   {text}")
        self.setCheckable(True)
        self.setMinimumHeight(48)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QPushButton { text-align: left; padding-left: 16px; font-size: 14px; border-radius: 10px; }")


# ═══════════════════════════════════════════════════════════════
#  PAGE 1: MAXWELL'S EQUATIONS
# ═══════════════════════════════════════════════════════════════
class MaxwellPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Maxwell's Equations")
        header.setFont(QFont("Segoe UI", 28, QFont.Bold))
        header.setStyleSheet("color: #58a6ff; margin-bottom: 4px;")
        layout.addWidget(header)

        subtitle = QLabel("The four fundamental equations that unify electricity, magnetism, and optics.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8b949e; font-size: 15px; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        equations = [
            {"title": "I. Gauss's Law for Electric Fields", "color": "#f97583",
             "integral": "∮ <b>E</b> · d<b>A</b> = Q<sub>enc</sub> / ε₀", "differential": "∇ · <b>E</b> = ρ / ε₀",
             "desc": "Electric charges produce electric fields. The total electric flux through any closed surface equals the net charge enclosed divided by the permittivity of free space."},
            {"title": "II. Gauss's Law for Magnetic Fields", "color": "#79c0ff",
             "integral": "∮ <b>B</b> · d<b>A</b> = 0", "differential": "∇ · <b>B</b> = 0",
             "desc": "There are no magnetic monopoles. The net magnetic flux through any closed surface is always zero. Magnetic field lines always form closed loops."},
            {"title": "III. Faraday's Law of Induction", "color": "#d2a8ff",
             "integral": "∮ <b>E</b> · d<b>l</b> = − dΦ<sub>B</sub> / dt", "differential": "∇ × <b>E</b> = − ∂<b>B</b> / ∂t",
             "desc": "A changing magnetic field creates an electric field. The induced EMF equals the negative rate of change of magnetic flux (Lenz's law)."},
            {"title": "IV. Ampère–Maxwell Law", "color": "#56d364",
             "integral": "∮ <b>B</b> · d<b>l</b> = μ₀ I<sub>enc</sub> + μ₀ ε₀ dΦ<sub>E</sub>/dt", "differential": "∇ × <b>B</b> = μ₀ <b>J</b> + μ₀ ε₀ ∂<b>E</b>/∂t",
             "desc": "Magnetic fields are produced by electric currents AND by changing electric fields (displacement current). This predicted electromagnetic waves."},
        ]

        for eq in equations:
            card = QFrame()
            card.setStyleSheet(f"QFrame {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; border-left: 4px solid {eq['color']}; padding: 4px; }}")
            cl = QVBoxLayout(card)
            cl.setSpacing(8)
            cl.setContentsMargins(20, 16, 20, 16)
            t = QLabel(eq["title"]); t.setFont(QFont("Segoe UI", 16, QFont.Bold)); t.setStyleSheet(f"color: {eq['color']}; border: none;"); cl.addWidget(t)
            for form_type, form_key in [("Integral form", "integral"), ("Differential form", "differential")]:
                cl.addWidget(QLabel(f"<b>{form_type}:</b>"))
                ie = QLabel(eq[form_key]); ie.setFont(QFont("Cambria", 20)); ie.setAlignment(Qt.AlignCenter)
                ie.setStyleSheet(f"color: {eq['color']}; background-color: #0d1117; border-radius: 8px; padding: 10px; border: 1px solid #21262d;")
                cl.addWidget(ie)
            cl.addWidget(QLabel(eq["desc"]))
            layout.addWidget(card)

        layout.addStretch()
        scroll.setWidget(content)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(scroll)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2: ELECTRIC FIELD SIMULATOR (WITH INSPECTOR)
# ═══════════════════════════════════════════════════════════════
class ElectricFieldCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.charges = []
        self.charge_magnitude = 1.0
        self.show_potential = True
        self.show_arrows = True
        self.mouse_pos = None
        self.probe_data = None
        self.setMinimumSize(600, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._pot_img = None

    def add_charge(self, x, y, q): self.charges.append((x, y, q)); self._recompute_potential(); self.update()
    def clear_charges(self): self.charges.clear(); self._pot_img = None; self.update()

    def _recompute_potential(self):
        if not self.charges or not self.show_potential: self._pot_img = None; return
        w, h = self.width(), self.height()
        if w < 10 or h < 10: return
        scale = 4; sw, sh = w // scale, h // scale
        xs = np.arange(sw) * scale + scale / 2; ys = np.arange(sh) * scale + scale / 2
        X, Y = np.meshgrid(xs, ys)
        V = np.zeros((sh, sw), dtype=np.float64)
        for cx, cy, q in self.charges:
            r = np.sqrt((X - cx)**2 + (Y - cy)**2); r = np.maximum(r, 8.0); V += q / r
        V_norm = np.clip(np.tanh(V * 0.15), -1, 1)
        colors = np.zeros((sh, sw, 4), dtype=np.uint8); colors[:, :, 3] = 255
        neg = V_norm < 0; pos = ~neg; t_n = np.abs(V_norm[neg]); t_p = np.abs(V_norm[pos])
        colors[neg, 0] = (20 + 15*(1-t_n)).astype(np.uint8); colors[neg, 1] = (25 + 40*(1-t_n)).astype(np.uint8); colors[neg, 2] = (50 + 170*t_n).astype(np.uint8)
        colors[pos, 0] = (50 + 185*t_p).astype(np.uint8); colors[pos, 1] = (30 + 20*(1-t_p)).astype(np.uint8); colors[pos, 2] = (35 + 15*(1-t_p)).astype(np.uint8)
        colors = np.ascontiguousarray(colors)
        img = QImage(colors.data, sw, sh, sw * 4, QImage.Format_RGBA8888).copy()
        self._pot_img = img.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def resizeEvent(self, event): super().resizeEvent(event); self._recompute_potential()

    def _calc_field_at(self, px, py):
        ex, ey, V = 0.0, 0.0, 0.0
        for cx, cy, q in self.charges:
            dx, dy = px - cx, py - cy; r2 = dx*dx + dy*dy
            if r2 < 1: return None
            r = math.sqrt(r2); r3 = r2 * r
            ex += q * dx / r3; ey += q * dy / r3; V += q / r
        return ex, ey, math.sqrt(ex*ex + ey*ey), V

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.position()
        if self.charges:
            self.probe_data = self._calc_field_at(self.mouse_pos.x(), self.mouse_pos.y())
        self.update()

    def leaveEvent(self, event):
        self.mouse_pos = None; self.probe_data = None; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0d1117"))
        if self.show_potential and self._pot_img and self.charges: p.drawImage(0, 0, self._pot_img)
        p.setPen(QPen(QColor(48, 54, 61, 60), 1))
        for x in range(0, w, 50): p.drawLine(x, 0, x, h)
        for y in range(0, h, 50): p.drawLine(0, y, w, y)

        if self.charges and self.show_arrows:
            spacing = 35
            for gx in range(spacing, w, spacing):
                for gy in range(spacing, h, spacing):
                    ex, ey, too_close = 0.0, 0.0, False
                    for cx, cy, q in self.charges:
                        dx, dy = gx - cx, gy - cy; r2 = dx*dx + dy*dy
                        if r2 < 400: too_close = True; break
                        r = math.sqrt(r2); r3 = r2 * r; ex += q * dx / r3; ey += q * dy / r3
                    if too_close: continue
                    mag = math.sqrt(ex*ex + ey*ey)
                    if mag < 1e-8: continue
                    arrow_len = min(18, 3 * math.log(1 + mag * 500)); nx, ny = ex/mag * arrow_len, ey/mag * arrow_len
                    intensity = min(255, int(80 + 175 * min(1.0, mag * 300)))
                    p.setPen(QPen(QColor(intensity, intensity, 100, 200), 1.3))
                    draw_arrow(p, gx - nx*0.5, gy - ny*0.5, gx + nx*0.5, gy + ny*0.5, head_size=4)

        for cx, cy, q in self.charges:
            radius = 18; grad = QRadialGradient(cx, cy, radius)
            if q > 0: grad.setColorAt(0, QColor(255, 120, 100)); grad.setColorAt(1, QColor(200, 40, 40, 180)); p.setPen(QPen(QColor("#ff6666"), 2))
            else: grad.setColorAt(0, QColor(100, 160, 255)); grad.setColorAt(1, QColor(30, 80, 200, 180)); p.setPen(QPen(QColor("#6699ff"), 2))
            p.setBrush(QBrush(grad)); p.drawEllipse(QPointF(cx, cy), radius, radius)
            p.setPen(QPen(Qt.white, 2.5)); p.setFont(QFont("Segoe UI", 16, QFont.Bold)); p.drawText(QPointF(cx - 6, cy + 7), "+" if q > 0 else "−")

        if not self.charges:
            p.setPen(QPen(QColor("#484f58"), 1)); p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignCenter, "Left-click: +charge\nRight-click: −charge\nHover to inspect field")

        # Inspector overlay
        if self.mouse_pos and self.probe_data:
            ex, ey, mag, V = self.probe_data
            p.setPen(Qt.NoPen); p.setBrush(QColor(13, 17, 23, 200))
            px, py = self.mouse_pos.x() + 20, self.mouse_pos.y() - 60
            if px + 180 > w: px = self.mouse_pos.x() - 200
            if py < 10: py = self.mouse_pos.y() + 20
            p.drawRoundedRect(QRectF(px, py, 180, 55), 8, 8)
            p.setPen(QPen(QColor("#58a6ff"))); p.setFont(QFont("Consolas", 10))
            p.drawText(QPointF(px+8, py+18), f"|E| = {mag:.3e}")
            p.drawText(QPointF(px+8, py+34), f" V  = {V:.3e}")
            # Draw E vector at cursor
            if mag > 1e-6:
                scale_v = min(40, 40 * math.log(1 + mag * 100))
                p.setPen(QPen(QColor("#ff4444"), 2.5))
                draw_arrow(p, self.mouse_pos.x(), self.mouse_pos.y(), self.mouse_pos.x() + (ex/mag)*scale_v, self.mouse_pos.y() + (ey/mag)*scale_v, head_size=6)

    def mousePressEvent(self, event):
        q = self.charge_magnitude if event.button() == Qt.LeftButton else -self.charge_magnitude
        self.add_charge(event.position().x(), event.position().y(), q)


class ElectricFieldPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self); layout.setContentsMargins(12,12,12,12); layout.setSpacing(12)
        self.canvas = ElectricFieldCanvas(); layout.addWidget(self.canvas, stretch=3)
        ctrl = QVBoxLayout(); ctrl.setSpacing(10)
        title = QLabel("Electric Field"); title.setFont(QFont("Segoe UI", 20, QFont.Bold)); title.setStyleSheet("color: #f97583;"); ctrl.addWidget(title)
        info = QLabel("Click to place charges.\nHover to inspect field values."); info.setWordWrap(True); info.setStyleSheet("color: #8b949e; font-size: 12px;"); ctrl.addWidget(info)
        mag_group = QGroupBox("Charge Magnitude"); mg_lay = QVBoxLayout(mag_group)
        self.mag_label = QLabel("q = 1.0"); self.mag_label.setAlignment(Qt.AlignCenter); self.mag_label.setStyleSheet("color: #f97583; font-size: 14px; font-weight: bold;"); mg_lay.addWidget(self.mag_label)
        self.mag_slider = QSlider(Qt.Horizontal); self.mag_slider.setRange(1, 50); self.mag_slider.setValue(10); self.mag_slider.valueChanged.connect(self._on_mag_changed); mg_lay.addWidget(self.mag_slider); ctrl.addWidget(mag_group)
        opt_group = QGroupBox("Display Options"); og_lay = QVBoxLayout(opt_group)
        self.cb_pot = QCheckBox("Show Potential Map"); self.cb_pot.setChecked(True); self.cb_pot.toggled.connect(self._on_toggle_pot); og_lay.addWidget(self.cb_pot)
        self.cb_arr = QCheckBox("Show Field Arrows"); self.cb_arr.setChecked(True); self.cb_arr.toggled.connect(self._on_toggle_arr); og_lay.addWidget(self.cb_arr); ctrl.addWidget(opt_group)
        info_group = QGroupBox("Key Concepts"); ig_lay = QVBoxLayout(info_group)
        for c in ["E = kq/r²", "V = kq/r", "F = kq₁q₂/r²", "Field lines: + → −", "Superposition: E = ΣEᵢ"]:
            lb = QLabel(f"• {c}"); lb.setStyleSheet("color: #8b949e; font-size: 12px;"); ig_lay.addWidget(lb)
        ctrl.addWidget(info_group)
        self.clear_btn = QPushButton("🗑  Clear All"); self.clear_btn.setStyleSheet("QPushButton{background:#2d1117;border-color:#f97583;color:#f97583;}QPushButton:hover{background:#3d1520;}")
        self.clear_btn.clicked.connect(self.canvas.clear_charges); ctrl.addWidget(self.clear_btn)
        self.preset_btn = QPushButton("⚡ Load Dipole"); self.preset_btn.clicked.connect(self._load_dipole); ctrl.addWidget(self.preset_btn)
        self.preset2_btn = QPushButton("⚡ Load Quadrupole"); self.preset2_btn.clicked.connect(self._load_quadrupole); ctrl.addWidget(self.preset2_btn)
        ctrl.addStretch(); ctrl_widget = QWidget(); ctrl_widget.setLayout(ctrl); ctrl_widget.setFixedWidth(260); layout.addWidget(ctrl_widget)

    def _on_mag_changed(self, val): q = val / 10.0; self.mag_label.setText(f"q = {q:.1f}"); self.canvas.charge_magnitude = q
    def _on_toggle_pot(self, checked): self.canvas.show_potential = checked; self.canvas._recompute_potential(); self.canvas.update()
    def _on_toggle_arr(self, checked): self.canvas.show_arrows = checked; self.canvas.update()
    def _load_dipole(self): self.canvas.clear_charges(); cx, cy = self.canvas.width()/2, self.canvas.height()/2; self.canvas.add_charge(cx-100, cy, 2.0); self.canvas.add_charge(cx+100, cy, -2.0)
    def _load_quadrupole(self): self.canvas.clear_charges(); cx, cy = self.canvas.width()/2, self.canvas.height()/2; self.canvas.add_charge(cx-80,cy-80,1.5); self.canvas.add_charge(cx+80,cy-80,-1.5); self.canvas.add_charge(cx-80,cy+80,-1.5); self.canvas.add_charge(cx+80,cy+80,1.5)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3: MAGNETIC FIELD SIMULATOR (MULTI-WIRE)
# ═══════════════════════════════════════════════════════════════
class MagneticFieldCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current = 5.0
        self.mode = 0  # 0: single, 1: attract, 2: repel
        self.show_field_arrows = True
        self.show_field_lines = True
        self.setMinimumSize(500, 400); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _get_wires(self):
        cx, cy = self.width()/2, self.height()/2
        if self.mode == 0: return [(cx, cy, self.current)]
        elif self.mode == 1: return [(cx-80, cy, self.current), (cx+80, cy, self.current)] # same direction -> attract
        else: return [(cx-80, cy, self.current), (cx+80, cy, -self.current)] # opposite -> repel

    def _calc_B_at(self, px, py):
        bx, by = 0.0, 0.0
        for wx, wy, I in self._get_wires():
            dx, dy = px - wx, py - wy; r2 = dx*dx + dy*dy
            if r2 < 25: return None
            factor = MU_0 * I / (2 * math.pi * r2)
            bx += -factor * dy; by += factor * dx
        return bx, by

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0d1117"))
        p.setPen(QPen(QColor(48, 54, 61, 40), 1))
        for x in range(0, w, 50): p.drawLine(x, 0, x, h)
        for y in range(0, h, 50): p.drawLine(0, y, w, y)

        wires = self._get_wires()

        # Field Arrows
        if self.show_field_arrows:
            spacing = 30
            for gx in range(spacing, w, spacing):
                for gy in range(spacing, h, spacing):
                    res = self._calc_B_at(gx, gy)
                    if res is None: continue
                    bx, by = res; mag = math.sqrt(bx*bx + by*by)
                    if mag < 1e-8: continue
                    arrow_len = min(14, 3 * math.log(1 + mag * 1e6))
                    nx, ny = bx/mag * arrow_len, by/mag * arrow_len
                    intensity = min(255, int(80 + 175 * min(1.0, mag * 5e5)))
                    p.setPen(QPen(QColor(100, 180, 255, intensity), 1.2))
                    draw_arrow(p, gx - nx*0.5, gy - ny*0.5, gx + nx*0.5, gy + ny*0.5, head_size=3)

        # Field Lines (Concentric circles for single wire)
        if self.show_field_lines and self.mode == 0:
            wx, wy, I = wires[0]; max_r = min(w, h) * 0.44; direction = 1 if I > 0 else -1
            for i in range(1, 9):
                r = max_r * i / 9.5; alpha = max(60, 220 - i * 18)
                color = QColor(100, 180, 255, alpha) if I > 0 else QColor(255, 160, 80, alpha)
                p.setPen(QPen(color, 1.8, Qt.DashLine)); p.setBrush(Qt.NoBrush); p.drawEllipse(QPointF(wx, wy), r, r)
                for j in range(max(4, i*2)):
                    angle = 2*math.pi*j/(max(4, i*2)) + 0.1*i
                    ax, ay = wx + r*math.cos(angle), wy + r*math.sin(angle); tang_angle = angle + direction*math.pi/2; al = 10
                    p.setPen(QPen(color, 2)); draw_arrow(p, ax, ay, ax + al*math.cos(tang_angle), ay + al*math.sin(tang_angle), head_size=4)

        # Wires
        for wx, wy, I in wires:
            wire_r = 22; wire_grad = QRadialGradient(wx, wy, wire_r)
            if I > 0:
                wire_grad.setColorAt(0, QColor(130, 190, 255)); wire_grad.setColorAt(1, QColor(50, 100, 200))
                p.setBrush(QBrush(wire_grad)); p.setPen(QPen(QColor("#79c0ff"), 2.5)); p.drawEllipse(QPointF(wx, wy), wire_r, wire_r)
                p.setBrush(QBrush(QColor("#0d1117"))); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(wx, wy), 5, 5)
            else:
                wire_grad.setColorAt(0, QColor(255, 180, 130)); wire_grad.setColorAt(1, QColor(200, 100, 40))
                p.setBrush(QBrush(wire_grad)); p.setPen(QPen(QColor("#ffa657"), 2.5)); p.drawEllipse(QPointF(wx, wy), wire_r, wire_r)
                p.setPen(QPen(QColor("#0d1117"), 3)); s = 9
                p.drawLine(QPointF(wx-s, wy-s), QPointF(wx+s, wy+s)); p.drawLine(QPointF(wx-s, wy+s), QPointF(wx+s, wy-s))

        p.setPen(QPen(QColor("#8b949e"), 1)); p.setFont(QFont("Segoe UI", 11))
        modes = ["Single Wire", "Attracting Wires", "Repelling Wires"]
        p.drawText(20, h - 15, f"I = {self.current:.1f} A  |  Mode: {modes[self.mode]}  |  B = μ₀I/(2πr)")


class MagneticFieldPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self); layout.setContentsMargins(12,12,12,12); layout.setSpacing(12)
        self.canvas = MagneticFieldCanvas(); layout.addWidget(self.canvas, stretch=3)
        ctrl = QVBoxLayout(); ctrl.setSpacing(10)
        title = QLabel("Magnetic Field"); title.setFont(QFont("Segoe UI", 20, QFont.Bold)); title.setStyleSheet("color: #79c0ff;"); ctrl.addWidget(title)
        
        config_group = QGroupBox("Wire Configuration"); cg_lay = QVBoxLayout(config_group)
        self.config_combo = QComboBox(); self.config_combo.addItems(["Single Wire", "Parallel (Attract)", "Anti-parallel (Repel)"])
        self.config_combo.currentIndexChanged.connect(lambda i: (setattr(self.canvas, 'mode', i), self.canvas.update())); cg_lay.addWidget(self.config_combo); ctrl.addWidget(config_group)
        
        cur_group = QGroupBox("Current (A)"); cur_l = QVBoxLayout(cur_group)
        self.cur_label = QLabel("I = 5.0 A"); self.cur_label.setAlignment(Qt.AlignCenter); self.cur_label.setStyleSheet("color: #79c0ff; font-size: 14px; font-weight: bold;"); cur_l.addWidget(self.cur_label)
        self.cur_slider = QSlider(Qt.Horizontal); self.cur_slider.setRange(1, 100); self.cur_slider.setValue(50); self.cur_slider.valueChanged.connect(self._on_current_changed); cur_l.addWidget(self.cur_slider); ctrl.addWidget(cur_group)
        
        self.cb_arrows = QCheckBox("Show Field Arrows"); self.cb_arrows.setChecked(True); self.cb_arrows.toggled.connect(lambda c: (setattr(self.canvas, 'show_field_arrows', c), self.canvas.update())); ctrl.addWidget(self.cb_arrows)
        self.cb_lines = QCheckBox("Show Field Lines (Single)"); self.cb_lines.setChecked(True); self.cb_lines.toggled.connect(lambda c: (setattr(self.canvas, 'show_field_lines', c), self.canvas.update())); ctrl.addWidget(self.cb_lines)
        
        concept_group = QGroupBox("Key Concepts"); kg_lay = QVBoxLayout(concept_group)
        for c in ["B = μ₀I / (2πr)", "F/L = μ₀I₁I₂ / (2πd)", "Parallel currents attract", "Anti-parallel currents repel", "F = qv × B (Lorentz)"]:
            lb = QLabel(f"• {c}"); lb.setStyleSheet("color: #8b949e; font-size: 12px;"); kg_lay.addWidget(lb)
        ctrl.addWidget(concept_group); ctrl.addStretch()
        ctrl_widget = QWidget(); ctrl_widget.setLayout(ctrl); ctrl_widget.setFixedWidth(260); layout.addWidget(ctrl_widget)

    def _on_current_changed(self, val): current = val / 10.0; self.cur_label.setText(f"I = {current:.1f} A"); self.canvas.current = current; self.canvas.update()


# ═══════════════════════════════════════════════════════════════
#  PAGE 4: EM WAVE ANIMATION (WITH POYNTING VECTOR)
# ═══════════════════════════════════════════════════════════════
class EMWaveCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0.0; self.show_e = True; self.show_b = True; self.show_s = False
        self.amplitude = 0.8; self.frequency = 1.0; self.playing = True
        self.setMinimumSize(600, 400); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(30)

    def _tick(self):
        if self.playing: self.phase += 0.06 * self.frequency; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        cy = h / 2; margin = 60; plot_w = w - 2 * margin
        p.fillRect(0, 0, w, h, QColor("#0d1117"))
        p.setPen(QPen(QColor(33, 38, 45), 1))
        for x in range(margin, w - margin, 40): p.drawLine(x, int(cy - h*0.4), x, int(cy + h*0.4))
        p.drawLine(margin, int(cy), w - margin, int(cy))
        if plot_w < 20: return
        wavelength = plot_w / (1.5 * self.frequency + 0.5); k = 2 * math.pi / max(wavelength, 20); amp = h * 0.3 * self.amplitude

        if self.show_b:
            b_angle = math.radians(30); b_cos, b_sin = math.cos(b_angle), math.sin(b_angle); b_amp = amp * 0.75
            for px_i in range(0, int(plot_w), 4):
                bv = b_amp * math.sin(k * px_i - self.phase); sx = margin + px_i
                alpha = max(15, min(60, int(abs(bv)*0.8))); p.setPen(QPen(QColor(68, 136, 255, alpha), 1))
                p.drawLine(QPointF(sx, cy), QPointF(sx + bv*b_cos*0.6, cy + bv*b_sin*0.6))
            b_path = QPainterPath(); first = True
            for px_i in range(0, int(plot_w)+1, 2):
                bv = b_amp * math.sin(k * px_i - self.phase); sx = margin + px_i + bv*b_cos*0.6; sy = cy + bv*b_sin*0.6
                if first: b_path.moveTo(sx, sy); first = False
                else: b_path.lineTo(sx, sy)
            p.setPen(QPen(QColor("#4488ff"), 2.8)); p.setBrush(Qt.NoBrush); p.drawPath(b_path)
            p.setFont(QFont("Segoe UI", 14, QFont.Bold)); p.setPen(QPen(QColor("#4488ff"))); p.drawText(margin + 5, int(cy + h*0.38), "B")

        if self.show_e:
            e_fill = QPainterPath(); e_fill.moveTo(margin, cy)
            for px_i in range(0, int(plot_w)+1, 2): ev = amp * math.sin(k * px_i - self.phase); e_fill.lineTo(margin + px_i, cy - ev)
            e_fill.lineTo(margin + int(plot_w), cy); e_fill.closeSubpath(); p.setPen(Qt.NoPen); p.setBrush(QColor(255, 60, 60, 30)); p.drawPath(e_fill)
            e_path = QPainterPath(); first = True
            for px_i in range(0, int(plot_w)+1, 2):
                ev = amp * math.sin(k * px_i - self.phase)
                if first: e_path.moveTo(margin + px_i, cy - ev); first = False
                else: e_path.lineTo(margin + px_i, cy - ev)
            p.setPen(QPen(QColor("#ff4444"), 2.8)); p.setBrush(Qt.NoBrush); p.drawPath(e_path)
            p.setFont(QFont("Segoe UI", 14, QFont.Bold)); p.setPen(QPen(QColor("#ff4444"))); p.drawText(margin + 5, int(cy - h*0.35), "E")

        # Poynting Vector S = E x B (Points in propagation direction, magnitude ~ E*B)
        if self.show_s:
            for px_i in range(30, int(plot_w)-10, 25):
                val = math.sin(k * px_i - self.phase)
                mag_s = abs(val * val) # Proportional to sin^2
                if mag_s > 0.05:
                    s_len = mag_s * 20
                    sx = margin + px_i
                    p.setPen(QPen(QColor("#56d364"), 2.5))
                    draw_arrow(p, sx - s_len/2, cy, sx + s_len/2, cy, head_size=5)
            p.setFont(QFont("Segoe UI", 12, QFont.Bold)); p.setPen(QPen(QColor("#56d364"))); p.drawText(margin + 50, int(cy + 15), "S")

        arr_y = cy + h * 0.42; p.setPen(QPen(QColor("#8b949e"), 2)); draw_arrow(p, margin, arr_y, w - margin, arr_y, head_size=10)
        p.setFont(QFont("Segoe UI", 12)); p.drawText(w//2 - 40, arr_y + 18, "Propagation (k)")


class EMWavePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(12,12,12,12); layout.setSpacing(12)
        header = QHBoxLayout(); title = QLabel("Electromagnetic Wave"); title.setFont(QFont("Segoe UI", 22, QFont.Bold)); title.setStyleSheet("color: #d2a8ff;"); header.addWidget(title); header.addStretch()
        self.play_btn = QPushButton("⏸  Pause"); self.play_btn.setFixedWidth(120); self.play_btn.clicked.connect(self._toggle_play); header.addWidget(self.play_btn); layout.addLayout(header)
        self.canvas = EMWaveCanvas(); layout.addWidget(self.canvas, stretch=1)
        ctrl = QHBoxLayout(); ctrl.setSpacing(20)
        ctrl.addWidget(QLabel("Amplitude:")); self.amp_slider = QSlider(Qt.Horizontal); self.amp_slider.setRange(20, 100); self.amp_slider.setValue(80); self.amp_slider.setFixedWidth(150)
        self.amp_slider.valueChanged.connect(lambda v: setattr(self.canvas, 'amplitude', v/100.0)); ctrl.addWidget(self.amp_slider)
        ctrl.addWidget(QLabel("Frequency:")); self.freq_slider = QSlider(Qt.Horizontal); self.freq_slider.setRange(5, 30); self.freq_slider.setValue(10); self.freq_slider.setFixedWidth(150)
        self.freq_slider.valueChanged.connect(lambda v: setattr(self.canvas, 'frequency', v/10.0)); ctrl.addWidget(self.freq_slider)
        self.cb_e = QCheckBox("E field"); self.cb_e.setChecked(True); self.cb_e.toggled.connect(lambda c: setattr(self.canvas, 'show_e', c)); ctrl.addWidget(self.cb_e)
        self.cb_b = QCheckBox("B field"); self.cb_b.setChecked(True); self.cb_b.toggled.connect(lambda c: setattr(self.canvas, 'show_b', c)); ctrl.addWidget(self.cb_b)
        self.cb_s = QCheckBox("S (Poynting)"); self.cb_s.toggled.connect(lambda c: setattr(self.canvas, 'show_s', c)); ctrl.addWidget(self.cb_s)
        ctrl.addStretch(); info = QLabel("E ⊥ B ⊥ S  |  S = (1/μ₀) E × B"); info.setStyleSheet("color: #8b949e; font-size: 12px;"); ctrl.addWidget(info)
        layout.addLayout(ctrl)

    def _toggle_play(self): self.canvas.playing = not self.canvas.playing; self.play_btn.setText("▶  Play" if not self.canvas.playing else "⏸  Pause")


# ═══════════════════════════════════════════════════════════════
#  PAGE 5: EM SPECTRUM
# ═══════════════════════════════════════════════════════════════
EM_BANDS = [
    {"name": "Radio", "freq_range": "3 Hz – 300 MHz", "color": "#e05050", "freq_lo": 3, "freq_hi": 3e8, "icon": "📻",
     "sources": "Antennas, astronomical objects", "applications": "Broadcasting, communication, MRI", "dangers": "Tissue heating (thermal)", "energy": "peV – μeV", "discovered_by": "Hertz (1887)", "fun_fact": "The 21 cm hydrogen line maps the universe."},
    {"name": "Microwave", "freq_range": "300 MHz – 300 GHz", "color": "#e08840", "freq_lo": 3e8, "freq_hi": 3e11, "icon": "📡",
     "sources": "Klystrons, CMB", "applications": "Ovens, WiFi, radar", "dangers": "Burns, cataracts", "energy": "μeV – meV", "discovered_by": "Bose (1895)", "fun_fact": "CMB is the oldest light (~13.8 Billion years)."},
    {"name": "Infrared", "freq_range": "300 GHz – 430 THz", "color": "#d4a030", "freq_lo": 3e11, "freq_hi": 4.3e14, "icon": "🔥",
     "sources": "Thermal radiation", "applications": "Night vision, remotes", "dangers": "Eye/skin burns", "energy": "meV – eV", "discovered_by": "Herschel (1800)", "fun_fact": "Human body peaks at ~10 μm IR emission."},
    {"name": "Visible", "freq_range": "430 – 750 THz", "color": "#40d040", "freq_lo": 4.3e14, "freq_hi": 7.5e14, "icon": "🌈",
     "sources": "Sun, LEDs, lasers", "applications": "Vision, fiber optics", "dangers": "Retinal damage", "energy": "1.8 – 3.1 eV", "discovered_by": "Newton (1666)", "fun_fact": "Less than 0.0035% of the full EM spectrum."},
    {"name": "Ultraviolet", "freq_range": "750 THz – 30 PHz", "color": "#7060e0", "freq_lo": 7.5e14, "freq_hi": 3e16, "icon": "☀️",
     "sources": "Sun, synchrotrons", "applications": "Sterilization, forensics", "dangers": "Cancer, DNA damage", "energy": "3.1 – 124 eV", "discovered_by": "Ritter (1801)", "fun_fact": "Bees can see UV patterns on flowers."},
    {"name": "X-Ray", "freq_range": "30 PHz – 30 EHz", "color": "#3090e0", "freq_lo": 3e16, "freq_hi": 3e19, "icon": "🩻",
     "sources": "X-ray tubes, black holes", "applications": "Medical imaging, security", "dangers": "Ionizing, mutations", "energy": "0.1 – 100 keV", "discovered_by": "Röntgen (1895)", "fun_fact": "'X' stands for 'Unknown' in mathematics."},
    {"name": "Gamma", "freq_range": "> 30 EHz", "color": "#c040c0", "freq_lo": 3e19, "freq_hi": 3e22, "icon": "☢️",
     "sources": "Radioactive decay, supernovae", "applications": "Cancer treatment, PET scans", "dangers": "Extreme DNA damage", "energy": "> 100 keV", "discovered_by": "Villard (1900)", "fun_fact": "GRBs release more energy in seconds than the Sun will in its lifetime."},
]

class SpectrumBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.selected_band = 3; self.hover_band = -1; self.setMouseTracking(True); self.setMinimumHeight(100); self.on_band_selected = None

    def _band_rects(self):
        w = self.width() - 40; h_bar = 52; y0 = 28; rects = []; n = len(EM_BANDS)
        # Logarithmic spacing approximation: give wider bands slightly more width
        widths = [math.log10(b["freq_hi"]/b["freq_lo"]) for b in EM_BANDS]
        total_w = sum(widths)
        x = 20
        for i in range(n):
            rw = max(60, (widths[i] / total_w) * w) # min width 60 for clickability
            rects.append(QRectF(x, y0, rw, h_bar)); x += rw
        # Scale down if overflow
        if x > self.width() - 20:
            scale = (self.width() - 40) / (x - 20)
            for r in rects: r = QRectF(20 + (r.left()-20)*scale, r.top(), r.width()*scale, r.height())
        return rects

    def _band_at(self, pos):
        for i, r in enumerate(self._band_rects()):
            if r.contains(pos): return i
        return -1

    def mouseMoveEvent(self, event):
        band = self._band_at(event.position())
        if band != self.hover_band: self.hover_band = band; self.update()

    def mousePressEvent(self, event):
        band = self._band_at(event.position())
        if band >= 0: self.selected_band = band; self.update()
        if self.on_band_selected: self.on_band_selected(band)

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0d1117")); rects = self._band_rects()
        for i, r in enumerate(rects):
            band = EM_BANDS[i]; grad = QLinearGradient(r.left(), r.top(), r.right(), r.bottom())
            if band["name"] == "Visible":
                for s in range(11): wl = 400 + s * 30; grad.setColorAt(s/10.0, wavelength_to_rgb(wl))
            else:
                grad.setColorAt(0, QColor(band["color"]).lighter(130)); grad.setColorAt(1, QColor(band["color"]))
            pen_width = 3 if i == self.selected_band else 1
            border_color = QColor("#ffffff") if i == self.selected_band else QColor("#30363d")
            if i == self.hover_band and i != self.selected_band: border_color = QColor("#8b949e"); pen_width = 2
            p.setPen(QPen(border_color, pen_width)); p.setBrush(QBrush(grad)); p.drawRoundedRect(r, 6, 6)
            p.setPen(QPen(Qt.white if i == self.selected_band else QColor("#c9d1d9"))); font_size = 9 if r.width() < 90 else 11
            p.setFont(QFont("Segoe UI", font_size, QFont.Bold)); p.drawText(QRectF(r.left(), r.top()+4, r.width(), r.height()*0.45), Qt.AlignCenter, f"{band['icon']} {band['name']}")
            p.setFont(QFont("Segoe UI", 7)); p.setPen(QColor(220,220,220,180)); p.drawText(QRectF(r.left(), r.top()+r.height()*0.48, r.width(), r.height()*0.5), Qt.AlignCenter, band["freq_range"])
        p.setPen(QPen(QColor("#484f58"), 1)); p.setFont(QFont("Segoe UI", 9)); y_scale = rects[0].bottom() + 16
        p.drawText(QRectF(20, y_scale, 150, 20), Qt.AlignLeft, "← Longer λ   Lower f")
        p.drawText(QRectF(w-170, y_scale, 150, 20), Qt.AlignRight, "Shorter λ   Higher f →")
        y_arrow = y_scale + 22; p.setPen(QPen(QColor("#484f58"), 1.5)); p.drawLine(20, int(y_arrow), w-20, int(y_arrow))
        p.drawLine(w-20, int(y_arrow), w-28, int(y_arrow-4)); p.drawLine(w-20, int(y_arrow), w-28, int(y_arrow+4))
        p.setFont(QFont("Segoe UI", 8)); p.drawText(20, int(y_arrow-6), "Low energy"); p.drawText(w-90, int(y_arrow-6), "High energy")


class SpectrumPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(20,16,20,20); layout.setSpacing(12)
        hdr = QHBoxLayout(); title = QLabel("Electromagnetic Spectrum"); title.setFont(QFont("Segoe UI", 26, QFont.Bold)); title.setStyleSheet("color: #f0c040;"); hdr.addWidget(title); hdr.addStretch(); layout.addLayout(hdr)
        subtitle = QLabel("All EM radiation travels at c ≈ 3×10⁸ m/s. Click a band to explore it."); subtitle.setWordWrap(True); subtitle.setStyleSheet("color: #8b949e; font-size: 13px; margin-bottom: 4px;"); layout.addWidget(subtitle)
        self.spectrum_bar = SpectrumBarWidget(); self.spectrum_bar.on_band_selected = self._on_band_selected; self.spectrum_bar.setMinimumHeight(110); layout.addWidget(self.spectrum_bar)
        self.detail_card = QFrame(); self.detail_card.setStyleSheet("QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 2px; }")
        detail_layout = QHBoxLayout(self.detail_card); detail_layout.setContentsMargins(20,16,20,16); detail_layout.setSpacing(20)
        left = QVBoxLayout(); left.setSpacing(6)
        self.band_title = QLabel(); self.band_title.setFont(QFont("Segoe UI", 22, QFont.Bold)); left.addWidget(self.band_title)
        self.band_freq = QLabel(); self.band_freq.setFont(QFont("Segoe UI", 13)); self.band_freq.setStyleSheet("color: #8b949e;"); left.addWidget(self.band_freq)
        self.band_energy = QLabel(); self.band_energy.setFont(QFont("Segoe UI", 13)); self.band_energy.setStyleSheet("color: #8b949e;"); left.addWidget(self.band_energy)
        self.band_discovered = QLabel(); self.band_discovered.setFont(QFont("Segoe UI", 12)); self.band_discovered.setStyleSheet("color: #6e7681;"); left.addWidget(self.band_discovered)
        left.addStretch(); detail_layout.addLayout(left, stretch=2)
        right = QVBoxLayout(); right.setSpacing(8)
        self.band_sources = QLabel(); self.band_applications = QLabel(); self.band_dangers = QLabel(); self.band_fun_fact = QLabel()
        for key, header_text, widget in [("s", "💡 Sources", self.band_sources), ("a", "🔧 Applications", self.band_applications), ("d", "⚠️ Dangers", self.band_dangers), ("f", "🧠 Fun Fact", self.band_fun_fact)]:
            grp = QFrame(); grp.setStyleSheet("QFrame { background-color: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 8px; }"); gl = QVBoxLayout(grp); gl.setContentsMargins(10,8,10,8); gl.setSpacing(4)
            hdr_lbl = QLabel(header_text); hdr_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold)); hdr_lbl.setStyleSheet("color: #58a6ff; border: none;"); gl.addWidget(hdr_lbl)
            widget.setWordWrap(True); widget.setStyleSheet("color: #8b949e; font-size: 12px; border: none;"); gl.addWidget(widget); right.addWidget(grp)
        detail_layout.addLayout(right, stretch=3); layout.addWidget(self.detail_card, stretch=1)
        
        conv_frame = QFrame(); conv_frame.setStyleSheet("QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 2px; }")
        conv_lay = QHBoxLayout(conv_frame); conv_lay.setContentsMargins(16,10,16,10); conv_lay.setSpacing(12)
        conv_lay.addWidget(QLabel("⚡ Converter:")); conv_lay.addWidget(QLabel("f ="))
        self.conv_freq_spin = QDoubleSpinBox(); self.conv_freq_spin.setRange(1e-30, 1e30); self.conv_freq_spin.setDecimals(4); self.conv_freq_spin.setValue(5e14); self.conv_freq_spin.setMinimumWidth(110); self.conv_freq_spin.valueChanged.connect(self._freq_to_wave); conv_lay.addWidget(self.conv_freq_spin)
        self.conv_freq_unit = QComboBox(); self.conv_freq_unit.addItems(["Hz", "kHz", "MHz", "GHz", "THz", "PHz", "EHz"]); self.conv_freq_unit.setCurrentIndex(4); self.conv_freq_unit.currentIndexChanged.connect(self._freq_to_wave); conv_lay.addWidget(self.conv_freq_unit)
        conv_lay.addWidget(QLabel("⟶ λ =")); self.conv_wave_label = QLabel("599.6 nm"); self.conv_wave_label.setFont(QFont("Segoe UI", 13, QFont.Bold)); self.conv_wave_label.setStyleSheet("color: #f0c040; min-width: 100px;"); conv_lay.addWidget(self.conv_wave_label)
        conv_lay.addWidget(QLabel("E =")); self.conv_energy_label = QLabel("2.07 eV"); self.conv_energy_label.setFont(QFont("Segoe UI", 13, QFont.Bold)); self.conv_energy_label.setStyleSheet("color: #56d364; min-width: 80px;"); conv_lay.addWidget(self.conv_energy_label)
        self.conv_band_label = QLabel("Visible 🌈"); self.conv_band_label.setFont(QFont("Segoe UI", 12, QFont.Bold)); self.conv_band_label.setStyleSheet("color: #8b949e;"); conv_lay.addWidget(self.conv_band_label)
        self.color_swatch = QLabel(); self.color_swatch.setFixedSize(30, 30); self.color_swatch.setStyleSheet("background-color: #40d040; border-radius: 6px; border: 1px solid #30363d;"); conv_lay.addWidget(self.color_swatch)
        layout.addWidget(conv_frame)
        self._on_band_selected(3); self._freq_to_wave()

    def _on_band_selected(self, idx):
        band = EM_BANDS[idx]
        self.band_title.setText(f"{band['icon']}  {band['name']}"); self.band_title.setStyleSheet(f"color: {band['color']};")
        self.band_freq.setText(f"Frequency:  {band['freq_range']}"); self.band_energy.setText(f"Photon Energy:  {band['energy']}"); self.band_discovered.setText(f"Discovered by: {band['discovered_by']}")
        self.band_sources.setText(band["sources"]); self.band_applications.setText(band["applications"]); self.band_dangers.setText(band["dangers"]); self.band_fun_fact.setText(band["fun_fact"])

    def _freq_to_wave(self):
        try:
            freq = self.conv_freq_spin.value(); unit_multipliers = [1, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18]
            freq_hz = freq * unit_multipliers[self.conv_freq_unit.currentIndex()]
            if freq_hz <= 0: return
            wl_m = C_LIGHT / freq_hz; wl_nm = wl_m * 1e9; energy_eV = (H_PLANCK * freq_hz) / 1.602e-19
            if wl_m >= 1: wl_str = f"{wl_m:.4g} m"
            elif wl_m >= 1e-3: wl_str = f"{wl_m*1e3:.4g} mm"
            elif wl_m >= 1e-6: wl_str = f"{wl_m*1e6:.4g} μm"
            elif wl_m >= 1e-9: wl_str = f"{wl_m*1e9:.4g} nm"
            else: wl_str = f"{wl_m*1e12:.4g} pm"
            if energy_eV < 1e-3: e_str = f"{energy_eV*1e6:.4g} μeV"
            elif energy_eV < 1: e_str = f"{energy_eV*1e3:.4g} meV"
            elif energy_eV < 1e3: e_str = f"{energy_eV:.4g} eV"
            elif energy_eV < 1e6: e_str = f"{energy_eV/1e3:.4g} keV"
            else: e_str = f"{energy_eV/1e6:.4g} MeV"
            self.conv_wave_label.setText(wl_str); self.conv_energy_label.setText(e_str)
            band_name, band_icon, band_color = "Unknown", "❓", "#8b949e"
            for b in EM_BANDS:
                if b["freq_lo"] <= freq_hz <= b["freq_hi"] * 10: band_name, band_icon, band_color = b["name"], b["icon"], b["color"]; break
            self.conv_band_label.setText(f"{band_icon} {band_name}"); self.conv_band_label.setStyleSheet(f"color: {band_color};")
            if 380 <= wl_nm <= 780: c = wavelength_to_rgb(wl_nm); self.color_swatch.setStyleSheet(f"background-color: {c.name()}; border-radius: 6px; border: 1px solid #30363d;")
            else: self.color_swatch.setStyleSheet(f"background-color: {band_color}; border-radius: 6px; border: 1px solid #30363d;")
        except Exception: pass


# ═══════════════════════════════════════════════════════════════
#  PAGE 6: BLACKBODY RADIATION (NEW)
# ═══════════════════════════════════════════════════════════════
class BlackbodyCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.temp = 5778  # K (Sun)
        self.setMinimumSize(500, 350)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def planck_wavelength(self, lam_m, T):
        """Spectral radiance B(λ, T)"""
        try:
            exp_term = math.exp(H_PLANCK * C_LIGHT / (lam_m * K_BOLTZ * T))
            return (2 * H_PLANCK * C_LIGHT**2 / lam_m**5) / (exp_term - 1)
        except OverflowError:
            return 0.0

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0d1117"))

        margin_l, margin_r, margin_t, margin_b = 70, 30, 30, 50
        plot_w = w - margin_l - margin_r; plot_h = h - margin_t - margin_b

        # Axes
        p.setPen(QPen(QColor("#8b949e"), 1.5))
        p.drawLine(margin_l, margin_t, margin_l, h - margin_b)
        p.drawLine(margin_l, h - margin_b, w - margin_r, h - margin_b)

        # Compute Planck curve
        lam_min, lam_max = 100e-9, 3000e-9  # 100nm to 3000nm
        points = []
        max_B = 0
        for i in range(200):
            lam = lam_min + (lam_max - lam_min) * i / 199
            B = self.planck_wavelength(lam, self.temp)
            if B > max_B: max_B = B
            px = margin_l + plot_w * i / 199
            points.append((px, B, lam))

        if max_B == 0: return

        # Visible spectrum band
        for i in range(200):
            lam = lam_min + (lam_max - lam_min) * i / 199
            lam_nm = lam * 1e9
            if 380 <= lam_nm <= 780:
                px = margin_l + plot_w * i / 199
                color = wavelength_to_rgb(lam_nm); color.setAlpha(40)
                p.setPen(Qt.NoPen); p.setBrush(color)
                p.drawRect(QRectF(px, margin_t, plot_w/199 + 1, plot_h))

        # Draw Curve
        path = QPainterPath()
        fill_path = QPainterPath(); fill_path.moveTo(margin_l, h - margin_b)
        for px, B, _ in points:
            py = h - margin_b - (B / max_B) * plot_h * 0.9
            if px == margin_l: path.moveTo(px, py); fill_path.lineTo(px, py)
            else: path.lineTo(px, py); fill_path.lineTo(px, py)
        fill_path.lineTo(w - margin_r, h - margin_b); fill_path.closeSubpath()
        
        p.setPen(Qt.NoPen); p.setBrush(QColor(255, 200, 100, 40)); p.drawPath(fill_path)
        p.setPen(QPen(QColor("#f0c040"), 2.5)); p.setBrush(Qt.NoBrush); p.drawPath(path)

        # Wien's displacement law marker
        lam_peak_m = WIEN_B / self.temp
        if lam_min <= lam_peak_m <= lam_max:
            peak_x = margin_l + plot_w * (lam_peak_m - lam_min) / (lam_max - lam_min)
            p.setPen(QPen(QColor("#ff4444"), 1.5, Qt.DashLine))
            p.drawLine(int(peak_x), margin_t, int(peak_x), h - margin_b)
            p.setFont(QFont("Segoe UI", 10)); p.setPen(QPen(QColor("#ff4444")))
            lam_peak_nm = lam_peak_m * 1e9
            txt = f"λ_max = {lam_peak_nm:.1f} nm"
            if lam_peak_nm > 1000: txt = f"λ_max = {lam_peak_nm/1000:.2f} μm"
            p.drawText(int(peak_x) + 5, margin_t + 15, txt)

        # Axis labels
        p.setPen(QPen(QColor("#8b949e"))); p.setFont(QFont("Segoe UI", 10))
        for i in range(5):
            lam = lam_min + (lam_max - lam_min) * i / 4; lam_nm = lam * 1e9
            px = margin_l + plot_w * i / 4
            p.drawLine(int(px), h - margin_b, int(px), h - margin_b + 5)
            if lam_nm >= 1000: txt = f"{lam_nm/1000:.1f}μm"
            else: txt = f"{lam_nm:.0f}nm"
            p.drawText(int(px) - 20, h - margin_b + 20, txt)
        
        p.drawText(w // 2 - 40, h - 5, "Wavelength")
        p.save(); p.translate(15, h // 2); p.rotate(-90); p.drawText(0, 0, "Spectral Radiance"); p.restore()

        # Temp label
        p.setPen(QPen(QColor("#f0c040"))); p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(w - 150, margin_t + 20, f"T = {self.temp} K")


class BlackbodyPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self); layout.setContentsMargins(12,12,12,12); layout.setSpacing(12)
        self.canvas = BlackbodyCanvas(); layout.addWidget(self.canvas, stretch=3)
        ctrl = QVBoxLayout(); ctrl.setSpacing(10)
        title = QLabel("Blackbody Radiation"); title.setFont(QFont("Segoe UI", 20, QFont.Bold)); title.setStyleSheet("color: #f0c040;"); ctrl.addWidget(title)
        info = QLabel("Planck's Law describes the spectral density of electromagnetic radiation emitted by a black body in thermal equilibrium."); info.setWordWrap(True); info.setStyleSheet("color: #8b949e; font-size: 12px;"); ctrl.addWidget(info)
        
        temp_group = QGroupBox("Temperature"); tg_lay = QVBoxLayout(temp_group)
        self.temp_label = QLabel("T = 5778 K (Sun)"); self.temp_label.setAlignment(Qt.AlignCenter); self.temp_label.setStyleSheet("color: #f0c040; font-size: 14px; font-weight: bold;"); tg_lay.addWidget(self.temp_label)
        self.temp_slider = QSlider(Qt.Horizontal); self.temp_slider.setRange(300, 25000); self.temp_slider.setValue(5778); self.temp_slider.valueChanged.connect(self._on_temp_changed); tg_lay.addWidget(self.temp_slider)
        presets = QHBoxLayout()
        for name, t in [("CMB", 2725), ("Room", 300), ("Iron", 1811), ("Sun", 5778), ("Sirius", 9940)]:
            btn = QPushButton(name); btn.setFixedWidth(50); btn.clicked.connect(lambda checked, val=t: (self.temp_slider.setValue(val), self._on_temp_changed(val))); presets.addWidget(btn)
        tg_lay.addLayout(presets); ctrl.addWidget(temp_group)

        result_group = QGroupBox("Calculated Properties"); rg_lay = QVBoxLayout(result_group)
        self.peak_label = QLabel("Peak λ: 500 nm"); self.peak_label.setStyleSheet("color: #c9d1d9;"); rg_lay.addWidget(self.peak_label)
        self.power_label = QLabel("Power: 6.32×10⁷ W/m²"); self.power_label.setStyleSheet("color: #c9d1d9;"); rg_lay.addWidget(self.power_label)
        ctrl.addWidget(result_group)

        concept_group = QGroupBox("Key Concepts"); kg_lay = QVBoxLayout(concept_group)
        for c in ["B(λ,T) = (2hc²/λ⁵) · 1/(e^(hc/λkT)-1)", "Wien's Law: λ_max = b/T", "Stefan-Boltzmann: P = σT⁴", "Quantum hypothesis: E = nhf"]:
            lb = QLabel(f"• {c}"); lb.setWordWrap(True); lb.setStyleSheet("color: #8b949e; font-size: 11px;"); kg_lay.addWidget(lb)
        ctrl.addWidget(concept_group); ctrl.addStretch()
        ctrl_widget = QWidget(); ctrl_widget.setLayout(ctrl); ctrl_widget.setFixedWidth(260); layout.addWidget(ctrl_widget)

    def _on_temp_changed(self, val):
        self.canvas.temp = val; self.canvas.update()
        if val == 300: name = "Room Temp"
        elif val == 2725: name = "Cosmic Microwaves"
        elif val == 1811: name = "Melting Iron"
        elif val == 5778: name = "Sun's Surface"
        elif val == 9940: name = "Sirius (Star)"
        else: name = ""
        self.temp_label.setText(f"T = {val} K" + (f" ({name})" if name else ""))
        lam_peak = WIEN_B / val; lam_peak_nm = lam_peak * 1e9
        self.peak_label.setText(f"Peak λ: {lam_peak_nm:.1f} nm" if lam_peak_nm < 1000 else f"Peak λ: {lam_peak_nm/1000:.2f} μm")
        power = 5.670e-8 * val**4
        if power < 1e3: self.power_label.setText(f"Power: {power:.2f} W/m²")
        elif power < 1e6: self.power_label.setText(f"Power: {power/1e3:.2f} kW/m²")
        else: self.power_label.setText(f"Power: {power/1e6:.2e} MW/m²")


# ═══════════════════════════════════════════════════════════════
#  PAGE 7: FORMULA CALCULATOR (EXPANDED)
# ═══════════════════════════════════════════════════════════════
FORMULAS = [
    {"name": "Coulomb's Law", "formula": "F = k |q₁ q₂| / r²", "inputs": [("q₁", "μC", 1.0), ("q₂", "μC", 1.0), ("r", "m", 0.1)], "calc": lambda v: K_COULOMB*abs(v[0]*1e-6*v[1]*1e-6)/v[2]**2, "unit": "N", "result_name": "Force"},
    {"name": "Electric Field (point charge)", "formula": "E = k q / r²", "inputs": [("q", "μC", 1.0), ("r", "m", 0.1)], "calc": lambda v: K_COULOMB*abs(v[0]*1e-6)/v[1]**2, "unit": "N/C", "result_name": "Electric Field"},
    {"name": "Electric Potential (point charge)", "formula": "V = k q / r", "inputs": [("q", "μC", 1.0), ("r", "m", 0.1)], "calc": lambda v: K_COULOMB*v[0]*1e-6/v[1], "unit": "V", "result_name": "Potential"},
    {"name": "Parallel-Plate Capacitor", "formula": "C = ε₀ A / d", "inputs": [("A", "cm²", 100.0), ("d", "mm", 1.0)], "calc": lambda v: EPSILON_0*(v[0]*1e-4)/(v[1]*1e-3), "unit": "pF", "result_name": "Capacitance", "transform": lambda x: x*1e12},
    {"name": "Magnetic Field (straight wire)", "formula": "B = μ₀ I / (2π r)", "inputs": [("I", "A", 10.0), ("r", "cm", 5.0)], "calc": lambda v: MU_0*v[0]/(2*math.pi*v[1]*1e-2), "unit": "μT", "result_name": "Magnetic Field", "transform": lambda x: x*1e6},
    {"name": "Lorentz Force", "formula": "F = q v B sin(θ)", "inputs": [("q", "μC", 1.0), ("v", "m/s", 1e6), ("B", "T", 0.5), ("θ", "°", 90.0)], "calc": lambda v: abs(v[0]*1e-6)*v[1]*v[2]*math.sin(math.radians(v[3])), "unit": "N", "result_name": "Force"},
    {"name": "Energy in Capacitor", "formula": "U = ½ C V²", "inputs": [("C", "μF", 100.0), ("V", "V", 12.0)], "calc": lambda v: 0.5*(v[0]*1e-6)*v[1]**2, "unit": "mJ", "result_name": "Energy", "transform": lambda x: x*1e3},
    {"name": "Energy in Inductor", "formula": "U = ½ L I²", "inputs": [("L", "mH", 10.0), ("I", "A", 5.0)], "calc": lambda v: 0.5*(v[0]*1e-3)*v[1]**2, "unit": "mJ", "result_name": "Energy", "transform": lambda x: x*1e3},
    {"name": "EM Wave: c = f λ", "formula": "c = f λ", "inputs": [("f", "MHz", 100.0)], "calc": lambda v: C_LIGHT/(v[0]*1e6), "unit": "m", "result_name": "Wavelength"},
    {"name": "Poynting Vector", "formula": "S = E² / (μ₀ c)", "inputs": [("E", "V/m", 1000.0)], "calc": lambda v: v[0]**2/(MU_0*C_LIGHT), "unit": "W/m²", "result_name": "Intensity"},
    {"name": "RC Time Constant", "formula": "τ = R C", "inputs": [("R", "kΩ", 1.0), ("C", "μF", 100.0)], "calc": lambda v: (v[0]*1e3)*(v[1]*1e-6), "unit": "ms", "result_name": "Time Constant", "transform": lambda x: x*1e3},
    {"name": "RL Time Constant", "formula": "τ = L / R", "inputs": [("L", "mH", 10.0), ("R", "Ω", 100.0)], "calc": lambda v: (v[0]*1e-3)/v[1], "unit": "μs", "result_name": "Time Constant", "transform": lambda x: x*1e6},
    {"name": "LC Resonant Frequency", "formula": "f = 1 / (2π √(LC))", "inputs": [("L", "mH", 10.0), ("C", "nF", 100.0)], "calc": lambda v: 1/(2*math.pi*math.sqrt((v[0]*1e-3)*(v[1]*1e-9))), "unit": "kHz", "result_name": "Frequency", "transform": lambda x: x*1e-3},
    {"name": "Magnetic Flux", "formula": "Φ = B A cos(θ)", "inputs": [("B", "T", 0.5), ("A", "cm²", 100.0), ("θ", "°", 0.0)], "calc": lambda v: v[0]*(v[1]*1e-4)*math.cos(math.radians(v[2])), "unit": "mWb", "result_name": "Flux", "transform": lambda x: x*1e3},
    {"name": "Motional EMF", "formula": "ε = B L v", "inputs": [("B", "T", 0.5), ("L", "cm", 10.0), ("v", "m/s", 5.0)], "calc": lambda v: v[0]*(v[1]*1e-2)*v[2], "unit": "mV", "result_name": "EMF", "transform": lambda x: x*1e3},
]

class CalculatorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(24,20,24,20); layout.setSpacing(16)
        title = QLabel("Formula Calculator"); title.setFont(QFont("Segoe UI", 24, QFont.Bold)); title.setStyleSheet("color: #56d364;"); layout.addWidget(title)
        sel_layout = QHBoxLayout(); sel_layout.addWidget(QLabel("Select formula:"))
        self.combo = QComboBox(); 
        for f in FORMULAS: self.combo.addItem(f["name"])
        self.combo.currentIndexChanged.connect(self._on_formula_changed); sel_layout.addWidget(self.combo, stretch=1); sel_layout.addStretch(); layout.addLayout(sel_layout)
        self.formula_label = QLabel(); self.formula_label.setFont(QFont("Cambria", 22)); self.formula_label.setAlignment(Qt.AlignCenter)
        self.formula_label.setStyleSheet("color: #56d364; background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px;"); layout.addWidget(self.formula_label)
        self.input_grid = QGridLayout(); self.input_grid.setSpacing(12); self.input_spins = []; layout.addLayout(self.input_grid)
        self.calc_btn = QPushButton("⚡  Calculate"); self.calc_btn.setFixedHeight(48); self.calc_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.calc_btn.setStyleSheet("QPushButton { background-color: #1f3a1f; border: 2px solid #56d364; color: #56d364; border-radius: 10px; } QPushButton:hover { background-color: #264d26; } QPushButton:pressed { background-color: #2d662d; }")
        self.calc_btn.clicked.connect(self._calculate); layout.addWidget(self.calc_btn)
        self.result_label = QLabel("—"); self.result_label.setFont(QFont("Segoe UI", 28, QFont.Bold)); self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #e6edf3; background-color: #161b22; border: 2px solid #30363d; border-radius: 14px; padding: 20px;"); layout.addWidget(self.result_label)
        layout.addStretch(); self._on_formula_changed(0)

    def _on_formula_changed(self, idx):
        f = FORMULAS[idx]; self.formula_label.setText(f["formula"])
        for w in self.input_spins: w.setParent(None)
        self.input_spins.clear()
        for i, (name, unit, default) in enumerate(f["inputs"]):
            lbl = QLabel(f"{name} ({unit}):"); lbl.setStyleSheet("color: #c9d1d9; font-size: 14px;")
            spin = QDoubleSpinBox(); spin.setRange(-1e12, 1e12); spin.setDecimals(4); spin.setValue(default); spin.setMinimumWidth(140)
            self.input_spins.append(spin); self.input_grid.addWidget(lbl, i, 0); self.input_grid.addWidget(spin, i, 1)
        self.result_label.setText("—")

    def _calculate(self):
        f = FORMULAS[self.combo.currentIndex()]; values = [spin.value() for spin in self.input_spins]
        try:
            result = f["calc"](values)
            if "transform" in f: result = f["transform"](result)
            if abs(result) < 1e-3 or abs(result) > 1e6: text = f"{f['result_name']} = {result:.4e} {f['unit']}"
            else: text = f"{f['result_name']} = {result:.4f} {f['unit']}"
            self.result_label.setText(text); self.result_label.setStyleSheet("color: #56d364; background-color: #161b22; border: 2px solid #56d364; border-radius: 14px; padding: 20px;")
        except Exception as e: self.result_label.setText(f"Error: {e}"); self.result_label.setStyleSheet("color: #f97583; background-color: #161b22; border: 2px solid #f97583; border-radius: 14px; padding: 20px;")


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ Electromagnetic Field & Equations")
        self.setMinimumSize(1100, 700); self.resize(1350, 850)
        self.setStatusBar(QStatusBar())

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); main_layout.setContentsMargins(0,0,0,0); main_layout.setSpacing(0)

        sidebar = QWidget(); sidebar.setFixedWidth(240); sidebar.setStyleSheet("background-color: #0a0e14; border-right: 1px solid #21262d;")
        side_layout = QVBoxLayout(sidebar); side_layout.setContentsMargins(12,20,12,20); side_layout.setSpacing(8)
        logo = QLabel("⚡ EM Field"); logo.setFont(QFont("Segoe UI", 20, QFont.Bold)); logo.setStyleSheet("color: #58a6ff; padding: 8px;"); side_layout.addWidget(logo)
        desc = QLabel("Interactive Learning"); desc.setStyleSheet("color: #484f58; font-size: 12px; padding: 0 0 12px 8px;"); side_layout.addWidget(desc)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("border: none; border-top: 1px solid #21262d;"); side_layout.addWidget(sep)

        pages = [("Maxwell's Eqns", "📖"), ("Electric Field", "⚡"), ("Magnetic Field", "🧲"), ("EM Wave", "🌊"), ("EM Spectrum", "🌈"), ("Blackbody Rad.", "🌡️"), ("Calculator", "🧮")]
        self.nav_buttons = []; self.stack = QStackedWidget()
        for i, (name, icon) in enumerate(pages):
            btn = NavButton(name, icon); btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx)); side_layout.addWidget(btn); self.nav_buttons.append(btn)
        self.nav_buttons[0].setChecked(True); side_layout.addStretch()
        footer = QLabel("v2.0  •  PySide6 + NumPy"); footer.setStyleSheet("color: #30363d; font-size: 11px;"); footer.setAlignment(Qt.AlignCenter); side_layout.addWidget(footer)
        main_layout.addWidget(sidebar)

        self.stack.addWidget(MaxwellPage()); self.stack.addWidget(ElectricFieldPage()); self.stack.addWidget(MagneticFieldPage())
        self.stack.addWidget(EMWavePage()); self.stack.addWidget(SpectrumPage()); self.stack.addWidget(BlackbodyPage()); self.stack.addWidget(CalculatorPage())
        main_layout.addWidget(self.stack, stretch=1)
        self.statusBar().showMessage("Ready  |  Explore electromagnetic phenomena interactively")

    def _switch_page(self, idx):
        for i, btn in enumerate(self.nav_buttons): btn.setChecked(i == idx)
        self.stack.setCurrentIndex(idx)
        tips = ["The foundation of classical electromagnetism", "Click to place charges, hover to inspect field values",
                "Switch wire configurations to see field interactions", "Animated EM wave with Poynting vector option",
                "Click a band to explore its properties", "Plot Planck's law for any temperature",
                "Compute common EM formulas with units"]
        self.statusBar().showMessage(tips[idx] if idx < len(tips) else "")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())