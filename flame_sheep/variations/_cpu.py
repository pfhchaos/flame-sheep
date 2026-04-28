"""CPU approximation of variation functions for viability testing and scoring.

Only implements variations that can blow up or produce large excursions.
Safe/bounded variations fall through to linear (identity * w).

Used by Genome.is_viable() and BackgroundScorer for CPU chaos game.
"""
from __future__ import annotations

import numpy as np

# Module-level var_params for CPU path (set by caller before apply)
_current_var_params: dict = {}


def apply_variations_cpu(variations: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply all active variations weighted, matching GPU behavior.

    Args:
        variations: per-variation weights (NUM_VARIATIONS,)
        x, y: input point (post-affine)

    Returns:
        (x', y') blended result
    """
    rx, ry = 0.0, 0.0
    for var_idx in range(len(variations)):
        w = float(variations[var_idx])
        if w < 1e-6:
            continue
        vx, vy = apply_variation_cpu(var_idx, x, y, w)
        rx += vx
        ry += vy
    return rx, ry


def apply_variation_cpu(var_idx: int, x: float, y: float, w: float) -> tuple[float, float]:
    """Apply a single variation on the CPU. Returns (x', y')."""
    r = np.sqrt(x*x + y*y) + 1e-10
    th = np.arctan2(x, y)  # flam3 convention

    if var_idx == 0:   # linear
        return w*x, w*y
    elif var_idx == 1: # sinusoidal
        return w*np.sin(x), w*np.sin(y)
    elif var_idx == 4: # horseshoe
        ri = 1.0 / max(r, 1e-6)
        return w*ri*((x - y)*(x + y)), w*ri*(2.0*x*y)
    elif var_idx == 5: # polar
        return w*(th / np.pi), w*(r - 1.0)
    elif var_idx == 6: # handkerchief
        return w*r*np.sin(th + r), w*r*np.cos(th - r)
    elif var_idx == 7: # heart
        return w*r*np.sin(th*r), w*r*(-np.cos(th*r))
    elif var_idx == 8: # disk
        pir = np.pi * r
        return w*(th/np.pi)*np.sin(pir), w*(th/np.pi)*np.cos(pir)
    elif var_idx == 11: # diamond
        return w*np.sin(th)*np.cos(r), w*np.cos(th)*np.sin(r)
    elif var_idx == 12: # ex
        p0 = np.sin(th + r)
        p1 = np.cos(th - r)
        return w*r*(p0**3 + p1**3), w*r*(p0**3 - p1**3)
    elif var_idx == 14: # bent
        qx = x * 2.0 if x < 0.0 else x
        qy = y * 0.5 if y < 0.0 else y
        return w*qx, w*qy
    elif var_idx == 16: # fisheye
        ri = 2.0 / (r + 1.0)
        return w*ri*y, w*ri*x
    elif var_idx == 15: # waves
        fx = _current_var_params.get('waves_freq_x', 0.5)
        fy = _current_var_params.get('waves_freq_y', 0.5)
        ax = _current_var_params.get('waves_amp_x', 0.5)
        ay = _current_var_params.get('waves_amp_y', 0.5)
        return w*(x + fx * np.sin(y / max(ax*ax, 1e-6))), w*(y + fy * np.sin(x / max(ay*ay, 1e-6)))
    elif var_idx == 17: # popcorn
        cx = _current_var_params.get('popcorn_cx', 0.5)
        cy = _current_var_params.get('popcorn_cy', 0.5)
        return w*(x + cx * np.sin(np.tan(3*y))), w*(y + cy * np.sin(np.tan(3*x)))
    elif var_idx == 21: # rings
        c = _current_var_params.get('rings_c', 0.5)
        cc = c*c + 1e-6
        rr = ((r + cc) % (2*cc)) - cc + r * (1 - cc)
        return w*rr*np.cos(th), w*rr*np.sin(th)
    elif var_idx == 22: # fan
        c = _current_var_params.get('fan_c', 0.5)
        f = _current_var_params.get('fan_f', 0.5)
        t = np.pi * c*c + 1e-6
        th2 = th - t if ((th + f) % (2*t)) > t else th + t
        return w*r*np.cos(th2), w*r*np.sin(th2)
    elif var_idx == 23: # blob
        low = _current_var_params.get('blob_low', 0.3)
        high = _current_var_params.get('blob_high', 1.2)
        waves = _current_var_params.get('blob_waves', 6.0)
        rr = r * (low + (high - low) * (0.5 + 0.5 * np.sin(waves * th)))
        return w*rr*np.sin(th), w*rr*np.cos(th)
    elif var_idx == 24: # pdj
        a = _current_var_params.get('pdj_a', 1.4)
        b = _current_var_params.get('pdj_b', 2.3)
        c = _current_var_params.get('pdj_c', 2.4)
        d = _current_var_params.get('pdj_d', 2.2)
        return w*(np.sin(a*y) - np.cos(b*x)), w*(np.sin(c*x) - np.cos(d*y))
    elif var_idx == 25: # fan2
        fx = _current_var_params.get('fan2_x', 0.5)
        fy = _current_var_params.get('fan2_y', 1.2)
        dx = np.pi * fx * fx + 1e-6
        dx2 = dx * 0.5
        t = th + fy - np.floor((th + fy) / dx) * dx
        a = th - dx2 if t > dx2 else th + dx2
        return w*r*np.sin(a), w*r*np.cos(a)
    elif var_idx == 26: # rings2
        val = _current_var_params.get('rings2_val', 0.01)
        _dx = val * val + 1e-6
        if r < 1e-10:
            return w*x, w*y
        k = int((r / _dx + 1) / 2)
        rr = 2.0 - _dx * (k * 2.0 / r + 1.0)
        return w*rr*x, w*rr*y
    elif var_idx == 46: # rings3
        val = _current_var_params.get('rings3_val', 0.01)
        n = _current_var_params.get('rings3_n', 0.0)
        _dx = val * val + 1e-6
        c = 2.0 * (_dx - _dx * _dx)
        if r < 1e-10:
            return w*x, w*y
        k = int((r / _dx + 1) / 2)
        rr = 2.0 - _dx * (k * 2.0 / r + 1.0) - n * (k * c - 1.0) / r
        return w*rr*x, w*rr*y
    elif var_idx == 47: # mobius — (az+b)/(cz+d), can blow up at poles
        re_a = _current_var_params.get('mobius_re_a', 0.1)
        re_b = _current_var_params.get('mobius_re_b', 0.2)
        re_c = _current_var_params.get('mobius_re_c', -0.15)
        re_d = _current_var_params.get('mobius_re_d', 0.21)
        im_a = _current_var_params.get('mobius_im_a', 0.2)
        im_b = _current_var_params.get('mobius_im_b', -0.12)
        im_c = _current_var_params.get('mobius_im_c', -0.15)
        im_d = _current_var_params.get('mobius_im_d', 0.1)
        re_u = re_a * x - im_a * y + re_b
        im_u = re_a * y + im_a * x + im_b
        re_v = re_c * x - im_c * y + re_d
        im_v = re_c * y + im_c * x + im_d
        d = re_v * re_v + im_v * im_v
        if d < 1e-10:
            return w*x, w*y
        inv_d = w / d
        return inv_d * (re_u * re_v + im_u * im_v), inv_d * (im_u * re_v - re_u * im_v)
    elif var_idx == 48: # cpow — complex power, can blow up
        cr = _current_var_params.get('cpow_r', 1.0)
        ci = _current_var_params.get('cpow_i', 0.1)
        power = _current_var_params.get('cpow_power', 1.5)
        a = np.arctan2(x, y)  # flam3 convention
        lnr = 0.5 * np.log(max(x*x + y*y, 1e-10))
        va = 2 * np.pi / power
        vc = cr / power
        vd = ci / power
        ang = vc * a + vd * lnr + va * np.floor(power * np.random.random())
        m = w * np.exp(vc * lnr - vd * a)
        return m * np.cos(ang), m * np.sin(ang)
    elif var_idx == 49: # ngon
        circle = _current_var_params.get('ngon_circle', 1.0)
        corners = _current_var_params.get('ngon_corners', 2.0)
        power = _current_var_params.get('ngon_power', 3.0)
        sides = _current_var_params.get('ngon_sides', 5.0)
        rf = max(x*x + y*y, 1e-10) ** (power * 0.5)
        th_std = np.arctan2(y, x)
        b = 2 * np.pi / sides
        ph = th_std - b * np.floor(th_std / b)
        if ph > b * 0.5:
            ph -= b
        amp = (corners * (1.0 / max(np.cos(ph), 1e-6) - 1.0) + circle) / max(rf, 1e-6)
        return w * amp * x, w * amp * y
    elif var_idx == 50: # loonie
        rr = x*x + y*y
        if rr < 1.0 and rr > 1e-10:
            s = np.sqrt(1.0 / rr - 1.0)
            return w * s * x, w * s * y
        return w*x, w*y
    elif var_idx == 51: # scry
        rr = x*x + y*y
        ri = np.sqrt(rr)
        d = ri * (rr + 1.0)
        if d < 1e-10:
            return w*x, w*y
        return w * x / d, w * y / d
    elif var_idx == 52: # epispiral
        n = _current_var_params.get('epispiral_n', 6.0)
        thickness = _current_var_params.get('epispiral_thickness', 0.0)
        holes = _current_var_params.get('epispiral_holes', 1.0)
        th_std = np.arctan2(y, x)
        d = np.cos(n * th_std)
        if abs(d) < 1e-6:
            return w*x, w*y
        t = -holes
        if abs(thickness) > 1e-6:
            t += (np.random.random() * thickness) / d
        else:
            t += 1.0 / d
        return w * t * np.cos(th_std), w * t * np.sin(th_std)
    elif var_idx == 53: # waves3
        scalex = _current_var_params.get('waves3_scalex', 0.05)
        scaley = _current_var_params.get('waves3_scaley', 0.05)
        freqx = _current_var_params.get('waves3_freqx', 7.0)
        freqy = _current_var_params.get('waves3_freqy', 13.0)
        sx_freq = _current_var_params.get('waves3_sx_freq', 0.0)
        sy_freq = _current_var_params.get('waves3_sy_freq', 2.0)
        scalexx = 0.5 * scalex * (1.0 + np.sin(y * sx_freq))
        scaleyy = 0.5 * scaley * (1.0 + np.sin(x * sy_freq))
        return w * (x + np.sin(y * freqx) * scalexx), w * (y + np.sin(x * freqy) * scaleyy)
    elif var_idx == 20: # cosine
        return w*np.cos(np.pi*x)*np.cosh(y), w*(-np.sin(np.pi*x)*np.sinh(y))
    elif var_idx == 27: # eyefish
        ri = 2.0 / (r + 1.0)
        return w*ri*x, w*ri*y
    elif var_idx == 28: # bubble
        d = 4.0 / (x*x + y*y + 4.0)
        return w*d*x, w*d*y
    elif var_idx == 29: # cylinder
        return w*np.sin(x), w*y
    elif var_idx == 30: # splits
        sx = _current_var_params.get('splits_x', 0.5)
        sy = _current_var_params.get('splits_y', 0.5)
        ox = x + sx if x >= 0.0 else x - sx
        oy = y + sy if y >= 0.0 else y - sy
        return w*ox, w*oy
    elif var_idx == 31: # cloverleaf
        a = np.arctan2(y, x)  # phi (standard atan2)
        rr = r * (np.sin(2.0*a) + 0.25*np.sin(6.0*a))
        return w*rr*np.cos(a), w*rr*np.sin(a)
    elif var_idx == 36: # butterfly
        w_factor = 1.3029400
        ri = max(r, 1e-6)
        return w*w_factor*y*(2.0*x/ri), w*w_factor*ri
    elif var_idx == 37: # curl
        c1 = _current_var_params.get('curl_c1', 0.5)
        c2 = _current_var_params.get('curl_c2', 0.0)
        x2 = x*x
        y2 = y*y
        re = 1.0 + c1*x + c2*(x2 - y2)
        im = c1*y + c2*2.0*x*y
        d = max(re*re + im*im, 1e-6)
        return w*(x*re + y*im)/d, w*(y*re - x*im)/d
    elif var_idx == 38: # rectangles
        rx = _current_var_params.get('rect_x', 0.5)
        ry = _current_var_params.get('rect_y', 0.5)
        ox = x if abs(rx) < 1e-6 else (2.0*np.floor(x/rx) + 1.0)*rx - x
        oy = y if abs(ry) < 1e-6 else (2.0*np.floor(y/ry) + 1.0)*ry - y
        return w*ox, w*oy
    elif var_idx == 39: # checks
        cs = _current_var_params.get('check_size', 1.0)
        cx = _current_var_params.get('check_x', 0.5)
        cy = _current_var_params.get('check_y', 0.5)
        ics = 1.0 / max(cs, 1e-6)
        cell = int(round(x*ics)) + int(round(y*ics))
        if cell % 2 == 0:
            return w*(x + cx), w*y
        else:
            return w*x, w*(y + cy)
    elif var_idx == 40: # hex_modulus
        size = _current_var_params.get('hex_modulus_size', 1.0)
        hsize = 0.86602540 / max(size, 1e-6)
        weight = 1.0 / 0.86602540
        hx = 0.57735027*x*hsize - y*hsize/3.0
        hz = 2.0*y*hsize/3.0
        hy = -hx - hz
        rx_h = round(hx)
        ry_h = round(hy)
        rz_h = round(hz)
        xd = abs(rx_h - hx)
        yd = abs(ry_h - hy)
        zd = abs(rz_h - hz)
        if xd > yd and xd > zd:
            rx_h = -ry_h - rz_h
        elif yd > zd:
            ry_h = -rx_h - rz_h
        fx = hx - rx_h
        fy = hz - rz_h
        return w*fx*weight, w*fy*weight
    elif var_idx == 41: # kaleidoscope
        pull = _current_var_params.get('kal_pull', 0.0)
        rot = _current_var_params.get('kal_rotate', 0.0)
        n = max(_current_var_params.get('kal_n', 6.0), 2.0)
        cr = np.cos(rot)
        sr = np.sin(rot)
        rpx = cr*x - sr*y
        rpy = sr*x + cr*y
        a = np.arctan2(rpy, rpx)
        ri = np.sqrt(rpx*rpx + rpy*rpy) + pull
        sector = 2.0*np.pi / n
        a = a % sector
        if a > sector*0.5:
            a = sector - a
        return w*ri*np.cos(a), w*ri*np.sin(a)
    elif var_idx == 2: # spherical — 1/r², can blow up near origin
        r2 = x*x + y*y + 1e-10
        return w*x/r2, w*y/r2
    elif var_idx == 3: # swirl
        rr = x*x + y*y
        return w*(x*np.sin(rr) - y*np.cos(rr)), w*(x*np.cos(rr) + y*np.sin(rr))
    elif var_idx == 9: # spiral — w/r, blows up near origin
        return (w/r)*(np.cos(th) + np.sin(r)), (w/r)*(np.sin(th) - np.cos(r))
    elif var_idx == 10: # hyperbolic — sin/r, can be large
        return w*np.sin(th)/r, w*np.cos(th)*r
    elif var_idx == 13: # julia
        sqr = w * np.sqrt(r)
        t2  = th * 0.5
        return sqr*np.cos(t2), sqr*np.sin(t2)
    elif var_idx == 18: # exponential — exp(x), very dangerous
        scale = w * np.exp(min(x - 1.0, 10.0))  # clamp to avoid overflow
        return scale * np.cos(np.pi * y), scale * np.sin(np.pi * y)
    elif var_idx == 19: # power
        rp = np.power(max(r, 1e-10), np.sin(th))
        return w*rp*np.cos(th), w*rp*np.sin(th)
    elif var_idx == 34: # tangent — tan(y) blows up at pi/2
        cy = np.cos(y)
        if abs(cy) < 1e-6:
            return w*x, w*10.0  # clamp
        return w*np.sin(x)/max(abs(cy), 1e-6), w*np.tan(y)
    elif var_idx == 35: # cross — 1/(x²-y²)², blows up on diagonals
        d = x*x - y*y
        if abs(d) < 1e-6:
            return w*x, w*y
        s = 1.0 / (d*d + 1e-6)
        return w*s*x, w*s*y
    elif var_idx == 32: # julian — like julia but with nth root
        sqr = w * np.sqrt(r)
        t2 = th * 0.5  # simplified: power=2
        return sqr*np.cos(t2), sqr*np.sin(t2)
    elif var_idx == 33: # juliascope — same danger profile as julian
        sqr = w * np.sqrt(r)
        t2 = th * 0.5
        return sqr*np.cos(t2), sqr*np.sin(t2)
    elif var_idx == 42: # icon — complex polynomial, can blow up
        # Simplified: just rotate by degree, bounded approximation
        return w*x, w*y
    elif var_idx == 43: # sattractor — pure rotation, always bounded
        return w*x, w*y
    elif var_idx == 44: # wallpaper — random group element
        from ._symmetry_groups import WALLPAPER_GROUPS
        group = int(_current_var_params.get('wallpaper_group', 0))
        group = max(0, min(16, group))
        elements = WALLPAPER_GROUPS[group]
        elem = elements[int(np.random.random() * len(elements))]
        a, b, c, d, e, f = elem
        return w*(a*x + b*y + c), w*(d*x + e*y + f)
    elif var_idx == 45: # frieze — random group element
        from ._symmetry_groups import FRIEZE_GROUPS
        group = int(_current_var_params.get('frieze_group', 0))
        group = max(0, min(6, group))
        elements = FRIEZE_GROUPS[group]
        elem = elements[int(np.random.random() * len(elements))]
        a, b, c, d, e, f = elem
        return w*(a*x + b*y + c), w*(d*x + e*y + f)
    else:
        # treat unknown/safe variations as linear for viability purposes
        return w*x, w*y
