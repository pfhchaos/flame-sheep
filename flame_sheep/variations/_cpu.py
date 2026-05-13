"""CPU approximation of variation functions for viability testing and scoring.

Only implements variations that can blow up or produce large excursions.
Safe/bounded variations fall through to linear (identity * w).

Used by Genome.is_viable() and BackgroundScorer for CPU chaos game.
"""
from __future__ import annotations

import numpy as np

# Module-level var_params for CPU path (set by caller before apply)
_current_var_params: dict = {}


class Xorshift32:
    """Deterministic RNG matching the GPU's xorshift32 in rng.glsl.

    Allows CPU variations to produce identical random sequences as the GPU
    when seeded with the same initial state.
    """

    def __init__(self, seed: int = 1) -> None:
        self.state: int = seed & 0xFFFFFFFF

    def next(self) -> int:
        self.state ^= (self.state << 13) & 0xFFFFFFFF
        self.state ^= self.state >> 17
        self.state ^= (self.state << 5) & 0xFFFFFFFF
        return self.state

    def float(self) -> float:
        return self.next() / 0xFFFFFFFF


# Module-level RNG — when set, CPU variations use this instead of np.random.
# Set to None for normal behavior (np.random), or a Xorshift32 for GPU-matched testing.
_rng: Xorshift32 | None = None


def _rand() -> float:
    """Get next random float — from deterministic xorshift if set, else np.random."""
    if _rng is not None:
        return _rng.float()
    return np.random.random()


def _rand_int_mod(n: int) -> int:
    """Get next random uint mod n — from deterministic xorshift if set."""
    if _rng is not None:
        return _rng.next() % n
    return int(np.random.random() * n)


def apply_variations_cpu(variations: np.ndarray, x: float, y: float,
                         affine: np.ndarray | None = None) -> tuple[float, float]:
    """Apply all active variations weighted, matching GPU behavior.

    Args:
        variations: per-variation weights (NUM_VARIATIONS,)
        x, y: input point (post-affine)
        affine: [a,b,c,d,e,f] coefficients for the current transform.
                Required by waves/popcorn/rings/fan (idx 15/17/21/22)
                which read parameters from the live affine.

    Returns:
        (x', y') blended result
    """
    rx, ry = 0.0, 0.0
    for var_idx in range(len(variations)):
        w = float(variations[var_idx])
        if w < 1e-6:
            continue
        try:
            vx, vy = apply_variation_cpu(var_idx, x, y, w, affine)
        except (OverflowError, ValueError):
            vx, vy = w * x, w * y  # fallback to linear
        rx += vx
        ry += vy
    return rx, ry


def apply_variation_cpu(var_idx: int, x: float, y: float, w: float,
                        affine: np.ndarray | None = None) -> tuple[float, float]:
    """Apply a single variation on the CPU. Returns (x', y').

    Args:
        var_idx: variation function index
        x, y: input point
        w: variation weight
        affine: [a,b,c,d,e,f] — needed by affine-reading variations (15/17/21/22)
    """
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
    elif var_idx == 15: # waves — reads b,c,e,f from affine
        # flam3: x' = x + b*sin(y/c²), y' = y + e*sin(x/f²)
        b = affine[1] if affine is not None else 0.5
        c = affine[2] if affine is not None else 0.5
        e = affine[4] if affine is not None else 0.5
        f = affine[5] if affine is not None else 0.5
        return w*(x + b * np.sin(y / max(c*c, 1e-6))), w*(y + e * np.sin(x / max(f*f, 1e-6)))
    elif var_idx == 17: # popcorn — reads c,f from affine
        # flam3: x' = x + c*sin(tan(3y)), y' = y + f*sin(tan(3x))
        c = affine[2] if affine is not None else 0.5
        f = affine[5] if affine is not None else 0.5
        return w*(x + c * np.sin(np.tan(3*y))), w*(y + f * np.sin(np.tan(3*x)))
    elif var_idx == 21: # rings — reads c from affine
        # flam3: ring spacing from affine coefficient c
        c = affine[2] if affine is not None else 0.5
        cc = c*c + 1e-6
        rr = ((r + cc) % (2*cc)) - cc + r * (1 - cc)
        return w*rr*np.cos(th), w*rr*np.sin(th)
    elif var_idx == 22: # fan — reads c,f from affine
        # flam3: dx = π*c², dy = f, dx2 = dx/2
        # a += (fmod(a+dy,dx) > dx2) ? -dx2 : dx2
        c = affine[2] if affine is not None else 0.5
        f = affine[5] if affine is not None else 0.5
        dx = np.pi * c*c + 1e-6
        dx2 = 0.5 * dx
        a = th  # atan2(x, y) = flam3's precalc_atan
        a += -dx2 if (np.fmod(a + f, dx) > dx2) else dx2
        return w*r*np.cos(a), w*r*np.sin(a)
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
        if r < 1e-10 or not np.isfinite(r):
            return w*x, w*y
        k = int(min(1e6, (r / _dx + 1) / 2))
        rr = 2.0 - _dx * (k * 2.0 / r + 1.0)
        return w*rr*x, w*rr*y
    elif var_idx == 46: # rings3
        val = _current_var_params.get('rings3_val', 0.01)
        n = _current_var_params.get('rings3_n', 0.0)
        _dx = val * val + 1e-6
        c = 2.0 * (_dx - _dx * _dx)
        if r < 1e-10 or not np.isfinite(r):
            return w*x, w*y
        k = int(min(1e6, (r / _dx + 1) / 2))
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
        ang = vc * a + vd * lnr + va * np.floor(power * _rand())
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
            t += (_rand() * thickness) / d
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
    elif var_idx == 36: # butterfly — flam3/JWildfire formula
        wx = w * 1.3029400317411197908970256609023
        y2 = y * 2.0
        ri = wx * np.sqrt(abs(y * x) / (1e-10 + x*x + y2*y2))
        return ri * x, ri * y2
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
        size = _current_var_params.get('hex_size', 1.0)
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
        if _rand_int_mod(2) == 0:
            t2 += np.pi
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
    elif var_idx == 35: # cross — flam3: w * sqrt(1/(s²+eps)) * (x, y)
        s = x*x - y*y
        ri = w * np.sqrt(1.0 / (s*s + 1e-10))
        return ri*x, ri*y
    elif var_idx == 32: # julian
        power = _current_var_params.get('julian_power', 2.0)
        dist = _current_var_params.get('julian_dist', 1.0)
        abs_n = abs(power)
        cn = dist / power * 0.5
        t_rand = np.floor(abs_n * _rand())
        phi_val = np.arctan2(y, x)  # standard atan2 for phi
        a = (phi_val + 2*np.pi * t_rand) / power
        # flam3 uses pow(r², cn) = pow(sumsq, cn), NOT pow(r, cn)
        ri = max(x*x + y*y, 1e-10) ** cn
        return w*ri*np.cos(a), w*ri*np.sin(a)
    elif var_idx == 33: # juliascope
        power = _current_var_params.get('julian_power', 2.0)
        dist = _current_var_params.get('julian_dist', 1.0)
        abs_n = abs(power)
        cn = dist / power * 0.5
        t_rand = np.floor(abs_n * _rand())
        phi_val = np.arctan2(y, x)
        if _rand_int_mod(2) == 0:
            phi_val = -phi_val
        a = (phi_val + 2*np.pi * t_rand) / power
        # flam3 uses pow(r², cn), not pow(r, cn)
        ri = max(x*x + y*y, 1e-10) ** cn
        return w*ri*np.cos(a), w*ri*np.sin(a)
    elif var_idx == 42: # icon — complex polynomial with n-fold rotational symmetry
        n = _current_var_params.get('icon_degree', 4.0)
        lam = _current_var_params.get('icon_lambda', 1.5)
        alp = _current_var_params.get('icon_alpha', 0.5)
        bet = _current_var_params.get('icon_beta', 0.3)
        gam = _current_var_params.get('icon_gamma', 0.1)
        ome = _current_var_params.get('icon_omega', 0.2)
        # conj(z) = (x, -y)
        cr, ci = x, -y
        cr_r = np.sqrt(cr*cr + ci*ci)
        cr_a = np.arctan2(ci, cr)
        # conj(z)^(n-1)
        nm1 = n - 1.0
        cpow_r = max(cr_r, 1e-10) ** nm1
        c1r = cpow_r * np.cos(nm1 * cr_a)
        c1i = cpow_r * np.sin(nm1 * cr_a)
        # conj(z)^(n-3)
        nm3 = n - 3.0
        cpow3_r = max(cr_r, 1e-10) ** nm3
        c3r = cpow3_r * np.cos(nm3 * cr_a)
        c3i = cpow3_r * np.sin(nm3 * cr_a)
        # exp(i*omega) * conj(z)^(n-3)
        eor, eoi = np.cos(ome), np.sin(ome)
        ec3r = eor * c3r - eoi * c3i
        ec3i = eor * c3i + eoi * c3r
        # z' = lam*z + alp*conj(z)^(n-1) + bet*exp(i*ome)*conj(z)^(n-3)
        rx = lam * x + alp * c1r + bet * ec3r
        ry = lam * y + alp * c1i + bet * ec3i
        return w*rx, w*ry
    elif var_idx == 43: # sattractor — m-fold rotation
        m = max(_current_var_params.get('sat_m', 6.0), 2.0)
        k = np.floor(_rand() * m)
        angle = k * 2*np.pi / m
        ca, sa = np.cos(angle), np.sin(angle)
        return w*(ca*x - sa*y), w*(sa*x + ca*y)
    elif var_idx == 44: # wallpaper — random group element
        from ._symmetry_groups import WALLPAPER_GROUPS
        group = int(_current_var_params.get('wallpaper_group', 0))
        group = max(0, min(16, group))
        elements = WALLPAPER_GROUPS[group]
        idx = int(_rand() * len(elements))
        idx = max(0, min(idx, len(elements) - 1))
        elem = elements[idx]
        a, b, c, d, e, f = elem
        return w*(a*x + b*y + c), w*(d*x + e*y + f)
    elif var_idx == 45: # frieze — random group element
        from ._symmetry_groups import FRIEZE_GROUPS
        group = int(_current_var_params.get('frieze_group', 0))
        group = max(0, min(6, group))
        elements = FRIEZE_GROUPS[group]
        idx = int(_rand() * len(elements))
        idx = max(0, min(idx, len(elements) - 1))
        elem = elements[idx]
        a, b, c, d, e, f = elem
        return w*(a*x + b*y + c), w*(d*x + e*y + f)
    elif var_idx == 54: # boarders
        c = _current_var_params.get('boarders_c', 0.5)
        cl = _current_var_params.get('boarders_cl', 0.25)
        cr = _current_var_params.get('boarders_cr', 0.75)
        rx, ry = np.floor(x + 0.5), np.floor(y + 0.5)
        ox, oy = x - rx, y - ry
        if _rand() >= cr:
            return w*(ox*c + rx), w*(oy*c + ry)
        if abs(ox) >= abs(oy):
            s = 1.0 if ox >= 0 else -1.0
            return w*(ox*c + rx + s*cl), w*(oy*c + ry + s*cl*oy/(ox+1e-10))
        s = 1.0 if oy >= 0 else -1.0
        return w*(ox*c + rx + s*cl*ox/(oy+1e-10)), w*(oy*c + ry + s*cl)
    elif var_idx == 55: # hypertile
        re = _current_var_params.get('hypertile_re', 0.3)
        im = _current_var_params.get('hypertile_im', 0.0)
        a = x + re; b = y - im
        c = re*x - im*y + 1.0; d = re*y + im*x
        vr = 1.0 / max(c*c + d*d, 1e-6)
        return w*vr*(a*c + b*d), w*vr*(b*c - a*d)
    elif var_idx == 56: # cell
        sz = _current_var_params.get('cell_size', 1.0)
        inv = 1.0 / max(abs(sz), 1e-6)
        ix, iy = int(np.floor(x*inv)), int(np.floor(y*inv))
        dx, dy = x - ix*sz, y - iy*sz
        if iy >= 0:
            if ix >= 0: iy *= 2; ix *= 2
            else: iy *= 2; ix = -(2*ix + 1)
        else:
            if ix >= 0: iy = -(2*iy + 1); ix *= 2
            else: iy = -(2*iy + 1); ix = -(2*ix + 1)
        return w*(dx + ix*sz), w*(-dy - iy*sz)
    elif var_idx == 57: # whorl — flam3 uses (weight - r) in BOTH branches
        ins = _current_var_params.get('whorl_inside', 0.5)
        out = _current_var_params.get('whorl_outside', 0.5)
        rr = np.sqrt(x*x + y*y) + 1e-10
        denom = w - rr  # weight - r, same sign convention both branches
        if abs(denom) < 1e-10:
            denom = 1e-10 if denom >= 0 else -1e-10
        if rr < w:
            a = np.arctan2(y, x) + ins / denom
        else:
            a = np.arctan2(y, x) + out / denom
        return w*rr*np.cos(a), w*rr*np.sin(a)
    elif var_idx == 58: # disc2 — flam3 precalc: timespi=rot*π, cosadd=cos(twist)-1, sinadd=sin(twist)
        rot = _current_var_params.get('disc2_rot', 1.0)
        twist = _current_var_params.get('disc2_twist', 0.5)
        timespi = rot * np.pi
        add = twist
        sinadd = np.sin(add)
        cosadd = np.cos(add) - 1.0
        if add > 2*np.pi:
            k = 1.0 + add - 2*np.pi
            cosadd *= k; sinadd *= k
        elif add < -2*np.pi:
            k = 1.0 + add + 2*np.pi
            cosadd *= k; sinadd *= k
        t = timespi * (x + y)
        rv = w * np.arctan2(x, y) / np.pi  # precalc_atan = atan2(tx,ty)
        return (np.sin(t) + cosadd)*rv, (np.cos(t) + sinadd)*rv
    elif var_idx == 59: # flower
        holes = _current_var_params.get('flower_holes', 0.5)
        petals = _current_var_params.get('flower_petals', 6.0)
        th = np.arctan2(y, x)
        d = np.sqrt(x*x + y*y) + 1e-10
        r = w * (_rand() - holes) * np.cos(petals * th) / d
        return r*x, r*y
    elif var_idx == 60: # blade
        rr = _rand() * w * np.sqrt(x*x + y*y)
        sr, cr = np.sin(rr), np.cos(rr)
        return w*x*(cr + sr), w*x*(cr - sr)
    elif var_idx == 61: # spiralwing
        c1 = x*x; c2 = y*y
        d = w / (c1 + c2 + 1e-10)
        s2 = np.sin(c2)
        return d*np.cos(c1)*s2, d*np.sin(c1)*s2
    elif var_idx == 62: # collideoscope
        ka = _current_var_params.get('collide_a', 0.5)
        num = max(_current_var_params.get('collide_num', 4.0), 1.0)
        kn_pi = num / np.pi
        pi_kn = np.pi / num
        ka_kn = ka / num
        a = np.arctan2(y, x)
        rr = w * np.sqrt(x*x + y*y)
        if a >= 0:
            alt = int(a * kn_pi)
            if alt % 2 == 0: a = alt*pi_kn + (ka_kn + a) % pi_kn
            else: a = alt*pi_kn + (-ka_kn + a) % pi_kn
        else:
            alt = int(-a * kn_pi)
            if alt % 2 != 0: a = -(alt*pi_kn + (-ka_kn - a) % pi_kn)
            else: a = -(alt*pi_kn + (ka_kn - a) % pi_kn)
        return rr*np.cos(a), rr*np.sin(a)
    elif var_idx == 63: # auger
        freq = _current_var_params.get('auger_freq', 3.0)
        wt = _current_var_params.get('auger_weight', 0.5)
        sym = _current_var_params.get('auger_sym', 0.5)
        scale = _current_var_params.get('auger_scale', 0.5)
        s = np.sin(freq * x); t = np.sin(freq * y)
        dy = y + wt * (scale * s * 0.5 + abs(y) * s)
        dx = x + wt * (scale * t * 0.5 + abs(x) * t)
        return w*(x + sym*(dx - x)), w*dy
    elif var_idx == 64: # flipcircle
        if x*x + y*y > 1.0: return w*x, w*y
        return w*x, -w*y
    elif var_idx == 65: # eclipse
        shift = _current_var_params.get('eclipse_shift', 0.0)
        if abs(y) <= 1.0:
            c2 = np.sqrt(max(0, 1.0 - y*y))
            if abs(x) <= c2:
                nx = x + shift
                if abs(nx) >= c2: return -w*x, w*y
                return w*nx, w*y
        return w*x, w*y
    elif var_idx == 66: # layered_spiral
        radius = _current_var_params.get('layered_spiral_radius', 1.0)
        a = x * radius
        t = x*x + y*y + 1e-10
        return w*a*np.cos(t), w*a*np.sin(t)
    elif var_idx == 67: # stripes
        space = _current_var_params.get('stripes_space', 0.5)
        warp = _current_var_params.get('stripes_warp', 0.5)
        rx = np.floor(x + 0.5)
        ox = x - rx
        return w*(ox*(1.0-space) + rx), w*(y + ox*ox*warp)
    elif var_idx == 68: # lissajous
        tmin = _current_var_params.get('liss_tmin', -np.pi)
        tmax = _current_var_params.get('liss_tmax', np.pi)
        la = _current_var_params.get('liss_a', 3.0)
        lb = _current_var_params.get('liss_b', 2.0)
        lc = _current_var_params.get('liss_c', 0.0)
        ld = _current_var_params.get('liss_d', 0.0)
        le = _current_var_params.get('liss_e', 0.0)
        t = (tmax - tmin) * _rand() + tmin
        yy = _rand() - 0.5
        return w*(np.sin(la*t + ld) + lc*t + le*yy), w*(np.sin(lb*t) + lc*t + le*yy)
    elif var_idx == 69: # ripple
        freq = _current_var_params.get('ripple_freq', 5.0)
        vel = _current_var_params.get('ripple_vel', 0.0)
        amp = _current_var_params.get('ripple_amp', 0.1)
        cx = _current_var_params.get('ripple_cx', 0.0)
        cy = _current_var_params.get('ripple_cy', 0.0)
        phase = _current_var_params.get('ripple_phase', 0.0)
        scale = _current_var_params.get('ripple_scale', 1.0)
        fixd = _current_var_params.get('ripple_fixd', 1.0)
        xx = x*scale - cx; yy = y*scale + cy
        d = np.sqrt(xx*xx + yy*yy) if fixd > 0.5 else np.sqrt(xx*xx * yy*yy)
        d = max(d, 1e-10)
        nx, ny = xx/d, yy/d
        wave = np.cos(freq*d - vel + phase)
        os = amp * wave
        return w*(x + nx*os), w*(y + ny*os)
    elif var_idx == 70: # waves_param — same formula as waves, explicit params
        fx = _current_var_params.get('waves_freq_x', 0.5)
        fy = _current_var_params.get('waves_freq_y', 0.5)
        ax = _current_var_params.get('waves_amp_x', 0.5)
        ay = _current_var_params.get('waves_amp_y', 0.5)
        return w*(x + fx * np.sin(y / max(ax*ax, 1e-6))), w*(y + fy * np.sin(x / max(ay*ay, 1e-6)))
    elif var_idx == 71: # popcorn_param — same formula as popcorn, explicit params
        cx = _current_var_params.get('popcorn_cx', 0.5)
        cy = _current_var_params.get('popcorn_cy', 0.5)
        return w*(x + cx * np.sin(np.tan(3*y))), w*(y + cy * np.sin(np.tan(3*x)))
    elif var_idx == 72: # rings_param — same formula as rings, explicit param
        c = _current_var_params.get('rings_c', 0.5)
        cc = c*c + 1e-6
        rr = ((r + cc) % (2*cc)) - cc + r * (1 - cc)
        return w*rr*np.cos(th), w*rr*np.sin(th)
    elif var_idx == 73: # fan_param — same formula as fan, explicit params
        c = _current_var_params.get('fan_c', 0.5)
        f = _current_var_params.get('fan_f', 0.5)
        dx = np.pi * c*c + 1e-6
        dx2 = 0.5 * dx
        a = th
        a += -dx2 if (np.fmod(a + f, dx) > dx2) else dx2
        return w*r*np.cos(a), w*r*np.sin(a)
    elif var_idx == 74: # waves2 — different formula: x + scalex*sin(y*freqx)
        sx = _current_var_params.get('waves2_scalex', 0.05)
        sy = _current_var_params.get('waves2_scaley', 0.05)
        fx = _current_var_params.get('waves2_freqx', 7.0)
        fy = _current_var_params.get('waves2_freqy', 13.0)
        return w*(x + sx * np.sin(y * fx)), w*(y + sy * np.sin(x * fy))
    # --- flam3 standard batch 2 (75-121) ---
    elif var_idx == 75: # blur — random point on disk
        angle = _rand() * 2.0 * np.pi
        r2 = w * _rand()
        return r2 * np.cos(angle), r2 * np.sin(angle)
    elif var_idx == 76: # gaussian_blur
        angle = _rand() * 2.0 * np.pi
        r = w * (_rand() + _rand() + _rand() + _rand() - 2.0)
        return r * np.cos(angle), r * np.sin(angle)
    elif var_idx == 77: # radial_blur
        ang = _current_var_params.get('radial_blur_angle', 0.5)
        rndg = _rand() + _rand() + _rand() + _rand() - 2.0
        spin = w * np.sin(ang * np.pi * 0.5)
        zoom = w * np.cos(ang * np.pi * 0.5)
        ra = np.sqrt(x*x + y*y)
        alpha = np.arctan2(y, x) + spin * rndg
        rz = zoom * rndg - 1.0
        return ra * np.cos(alpha) + rz * x, ra * np.sin(alpha) + rz * y
    elif var_idx == 78: # perspective
        ang = _current_var_params.get('perspective_angle', 0.5)
        dist = _current_var_params.get('perspective_dist', 1.5)
        vsin = np.sin(ang * np.pi * 0.5)
        vfcos = dist * np.cos(ang * np.pi * 0.5)
        d = dist - y * vsin
        if abs(d) < 1e-10: return w*x, w*y
        t = 1.0 / d
        return w * dist * x * t, w * vfcos * y * t
    elif var_idx == 79: # super_shape
        rnd = _current_var_params.get('super_shape_rnd', 0.0)
        m = _current_var_params.get('super_shape_m', 4.0)
        n1 = _current_var_params.get('super_shape_n1', 2.0)
        n2 = _current_var_params.get('super_shape_n2', 2.0)
        n3 = _current_var_params.get('super_shape_n3', 2.0)
        holes = _current_var_params.get('super_shape_holes', 0.0)
        theta = m * 0.25 * np.arctan2(y, x) + np.pi * 0.25
        t1 = abs(np.cos(theta)) ** n2
        t2 = abs(np.sin(theta)) ** n3
        ri = np.sqrt(x*x + y*y)
        if abs(n1) < 1e-10 or ri < 1e-10: return w*x, w*y
        rr = w * ((rnd * _rand() + (1.0 - rnd) * ri) - holes) * (t1 + t2) ** (-1.0/n1) / ri
        return rr * x, rr * y
    elif var_idx == 80: # noise
        angle = _rand() * 2.0 * np.pi
        r = _rand()
        return w * x * r * np.cos(angle), w * y * r * np.sin(angle)
    elif var_idx == 81: # secant2
        ri = w * np.sqrt(x*x + y*y)
        cr = np.cos(ri)
        if abs(cr) < 1e-10: return w*x, w*y
        icr = 1.0 / cr
        return w * x, w * (icr + 1.0 if cr < 0.0 else icr - 1.0)
    elif var_idx == 82: # pie
        slices = _current_var_params.get('pie_slices', 6.0)
        rotation = _current_var_params.get('pie_rotation', 0.0)
        thickness = _current_var_params.get('pie_thickness', 0.5)
        sl = int(_rand() * slices + 0.5)
        a = rotation + 2.0 * np.pi * (sl + _rand() * thickness) / slices
        ri = w * _rand()
        return ri * np.cos(a), ri * np.sin(a)
    elif var_idx == 83: # arch
        ang = _rand() * w * np.pi
        sr = np.sin(ang)
        cr = np.cos(ang)
        if abs(cr) < 1e-10: return w * sr, w * sr
        return w * sr, w * sr * sr / cr
    elif var_idx == 84: # parabola
        height = _current_var_params.get('parabola_height', 1.0)
        width = _current_var_params.get('parabola_width', 1.0)
        ri = np.sqrt(x*x + y*y)
        sr = np.sin(ri)
        cr = np.cos(ri)
        return w * height * sr * sr * _rand(), w * width * cr * _rand()
    elif var_idx == 85: # rays
        ang = _rand() * np.pi * w
        ri = x*x + y*y + 1e-10
        tanr = w * np.tan(ang) / ri
        return tanr * np.cos(x), tanr * np.sin(y)
    elif var_idx == 86: # conic
        ecc = _current_var_params.get('conic_eccentricity', 1.0)
        holes = _current_var_params.get('conic_holes', 0.0)
        ri = np.sqrt(x*x + y*y) + 1e-10
        ct = x / ri
        rr = w * (_rand() - holes) * ecc / (1.0 + ecc * ct) / ri
        return rr * x, rr * y
    elif var_idx == 87: # escher
        beta = _current_var_params.get('escher_beta', 0.3)
        seb = np.sin(beta)
        ceb = np.cos(beta)
        vc = 0.5 * (1.0 + ceb)
        vd = 0.5 * seb
        a = np.arctan2(y, x)
        lnr = 0.5 * np.log(x*x + y*y + 1e-10)
        m = w * np.exp(vc * lnr - vd * a)
        n = vc * a + vd * lnr
        return m * np.cos(n), m * np.sin(n)
    elif var_idx == 88: # elliptic
        sq = y*y + x*x
        x2 = 2.0 * x
        xmax = 0.5 * (np.sqrt(sq + x2 + 1e-10) + np.sqrt(abs(sq - x2) + 1e-10))
        xmax = max(xmax, 1.0)
        a_val = x / xmax
        a_val = max(-1.0, min(1.0, a_val))
        ssx = np.sqrt(xmax - 1.0) if xmax >= 1.0 else 0.0
        sign = 1.0 if y > 0.0 else -1.0
        v = w * 2.0 / np.pi
        return v * np.arcsin(a_val), sign * v * np.log(xmax + ssx + 1e-10)
    elif var_idx == 89: # oscilloscope
        sep = _current_var_params.get('osc_separation', 1.0)
        freq = _current_var_params.get('osc_frequency', np.pi)
        amp = _current_var_params.get('osc_amplitude', 1.0)
        damp = _current_var_params.get('osc_damping', 0.0)
        tpf = 2.0 * np.pi * freq
        t = amp * np.exp(-abs(x) * damp) * np.cos(tpf * x) + sep
        dy = -y if abs(y) <= t else y
        return w * x, w * dy
    elif var_idx == 90: # edisc
        tmp = x*x + y*y + 1.0
        tmp2 = 2.0 * x
        r1 = np.sqrt(abs(tmp + tmp2))
        r2 = np.sqrt(abs(tmp - tmp2))
        xmax = max((r1 + r2) * 0.5, 1.0)
        a1 = np.log(xmax + np.sqrt(max(xmax - 1.0, 0.0)))
        a2 = -np.arccos(max(-1.0, min(1.0, x / xmax)))
        snv = np.sin(a1)
        if y > 0.0: snv = -snv
        cst = w / 11.57034632
        return cst * np.cosh(a2) * np.cos(a1), cst * np.sinh(a2) * snv
    elif var_idx == 91: # square
        return w * (_rand() - 0.5), w * (_rand() - 0.5)
    elif var_idx == 92: # curve
        xamp = _current_var_params.get('curve_xamp', 0.0)
        yamp = _current_var_params.get('curve_yamp', 0.0)
        xl = _current_var_params.get('curve_xlength', 1.0)
        yl = _current_var_params.get('curve_ylength', 1.0)
        xl2 = max(xl * xl, 1e-10)
        yl2 = max(yl * yl, 1e-10)
        return w * (x + xamp * np.exp(-y*y / xl2)), w * (y + yamp * np.exp(-x*x / yl2))
    elif var_idx == 93: # twintrian
        ri = _rand() * w * np.sqrt(x*x + y*y)
        sr = np.sin(ri)
        cr = np.cos(ri)
        diff = np.log10(max(sr * sr, 1e-20)) + cr
        return w * x * diff, w * x * (diff - sr * np.pi)
    elif var_idx == 94: # wedge_julia
        ang = _current_var_params.get('wedge_julia_angle', 0.0)
        count = _current_var_params.get('wedge_julia_count', 1.0)
        power = _current_var_params.get('wedge_julia_power', 2.0)
        dist = _current_var_params.get('wedge_julia_dist', 1.0)
        abs_n = abs(power)
        cn = dist / power * 0.5
        cf = 1.0 - ang * count / np.pi * 0.5
        t_rnd = int(abs_n * _rand())
        a = (np.arctan2(y, x) + 2.0 * np.pi * t_rnd) / power
        c = np.floor((count * a + np.pi) / np.pi * 0.5)
        a = a * cf + c * ang
        # flam3 uses pow(r², cn) via sumsq — already using x*x+y*y here
        ri = max(x*x + y*y, 1e-10) ** cn
        return w * ri * np.cos(a), w * ri * np.sin(a)
    elif var_idx == 95: # wedge
        ang = _current_var_params.get('wedge_angle', 0.0)
        hole = _current_var_params.get('wedge_hole', 0.0)
        count = _current_var_params.get('wedge_count', 1.0)
        swirl = _current_var_params.get('wedge_swirl', 0.0)
        ri = np.sqrt(x*x + y*y)
        a = np.arctan2(y, x) + swirl * ri
        c = np.floor((count * a + np.pi) / np.pi * 0.5)
        cf = 1.0 - ang * count / np.pi * 0.5
        a = a * cf + c * ang
        ri = w * (ri + hole)
        return ri * np.cos(a), ri * np.sin(a)
    elif var_idx == 96: # wedge_sph
        ang = _current_var_params.get('wedge_sph_angle', 0.0)
        hole = _current_var_params.get('wedge_sph_hole', 0.0)
        count = _current_var_params.get('wedge_sph_count', 1.0)
        swirl = _current_var_params.get('wedge_sph_swirl', 0.0)
        ri = 1.0 / (np.sqrt(x*x + y*y) + 1e-10)
        a = np.arctan2(y, x) + swirl * ri
        c = np.floor((count * a + np.pi) / np.pi * 0.5)
        cf = 1.0 - ang * count / np.pi * 0.5
        a = a * cf + c * ang
        ri = w * (ri + hole)
        return ri * np.cos(a), ri * np.sin(a)
    elif var_idx == 97: # lazysusan
        lx = _current_var_params.get('lazysusan_x', 0.0)
        ly = _current_var_params.get('lazysusan_y', 0.0)
        spin = _current_var_params.get('lazysusan_spin', 0.0)
        space = _current_var_params.get('lazysusan_space', 0.0)
        twist = _current_var_params.get('lazysusan_twist', 0.0)
        xx = x - lx
        yy = y + ly
        rr = np.sqrt(xx*xx + yy*yy)
        if rr < w:
            a = np.arctan2(yy, xx) + spin + twist * (w - rr)
            rr = w * rr
            return rr * np.cos(a) + lx, rr * np.sin(a) - ly
        else:
            rr = w * (1.0 + space / max(rr, 1e-10))
            return rr * xx + lx, rr * yy - ly
    elif var_idx == 98: # modulus
        mx = _current_var_params.get('modulus_x', 1.0)
        my = _current_var_params.get('modulus_y', 1.0)
        xr = 2.0 * mx
        yr = 2.0 * my
        if x > mx: ox = -mx + ((x + mx) % xr)
        elif x < -mx: ox = mx - ((mx - x) % xr)
        else: ox = x
        if y > my: oy = -my + ((y + my) % yr)
        elif y < -my: oy = my - ((my - y) % yr)
        else: oy = y
        return w * ox, w * oy
    elif var_idx == 99: # bent2
        bx = _current_var_params.get('bent2_x', 1.0)
        by = _current_var_params.get('bent2_y', 1.0)
        nx = x * bx if x < 0.0 else x
        ny = y * by if y < 0.0 else y
        return w * nx, w * ny
    elif var_idx == 100: # bipolar
        shift = _current_var_params.get('bipolar_shift', 0.0)
        x2y2 = x*x + y*y
        ps = -np.pi * 0.5 * shift
        y2 = 0.5 * np.arctan2(2.0 * y, x2y2 - 1.0) + ps
        if y2 > np.pi * 0.5: y2 = -np.pi * 0.5 + (y2 + np.pi * 0.5) % np.pi
        elif y2 < -np.pi * 0.5: y2 = np.pi * 0.5 - (np.pi * 0.5 - y2) % np.pi
        t = x2y2 + 1.0
        tp = t + 2.0 * x
        tm = t - 2.0 * x
        if abs(tm) < 1e-10: tm = 1e-10
        return w * 0.25 * 2.0 / np.pi * np.log(tp / tm), w * 2.0 / np.pi * y2
    elif var_idx == 101: # flux
        spread = _current_var_params.get('flux_spread', 0.0)
        xpw = x + w
        xmw = x - w
        avgr = w * (2.0 + spread) * np.sqrt(np.sqrt(max(y*y + xpw*xpw, 1e-10)) / np.sqrt(max(y*y + xmw*xmw, 1e-10)))
        avga = (np.arctan2(y, xmw) - np.arctan2(y, xpw)) * 0.5
        return avgr * np.cos(avga), avgr * np.sin(avga)
    elif var_idx == 102: # split (different from splits!)
        xs = _current_var_params.get('split_xsize', 0.5)
        ys = _current_var_params.get('split_ysize', 0.5)
        sx = 1.0 if np.cos(x * xs * np.pi) >= 0.0 else -1.0
        sy = 1.0 if np.cos(y * ys * np.pi) >= 0.0 else -1.0
        return w * x * sy, w * y * sx
    elif var_idx == 103: # separation
        sepx = _current_var_params.get('separation_x', 0.5)
        sepy = _current_var_params.get('separation_y', 0.5)
        sexi = _current_var_params.get('separation_xinside', 0.0)
        seyi = _current_var_params.get('separation_yinside', 0.0)
        sx2 = sepx * sepx
        sy2 = sepy * sepy
        if x > 0.0: ox = w * (np.sqrt(x*x + sx2) - x * sexi)
        elif x < 0.0: ox = -w * (np.sqrt(x*x + sx2) + x * sexi)
        else: ox = 0.0
        if y > 0.0: oy = w * (np.sqrt(y*y + sy2) - y * seyi)
        elif y < 0.0: oy = -w * (np.sqrt(y*y + sy2) + y * seyi)
        else: oy = 0.0
        return ox, oy
    elif var_idx == 104: # polar2
        p2v = w / np.pi
        return p2v * np.arctan2(y, x), p2v * 0.5 * np.log(x*x + y*y + 1e-10)
    elif var_idx == 105: # foci
        expx = np.exp(min(x, 20.0)) * 0.5
        expnx = 0.25 / max(expx, 1e-10)
        tmp = expx + expnx - np.cos(y)
        if abs(tmp) < 1e-10: return w*x, w*y
        tmp = w / tmp
        return (expx - expnx) * tmp, np.sin(y) * tmp
    elif var_idx == 106: # popcorn2
        px = _current_var_params.get('popcorn2_x', 0.1)
        py = _current_var_params.get('popcorn2_y', 0.1)
        pc_ = _current_var_params.get('popcorn2_c', 3.0)
        return w * (x + px * np.sin(np.tan(y * pc_))), w * (y + py * np.sin(np.tan(x * pc_)))
    elif var_idx == 107: # secant (original)
        ri = w * np.sqrt(x*x + y*y)
        cr = np.cos(ri)
        if abs(cr) < 1e-10: return w*x, w*y
        return w * x, w / cr
    # --- complex trig (z = x+iy) ---
    elif var_idx == 108: # sin(z)
        return w * np.sin(x) * np.cosh(y), w * np.cos(x) * np.sinh(y)
    elif var_idx == 109: # cos(z)
        return w * np.cos(x) * np.cosh(y), -w * np.sin(x) * np.sinh(y)
    elif var_idx == 110: # tan(z)
        d = np.cos(2.0*x) + np.cosh(2.0*y)
        if abs(d) < 1e-10: return w*x, w*y
        return w * np.sin(2.0*x) / d, w * np.sinh(2.0*y) / d
    elif var_idx == 111: # sec(z)
        d = np.cos(2.0*x) + np.cosh(2.0*y)
        if abs(d) < 1e-10: return w*x, w*y
        s = 2.0 / d
        return w * s * np.cos(x) * np.cosh(y), w * s * np.sin(x) * np.sinh(y)
    elif var_idx == 112: # csc(z)
        d = np.cosh(2.0*y) - np.cos(2.0*x)
        if abs(d) < 1e-10: return w*x, w*y
        s = 2.0 / d
        return w * s * np.sin(x) * np.cosh(y), -w * s * np.cos(x) * np.sinh(y)
    elif var_idx == 113: # cot(z)
        d = np.cosh(2.0*y) - np.cos(2.0*x)
        if abs(d) < 1e-10: return w*x, w*y
        return w * np.sin(2.0*x) / d, -w * np.sinh(2.0*y) / d
    elif var_idx == 114: # sinh(z)
        return w * np.sinh(x) * np.cos(y), w * np.cosh(x) * np.sin(y)
    elif var_idx == 115: # cosh(z)
        return w * np.cosh(x) * np.cos(y), w * np.sinh(x) * np.sin(y)
    elif var_idx == 116: # tanh(z)
        d = np.cos(2.0*y) + np.cosh(2.0*x)
        if abs(d) < 1e-10: return w*x, w*y
        return w * np.sinh(2.0*x) / d, w * np.sin(2.0*y) / d
    elif var_idx == 117: # sech(z)
        d = np.cos(2.0*y) + np.cosh(2.0*x)
        if abs(d) < 1e-10: return w*x, w*y
        s = 2.0 / d
        return w * s * np.cos(y) * np.cosh(x), -w * s * np.sin(y) * np.sinh(x)
    elif var_idx == 118: # csch(z)
        d = np.cosh(2.0*x) - np.cos(2.0*y)
        if abs(d) < 1e-10: return w*x, w*y
        s = 2.0 / d
        return w * s * np.sinh(x) * np.cos(y), -w * s * np.cosh(x) * np.sin(y)
    elif var_idx == 119: # coth(z)
        d = np.cosh(2.0*x) - np.cos(2.0*y)
        if abs(d) < 1e-10: return w*x, w*y
        return w * np.sinh(2.0*x) / d, w * np.sin(2.0*y) / d
    elif var_idx == 120: # exp(z) — complex exp, different from exponential (idx 18)
        expe = np.exp(min(x, 20.0))
        return w * expe * np.cos(y), w * expe * np.sin(y)
    elif var_idx == 121: # log(z) — complex log
        return w * 0.5 * np.log(x*x + y*y + 1e-10), w * np.arctan2(y, x)
    elif var_idx == 122: # splitbrdr — bubble + tiled border
        B = (x*x + y*y) / 4.0 + 1.0
        b = w / B
        rx = round(x)
        ry = round(y)
        ox = x - rx
        oy = y - ry
        sb_x = _current_var_params.get('splitbrdr_x', 0.25)
        sb_y = _current_var_params.get('splitbrdr_y', 0.25)
        sb_px = _current_var_params.get('splitbrdr_px', 0.0)
        sb_py = _current_var_params.get('splitbrdr_py', 0.0)
        out_x = x * b + x * sb_px
        out_y = y * b + y * sb_py
        if _rand() >= 0.75:
            out_x += w * (ox * 0.5 + rx)
            out_y += w * (oy * 0.5 + ry)
        else:
            if abs(ox) >= abs(oy):
                if ox >= 0.0:
                    out_x += w * (ox * 0.5 + rx + sb_x)
                    out_y += w * (oy * 0.5 + ry + sb_y * oy / (ox + 1e-10))
                else:
                    out_x += w * (ox * 0.5 + rx - sb_y)
                    out_y += w * (oy * 0.5 + ry - sb_y * oy / (ox - 1e-10))
            else:
                if oy >= 0.0:
                    out_y += w * (oy * 0.5 + ry + sb_y)
                    out_x += w * (ox * 0.5 + rx + ox / (oy + 1e-10) * sb_y)
                else:
                    out_y += w * (oy * 0.5 + ry - sb_y)
                    out_x += w * (ox * 0.5 + rx - ox / (oy - 1e-10) * sb_x)
        return out_x, out_y
    elif var_idx == 123: # phoenix_julia
        power = _current_var_params.get('phoenix_power', 3.0)
        dist = _current_var_params.get('phoenix_dist', 1.0)
        x_distort = _current_var_params.get('phoenix_x_distort', -0.5)
        y_distort = _current_var_params.get('phoenix_y_distort', 0.0)
        inv_n = dist / power
        inv_2pi_n = 2.0 * np.pi / power
        cn = dist / power / 2.0
        pre_x = x * (x_distort + 1.0)
        pre_y = y * (y_distort + 1.0)
        a = np.arctan2(pre_y, pre_x) * inv_n + int(_rand() * 32767) * inv_2pi_n
        r = w * (x*x + y*y + 1e-10) ** cn
        return r * np.cos(a), r * np.sin(a)
    elif var_idx == 124: # juliaq
        power = _current_var_params.get('juliaq_power', 3.0)
        divisor = _current_var_params.get('juliaq_divisor', 2.0)
        if abs(power) < 1e-10: power = 1.0
        inv_power = divisor / power
        inv_power_2pi = 2.0 * np.pi / power
        half_inv_power = 0.5 * divisor / power
        a = np.arctan2(y, x) * inv_power + int(_rand() * 10) * inv_power_2pi
        r = w * (x*x + y*y + 1e-10) ** half_inv_power
        return r * np.cos(a), r * np.sin(a)
    elif var_idx == 125: # minkowskope
        separation = _current_var_params.get('mskope_separation', 0.5)
        freq_x = _current_var_params.get('mskope_frequencyx', -2.0)
        freq_y = _current_var_params.get('mskope_frequencyy', 2.0)
        amplitude = _current_var_params.get('mskope_amplitude', 0.5)
        perturbation = _current_var_params.get('mskope_perturbation', 1.0)
        damping = _current_var_params.get('mskope_damping', 0.0)
        alt_wave = freq_x <= 0.0
        tpf = 0.5 * freq_x
        tpf2 = 0.5 * freq_y
        def _minkowski(xv):
            p, q, r, s = 0.0, 1.0, 1.0, 1.0
            d, yv = 1.0, 0.0
            for _ in range(20):
                d *= 0.5
                m = p + r
                n = q + s
                if xv < m / n:
                    r, s = m, n
                else:
                    yv += d
                    p, q = m, n
            return yv + d
        def _minkosine(xv):
            lp = abs(xv) % 4.0
            p = abs(xv) % 2.0
            if p > 1.0: p = 2.0 - p
            mink = _minkowski(p) - p if alt_wave else _minkowski(p)
            if (lp < 2.0) ^ (xv > 0): return mink
            return -mink
        pt = perturbation * _minkosine(tpf2 * y)
        if abs(damping) < 1e-6:
            t = amplitude * _minkosine(tpf * x + pt - 1.0) + separation
        else:
            t = amplitude * np.exp(-abs(x) * damping) * _minkosine(tpf * x + pt - 1.0) + separation
        if abs(y) <= t:
            return -w*x, -w*y
        return w*x, w*y
    elif var_idx == 126: # glynnia
        vvar2 = w * np.sqrt(2.0) / 2.0
        r = np.sqrt(x*x + y*y)
        if r >= 1.0:
            if _rand() > 0.5:
                d = np.sqrt(r + x)
                if abs(d) < 1e-10: return w*x, w*y
                return vvar2 * d, -vvar2 / d * y
            else:
                d = r + x
                dx = np.sqrt(r * (y*y + d*d))
                if abs(dx) < 1e-10: return w*x, w*y
                rr = w / dx
                return rr * d, rr * y
        else:
            if _rand() > 0.5:
                d = np.sqrt(r + x)
                if abs(d) < 1e-10: return w*x, w*y
                return -vvar2 * d, -vvar2 / d * y
            else:
                d = r + x
                dx = np.sqrt(r * (y*y + d*d))
                if abs(dx) < 1e-10: return w*x, w*y
                rr = w / dx
                return -rr * d, rr * y
    else:
        # treat unknown/safe variations as linear for viability purposes
        return w*x, w*y
