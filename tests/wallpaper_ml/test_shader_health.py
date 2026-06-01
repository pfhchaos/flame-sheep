"""Static analysis tests for ML compute shaders.

Catches the software-atomic-float anti-pattern (`atomicCompSwap` inside
a loop body) that triggered the GRU backward GuC wedge on 2026-05-31.

The pattern looks like:

    while (true) {
        uint des = floatBitsToUint(uintBitsToFloat(exp) + delta);
        uint old = atomicCompSwap(grad[tgt], exp, des);
        if (old == exp) break;
        exp = old;
    }

Each iteration retries on contention. When B × T × N_params threads all
target the same parameter buffer (every backward pass), contention is
100% by construction and threads thrash. Mesa-Xe's worst-case cycle
estimate for such blocks runs into the millions; in practice they wedge
the GuC scheduler under sustained training.

The structural fix is workgroup-local shared-memory accumulation
followed by one hardware `atomicAdd` per parameter per workgroup
(standard cuDNN/oneDNN pattern). Tracked in task #69 — until that ships,
the six currently-affected shaders are listed in
SOFTWARE_ATOMIC_FLOAT_DEBT.

The detector is structural (any CAS in any loop body), not pattern-
matched on `while (true)` — float CAS is the wrong primitive *anywhere*
in a hot loop, regardless of the surrounding control flow.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


SHADERS_ROOT = (Path(__file__).resolve().parents[2]
                / 'wallpaper_ml' / 'src' / 'wallpaper_ml' / 'shaders')


# Shaders that currently contain the anti-pattern. New shaders MUST
# NOT be added here — they have to be written with workgroup-shared
# accumulation from the start. Existing entries are removed as
# task #69 ports them over.
SOFTWARE_ATOMIC_FLOAT_DEBT: set[str] = {
    'cnn/conv2d_backward_weights.comp',
    'cnn/gap_linear_backward.comp',
    'cnn/gap_mlp_backward.comp',
    'rnn/gru_backward.comp',
    'rnn/gru_seq_backward.comp',
    'rnn/linear_backward.comp',
}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    """Remove // line and /* block */ comments so loop/CAS tokens inside
    them don't confuse the parser."""
    src = re.sub(r'//[^\n]*', '', src)
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)
    return src


def find_cas_in_loops(source: str) -> list[tuple[int, str]]:
    """Walk shader source tracking brace depth + currently-open loop
    bodies. For each `atomicCompSwap` token encountered while inside
    one or more loop bodies, return (line_number, line_text).

    Loop bodies are tracked by recording brace depth at the point each
    for/while/do opens its body. When the closing brace drops depth
    back, the loop frame is popped. Single-statement loop bodies
    without braces (`for(...) stmt;`) are recognized but don't push a
    frame; the CAS would have to be in the bare statement, which
    doesn't match any real-world pattern we care about.

    Note: parser is pragmatic, not a full GLSL frontend. It doesn't
    handle preprocessor directives that hide braces, lambda-like
    constructs (don't exist in GLSL), or other esoterica. Good enough
    for production shader hygiene.
    """
    src = _strip_comments(source)
    findings: list[tuple[int, str]] = []
    lines = source.split('\n')

    brace_depth = 0
    loop_body_depths: list[int] = []  # brace depths where each open loop body started
    pending_loop_body = False
    line = 1
    i = 0
    N = len(src)

    while i < N:
        c = src[i]
        if c == '\n':
            line += 1
            i += 1
            continue
        if c == '{':
            brace_depth += 1
            if pending_loop_body:
                loop_body_depths.append(brace_depth)
                pending_loop_body = False
            i += 1
            continue
        if c == '}':
            while loop_body_depths and loop_body_depths[-1] == brace_depth:
                loop_body_depths.pop()
            brace_depth -= 1
            i += 1
            continue
        if c == ';':
            if pending_loop_body:
                # Single-statement loop body — no frame to track.
                pending_loop_body = False
            i += 1
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < N and (src[j].isalnum() or src[j] == '_'):
                j += 1
            tok = src[i:j]
            if tok in ('for', 'while'):
                # Skip the condition paren to avoid spurious tokens.
                k = j
                while k < N and src[k] != '(':
                    if src[k] == '\n':
                        line += 1
                    k += 1
                if k < N:
                    depth = 1
                    k += 1
                    while k < N and depth > 0:
                        if src[k] == '(':
                            depth += 1
                        elif src[k] == ')':
                            depth -= 1
                        elif src[k] == '\n':
                            line += 1
                        k += 1
                pending_loop_body = True
                i = k
                continue
            if tok == 'do':
                pending_loop_body = True
                i = j
                continue
            if tok == 'atomicCompSwap' and loop_body_depths:
                text = lines[line - 1].rstrip() if 0 < line <= len(lines) else tok
                findings.append((line, text))
            i = j
            continue
        i += 1

    return findings


# ---------------------------------------------------------------------------
# Self-tests for the scanner (so the test that uses it is itself trusted)
# ---------------------------------------------------------------------------

def test_scanner_finds_cas_in_while_loop():
    src = """
    void main() {
        while (true) {
            atomicCompSwap(buf[0], 0u, 1u);
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert len(findings) == 1
    assert findings[0][0] == 4


def test_scanner_finds_cas_in_for_loop():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            atomicCompSwap(buf[i], 0u, 1u);
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert len(findings) == 1


def test_scanner_finds_cas_in_nested_loops():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                while (true) {
                    atomicCompSwap(buf[0], 0u, 1u);
                }
            }
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert len(findings) == 1


def test_scanner_does_not_flag_cas_outside_any_loop():
    src = """
    void main() {
        atomicCompSwap(buf[0], 0u, 1u);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_does_not_flag_cas_after_loop_closes():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            // unrelated work
            buf[i] += 1u;
        }
        atomicCompSwap(buf[0], 0u, 1u);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_ignores_cas_in_comment():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            // atomicCompSwap(buf[0], 0u, 1u);
            /* atomicCompSwap(buf[0], 0u, 1u); */
            buf[i] += 1u;
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_handles_for_keyword_in_identifier():
    """Words like 'before' or 'format' contain 'for' but aren't loops."""
    src = """
    void main() {
        int format = 0;
        atomicCompSwap(buf[0], 0u, 1u);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


# ---------------------------------------------------------------------------
# Whole-shader-tree scan
# ---------------------------------------------------------------------------

def _all_shader_files() -> list[Path]:
    if not SHADERS_ROOT.exists():
        pytest.skip(f'shader root not found: {SHADERS_ROOT}')
    return sorted(SHADERS_ROOT.rglob('*.comp'))


def _relative_key(path: Path) -> str:
    """Stable name for the debt allowlist: 'cnn/foo.comp'."""
    return str(path.relative_to(SHADERS_ROOT))


@pytest.mark.parametrize('shader_path', _all_shader_files(),
                          ids=lambda p: _relative_key(p))
def test_no_new_software_atomic_float(shader_path: Path):
    """Every .comp must either be free of the anti-pattern, or be
    listed in SOFTWARE_ATOMIC_FLOAT_DEBT with an explanatory entry
    while task #69 is in flight.

    If you wrote a new shader and this test fails, DO NOT add it to
    the debt list — use the workgroup-shared accumulation pattern
    instead. See `tests/wallpaper_ml/test_shader_health.py` docstring
    for the canonical fix.

    If you ported a debt shader to the new pattern, REMOVE it from
    SOFTWARE_ATOMIC_FLOAT_DEBT and this test will then enforce that
    it stays clean.
    """
    key = _relative_key(shader_path)
    findings = find_cas_in_loops(shader_path.read_text())

    if key in SOFTWARE_ATOMIC_FLOAT_DEBT:
        if not findings:
            pytest.fail(
                f'{key} is listed in SOFTWARE_ATOMIC_FLOAT_DEBT but no '
                f'atomicCompSwap-in-loop was found. The debt has been '
                f'resolved — remove this shader from the allowlist.')
        # Debt acknowledged; pass.
        return

    if findings:
        msg = [
            f'{key} contains atomicCompSwap inside a loop body '
            f'(software-atomic-float anti-pattern):'
        ]
        for line_num, text in findings:
            msg.append(f'  line {line_num}: {text}')
        msg.append('')
        msg.append('Fix: workgroup-local shared-memory accumulation + '
                   'single atomicAdd per workgroup. See task #69 and '
                   'the gru_seq_backward.comp port for the reference '
                   'implementation pattern.')
        pytest.fail('\n'.join(msg))


def test_debt_list_matches_reality():
    """Every entry in SOFTWARE_ATOMIC_FLOAT_DEBT must correspond to a
    real shader file. Stale entries make the debt look bigger than it
    is and break the resolution check above."""
    for key in SOFTWARE_ATOMIC_FLOAT_DEBT:
        path = SHADERS_ROOT / key
        assert path.exists(), (
            f'SOFTWARE_ATOMIC_FLOAT_DEBT references {key!r} but no '
            f'such file exists. Was the shader renamed or deleted?')
