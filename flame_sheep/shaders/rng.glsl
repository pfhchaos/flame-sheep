// ------------------------------------------------------------
// Pseudo-random number generator
// We need per-invocation randomness. No stdlib rand() in GLSL.
// This is a simple hash-based LCG seeded from invocation ID + position.
// Good enough for the chaos game — we just need "random enough".
// ------------------------------------------------------------
uint rng_state;

void rng_seed(uint seed) {
    rng_state = seed;
}

uint rng_next() {
    // Xorshift32 — fast, decent quality
    rng_state ^= rng_state << 13u;
    rng_state ^= rng_state >> 17u;
    rng_state ^= rng_state << 5u;
    return rng_state;
}

float rng_float() {
    // 0..1
    return float(rng_next()) / float(0xFFFFFFFFu);
}
