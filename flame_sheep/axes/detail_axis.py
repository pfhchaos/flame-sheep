"""Detail axis — maps audio energy to chaos game iteration count.

Uses the slow-envelope harmonic RMS from the audio engine so that
sustained musical energy drives detail, not individual hits.
"""

from flame_sheep_audio import AudioState


class DetailAxis:
    """Energy -> iterations. Quiet=sparse, loud=dense and detailed."""

    def __init__(self, min_iters: int = 100, max_iters: int = 500,
                 rms_scale: float = 0.01):
        self.enabled = True
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.rms_scale = rms_scale
        self.iterations = min_iters

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = max(audio.bands['subbass'].slow_harmonic_rms,
                     audio.centroid_harmonic_rms)
        t = min(energy / self.rms_scale, 1.0) ** 0.5
        self.iterations = int(self.min_iters + t * (self.max_iters - self.min_iters))

    def contribute(self, frame) -> None:
        frame.iterations = self.iterations
