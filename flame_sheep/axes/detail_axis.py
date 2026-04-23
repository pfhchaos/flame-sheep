"""Detail axis — maps audio energy to chaos game iteration count."""

from flame_sheep.audio._types import AudioState


class DetailAxis:
    """Energy -> iterations. Quiet=sparse, loud=dense and detailed.

    Uses centroid RMS (energy around the dominant frequency) so that
    any strong musical content drives detail, not just bass.
    """

    def __init__(self, min_iters: int = 100, max_iters: int = 500,
                 rms_scale: float = 0.01):
        self.enabled = True
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.rms_scale = rms_scale
        self.iterations = min_iters

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        energy = max(audio.rms, audio.centroid_rms)
        t = min(energy / self.rms_scale, 1.0) ** 0.5
        self.iterations = int(self.min_iters + t * (self.max_iters - self.min_iters))

    def contribute(self, frame) -> None:
        frame.iterations = self.iterations
