"""Drift axis — enters slow morph mode when audio is quiet."""

import logging

log = logging.getLogger(__name__)


from flame_sheep.audio._types import AudioState


class DriftAxis:
    """Silence -> slow genome drift + loop cycling.

    When RMS stays below threshold for SWAP_FRAMES, triggers a genome
    swap on the GenomeAxis. After 3 full loop cycles, switches to a
    new loop.
    """

    RMS_THRESHOLD   = 0.0001
    SWAP_FRAMES     = 60 * 8   # ~8 seconds at 60fps
    DRIFT_MORPH_SPEED = 0.003
    CYCLES_PER_LOOP = 3

    def __init__(self, genome_axis):
        self.enabled = True
        self._genome_axis = genome_axis
        self._quiet_frames = 0
        self._loop_cycles = 0
        self.drifting = False  # True when in sustained silence

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        if audio.rms < self.RMS_THRESHOLD:
            self._quiet_frames += 1
            self.drifting = self._quiet_frames >= self.SWAP_FRAMES
            if self._quiet_frames >= self.SWAP_FRAMES:
                self._quiet_frames = 0
                self._genome_axis._swap_next_genome()
                self._genome_axis.morph_t     = 0.0
                self._genome_axis.morph_speed = self.DRIFT_MORPH_SPEED
                dist = self._genome_axis.current_genome.distance(
                    self._genome_axis.target_genome)
                log.debug(f'[drift] rms={audio.rms:.5f}  dist={dist:.3f}')

                # After full loop cycles, switch loops
                if (self._genome_axis._loop_genomes
                        and self._genome_axis._loop_pos == 0
                        and self._genome_axis._lib is not None):
                    self._loop_cycles += 1
                    if self._loop_cycles >= self.CYCLES_PER_LOOP:
                        self._loop_cycles = 0
                        self._genome_axis.next_loop()
                        log.info(f'[drift] switched to loop '
                              f'#{self._genome_axis.active_loop_id}')
        else:
            self._quiet_frames = 0
            self.drifting = False

    def contribute(self, frame) -> None:
        pass  # drift affects genome axis state directly, no frame output
