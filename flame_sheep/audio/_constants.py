"""Audio processing constants."""

import numpy as np

SAMPLE_RATE    = 48000
DEFAULT_DEVICE = 'Companion Speaker Analog Surround 5.1'
BLOCK_SIZE     = 1024   # frames per sounddevice callback
FFT_SIZE       = 2048   # FFT window size
N_BINS         = FFT_SIZE // 2 + 1
HISTORY_LEN    = 20     # ~0.33s of flux history at 60fps

# Frequency bin array — shared across all instances
FREQS = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
