# Acknowledgments

## Libraries & Algorithms

**rt-cqt** — Real-time Constant Q Transform
- Author: Jonas Merkt
- License: BSD 3-Clause
- https://github.com/jmerkt/rt-cqt
- Used for: CQT spectrum engine (SlidingCqt)

**pocketfft** — Header-only FFT library
- Author: Martin Reinecke
- License: BSD 3-Clause
- https://github.com/mreineck/pocketfft
- Used for: C++ octave bank FFT engine (vendored)

**pffft** — Pretty Fast FFT (used by rt-cqt)
- https://github.com/marton78/pffft
- Used for: FFT within rt-cqt's CQT implementation

## Papers

**Tempo, Beat and Downbeat Estimation Tutorial**
- https://tempobeatdownbeat.github.io/tutorial/ch2_basics/baseline.html
- Comprehensive overview of beat tracking approaches, fixed-frame vs beat-synchronous

**BTrack** — Real-time beat tracker
- Author: Adam Stark
- License: GPL-3
- https://github.com/adamstark/BTrack
- Used for: tempo estimation (BTrackTempoTracker)
- Our patch: configurable sample rate (upstream hardcodes 44100)

**Gist** — Audio analysis library (reference)
- Author: Adam Stark
- License: GPL-3
- https://github.com/adamstark/Gist
- Referenced for: complex spectral difference, onset detection functions

**Streamlined Tempo Estimation Based on Autocorrelation and Cross-correlation With Pulses**
- Graham Percival & George Tzanetakis (2014)
- IEEE/ACM Transactions on Audio, Speech, and Language Processing
- https://webhome.csc.uvic.ca/~gtzan/output/taslp2014-tempo-gtzan.pdf
- Basis for: tempo tracker architecture (log OSS, enhanced ACF, pulse train scoring)

**Sliding with a Constant Q**
- Russell Bradford, John ffitch & Richard Dobson (2008)
- DAFx-08, Espoo, Finland
- Basis for: understanding the SlidingCqt algorithm in rt-cqt

**Tempo, Beat and Downbeat Estimation Tutorial**
- ISMIR 2021
- https://tempobeatdownbeat.github.io/tutorial/ch2_basics/baseline.html
- Reference for: beat tracking approaches, fixed-frame vs beat-synchronous
