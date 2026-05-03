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

**Streamlined Tempo Estimation Based on Autocorrelation and Cross-correlation With Pulses**
- Graham Percival & George Tzanetakis (2014)
- IEEE/ACM Transactions on Audio, Speech, and Language Processing
- https://webhome.csc.uvic.ca/~gtzan/output/taslp2014-tempo-gtzan.pdf
- Basis for: tempo tracker architecture (log OSS, enhanced ACF, pulse train scoring)
