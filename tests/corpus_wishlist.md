# Corpus Wishlist

Tracks coverage gaps in the test corpus. Check items off as beatmaps
or clips are added to corpus.toml.

## Genre coverage
- [x] EDM/electronic (dubstep, future bass, hardcore, DnB)
- [x] Rock/pop (j-rock, alt rock)
- [x] Jazz (big band, bossa nova, fusion)
- [x] Classical (piano etudes)
- [ ] Hip-hop/trap (half-time feel, 808 sub-bass)
- [x] Metal (death metal, symphonic)
- [ ] Acoustic/folk (sparse, voice + guitar)
- [ ] Ambient/drone (tests idle detection)

## Problem cases
- [ ] Vibrato vocals (opera, R&B melisma)
- [ ] Wall-of-bass compressed masters
- [ ] Tempo changes within song (accelerando/ritardando)
- [ ] Polyrhythm (3 against 4)
- [ ] Live recording with crowd noise
- [ ] DJ transition (two songs overlapping)
- [ ] Fade-in/fade-out
- [ ] Heavily sidechain-compressed EDM (pumping)

## Speech discrimination
- [x] Pure speech (Patrick Boyle clip)
- [ ] Speech over background music (podcast intro, talk show)
- [ ] Sung vs spoken transition (musical theater, rap)
- [ ] ASMR / whisper (very low energy)

## Tempo edge cases
- [x] Very slow (<70 BPM): Yiruma piano
- [x] Very fast (>180 BPM): UNDEAD CORPORATION, Big Black
- [ ] Half-time feel (sounds slow, actual tempo is fast)
- [x] Rubato (Chopin etudes)
- [ ] Metric modulation (prog rock)
- [x] Musical theater tango (Cell Block Tango)

## HPSS quality
- [x] Clean separation: piano solo (Yiruma, Chopin)
- [ ] Muddy separation: compressed EDM
- [ ] Vibrato-heavy: cello, opera
- [ ] Cymbal wash: jazz ride, orchestral crash

## Notes
- osu! beatmaps should be ranked (timing quality verified by nominators)
- YouTube clips are for edge cases without osu! coverage
- All audio stays in ~/.cache/flame-sheep/corpus/ (gitignored)
- Only corpus.toml goes in the repo
