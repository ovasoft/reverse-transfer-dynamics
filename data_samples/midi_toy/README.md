# Toy MIDI corpus

Three short synthetic MIDI files (a repeated scale motif, a repeated
arpeggio, and an irregular pitch random-walk) generated with `symusic`,
NOT drawn from the real Lakh MIDI Dataset. They exist to exercise the
REMI tokenization + M1/M2/M3 pipeline end-to-end on something real but
tiny, and were chosen to span the metrics meaningfully: the scale/arpeggio
files should score high on M1 (repetitiveness) and M2 (regularity), the
random walk should score lower on both -- a good sanity check that your
metric implementations are pointed the right direction before trusting
them on real data.
