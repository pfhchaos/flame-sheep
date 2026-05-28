"""Loop playback sequencing — yield items in cyclic / palindrome / rondo
order, and compute the integer step count of one full cycle."""

from __future__ import annotations

STRUCTURES = ('cyclic', 'palindrome', 'rondo')


def loop_sequence(items: list, structure: str = 'cyclic'):
    """Yield items in playback order for the given structure, repeating forever.

    Works with any list type — genomes, genome IDs, indices, etc.

    Cyclic:      A B C D  A B C D  A B C D ...
    Palindrome:  A B C D  C B  A B C D  C B ...  (no doubled endpoints)
    Rondo:       A B A C A D  A B A C A D ...    (home=first, episodes=rest)
    """
    if not items:
        return

    if structure == 'palindrome':
        forward = items
        backward = items[-2:0:-1]  # exclude first and last
        while True:
            yield from forward
            yield from backward
    elif structure == 'rondo':
        home = items[0]
        episodes = items[1:]
        if not episodes:
            # Degenerate: single genome, just repeat
            while True:
                yield home
        while True:
            for ep in episodes:
                yield home
                yield ep
    else:  # cyclic (default)
        while True:
            yield from items


def cycle_length(n_items: int, structure: str = 'cyclic') -> int:
    """Number of steps in one full cycle of a loop structure.

    Cyclic:      n
    Palindrome:  2*(n-1)  (forward + backward, no doubled endpoints)
    Rondo:       2*(n-1)  (home + episode for each of n-1 episodes)
    """
    if n_items <= 1:
        return max(n_items, 1)
    if structure == 'palindrome':
        return 2 * (n_items - 1)
    elif structure == 'rondo':
        return 2 * (n_items - 1)
    else:
        return n_items
