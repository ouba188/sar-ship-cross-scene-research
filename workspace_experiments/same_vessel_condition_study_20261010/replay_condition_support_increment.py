"""Replay the isolated-increment child without changing the original receipt."""

import replay_condition_support_matching as replay


if __name__ == '__main__':
    replay.OUT = replay.BASE/'condition_support_increment_20261011'
    replay.main()
