import numpy as np

from go_pipeline.board import GameState, play_move
from go_pipeline.katago import encode_katago


def test_encode_katago_empty_board():
    spatial, globals_ = encode_katago(GameState(), global_count=14)
    assert spatial.shape == (22, 9, 9)
    assert globals_.shape == (14,)
    assert spatial[0].sum() == 81  # on-board plane
    assert spatial[1].sum() == spatial[2].sum() == 0
    assert globals_[5] == np.float32(-7.0 / 20.0)  # Black to move: self-komi = -7


def test_encode_katago_history_and_pass_globals():
    state = play_move(GameState(), 40)
    state = play_move(state, 81)  # White passes
    spatial, globals_ = encode_katago(state, global_count=19)
    assert spatial[1].sum() == 1 and spatial[2].sum() == 0  # Black stone, Black to move? perspective: own=white? next is Black
    assert globals_[0] == 1.0  # most recent move was a pass
    assert spatial[9].sum() == 0  # pass leaves no spatial history mark


def test_encode_katago_liberty_planes():
    state = GameState()
    for move in [10, 0, 9, 2, 19, 80, 11]:
        nxt = play_move(state, move)
        assert nxt is not None
        state = nxt
    spatial, _ = encode_katago(state, global_count=14)
    assert spatial[3].sum() + spatial[4].sum() + spatial[5].sum() > 0
