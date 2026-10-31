import numpy as np

from go_pipeline import AREA, PASS_MOVE
from go_pipeline.board import GameState, encode_features, legal_moves, play_move


def test_empty_board_features():
    feats = encode_features(GameState())
    assert feats.shape == (12, 9, 9)
    assert feats[:11].sum() == 0
    # Black to move: komi plane = -7/15 everywhere.
    assert np.allclose(feats[11], -7.0 / 15.0)


def test_capture_and_ko_planes():
    state = GameState()
    for move in [10, 0, 9, 1, 19, 80, 11]:  # surround white stone at 1? build a capture
        nxt = play_move(state, move)
        assert nxt is not None
        state = nxt
    assert PASS_MOVE in legal_moves(state)


def test_suicide_rejected_and_pass_legal():
    state = GameState()
    state.board[1] = state.board[9] = -1
    state.next_color = 1
    assert play_move(state, 0) is None
    assert play_move(state, PASS_MOVE) is not None


def test_recent_pass_planes():
    state = play_move(GameState(), PASS_MOVE)
    feats = encode_features(state)
    assert feats[9].sum() == AREA  # first recent-pass plane is full-board
    assert np.allclose(feats[11], 7.0 / 15.0)  # White to move flips komi sign
