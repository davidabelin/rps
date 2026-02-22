from rps_core.scoring import score_round


def test_scoring_truth_table():
    matrix = {
        (0, 0): 0,
        (0, 1): -1,
        (0, 2): 1,
        (1, 0): 1,
        (1, 1): 0,
        (1, 2): -1,
        (2, 0): -1,
        (2, 1): 1,
        (2, 2): 0,
    }
    for (left, right), expected in matrix.items():
        assert score_round(left, right) == expected
