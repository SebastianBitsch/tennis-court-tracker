import numpy as np
from dataclasses import dataclass

@dataclass
class Court:
    point_names: dict[int, str]
    point_positions: dict[int, tuple[float, float]]
    line_names: dict[int, str]
    lines: dict[int, tuple[int, int]]

    @property
    def points(self) -> np.ndarray:
        """ Get the point positions as an array """
        return np.array(list(self.point_positions.values()))


TENNISCOURT = Court(
    point_names = {
        0 : 'left-near-doubles-sideline',
        1 : 'left-near-singles-sideline',
        2 : 'right-near-singles-sideline',
        3 : 'right-near-doubles-sideline',
        4 : 'left-near-serviceline',
        5 : 'center-near-serviceline',
        6 : 'right-near-serviceline',
        7 : 'left-far-serviceline',
        8 : 'center-far-serviceline',
        9 : 'right-far-serviceline',
        10: 'left-far-doubles-sideline',
        11: 'left-far-singles-sideline',
        12: 'right-far-singles-sideline',
        13: 'right-far-doubles-sideline',
    },
    point_positions = {
        0 : (0.0,   0),
        1 : (13.7,  0),
        2 : (96.0,  0),
        3 : (109.7, 0),
        4 : (13.7,  54.8),
        5 : (54.8,  54.8),
        6 : (96.0,  54.8),
        7 : (13.7, 182.8),
        8 : (54.8, 182.8),
        9 : (96.0, 182.8),
        10: (0.0, 237.7),
        11: (13.7, 237.7),
        12: (96.0, 237.7),
        13: (109.7, 237.7)
    },
    line_names = {
        0 : 'left-doubles-sideline',
        1 : 'left-singles-sideline',
        2 : 'right-singles-sideline',
        3 : 'right-doubles-sideline',
        4 : 'near-baseline',
        5 : 'near-serviceline',
        6 : 'center-serviceline',
        7 : 'far-serviceline',
        8 : 'far-baseline'
    },
    lines = {
        0 : (0,  10),
        1 : (1,  11),
        2 : (2,  12),
        3 : (3,  13),
        4 : (0,  3),
        5 : (4,  6),
        6 : (5,  8),
        7 : (7,  9),
        8 : (10, 13)
    }
)


if __name__ == "__main__":
    print(TENNISCOURT.points)