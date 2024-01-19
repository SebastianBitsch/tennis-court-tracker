import numpy as np

class TennisCourt:
    court_point_name = {
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
    }
    court_point_positions = {
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
    }
    court_line_names = {
        0 : 'left-doubles-sideline',
        1 : 'left-singles-sideline',
        2 : 'right-singles-sideline',
        3 : 'right-doubles-sideline',
        4 : 'near-baseline',
        5 : 'near-serviceline',
        6 : 'center-serviceline',
        7 : 'far-serviceline',
        8 : 'far-baseline'
    }
    court_lines = {
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

    court_points = np.array(list(court_point_positions.values()))

a = TennisCourt()
print(a.court_line_names)