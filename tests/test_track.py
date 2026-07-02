import numpy as np

from f1lap.track import rotate_points


def test_rotate_points_90_degrees_matches_fastf1_formula():
    points = np.array([[1.0, 0.0]])
    rotated = rotate_points(points, angle_radians=np.pi / 2)

    assert np.allclose(rotated, np.array([[0.0, 1.0]]), atol=1e-7)
