import numpy as np
import pytest

from gm_simulator import Detector, Source, World


@pytest.mark.parametrize("dist", [-1, -0.5, 0.5, 1])
@pytest.mark.parametrize("q", [1.0, 2.0, 3.0])
def test_detector_singlesrc_count(dist, q):
    # q: Source intensity
    # dist: Distance between source and detector
    # TODO statistic method for testing

    np.random.seed(1234)
    N = 200
    # Source location
    pt = np.array([1.0, 0.0, 0.0])
    # Detector info
    duration = 2
    detect_factor = 5

    # Construct world
    world = World()
    world.add_source(Source(loc=pt.tolist(), intensity=q))

    # Expected values
    expected = duration * detect_factor * q / dist / dist
    for i in range(3):
        diff_vec = np.array([0.0, 0.0, 0.0])
        diff_vec[i] = dist
        d_loc = pt + diff_vec
        detector = Detector(loc=d_loc, duration=duration, factor=detect_factor)
        cnt_mean = np.array([world.get_measuments(detector) for _ in range(N)]).mean()
        error = abs((cnt_mean - expected)) / expected
        assert error < 0.05


@pytest.mark.parametrize("N_src", [2, 3])
@pytest.mark.parametrize("dist", [0.5, 1])
@pytest.mark.parametrize("q", [1.0, 2.0, 3.0])
def test_detector_multisrc_count(N_src, dist, q):
    # N_src: Number of source
    # q: Source intensity
    # dist: Distance between source and detector
    # TODO statistic method for testing

    np.random.seed(1234)
    N = 200
    # Source location
    pt = np.array([1.0, 0.0, 0.0])
    # Detector info
    duration = 2
    detect_factor = 5

    # Construct world
    world = World()
    for _ in range(N_src):
        world.add_source(Source(loc=pt.tolist(), intensity=q))

    # Expected values
    expected = N_src * duration * detect_factor * q / dist / dist
    for i in range(3):
        diff_vec = np.array([0.0, 0.0, 0.0])
        diff_vec[i] = dist
        d_loc = pt + diff_vec
        detector = Detector(loc=d_loc, duration=duration, factor=detect_factor)
        cnt_mean = np.array([world.get_measuments(detector) for _ in range(N)]).mean()
        error = abs((cnt_mean - expected)) / expected
        assert error < 0.05
