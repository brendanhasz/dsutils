"""Tests utils

"""



import numpy as np

from dsutils.utils import meshgridn, linspacen



def test_meshgridn():
    """Tests utils.meshgridn"""

    E = [np.linspace(0, 1, 3), np.linspace(2, 3, 3)]
    G = meshgridn(E)

    assert isinstance(G, np.ndarray)
    assert G.shape[0] == 9
    assert G.shape[1] == 2
    assert G[0,0] == 0
    assert G[1,0] == 0
    assert G[2,0] == 0
    assert G[3,0] == 0.5
    assert G[4,0] == 0.5
    assert G[5,0] == 0.5
    assert G[6,0] == 1
    assert G[7,0] == 1
    assert G[8,0] == 1
    assert G[0,1] == 2
    assert G[1,1] == 2.5
    assert G[2,1] == 3
    assert G[3,1] == 2
    assert G[4,1] == 2.5
    assert G[5,1] == 3
    assert G[6,1] == 2
    assert G[7,1] == 2.5
    assert G[8,1] == 3


def test_linspacen():
    """Tests utils.linspacen"""

    G = linspacen([0, 2], [1, 3], 9)

    assert isinstance(G, np.ndarray)
    assert G.shape[0] == 9
    assert G.shape[1] == 2
    assert G[0,0] == 0
    assert G[1,0] == 0
    assert G[2,0] == 0
    assert G[3,0] == 0.5
    assert G[4,0] == 0.5
    assert G[5,0] == 0.5
    assert G[6,0] == 1
    assert G[7,0] == 1
    assert G[8,0] == 1
    assert G[0,1] == 2
    assert G[1,1] == 2.5
    assert G[2,1] == 3
    assert G[3,1] == 2
    assert G[4,1] == 2.5
    assert G[5,1] == 3
    assert G[6,1] == 2
    assert G[7,1] == 2.5
    assert G[8,1] == 3

    # Should make the smallest grid w/ at least N points
    G = linspacen([0, 2], [1, 3], 5)

    assert isinstance(G, np.ndarray)
    assert G.shape[0] == 9
    assert G.shape[1] == 2
    