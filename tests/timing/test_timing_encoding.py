"""Tests timing of encoding classes

"""

import time

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from dsutils.encoding import MultiTargetEncoderLOO


def test_timing_MultiTargetEncoderLOO():
    """Tests timing of encoding.MultiTargetEncoderLOO"""

    # Dummy data
    N = 10000
    Nc = 100
    df = pd.DataFrame()
    cat1 = [str(e) for e in np.floor(Nc*np.random.randn(N))]
    cat2 = [str(e) for e in np.floor(Nc*np.random.randn(N))]
    df['a'] = [cat1[i]+','+cat2[i] for i in range(len(cat1))]
    df['b'] = np.random.randn(N)
    df['y'] = np.random.randn(N)

    # Encode the data 
    mte = MultiTargetEncoderLOO(cols='a')

    t0 = time.time()
    mte.fit_transform(df[['a', 'b']], df['y'])
    t1 = time.time()
    print('Elapsed time: ', t1-t0)



if __name__ == "__main__":
    test_timing_MultiTargetEncoderLOO()