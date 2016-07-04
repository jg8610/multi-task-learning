import pandas as pd
import numpy as np
import pdb

def pos_eval(path):
    import numpy as np
    data = pd.read_csv(path, sep=' ', header=None)
    targ = data[1].as_matrix()
    pred = data[3].as_matrix()
    return np.sum(targ == pred)/float(len(targ))
