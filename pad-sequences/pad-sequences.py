import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max(len(s) for s in seqs) if seqs else 0

    result = np.full((len(seqs), max_len), pad_value)

    for i, s in enumerate(seqs):
        s_arr = np.array(s)

        if len(s_arr) > max_len: 
            imp_val = s_arr[:max_len]
        else:
            imp_val = s_arr

        if len(imp_val) > 0:
            result[i, :len(imp_val)] = imp_val    
    
    return result
    pass