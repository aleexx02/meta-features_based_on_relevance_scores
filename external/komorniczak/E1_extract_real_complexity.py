# E1_extract_real.py

"""
Experiment 1 -- collect streams and metafetaures -- real-world streams
"""
import os
import sys
import numpy as np
from pymfe.mfe import MFE
from tqdm import tqdm
import strlearn as sl


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
 
INSECTS_STREAMS_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_streams')
INSECTS_GT_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_streams_gt')
 
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'external', 'komorniczak', 'results', 'real')
os.makedirs(RESULTS_DIR, exist_ok=True)


real_streams = [
    'poker-lsn-1-2vsAll-pruned',
    'INSECTS-abrupt_imbalanced_norm',
    'INSECTS-gradual_imbalanced_norm',
    'INSECTS-incremental_imbalanced_norm',
    ]

stream_static = { 'chunk_size': 300 }

measures = [
        "complexity"
        ]

pbar = tqdm(total=len(real_streams))

for m_id, measure_key in enumerate(measures):
    print(measure_key)
    
    for f_id, fname in enumerate(real_streams):

        drfs = np.load(os.path.join(INSECTS_GT_DIR, f'{fname}.npy'))
       
        concept=0
        out = []
        
        stream = sl.streams.NPYParser(
            os.path.join(INSECTS_STREAMS_DIR, f'{fname}.npy'),
            chunk_size=stream_static['chunk_size'],
            n_chunks=100000
        )
        
        for chunk in range(100000):
            # GET CONCEPT
            if chunk in drfs:
                concept+=1
                           
            # CALCULATE
            try:
                X, y = stream.get_chunk()
            except Exception as e:
                print(e)
                break
                    
            if len(np.unique(y))<2:
                continue
                                
            mfe = MFE(groups=[measure_key])
            mfe.fit(X,y)
            ft_labels, ft = mfe.extract()
            ft.append(concept)

            out.append(ft)
                
        out_path = os.path.join(RESULTS_DIR, f'komor_real_{fname}_{measure_key}.npy')
        np.save(out_path, np.array(out))
        print(f'  Saved: {out_path}  shape={np.array(out).shape}')
 
        pbar.update(1)
 
pbar.close()
print("\nE1 extraction complete.")
