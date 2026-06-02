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
 
KOMOR_DATA_DIR = os.path.expanduser('~/code_komor/data')
INSECTS_STREAMS_DIR = os.path.join(KOMOR_DATA_DIR, 'real_streams_pr')
INSECTS_GT_DIR = os.path.join(KOMOR_DATA_DIR, 'real_streams_gt')
 
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'real')
os.makedirs(RESULTS_DIR, exist_ok=True)


real_streams = [
    'real_streams/covtypeNorm-1-2vsAll-pruned.arff',
    'real_streams/electricity.npy',
    'real_streams/poker-lsn-1-2vsAll-pruned.arff',
    'real_streams/INSECTS-abrupt_imbalanced_norm.arff',
    'real_streams/INSECTS-gradual_imbalanced_norm.arff',
    'real_streams/INSECTS-incremental_imbalanced_norm.arff'
    ]

stream_static = { 'chunk_size': 300 }

measures = ["clustering",
        "complexity",
        "concept",
        "general",
        "info-theory",
        "itemset",
        "landmarking",
        "model-based",
        "statistical"
        ]

pbar = tqdm(total=len(real_streams))

for m_id, measure_key in enumerate(measures):
    print(measure_key)
    
    for f_id, f in enumerate(real_streams):
        fname=(f.split('/')[1]).split('.')[0]

        drfs = np.load('real_streams_gt/%s.npy' % fname)
       
        concept=0
        out = []
        
        stream = sl.streams.NPYParser('real_streams_pr/%s.npy' % fname, chunk_size=stream_static['chunk_size'], n_chunks=100000)
        
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
                
        np.save('res/real_%i_%s.npy' % (f_id, measure_key), np.array(out))
