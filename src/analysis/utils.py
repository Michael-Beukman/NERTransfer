from matplotlib import pyplot as plt
import pandas as pd
try:
   import cPickle as pickle
except:
   import pickle

import bz2
# ================================= Some utility functions regarding languages ================================= #

# Returns languages
def ORDER():
    return [ 'wol', 'pcm', 'yor', 'hau', 'ibo', 'luo', 'lug', 'kin', 'swa']

def LANGS(): return ORDER()

def REGIONS(): return {
        'wol': 'west', 
        'pcm': 'west', 
        'yor': 'west', 
        'hau': 'west', 
        'ibo': 'west', 

        'lug': 'east', 
        'luo': 'east', 
        'kin': 'east',
        'swa': 'east',
        'amh': 'east'
    }

def FAMILIES(): return {
    'hau': 'Afro­Asiatic­Chadic',
    
    'kin': 'Niger­Congo­Bantu',
    'lug': 'Niger­Congo­Bantu',
    'swa': 'Niger­Congo­Bantu',
    
    'luo': 'Nilo Saharan',
    'pcm': 'English Creole',
    'wol': 'Niger­CongoSenegambia',
    
    'ibo': 'Niger­Congo­Volta­Niger',
    'yor': 'Niger­Congo­Volta­Niger',
}

def FULLNAMES(): return {
    'yor':'Yoruba', 
    'hau':'Hausa', 
    'kin':'Kinyarwanda',           
    'lug':'luganda',  
    'pcm': 'Nigerian Pidgin',  
    'wol': 'Wolof',  
    'swa': 'Swahili',  
    'ibo': 'Igbo', 
    'luo': 'Luo',
    'base': "Base"
}
def order_columns_of_df(df: pd.DataFrame, rows=False, both=False):
    """Orders the df columns (or rows) based on the above order, for consistent tables + heatmaps.

    Args:
        df (pd.DataFrame): 
        rows (bool, optional): . Defaults to False.

    Returns:
        pd.DataFrame
    """
    if both:
        return order_columns_of_df(order_columns_of_df(df), rows=True)

    ordering =[(i, j) for i, j in enumerate(ORDER())]
    
    if rows:
        return df.reindex([o[1] for o in ordering])
    else:
        cols = list(df.columns)
        ordered = [i[1] for i in ordering if i[1] in cols]
        assert len(ordered) == len(cols)
        df = df[ordered]
        return df



# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def save_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_compressed_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


def savefig(name, pad=0.1):
    # consistent saving
    plt.savefig(name, bbox_inches='tight', pad_inches=pad)