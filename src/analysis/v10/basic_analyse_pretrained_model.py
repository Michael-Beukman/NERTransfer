from collections import defaultdict
import glob
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro

from analysis.utils import order_columns_of_df
DATA_PATH = "runs/v10/models"

def pretty(li, decimal: int = 2):
    assert len(li) == 5
    return f"{np.round(np.mean(li) * 100, decimal)} ({np.round(np.std(li) * 100, 1)})"

def do_pretty_latex(dic: Dict[str, Dict[str, List[float]]], save_name='../analysis/v10/results.tex'):
    """This takes in a dictionary of lists (results for 5 seeds), turns it into a latex table (through pandas)
        And makes it in the form mean (std).
        
        Also does statistical tests, specifically mannwhitneyu

    Args:
        dic (Dict[str, Dict[str, List[float]]]): [description]
        save_name (str, optional): [description]. Defaults to '../analysis/v10/results.tex'.
    """
    dic_to_print = defaultdict(lambda: dict())
    names = [
        'Pre-trained (same-language)', 'Pre-trained (base)', 'Pre-trained (swa)'
    ]
    replace = [
        r'$\text{base} \to X$',
        r'$\text{base}$',
        r'$\text{base} \to \text{swa}$'
    ]
    for n, r in zip(names, replace):
        # break
        dic[r] = dic[n]
        del dic[n]
    keys = sorted(list(dic.keys()))
    langs = list(dic[keys[0]].keys())
    for l in langs:
        alls = [np.mean(dic[XX][l]) for XX in keys]
        base_val = dic[replace[1]][l]
        for pretrained in keys:
            this_val = dic[pretrained][l]
            _, p = mannwhitneyu(this_val, base_val)
            prett = pretty(this_val, decimal=1)
            if p < 0.05:
                prett = r"$\text{" + prett + "}^*$"
            if max(alls) == np.mean(this_val):
                prett = prett.replace("text", 'textbf')
            dic_to_print[pretrained][l] = prett
    df = pd.DataFrame(dic_to_print).T
    df = order_columns_of_df(df)
    df.index.name = 'Starting point for NER fine-tune'
    df.to_latex(save_name, escape=False)    


def main():
    """
        What this tries to do is the following:
            - Which base models performed the best on the downstream NER task after being trained on that language.
        
        It compares base, lang-specific and swahili models after finetuning on the same NER data.
    """
    folders = glob.glob(os.path.join(DATA_PATH, '*_50'))
    # Base:  {target lang: score}
    dic = defaultdict(lambda: defaultdict(lambda: []))
    for path in folders:
        f = path.split("/")[-1]
        KK = os.path.join(path, 'test_results.txt')
        if not os.path.exists(KK): 
            print(f, 'does not exist')
            continue
        try:
            lang_finetune, _, _, lang_start, _, _, _, seed, _ = f.split("_")
        except Exception as e:
            print(f, e)
            continue

        with open(KK, 'r') as f:
            f1 = float(f.readlines()[0].strip().split(" = ")[1])
        print(f"Lang {lang_finetune} with starting on {lang_start} Got F1 = {f1}")
        dic[lang_start][lang_finetune].append(f1)
    new_dic = defaultdict(lambda: dict())
    for l, v in dic.items():
        for b, li in v.items():
            assert len(li) == 5
            new_dic[l][b] = pretty(li)
    df = pd.DataFrame(new_dic)
    print(df.T)
    os.makedirs('../analysis/v10/', exist_ok=True)
    df.T.to_csv('../analysis/v10/results_seed.csv')    
    
    # Make proper table for latex
    swahili_run = defaultdict(lambda: dict())
    new_dic = defaultdict(lambda: dict())
    new_dic_alls = defaultdict(lambda: dict())
    for l, v in dic.items():
        og_l = l
        if l not in ['base', 'swa']:
            l = 'same-language'
        for b, li in v.items():
            if l == 'same-language' and b == 'swa':
                swahili_run[og_l][b] = pretty(li)
            else:
                new_dic['Pre-trained (' + l + ")"][b] = pretty(li)
                new_dic_alls['Pre-trained (' + l + ")"][b] = li
            if l == b == 'swa':
                new_dic['Pre-trained (same-language)'][b] = pretty(li)
                new_dic_alls['Pre-trained (same-language)'][b] = li
                swahili_run[og_l][b] = pretty(li)

    df = pd.DataFrame(new_dic)
    os.makedirs('../analysis/v10/', exist_ok=True)
    df = order_columns_of_df(df.T)
    df.to_csv('../analysis/v10/results_compressed.csv')
    do_pretty_latex(dict(new_dic_alls))

if __name__ == '__main__':
    main()