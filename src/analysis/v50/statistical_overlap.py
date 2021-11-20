from collections import defaultdict
import glob
import os
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from analysis.utils import ORDER, order_columns_of_df, savefig
import seaborn as sns
from scipy.stats import pearsonr, linregress
save_dir = '../analysis/v50'

def get_common_things_between_two_langs(lang1:str, lang2:str,
                                        entities_to_use=['DATE', 'PER', 'LOC', 'ORG'], joint_all_sets=False):
    """This would like to analyse the input data specifically, and see how many similarities are there between the diff.
        Languages, to get a feel for why training on more data helped.
        We return number of tokens that overlap, as well as the number of unique entities that overlap. 
        They do seem quite well correlated, so for the plots we use the former.

    Args:
        lang1 (str): Source language, i.e. the amount to count
        lang2 (str): Target language, used to check overlap
        entities_to_use (list, optional): List of entities to consider. Defaults to ['DATE', 'PER', 'LOC', 'ORG'].
        joint_all_sets (bool, optional): If true, uses all datasets. Defaults to False.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]] -> (tokens_overlap, tokens_total), (entities_overlap, entities_total)
    """    
    mode = 'train'
    all_files = [f for f in glob.glob(f'../data/masakhane-ner/data/*/{mode}.txt')]
    assert len(all_files) == 10
    dic = {}
    for lang_file in all_files:
        lang = lang_file.split("/")[-2]
        if lang not in [lang1, lang2]: continue
        dic_of_ents_for_this_lang = defaultdict(lambda: defaultdict(lambda: 0))
        if joint_all_sets:
            lines = []

            for n in ['test', 'train', 'dev']:
                with open(lang_file.replace(mode, n), 'r') as f:
                    lines += f.readlines()
        else:
            with open(lang_file, 'r') as f:
                lines = f.readlines()
        lines_with_ents = [l.strip() for l in lines if l[-3:] != ' O\n']
        lines_with_ents = [l.replace('\n', '') for l in lines_with_ents if len(l) > 0]
        lines_with_ents = [l for l in lines_with_ents if len(l) > 0]
        for l in lines_with_ents:
            splits = l.split(" ")
            assert len(splits) == 2, f"splits '{l}' is bad with l={len(splits)}"
            a, entity = splits
            dic_of_ents_for_this_lang[entity.split("-")[-1]][a] += 1
        dic[lang] = dic_of_ents_for_this_lang

    dic1 = dic[lang1]
    dic2 = dic[lang2]

    S = 0
    total = 0
    total_ents = 0
    S_ents = 0
    for e in dic1.keys():
        if e not in entities_to_use: continue
        
        inter = set(dic1[e].keys()) & set(dic2[e].keys())
        # print("INTER", inter)
        total_common = 0
        for key in inter:
            # overlap between source and target - take how many inside source
            total_common += dic1[e][key]

        total_ents += len(dic1[e])
        total += sum(dic1[e].values())
        
        S_ents += len(inter)
        S += total_common
    e = "ALL"
    return (S, total), (S_ents, total_ents)


def statistical_overlap(mode='train'):
    """
        Finds the overlap in the data between different languages.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(15*2, 15/2))
    dic_overall = defaultdict(lambda: {})
    for e, ax in zip(['DATE', 'PER', 'LOC', 'ORG'], axs.flat):
        dic = defaultdict(lambda: {})
        for l1 in ORDER():
            for l2 in ORDER():
                (common, T), (common_ents, T_ents) = get_common_things_between_two_langs(l1, l2, entities_to_use=[e], joint_all_sets=True)
                C = common
                if l1 in dic and l2 in dic[l1]:
                    assert dic[l1][l2] == C, f"BAD {C} {dic[l1][l2]}, {l1}"
                dic[l1][l2] = C
                (common, T), (common_ents, T_ents) = get_common_things_between_two_langs(l1, l2, joint_all_sets=True)
                dic_overall[l1][l2] = common
        df = pd.DataFrame(dic)
        df = order_columns_of_df(order_columns_of_df(df), rows=True)
        df.to_csv(os.path.join(save_dir, "data_" + e + ".csv"))
        sns.heatmap(df, annot=True, ax=ax, fmt='.0f')
        ax.set_title(e)
    plt.suptitle("Data Overlap")
    savefig(os.path.join(save_dir, "data_joint.png"))
    plt.close()
    df = pd.DataFrame(dic_overall)
    plt.figure(figsize=((15 + 2.5) / 1.4, 15 / 1.4))
    sns.heatmap(df, annot=True); plt.title("Data overlap for all Categories")
    savefig(os.path.join(save_dir, "data_all.png"))
    df.to_csv(os.path.join(save_dir, "data_all.csv"))
            
def investigate_performance():
    """
        Plots a correlation between data overlap and F1 score when transferring.
    """
    df_data = pd.read_csv(os.path.join(save_dir, "data_all.csv"), index_col=0)
    font = {'family' : 'normal', 'size'   : 24}

    matplotlib.rc('font', **font)
    df_performance = pd.read_csv('../analysis/v20/results_base.csv', index_col=0)
    df_performance.columns = [c.split(" ")[0] for c in df_performance.columns]
    df_performance.index = [c.split(" ")[0] for c in df_performance.index]
    df_data = order_columns_of_df(df_data, both=True)
    df_performance = order_columns_of_df(df_performance, both=True)
    
    # Remove diagonal ones, as that isn't really transfer learning.
    mask = np.ones_like(df_data.to_numpy()); K = np.arange(len(mask))
    mask[K, K] = False
    mask = mask.flatten() > 0.5

    all_data        = df_data.to_numpy().flatten()[mask]
    all_performance = df_performance.to_numpy().flatten()[mask]

    plt.figure(figsize=(15, 15))
    ans = linregress(all_data, all_performance)
    plt.scatter(all_data, all_performance)
    xs = np.linspace(all_data.min(), all_data.max())
    plt.plot(xs, xs * ans.slope + ans.intercept)
    
    plt.title(f"Comparing F1 vs. Data overlap\nStarting from base and fine-tuning on one language, evaluating on another.\nR={np.round(ans.rvalue, 2)}, p={ans.pvalue:1.1e}")
    plt.xlabel("Data Amount Overlapping (Tokens)")
    plt.ylabel("F1")
    savefig(os.path.join(save_dir, 'correlation_base.png'))

if __name__ == '__main__':
    statistical_overlap()
    investigate_performance()