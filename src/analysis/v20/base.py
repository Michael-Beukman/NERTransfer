from collections import defaultdict
import glob
from io import TextIOWrapper
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from analysis.utils import ORDER, savefig

DATA_PATH = "runs/v10/models/v20/"

def get_f1(f: TextIOWrapper, 
           which_mode='all'):
    
    all_lines = f.readlines() 
    if which_mode == 'all':
        f1 = float(all_lines[0].strip().split(" = ")[1])
        return f1
    # Per category
    good_lines = all_lines[6:10]
    A = []
    B = []
    T = 0
    for l in good_lines:
        ans = [a for a in l.strip().split(" ") if a != '']
        assert len(ans) == 5
        cat = ans[0] 
        _prec, _rec, this_f1, count = map(float, ans[1:])
        count = float(count)
        if cat == which_mode:
            return float(this_f1)
    return sum(A) / T
    


def main(mode='all', ax=None, save=True, start = 0, end = 3, name_to_save=None, fig=None, do_std=False, use_diag_mask=False):
    """What this tries to do is the following:
        - After training NER on Y, how good is zero shot transfer to language X
            - When starting from different pretrained models

        It also looks at zero shot transfer between all other languages and plots a heatmap of this.

    Args:
        mode (str, optional): either all, or one of the NER categories to only plot one. Defaults to 'all'.
        ax ([type], optional): Axis to plot on. Makes own if not provided. Defaults to None.
        save (bool, optional): If true, runs savefig. Defaults to True.
        
        These two are only used because of how we set up the loops, to get individual category plots
            start (int, optional): Which pretrained model to start with. Defaults to 0.
            end (int, optional): Which pretrained model to end with. Defaults to 3.
        
        name_to_save ([type], optional): What to save the file as. Defaults to None.
        fig ([type], optional): Figure if provided, otherwise makes own. Defaults to None.
        do_std (bool, optional): If true, plots standard deviation instead of mean. Defaults to False.
        use_diag_mask (bool, optional): If true, plots only the diagonals instead of the entire heatmap. Defaults to False.
    """    
    folders = glob.glob(os.path.join(DATA_PATH, '*'))
    # Base:  {target lang: score}
    # Make DataFrame from all file results
    dic = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    NN = '' if mode == 'all' else mode
    NN2 = '' if mode == 'all' else f"\nCategory = {mode}"
    if name_to_save is not None:
        NN = name_to_save
    if do_std: 
        NN += "_std"
        NN2 = NN2 + ". Standard Deviation" if mode != 'all' else "Standard Deviation"
    
    # This might be somewhat convoluted, but simply gets the results from the files.
    for path in folders:
        if 'Icon' in path: continue
        f = path.split("/")[-1]
        lang_evaled_on = f.split("_")[0]
        startpoint = f.split("_zero_shot_")[1].split(f"_{lang_evaled_on}_50")[0]
        pretrained_model = startpoint.split('start_')[1].split("__")[0]
        lang_finetuned_on_ner = startpoint.split('_')[0]
        print(f"LANG EVAL: {lang_evaled_on}. Pretrained={pretrained_model}. NER finetune lang = {lang_finetuned_on_ner}. FILE = {f}")

        with open(os.path.join(path, 'test_results.txt'), 'r') as f:
            f1 = get_f1(f, mode)

        if pretrained_model == lang_finetuned_on_ner:  # lang specific
            assert lang_finetuned_on_ner == pretrained_model, f'{lang_finetuned_on_ner} != {pretrained_model}'
            dic['lang-specific'][lang_finetuned_on_ner][lang_evaled_on].append(f1)

        if lang_finetuned_on_ner == 'swa':             # finetune on swa
            dic['x-swa-y'][pretrained_model][lang_evaled_on].append(f1)

        if pretrained_model in ['swa', 'base']:         # otherwise, start from base or swa.
            dic[pretrained_model][lang_finetuned_on_ner][lang_evaled_on].append(f1)
    
    os.makedirs('../analysis/v20/', exist_ok=True)
    all_df_2s = []
    names = []
    ax_ = ax
    figsize = (5, 5)
    
    # This then actually creates the dataframes.
    for I, (pretrained_model, D) in enumerate(dic.items()):
        if not ( start <= I <= end): continue
        if ax_ is None: fig, ax = plt.subplots(1, 1, figsize=figsize) 
        names.append(pretrained_model)
        new_dic = defaultdict(lambda: dict())
        new_dic_clean = defaultdict(lambda: dict())
        for l, v in D.items():
            for b, li in v.items():
                new_dic[f"{l} (NER finetune lang)" ][f"{b} Evaled on"] = np.round(np.mean(li) * 100, 2)
                assert len(li) == 5, f"Problem: {len(li)} for {l} & {b}"
                if do_std:
                    new_dic_clean[f"{l}" ][f"{b}"] = np.round(np.std(li) * 100, 2)
                else:
                    new_dic_clean[f"{l}" ][f"{b}"] = np.round(np.mean(li) * 100, 2)
        df = pd.DataFrame(new_dic)
        df2 = pd.DataFrame(new_dic_clean)
        
        # order the columns and rows in a better way
        df2 = df2.reindex(ORDER(), axis=1)
        df2 = df2.reindex(ORDER(), axis=0)
        all_df_2s.append(df2)
        
        df2.loc['avg'] = df2.mean(axis=0)
        df2['avg'] = df2.mean(axis=1)

        # And this plots the heatmap
        cbar = False if mode == 'all' else save
        if save and mode != 'all':
            cbar_ax = fig.add_axes([1.01, 0.11, .01, 0.67])
            cbar_ax.set_xticks([])
            cbar_ax.set_yticks([])
        sns.heatmap(df2.round(1), 
                    annot=True, 
                    vmin=0, vmax = 100 * (1 - do_std) + do_std * 8, 
                    fmt=".2g", 
                    ax=ax, 
                    **({} if mode == 'all' else {'cbar_ax': cbar_ax if save  else None}),
                    **({'mask':1-np.eye(len(df2))} if use_diag_mask else {}),
                    square=True, cbar=cbar)
        if mode == 'all' or (name_to_save == '_joint' and mode == 'DATE'):
            ax.set_ylabel(f"Evaluated on")
        xlabel = "Fine-tuned on"
        if NN2 and mode != 'all': ax.set_title(NN2)
        func = plt.suptitle if mode != 'all' else ax.set_title
        nl = '\n'
        NN2_tmp = (nl + NN2) if NN2 else ''
        
        # Some different options and titles based on what exactly is selected.
        if mode != 'all': NN2_tmp = ''
        if pretrained_model == 'lang-specific':
            func(f"Pre-trained model = Same as NER fine-tune one{NN2_tmp}")
        elif pretrained_model == 'x-swa-y':
            func(f"Pre-trained model = X-axis. Fine-tune on Swa{NN2_tmp}")
            xlabel = "Pretrained model. Always finetuned NER on SWA"
        else:
            func(f"Pre-trained model = {pretrained_model}{NN2_tmp}")
        if mode == 'all' or name_to_save == '_joint':
            ax.set_xlabel(xlabel)
        if save:
            plt.tight_layout()
            savefig(f"../analysis/v20/results_{pretrained_model}{NN}{'masked' if use_diag_mask else ''}.png")
            plt.close()
        df.T.to_csv(f'../analysis/v20/results_{pretrained_model}{NN}.csv')

    
    # Relative differences, subtract 2 heatmaps.
    if mode == 'all':
        for i in range(len(names)):
            for j in range(len(names)):
                n1, n2 = names[i], names[j]
                if n1 == 'base' or n2 != 'base': continue
                fig, ax = plt.subplots(1, 1, figsize=figsize) 
                d1, d2 = all_df_2s[i], all_df_2s[j]
                sns.heatmap((d1 - d2).round(), annot=True, vmin=-31, vmax=47, ax=ax, square=True, cbar=cbar)
                # plt.title(f"Results: {n1} - {n2}{NN2}")
                letter = ''
                if n1 == 'x-swa-y': letter = '(c)'
                if n1 == 'lang-specific': letter = '(b)'
                if n1 == 'swa': letter = 'Swahili Pre-trained Model'
                
                plt.title(f"Results: {letter} - (a){NN2}")
                plt.ylabel("Evaluated on")
                plt.xlabel("Fine-tuned on")
                plt.tight_layout()
                savefig(f"../analysis/v20/results_{n1}_{n2}{NN}.png")
                plt.close()

    print("AT END")

if __name__ == '__main__':
    # plot all info, i.e. all categories (and 'all' itself), with and without std, etc.
    for std in [False, True]:
        for diag in [True, False]:
            main(mode='all', do_std=std, use_diag_mask=diag and not std)
        for i in range(4):
            fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10*2, 10/2))
            for mode, ax in zip(['DATE', 'ORG', 'PER', 'LOC'], axs.ravel()):
                main(mode, ax, save = mode == 'LOC', start=i, end=i, name_to_save='_joint', fig=fig, do_std=std)