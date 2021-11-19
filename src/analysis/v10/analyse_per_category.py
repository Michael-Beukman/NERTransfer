from collections import defaultdict
import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from analysis.utils import ORDER, REGIONS, order_columns_of_df
DATA_PATH = "runs/v10/models"
def main():
    """
        This function analyses the results, and looks at different categories of NER (e.g. PER, ORG, LOC, DATE) 
        and creates some tables for these, to see whether any categories do better / worse when changing the pre-trained model.
    """
    
    # This makes multiple tables, one for each category
    folders = glob.glob(os.path.join(DATA_PATH, '*_50'))
    print(f"We have {len(folders)} folders")
    
    # Base:  {target lang: score}
    dic = defaultdict(lambda: defaultdict(lambda: []))
    # Base: {targer lang: {cat: score}} -> All categories
    dic_of_all = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    
    for path in folders:
        f = path.split("/")[-1]
        # Split the paths
        lang_finetune, _, _, lang_start, _, _, _, seed, _ = f.split("_")

        # Get the results
        with open(os.path.join(path, 'test_results.txt'), 'r') as f:
            all_lines = f.readlines() 
            f1 = float(all_lines[0].strip().split(" = ")[1])
            # Per category
            good_lines = all_lines[6:10]
            for l in good_lines:
                ans = [a for a in l.strip().split(" ") if a != '']
                assert len(ans) == 5
                cat = ans[0] 
                _prec, _rec, this_f1, count = map(float, ans[1:])
                dic_of_all[lang_start][lang_finetune][cat].append(this_f1)
            dic_of_all[lang_start][lang_finetune]['overall'].append(f1)
        dic[lang_start][lang_finetune].append(f1)
    
    print("--")
    # Cleans up some of the results from above
    new_dic_of_all = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    # We make a 'lang-specific' item to compress the table somewhat.
    # This means the NER language used is the same as the pre-trained mode
    for A_, va in dic_of_all.items():
        if not A_ in ['base', 'swa']:
            A = 'lang-specific'
        else:
            A = A_
        for B, vb in va.items():
            for C, vc in vb.items():
                if A == 'lang-specific' and B == 'swa':
                    continue
                if A == 'swa' == B == 'swa':
                    new_dic_of_all['lang-specific'][B][C] = vc
                new_dic_of_all[A][B][C] = vc
    
    dic_of_all = new_dic_of_all
    # where to save
    dir = '../analysis/v10/categories'; os.makedirs(dir, exist_ok=True)
    
    total_df = None
    first_df = None
    second_df = None
    for my_l in ['base', 'swa', 'lang-specific']:
        new_dic = defaultdict(lambda: dict())
        new_dic_std = defaultdict(lambda: dict())
        D = dic_of_all[my_l]
        # For all languages
        for l, v in D.items():
            # For all categories
            for b, li in v.items():
                assert len(li) == 5, f"Bad, {len(li)}, {my_l}, {l}"
                li = np.array(li) * 100
                # mean and standard deviation
                new_dic[l][b] = np.mean(li)
                new_dic_std[l][b] = np.std(li)
        # Order consistently
        df = order_columns_of_df(pd.DataFrame(new_dic))
        df2 = order_columns_of_df(pd.DataFrame(new_dic_std))
        assert np.all(df.columns == df2.columns)
        new_df = df.copy()
        new_df_numbers = df.copy()
        for c in df.columns:
            new_df[c] = df[c].round(2).astype(str) + " (" + df2[c].round(1).astype(str) + ")"
            new_df_numbers[c] = df[c]
        new_df = order_columns_of_df(new_df)
        
        # Columnwise average
        new_df_numbers = order_columns_of_df(new_df_numbers)
        new_df_numbers['Average'] = 0
        for c in new_df_numbers.columns[:-1]:
            new_df_numbers['Average'] += new_df_numbers[c] / len(new_df_numbers.columns[:-1])
            
        if first_df is None: first_df = new_df_numbers
        elif second_df is None: second_df = new_df_numbers
        if total_df is None:
            total_df = new_df.copy()
            for c in new_df.columns:
                total_df[my_l + " " +c] = new_df[c]
                total_df = total_df.drop([c], axis=1)
        else:
            for c in new_df.columns:
                total_df[my_l + " " +c] = new_df[c]
                
        
        # Now just save everything
        new_df.to_csv(os.path.join(dir, my_l + ".csv"))
        new_df.to_latex(os.path.join(dir, my_l + ".tex"))
        sns.heatmap(new_df_numbers, annot=True);
        plt.title(f"Pre-trained model = {my_l}"); plt.ylabel("Category of NER")
        plt.xlabel("Language fine-tuned and evaluated on")
        plt.savefig(os.path.join(dir, my_l + ".png"), bbox_inches='tight', pad_inches=0.1); plt.close()
        if my_l == 'lang-specific':
            sns.heatmap(new_df_numbers - first_df, annot=True);
            plt.title(f"{my_l} - base"); plt.ylabel("Category of NER")
            plt.xlabel("Language fine-tuned and evaluated on")
            plt.savefig(os.path.join(dir, my_l+'-base' + ".png"), bbox_inches='tight', pad_inches=0.1); plt.close()
            
            sns.heatmap(new_df_numbers - second_df, annot=True);
            plt.title(f"{my_l} - swa"); plt.ylabel("Category of NER")
            plt.xlabel("Language fine-tuned and evaluated on")
            plt.savefig(os.path.join(dir, my_l+'-swa' + ".png"), bbox_inches='tight', pad_inches=0.1); plt.close()
    
    # And save tables.
    new_total_df = total_df.copy()
    new_total_df = new_total_df.drop(new_total_df.columns, axis=1)
    for c in new_df.columns:
        for a in ['base', 'swa', 'lang-specific']:
            new_total_df[a + " " + c] = total_df[a + " " + c]
            diff = total_df[a + " " + c].str.split(" ").apply(lambda x: float(x[0])) \
                - total_df['base' + " " + c].str.split(" ").apply(lambda x: float(x[0]))
            new_total_df[a + " " + c][diff < 0] = new_total_df[a + " " + c][diff < 0].apply(lambda s: r"\textcolor{red}{" + s + "}")
            
            
            new_total_df[a + " " + c][diff > 5] = new_total_df[a + " " + c][diff > 5].apply(lambda s: r"\textcolor{green}{" + s + "}")
            if a != 'base':
                diff = total_df[a + " " + c].str.split(" ").apply(lambda x: float(x[0])) \
                    - total_df['swa' + " " + c].str.split(" ").apply(lambda x: float(x[0]))

                new_total_df[a + " " + c][diff < 0] = new_total_df[a + " " + c][diff < 0].apply(lambda s: r"\textcolor{orange}{" + s + "}")
    total_df = new_total_df.T

    index = pd.MultiIndex.from_tuples([
        (j + " (" + REGIONS()[j] + ")", i) for j in ORDER() for i in ['base', 'swa', 'lang-specific']
    ], names=["Lang", "Pretrained"])
    total_df.index = (index)
    
    S = total_df.to_latex(escape=False)
    lines = S.split("\n")
    new_lines = []
    for i, l in enumerate(lines):
        if (i >= 8) and (i - 8) % 3 == 0 and (len(lines) - i > 3):
            new_lines.append(r"\midrule")
        new_lines.append(l)
    S = '\n'.join(new_lines)

    with open(os.path.join(dir, 'all_cats' + ".tex"), 'w+') as f:
        f.write(S)
if __name__ == '__main__':
    main()