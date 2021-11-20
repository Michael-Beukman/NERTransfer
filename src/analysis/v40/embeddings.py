"""
    This file deals with word embeddings, both getting them from a wide range of models, as well as analysing
    them, based on a few different criteria. The functions at the top of this file are mostly utility based, 
    to actually find the word embeddings. The most important functions are main, which finds the embeddings and stores them to a file,
    and analyse_data, which analyses them and plots many graphs.
"""
import copy
import glob
import os
from typing import Dict, List, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from collections import defaultdict
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from analysis.utils import FULLNAMES, ORDER, load_compressed_pickle, save_compressed_pickle, savefig
import seaborn as sns

DEVICE = torch.device('cuda:0')
WordVector = np.ndarray
EmbeddingDictionary = Dict[str, Dict[str, Dict[str, WordVector]]]
# Lots of this was from here: https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
# Primarily actually getting the embeddings themselves.



def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    encoded = encoded.to(DEVICE)
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)

def get_word_vector(sent, idx, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, token_ids_word, model, layers)

def get_sentences_and_words(dir: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """Reads the test.txt file of masakhaner for a specific language, and returns:
        List of sentences
        List of list of words in those sentences that have NER categories (i.e. not O)
        List of list of categories for those words.

    Args:
        dir (str): Dir to load from. Must have a test.txt file.

    Returns:
        Tuple[List[str], List[List[str]], List[List[str]]]:
    """
    with open(os.path.join(dir, 'test.txt')) as f:
        contents = f.read()
        sentences = contents.split("\n\n")
    
    words = []
    new_sents = []
    categories = []
    for s in sentences:
        s = s.split("\n")
        tmp = []
        tmp_cats = []
        for line in s:
            # split on space
            KK = line.split(" ")
            if len(KK) != 2:
                # invalid
                continue
            else:
                word, cat = KK
            # ignore O
            if cat == 'O': continue
            else:
                tmp.append(word)
                tmp_cats.append(cat.split('-')[1])
        if len(tmp):
            words.append(tmp)
            new_sents.append(' '.join(s))
            categories.append(tmp_cats)
    return new_sents, words, categories

def get_embeddings_for_lang(lang: str, layers: Union[List[int], None], model_name: str='xlm-roberta-base') -> Dict[str, List[np.ndarray]]:
    """This returns embeddings for a specific language, using a specific model, grouped by NER categories.

    Args:
        lang (str): Language to use, 3 letter code
        layers (Union[List[int], None]): either none, or a list of integers for layers. If none, uses [-4, -3, -2, -1], i.e. last 4.
        model_name (str, optional): Name or path of model to use. Defaults to 'xlm-roberta-base'.

    Returns:
        Dict[str, List[np.ndarray]]: {
            category: list_of_word_embeddings
        }
    """
    sents, words, categories = get_sentences_and_words(f'../data/masakhane-ner/data/{lang}')
    layers = [-4, -3, -2, -1] if layers is None else layers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(DEVICE)
    all_embeddings = defaultdict(lambda: [])
    for sent, list_of_words, list_of_cats in zip(sents, words, categories):
        for word, cat in zip(list_of_words, list_of_cats):
            idx = get_word_idx(sent, word)
            word_embedding = get_word_vector(sent, idx, tokenizer, model, layers)
            all_embeddings[cat].append(word_embedding.cpu().numpy())
    return all_embeddings


# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    print(np.round(cov))
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def clean(dic, name):
    assert len(dic) == 1
    if name in dic: return dic
    k = list(dic.keys())[0]
    dic[name] = dic[k]
    del dic[k]
    return dic


def main(layers=None, I = 0 ):
    """This function calculates word embeddinsg from the last 4 layers by defaul of many different language models, 
        Using the MasakhaNER dataset. It only calculates the embeddings for the actual NER categories, with context being the sentence in which
        the word finds itself in.

    Args:
        layers: Which layers, list of integers, e.g. [-3, -2, -1] for last 3. Defaults to None.
        I (int, optional): The seed of models to use, between 0 and 4 inclusive. Defaults to 0.
    """
    os.makedirs('analysis/v40', exist_ok=True)
    os.makedirs('results/v40', exist_ok=True)
    langs =     ['yor',     'hau',  'kin',                   'lug',      'pcm',    'wol',   'swa',     'ibo',  'luo']
    fullnames = ['yoruba', 'hausa', 'kinyarwanda',           'luganda',  'naija',  'wolof', 'swahili', 'igbo', 'luo']    
    # Models
    if I == 0:
        models = {'base': 'xlm-roberta-base'}
        # domain adaptive finetuned models
        models.update({
            l: f'Davlan/xlm-roberta-base-finetuned-{f}' for l, f in zip(langs, fullnames)
        })
    else:
        models = {}

    # Now, the ones we actually fine-tuned on different NER languages.
    for f in glob.glob(f'../../../../../from_cluster/new_structure/src/runs/v10/models/*__{I}*'):
        l = f.split('/')[-1]
        models[l] = f

    all_langs_to_use = ORDER()
    for pretrained_name, model_path in models.items():
        # For each model
        # don't rerun for those that exist already
        if os.path.exists(f'results/v40/{pretrained_name}.p.pbz2'): 
            print(pretrained_name, 'exists already')
            continue
        all_dics = {}
        all_dics[pretrained_name] = {}
        try:
            # For each language
            for l in all_langs_to_use:
                all_dics[pretrained_name][l] = {}
                # Get the embeddings
                embed = get_embeddings_for_lang(l, layers, model_path)
                for k in embed:
                    all_dics[pretrained_name][l][k] = embed[k]
            # Save
            save_compressed_pickle(f'results/v40/{pretrained_name}.p', all_dics)
        except Exception as e:
            print("An error occurred ", e, pretrained_name)


CACHE = {}
def load_cache(name):
    if name not in CACHE: 
        CACHE[name] = load_compressed_pickle(name)
    
    return copy.deepcopy(CACHE[name])

def analyse_data(
    DO_1 = True,
    DO_2 = True,
    DO_3 = True,
    mds = True,
    pca = True,
):
    """Actually uses the embeddings from main() to do some analysis.

        All of this works on a relatively similar basis structure, namely:
            - Get some embeddings
            - Within each language / model, take the mean per category to get only 4 'center of masses' for DATE, PER, LOC & ORG.
            - Take all of the center of masses and perform PCA or MDS on them.
            - Then plot the result, with clear legends, colours and markers.

        This does a few separate things, specifically:
            1. Same model, different languages
            2. Same language, different model
            3. Same language, different seeds for fine-tuning

        The arguments control which of the above are actually performed.
    """
    DIR = '../analysis/v40'
    which = -1
    DIR_1 = os.path.join(DIR, f'1_same_model/{which}')
    DIR_2 = os.path.join(DIR, f'2_same_lang/{which}')
    DIR_3 = os.path.join(DIR, f'3_diff_seeds/{which}')
    DO_TABLE = True
    os.makedirs(DIR_1, exist_ok=True)
    os.makedirs(DIR_2, exist_ok=True)
    os.makedirs(DIR_3, exist_ok=True)

    mode_array = []
    if pca: mode_array.append('pca')
    if mds: mode_array.append('mds')

    def plot_things(
        embeddings_dictionaries: List[EmbeddingDictionary],
        languages_to_use: Union[List[str], str],
        dir_path: str,
        name: str,
        mode: str = 'mds',
        legend: bool = False,
        pretty_name=None

    ):
        """This is the main function here, which plots all of the embeddings

        Args:
            embeddings_dictionaries (List[EmbeddingDictionary]): List of dictionaries of embeddings
            languages_to_use (Union[List[str], str]): Which languages' embeddings to plot. If a string, all models use the same language.
                    If a list, then corresponding elements of embeddings_dictionaries will use the language at that index.
            dir_path (str): Where to save
            name (str): filename to save as without .png
            mode (str, optional): Use MDS or PCA as dimensionality reduction method. Defaults to 'mds'.
            legend (bool, optional): If false, doesn't show colour legend for models / langs and just uses the shapes for categories.  Defaults to False.
            pretty_name ([type], optional): A more human readable name for plot titles. If none, defaults to name. Defaults to None.
        """
        single_dic: EmbeddingDictionary = {}
        for e in embeddings_dictionaries:
            single_dic = {**single_dic, **e}
        if type(languages_to_use) == str:
            languages_to_use = [languages_to_use] * len(embeddings_dictionaries)

        pca_things = []
        all_vectors_not_mean = []
        method_names = []
        cats = ['PER', "DATE", "LOC", "ORG"]
        if pretty_name is None: pretty_name = name
        for c1 in cats:
            for (p, li_of_things), lang in zip(single_dic.items(), languages_to_use):
                if c1 == cats[0]:
                    # pretty legend.
                    if "0" <= p <= "9": 
                        method_names.append(f"{lang}")
                    else:
                        if len(set(languages_to_use)) == 1:
                            method_names.append(f"{p}")
                        else:
                            method_names.append(f"{lang} from {p}")
                # center of mass
                A = np.array(li_of_things[lang][c1]).mean(axis=0)
                # append correct layers
                pca_things.append(A[:])
                all_vectors_not_mean.append(np.array(li_of_things[lang][c1]))
                
        if mode == 'mds':
            method = MDS(n_components=2)
        else:
            # We first take the center of masses, then perform the PCA.
            # The other way around (pca then mean) is possible, but would muddy the plots a bit.
            method = PCA(n_components=2)
        # transform
        lower_dimensional_answer = method.fit_transform(pca_things)
        i = 0
        markers = ['o', 'x', '^', 's']
        colors = sns.color_palette(n_colors=len(single_dic))
        # Plot
        assert len(pca_things) == len(markers) * len(colors)
        ax = plt.gca()
        for marker in markers:
            for color in colors:
                # Plot each point with a colour and marker.
                ax.plot([lower_dimensional_answer[i][0]], [lower_dimensional_answer[i][1]], marker=marker, color=color, ls='none')
                i += 1

        f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

        handles = [f("s", colors[i]) for i in range(len(colors))] if legend else []
        handles += [f(markers[i], "k") for i in range(len(markers))]
        # Stuff for better plotting.
        labels = method_names + cats
        if not legend:
            labels = cats
        plt.legend(handles, labels, framealpha=0.4, ncol = 1 + legend)
        plt.title(f'{pretty_name} ({mode.upper()})')
        savefig(os.path.join(dir_path, name + "_" + mode + ".png")); 
        plt.close()
    
    def do_distance_table(embeddings_dictionaries: List[EmbeddingDictionary],
        language_to_use: str,
        dir_path: str,
        name: str,
    ):
        """Makes a table of distance between different categories for the same seeds as well as
           different seeds for the same categories

        Args: Same as above function
            embeddings_dictionaries (List[EmbeddingDictionary]):
            language_to_use (str): 
            dir_path (str): 
            name (str): 
        """        

        single_dic: EmbeddingDictionary = {}
        for e in embeddings_dictionaries:
            single_dic = {**single_dic, **e}

        cats = ['PER', "DATE", "LOC", "ORG"]
        output_ans = defaultdict(lambda: {})
        for c1 in cats:
            keys = list(single_dic.keys())
            I = len(keys)
            all_dists = []
            all_dists_com = []
            for i in range(I):
                for j in range(i + 1, I):
                    # Get distance between 2 seeds, same category.
                    A = np.array(single_dic[keys[i]][language_to_use][c1])
                    B = np.array(single_dic[keys[j]][language_to_use][c1])
                    
                    A_com = np.array(single_dic[keys[i]][language_to_use][c1]).mean(axis=0)
                    B_com = np.array(single_dic[keys[j]][language_to_use][c1]).mean(axis=0)

                    assert A.shape == B.shape
                    assert A_com.shape == B_com.shape
                    assert A_com.shape == (192, ) or A_com.shape == (192 * 4, )
                    mean_distance = np.linalg.norm(A - B, axis=1)
                    mean_distance_com = np.linalg.norm(A_com - B_com)
                        

                    all_dists.append(mean_distance.mean())
                    # now center of mass
                    all_dists_com.append(mean_distance_com)
            output_ans[c1]['mean'] = np.mean(all_dists)
            output_ans[c1]['std'] = np.std(all_dists)

            output_ans[c1 + "_com"]['mean'] = np.mean(all_dists_com)
            output_ans[c1 + "_com"]['std'] = np.std(all_dists_com)
        """ 
            Because, you cant really compare dates vs per, as there are different seeds and they dont really correspond.
            A few solutions: 
                1. Compare center of masses, then for comparison, compare CoM above as well.
                2. Do something like all to all, so each date pt comparing against each person point, and averaging out.

            We do the first option.
        """
        # Now look at  distance within a seed
        assert I == 5
        dist_within = []
        for i in range(I):
            dist_here = []
            for index_1, c1 in enumerate(cats):
                for index_2 in range(index_1 + 1, len(cats)):
                    c2 = cats[index_2];
                    # same seed, diff cats
                    A_com = np.array(single_dic[keys[i]][language_to_use][c1]).mean(axis=0)
                    B_com = np.array(single_dic[keys[i]][language_to_use][c2]).mean(axis=0)
                    assert A_com.shape == B_com.shape
                    assert A_com.shape == (192, ) or A_com.shape == (192 * 4, )
                    mean_distance_com = np.linalg.norm(A_com - B_com)
                    dist_here.append(mean_distance_com)
            dist_within.append(np.mean(dist_here))

        output_ans["within"]['mean'] = np.mean(dist_within)
        output_ans["within"]['std'] = np.std(dist_within)


        df = pd.DataFrame(output_ans)
        df.to_csv(os.path.join(dir_path, name + ".csv")); 
        

    if DO_1:
        all_models = glob.glob('results/v40/*.pbz2')
        counter = 0
        for model in all_models:
            t = model.split("/")[-1]
            if '__' in model:
                a = t.split("_")[0]
                b = t.split("start_")[1].split("_")[0]
                name = f"base -> {b} -> {a}"
            else:
                name = "base" if 'base' in model else f"base -> {t.split('.')[0]}"
            dic_to_use = clean(load_cache(model), name)
            Ls = ORDER()
            K = list(dic_to_use.keys())[0]
            for i, _ in enumerate(Ls):
                dic_to_use[f"{i}"] = dic_to_use[f"{K}"]
            del dic_to_use[f"{K}"]
            for mode in  mode_array:
                counter += 1
                print(f"\rStep 1: {counter} / {len(all_models) * len(mode_array)}     ", end='')
                tmp_name = t.split('.')[0]
                if tmp_name in FULLNAMES():
                    tmp_name = f"base (pre-trained)\n{FULLNAMES()[tmp_name]} (fine-tune NER)"
                else:
                    # swa_v10_start_ibo__swa__1_50
                    pretrain_lang = tmp_name.split("__")[0].split("start_")[1]
                    finetune_lang = tmp_name.split("__")[1]
                    if pretrain_lang == 'base':
                        tmp_name = f"base (pre-trained)\n{FULLNAMES()[finetune_lang]} (fine-tune)"
                    else:
                        tmp_name = f"base (pre-trained) + {FULLNAMES()[pretrain_lang]} (pre-trained)\n{FULLNAMES()[finetune_lang]} (fine-tune)"
                plot_things([dic_to_use for i, _ in enumerate(Ls)], [l for l in Ls], mode=mode, dir_path=DIR_1, 
                            name=t.split(".")[0], legend=True, 
                            pretty_name=f"Model = {tmp_name}")
            
        print('Step 1 DONE\n')
    
    if DO_2:
        # 2. Different model, same lang
        counter = 0
        total = len(ORDER()) ** 2 * len(mode_array)
        for model in ORDER():
            models = [
                    clean(load_cache('results/v40/base.p.pbz2'), 'base'),
                    clean(load_cache(f'results/v40/{model}.p.pbz2'), f'base -> {model}'),
                    clean(load_cache(f'results/v40/{model}_v10_start_{model}__{model}__0_50.p.pbz2'), f'base -> {model} -> {model}'),
                    clean(load_cache(f'results/v40/{model}_v10_start_swa__{model}__0_50.p.pbz2'), f'base -> swa -> {model}'),
                    clean(load_cache(f'results/v40/swa_v10_start_{model}__swa__0_50.p.pbz2'), f'base -> {model} -> swa'),
                ]
            Ls = ORDER()
            for l in Ls:
                for mode in mode_array:
                    print(f"\rStep 2: {counter} / {total}     ", end='')
                    plot_things(models, l, mode=mode, dir_path=DIR_2, name=f'{model}_{l}', legend=True, 
                            pretty_name=f"{FULLNAMES()[l]} embeddings from various models")
        print('Step 2 DONE\n')
    if DO_3:
        # 3. special case of (2) -> same model, dif ferent seeds
        counter = 0
        total = len(ORDER()) ** 2 * len(mode_array) * 10
        for model in ORDER():
            all_ms = [[
                    clean(load_cache(f'results/v40/{model}_v10_start_base__{model}__0_50.p.pbz2'), f'base -> NER {model} 0'),
                    clean(load_cache(f'results/v40/{model}_v10_start_base__{model}__1_50.p.pbz2'), f'base -> NER {model} 1'),
                    clean(load_cache(f'results/v40/{model}_v10_start_base__{model}__2_50.p.pbz2'), f'base -> NER {model} 2'),
                    clean(load_cache(f'results/v40/{model}_v10_start_base__{model}__3_50.p.pbz2'), f'base -> NER {model} 3'),
                    clean(load_cache(f'results/v40/{model}_v10_start_base__{model}__4_50.p.pbz2'), f'base -> NER {model} 4'),
                ], 
                [
                    clean(load_cache(f'results/v40/{model}_v10_start_{model}__{model}__0_50.p.pbz2'), f'base -> {model} -> NER {model} 0'),
                    clean(load_cache(f'results/v40/{model}_v10_start_{model}__{model}__1_50.p.pbz2'), f'base -> {model} -> NER {model} 1'),
                    clean(load_cache(f'results/v40/{model}_v10_start_{model}__{model}__2_50.p.pbz2'), f'base -> {model} -> NER {model} 2'),
                    clean(load_cache(f'results/v40/{model}_v10_start_{model}__{model}__3_50.p.pbz2'), f'base -> {model} -> NER {model} 3'),
                    clean(load_cache(f'results/v40/{model}_v10_start_{model}__{model}__4_50.p.pbz2'), f'base -> {model} -> NER {model} 4'),
                ]
                ]
            DIR_3_csv = os.path.join(DIR_3, 'csvs')
            os.makedirs(DIR_3_csv, exist_ok=True)
            for i in range(len(all_ms)):
                models = all_ms[i]
                n = 'base_then_finetune' if i == 0 else 'lang_adap'
                pn = 'Base fine-tuned on' if i == 0 else f'Language Adaptive ({model}) fine-tuned on'
                Ls = ORDER()
                for l in Ls:
                    for mode in  mode_array:
                        counter += 1
                        print(f"\rStep 3: {counter} / {total}     ", end='')
                        plot_things(models, l, mode=mode, dir_path=DIR_3, 
                                    name=f'{n}_{model}_{l}',
                                    pretty_name = f"{FULLNAMES()[l]} embeddings from {pn.format(F=model) if i == 0 else pn} {FULLNAMES()[model]}",
                                    legend=False)
                        pass
                    if DO_TABLE and 'pca' in mode_array:
                        do_distance_table(models, l, dir_path=DIR_3_csv, name=f'{n}_{model}_{l}')
        print('Step 3 DONE\n')
    
def analyse_distance_tables():
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    idxs = None
    for N, ax in zip(['hau', 'swa', 'wol'], axs.ravel()):
        ins = np.zeros((len(ORDER()), 2))
        outs = np.zeros((len(ORDER()), 2))
        all_others = {
            c: np.zeros((len(ORDER()), 2)) for c in ['PER', 'DATE', 'ORG', "LOC"]
        }
        for i, l in enumerate(ORDER()):
            file = f"../analysis/v40/3_diff_seeds/-1/csvs/base_then_finetune_{l}_{N}.csv"
            df = pd.read_csv(file)
            inside = df['within']
            for k in all_others:
                all_others[k][i, :] = list(df[f'{k}_com'])
            outside = (df['PER_com'] + df['ORG_com'] + df['LOC_com'] + df['DATE_com']) / 4
            outside.iloc[1] = np.sqrt((df['PER_com']**2 + df['ORG_com']**2 + df['LOC_com']**2 + df['DATE_com']**2).iloc[1])
            ins[i, :] = list(inside)
            outs[i, :] = list(outside)
        if idxs is None:
            idxs = np.argsort(ins[:, 0])
        xs = np.array(ORDER())[idxs]
        ins = ins[idxs]
        outs = outs[idxs]
        ax.plot(xs, ins[:, 0], label='inside')
        ax.plot(xs, outs[:, 0], label='outside')
        ax.set_title(f"xlm-roberta-base fine-tuned on {N.title()}")
    plt.legend()
    plt.suptitle("Embedding Distances between the same categories in different seeds (outside) and different categories within the same seed (inside)")
    savefig("../analysis/v40/distances.png")
    plt.close()


if __name__ == '__main__':
    # To run main, which generates the word embeddings, you can do the following:
    # for i in range(1, 5):
    #     main(I=i)
    import multiprocessing
    from multiprocessing import Pool
    # This does the plotting
    def arr(i):
        ans = [False, False, False, False, False]
        if i >= 3:
            ans[3] = True
            i -= 3
        else:
            ans[4] = True
        ans[i] = True
        return ans
    def tmp_func(x):
        return analyse_data(*arr(x))
    with Pool(6) as p:
        print(p.map(tmp_func, range(6)))

    analyse_distance_tables()
