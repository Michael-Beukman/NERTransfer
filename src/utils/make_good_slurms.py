"""
    This file creates Slurm scripts that runs all of our code.
"""

import os

langs =     ['yor',     'hau',  'kin',                   'lug',      'pcm',    'wol',   'swa',     'ibo',  'luo']
fullnames = ['yoruba', 'hausa', 'kinyarwanda',           'luganda',  'naija',  'wolof', 'swahili', 'igbo', 'luo']

# Change this path
ROOT_PATH = '/home/USERNAME/path/to/root'

def write_finetune_slurm(
    pretrained_model: str,
    lang_to_finetune_on: str,
    seed_to_use: int
):
    slurm_text = \
"""#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J {LANG}_{NAME}_{MODEL}
#SBATCH -o {ROOT_PATH}/{SLURMLOC}/{LANG}_{NAME}_{CLEANMODEL}.%N.%j.out

source ~/.bashrc
cd {ROOT_PATH}/src/runs/v10
conda activate mner
echo "Using {MODEL} for {LANG}, with name = {NAME} and seed = {SEED}"
time ./train.sh {NAME} {MODEL} {LANG} {SEED}
"""
    clean_model = pretrained_model.replace("/","_")

    for l, full in zip(langs, fullnames):
        if l in clean_model or full in clean_model:
            clean_model = l
            break
    if 'base' in clean_model:
        clean_model = 'base'
    name = f'v10_start_{clean_model}__{lang_to_finetune_on}__{seed_to_use}'

    s = slurm_text.format(
        NAME=name,
        MODEL=pretrained_model,
        CLEANMODEL=clean_model,
        SLURMLOC='v10',
        SEED=seed_to_use,
        LANG=lang_to_finetune_on,
        ROOT_PATH=ROOT_PATH
    )
    try:
        os.makedirs(ROOT_PATH, exist_ok=True)
    except Exception as e:
        print("Failed to create directories", e)
    D = 'slurms/v10'
    os.makedirs(D, exist_ok=True)
    with open(os.path.join(D, name + ".batch"), 'w') as f:
        f.write(s)
    
def make_v10_slurms():
    # Wants to make a few job scripts to run things
    for seed in range(5):
        for lang, fullname in zip(langs, fullnames):
            starting_points = [f'{lang}_finetune_ner_{lang}', f'swa_finetune_ner_{lang}', f'base_finetune_ner_{lang}']
            models = set([
                f'Davlan/xlm-roberta-base-finetuned-{fullname}',
                'Davlan/xlm-roberta-base-finetuned-swahili',
                'xlm-roberta-base'
            ])

            if lang in langs:
                for name, model in zip(starting_points, models):
                    if fullname in model:
                        write_finetune_slurm(
                            model, lang, seed
                        )

    
def make_v11_slurms():
    # Wants to make a few job scripts to run things
    for seed in range(5):
        for lang, fullname in zip(langs, fullnames):
            starting_points = [f'{lang}_finetune_ner_swa']
            models = set([
                f'Davlan/xlm-roberta-base-finetuned-{fullname}',
            ])

            lang = 'swa'
            for name, model in zip(starting_points, models):
                if fullname in model:
                    write_finetune_slurm(
                        model, lang, seed
                    )

    print("We have", len(langs))

if __name__ == '__main__':
    make_v10_slurms()
    make_v11_slurms()