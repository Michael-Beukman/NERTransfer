# NER Transfer - Interpretation and Analysis
- [NER Transfer - Interpretation and Analysis](#ner-transfer---interpretation-and-analysis)
- [About](#about)
- [Code Structure](#code-structure)
- [Get Started](#get-started)
  - [Train Models](#train-models)
  - [Evaluate](#evaluate)
  - [Analysis](#analysis)
  - [Exploration](#exploration)
- [References / Sources / Libraries](#references--sources--libraries)
- [Model Cards](#model-cards)
  - [About](#about-1)
    - [Contact & More information](#contact--more-information)
    - [Training Resources](#training-resources)
  - [Data](#data)
  - [Intended Use](#intended-use)
  - [Limitations](#limitations)
    - [Privacy & Ethical Considerations](#privacy--ethical-considerations)
  - [Metrics](#metrics)
  - [Caveats and Recommendations](#caveats-and-recommendations)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
# About
This is an Honours course project that investigates transfer learning in low-resourced languages, specifically in a named entity recognition (NER) task. This repository contains the code, the trained models can be found [here](https://huggingface.co/mbeukman) and the report can be found here (still in progress).

# Code Structure
We mainly did the following in this project:
1. Fine-tune many different pre-trained models on different languages from the [MasakhaNER](https://arxiv.org/abs/2103.11811) dataset.
2. Using these fine-tuned models, we evaluated on all of the languages, to get an idea for zero-shot potential.
3. Then we moved onto analysis, which is the bulk of our contributions and code here. The specific analysis we did can be found in `./src/analysis`, with the following subfolders:
   1. `v10`: Analyse Pre-training effect on final performance.
   2. `v20`: Look at zero-shot transfer.
   3. `v40`: Investigate word embeddings, performing, among others, PCA on them and plotting them.
   4. `v50`: Statistically analysing the data overlap between the different languages, and investigating correlation between this and performance.
```
├── analysis                            -> This contains the bulk of our results, including plots, csv files and Latex tables.
│   ├── v10
│   ├── v20
│   ├── v40
│   └── v50
├── data
│   └── masakhane-ner
├── doc
│   ├── report.pdf                      -> Written Report
├── env.yml                             -> Environment File
├── logs                                -> Log files are written to here
├── run.sh                              -> script to run any code in this repository
└── src                                 -> All of the source code
    ├── analysis                        -> Analyses the results we obtained
    │   ├── utils.py                    -> General utilities
    │   ├── v10                         -> Effect of pre-trained models
    │   ├── v20                         -> Effect of fine-tuning and doing zero shot transfer
    │   ├── v40                         -> Interpretability, specifically word embedding analysis
    │   └── v50                         -> Statistical analysis including data overlap
    ├── results
    │   └── v40                         -> This contains some pickle files with precomputed embeddings.
    ├── run.sh                          -> Another run.sh
    ├── runs                            -> Code that ran all the fine-tuning. Also contains the results on the test/dev set.
    │   ├── v10                         -> Fine-tuning from different pre-trained models.
    │   └── v20                         -> Evaluating zero shot transfer.
    ├── slurms                          -> Job files are saved here.
    │   └── v10
    └── utils                           -> Overall utility code, lot of it from the MasakhaNER codebase
        ├── create_slurms.py            -> Creates Slurm job files for the fine-tuning
        ├── evaluate_ner.py             -> Evaluates a specific model on any language
        ├── train_ner.py                -> Trains the models
        └── utils_ner.py                -> NER Utilities
```
# Get Started
To get started, you can use our [conda](https://www.anaconda.com/) environment by running the following command:
```
conda env create -f env.yml
```
Then run `conda activate mner`, and you should be ready to go. If that fails, you can try to instead use `env_all.yml`, with pinned versions for all packages.

Now, to run **any** python file in this project, the recommended way to do so is to use the `./run.sh` script provided in the root of the project instead of `python` directly, as the former sets up the requisite paths, e.g. `./run.sh src/utils/create_slurms.py`
## Train Models
If you want to train the models, then you can create the [Slurm](https://slurm.schedmd.com/documentation.html) scripts used by running `./run.sh src/utils/create_slurms.py`. Note that you will need to change this file's paths at the top of the file to be applicable to your cluster. Then, you should be able to see the slurm scripts in `src/slurms/v10` and you can run all of them. Note, there are very many and your cluster scheduler may block you if you try running all of them at once.
## Evaluate
Next up is how to do evaluation, i.e. take the models from before, and evaluate them on all of the other languages. To do so you can navigate to `src/runs/v20` and run `./run_eval_all.sh`. Note, this will run all of the inference (4 in parallel) in one step, so it might take a long time.


In practice, we do not recommend this training and evaluation is done from scratch as it uses significant compute and storage resources. We do provide downloadable models (albeit just one seed instead of 5), so it is recommended to just use those.
## Analysis
The analysis is mostly self contained, and the vast majority requires just the result .txt files present and not the models at all.


TL; DR: You can probably just run the following commands
```
./run.sh src/analysis/v10/basic_analyse_pretrained_model.py
./run.sh src/analysis/v10/analyse_per_category.py

./run.sh src/analysis/v20/base.py

./run.sh src/analysis/v50/statistical_overlap.py
```
and most of the analysis should run.

The embeddings code is a bit more involved, as that requires the actual word embeddings, which are quite large, and are stored in a separate repository.
To download these, do the following:
```
mkdir -p src/results/v40
cd src/results/v40
git clone https://github.com/Michael-Beukman/NERTransfer-Embeddings .
cd -
```


Once those have been downloaded, you can run the code by simply running `./run.sh src/analysis/v40/embeddings.py`.

Code is also contained inside there to actually create the word embeddings, although that does require all of the fine-tuned models.


In the repository, there are many images of embeddings from different combinations of models and languages, specifically found in `analysis/v40`, so the code does not need to be rerun to just view them.


## Exploration
In `src/runs/v10/models` there are many different folders, consisting of the results (metrics and predictions) from different fine-tuned models. `src/runs/v10/models/v20` contains the results from the same models, when evaluating on different languages. There is lots of data here, so feel free to explore it.

Additionally, there are very many plots, images and tables in `analysis/*` that might be interesting.


The folder names might be slightly confusing, but, e.g. `src/runs/v10/models/kin_v10_start_swa__kin__2_50/` means it was fine-tuned on Kinyarwanda NER data, starting from the [Swahili pre-trained model](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili), trained using seed = 2 and epochs = 50. Similarly, `src/runs/v10/models/ibo_v10_start_base__ibo__0_50` is xlm-roberta-base fine-tuned on Igbo data.


For the results in `v20`, e.g. `src/runs/v10/models/v20/lug_zero_shot_pcm_v10_start_swa__pcm__0_50_lug_50` is fundamentally the `pcm_v10_start_swa__pcm__0_50` model (i.e. Swahili pre-trained and fine-tuned on Nigerian Pidgin) that was used to evaluate Luganda. Hence, these files follow the format `src/runs/v10/models/v20/<eval>_zero_shot_<model>_<eval>_50`.
# References / Sources / Libraries
- MasakhaNER ([Github](https://github.com/masakhane-io/masakhane-ner), [paper](https://arxiv.org/abs/2103.11811)). Their code and data can be found in `data/`, and is licensed under `Masakhane-LICENSE`
- Interspersed throughout the code are links to other sources that were used to solve a particular problem.

# Model Cards
We release some models that were fine-tuned on the [MasakhaNER](https://arxiv.org/abs/2103.11811) dataset, using specific languages. We start with 3 different pre-trained models, and then fine-tune on 9 of the languages in the dataset.
The pre-trained models we start with are:
- [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- [xlm-roberta-base-finetuned-swahili](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) -> Domain adaptive fine-tuning on a large, unlabelled monolingual corpus.
- [xlm-roberta-base-finetuned-X](https://huggingface.co/Davlan/) -> Domain adaptive fine-tuning on large, unlabelled monolingual corpora, for each of the languages under consideration.

Here we describe a joint model card for all of the models, as they were quite similar.
## About
These models are transformer based and were all fine-tuned on the MasakhaNER dataset. It is a named entity recognition dataset, containing mostly news articles in 10 different African languages. 
The models were all fine-tuned for 50 epochs, with a maximum sequence length of 200, 32 batch size, 5e-5 learning rate. Each model was fine-tuned 5 times (with different random seeds), and the ones we upload are the ones that did the best for the language it was evaluated on. 

These models were fine-tuned by me, Michael Beukman while doing a project at the University of the Witwatersrand, Johannesburg. This is version 1, as of 20 November 2021.
These models are licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### Contact & More information
For more information about the models, including training scripts, detailed results and further resources, you can visit the the [main Github repository](https://github.com/Michael-Beukman/NERTransfer). You can contact me by filing an issue on this repository.

### Training Resources
In the interest of openness, and reporting resources used, we list here how long the training process took, as well as what the minimum resources would be to reproduce this. Fine-tuning each model on the NER dataset took between 10 and 30 minutes, and was performed on a NVIDIA RTX3090 GPU. To use a batch size of 32, at least 14GB of GPU memory was required, although it was just possible to fit these models in around 6.5GB's of VRAM when using a batch size of 1.


## Data
The train, evaluation and test datasets were taken directly from the MasakhaNER [Github](https://github.com/masakhane-io/masakhane-ner) repository, with minimal to no preprocessing, as the original dataset is already of high quality.
The motivation for the use of this data is that it is the "first large, publicly available, high­ quality dataset for named entity recognition (NER) in ten African languages" ([source](https://arxiv.org/pdf/2103.11811.pdf)). The high-quality data, as well as the groundwork laid by the paper introducing it are some more reasons why this dataset was used. For evaluation, the dedicated test split was used, which is from the same distribution as the training data, so the models may not generalise to other distributions, and further testing would need to be done to investigate this. The exact distribution of the data is covered in detail [here](https://arxiv.org/abs/2103.11811).

## Intended Use
These models are intended to be used for NLP research into e.g. interpretability or transfer learning. Using these models in production is not supported, as generalisability and downright performance is limited. In particular, these models are not designed to be used in any important downstream task that could affect people, as harm could be caused by the limitations of the model, described next.

## Limitations
These models were only trained on one (relatively small) dataset, covering one task (NER) in one domain (news articles) and in a set span of time. The results may not generalise, and the model may perform badly, or in an unfair / biased way if used on other tasks. Although the purpose of this project was to investigate transfer learning, the performance on languages that the models were not trained for does suffer.


Because all of these models covered here used xlm-roberta-base as their starting point (potentially with domain adaptive fine-tuning on specific languages), this model's limitations can also apply here. These can include being biased towards the hegemonic viewpoint of most of its training data, being ungrounded and having subpar results on other languages (possibly due to unbalanced training data).

The embeddings we provide are meant purely for exploratory and analysis purposes, and might not be useful, or even dangerous when deployed in a high stakes environment that has tangible consequences.

As [Adelani et al. (2021)](https://arxiv.org/abs/2103.11811) showed, the models in general struggled with entities that were either longer than 3 words and entities that were not contained in the training data. This could bias the models towards not finding, e.g. names of people that have many words, possibly leading to a misrepresentation in the results. Similarly, names that are uncommon, and may not have been found in the training data (due to e.g. different languages) would also be predicted less often.


Additionally, these models have not been verified in practice, and other, more subtle problems may become prevalent if used without any verification that it does what it is supposed to.

### Privacy & Ethical Considerations
The data comes from only publicly available news sources, the only available data should cover public figures and those that agreed to be reported on. See the original MasakhaNER paper for more details.

No explicit ethical considerations or adjustments were made during fine-tuning of these models.

## Metrics
As shown in our report, and in the MasakhaNER paper, the language adaptive models achieve (mostly) superior performance over starting with xlm-roberta-base. We also demonstrated some better transfer capabilities for some of these models. Our main metric was the aggregate F1 score for all NER categories.

These metrics are on the test set for MasakhaNER, so the data distribution is similar to the training set, so these results do not directly indicate how well models generalise.
We do find large variation in results when starting from different seeds (5 different seeds were tested), indicating that the fine-tuning process for transfer might be unstable.

The metrics used were chosen to be consistent with previous work, and to facilitate research. Other metrics may be more appropriate for other purposes.
## Caveats and Recommendations
In general, these models performed worse on the 'date' category compared to others, so if dates are a critical factor, then that might need to be taken into account and addressed, by for example collecting and annotating more data.

## Model Structure
Here are some details regarding the specific models, like where to find them, what the starting points where and what they were evaluated on. Evaluation here is also the same language that the model was fine-tuned on.
All of these metrics were calculated on the test set, and the seed was chosen that gave the best overall F1 score. The first three result columns are averaged over all categories, and the latter 4 provide performance broken down by category.

These models can predict the following label for a token ([source](https://huggingface.co/Davlan/xlm-roberta-large-masakhaner)):


Abbreviation|Description
-|-
O|Outside of a named entity
B-DATE |Beginning of a DATE entity right after another DATE entity
I-DATE |DATE entity
B-PER |Beginning of a person’s name right after another person’s name
I-PER |Person’s name
B-ORG |Beginning of an organisation right after another organisation
I-ORG |Organisation
B-LOC |Beginning of a location right after another location
I-LOC |Location


| Model Name                                         | Staring point        | Evaluation / Fine-tune Language  | F1 | Precision | Recall | F1 (DATE) | F1 (LOC) | F1 (ORG) | F1 (PER) |
| -------------------------------------------------- | -------------------- | -------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| [xlm-roberta-base-finetuned-hausa-finetuned-ner-hausa](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-hausa-finetuned-ner-hausa) | [hau](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-hausa) | hau                  | 92.27          | 90.46          | 94.16          | 85.00          | 95.00          | 80.00          | 97.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-hausa](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-hausa) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | hau                  | 89.14          | 87.18          | 91.20          | 82.00          | 93.00          | 76.00          | 93.00          |
| [xlm-roberta-base-finetuned-ner-hausa](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-hausa) | [base](https://huggingface.co/xlm-roberta-base) | hau                  | 89.94          | 87.74          | 92.25          | 84.00          | 94.00          | 74.00          | 93.00          |
| [xlm-roberta-base-finetuned-igbo-finetuned-ner-igbo](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-igbo-finetuned-ner-igbo) | [ibo](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-igbo) | ibo                  | 88.39          | 87.08          | 89.74          | 74.00          | 91.00          | 90.00          | 91.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-igbo](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-igbo) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | ibo                  | 84.93          | 83.63          | 86.26          | 70.00          | 88.00          | 89.00          | 84.00          |
| [xlm-roberta-base-finetuned-ner-igbo](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-igbo) | [base](https://huggingface.co/xlm-roberta-base) | ibo                  | 86.06          | 85.20          | 86.94          | 76.00          | 86.00          | 90.00          | 87.00          |
| [xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-kinyarwanda](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-kinyarwanda) | [kin](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-kinyarwanda) | kin                  | 79.55          | 75.56          | 83.99          | 69.00          | 79.00          | 77.00          | 90.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-kinyarwanda](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-kinyarwanda) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | kin                  | 76.31          | 72.64          | 80.37          | 70.00          | 76.00          | 75.00          | 84.00          |
| [xlm-roberta-base-finetuned-ner-kinyarwanda](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-kinyarwanda) | [base](https://huggingface.co/xlm-roberta-base) | kin                  | 74.59          | 72.17          | 77.17          | 70.00          | 75.00          | 70.00          | 82.00          |
| [xlm-roberta-base-finetuned-luganda-finetuned-ner-luganda](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-luganda-finetuned-ner-luganda) | [lug](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-luganda) | lug                  | 85.37          | 82.75          | 88.17          | 78.00          | 82.00          | 80.00          | 92.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-luganda](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-luganda) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | lug                  | 82.57          | 80.38          | 84.89          | 75.00          | 80.00          | 82.00          | 87.00          |
| [xlm-roberta-base-finetuned-ner-luganda](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-luganda) | [base](https://huggingface.co/xlm-roberta-base) | lug                  | 80.91          | 78.59          | 83.37          | 73.00          | 78.00          | 77.00          | 86.00          |
| [xlm-roberta-base-finetuned-luo-finetuned-ner-luo](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-luo-finetuned-ner-luo) | [luo](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-luo) | luo                  | 78.71          | 78.91          | 78.52          | 72.00          | 84.00          | 59.00          | 87.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-luo](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-luo) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | luo                  | 78.13          | 77.75          | 78.52          | 65.00          | 82.00          | 61.00          | 89.00          |
| [xlm-roberta-base-finetuned-ner-luo](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-luo) | [base](https://huggingface.co/xlm-roberta-base) | luo                  | 75.99          | 76.18          | 75.80          | 71.00          | 76.00          | 62.00          | 85.00          |
| [xlm-roberta-base-finetuned-naija-finetuned-ner-naija](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-naija-finetuned-ner-naija) | [pcm](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-naija) | pcm                  | 88.06          | 87.04          | 89.12          | 90.00          | 88.00          | 81.00          | 92.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-naija](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-naija) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | pcm                  | 89.12          | 87.84          | 90.42          | 90.00          | 89.00          | 82.00          | 94.00          |
| [xlm-roberta-base-finetuned-ner-naija](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-naija) | [base](https://huggingface.co/xlm-roberta-base) | pcm                  | 88.89          | 88.13          | 89.66          | 92.00          | 87.00          | 82.00          | 94.00          |
| [xlm-roberta-base-finetuned-hausa-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-hausa-finetuned-ner-swahili) | [hau](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-hausa) | swa                  | 88.36          | 86.95          | 89.82          | 86.00          | 91.00          | 77.00          | 94.00          |
| [xlm-roberta-base-finetuned-igbo-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-igbo-finetuned-ner-swahili) | [ibo](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-igbo) | swa                  | 87.75          | 86.55          | 88.97          | 85.00          | 92.00          | 77.00          | 91.00          |
| [xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-swahili) | [kin](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-kinyarwanda) | swa                  | 87.26          | 85.15          | 89.48          | 83.00          | 91.00          | 75.00          | 93.00          |
| [xlm-roberta-base-finetuned-luganda-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-luganda-finetuned-ner-swahili) | [lug](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-luganda) | swa                  | 88.93          | 87.64          | 90.25          | 83.00          | 92.00          | 79.00          | 95.00          |
| [xlm-roberta-base-finetuned-luo-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-luo-finetuned-ner-swahili) | [luo](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-luo) | swa                  | 87.93          | 86.91          | 88.97          | 83.00          | 91.00          | 76.00          | 94.00          |
| [xlm-roberta-base-finetuned-naija-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-naija-finetuned-ner-swahili) | [pcm](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-naija) | swa                  | 87.26          | 85.15          | 89.48          | 83.00          | 91.00          | 75.00          | 93.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-swahili) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | swa                  | 90.36          | 88.59          | 92.20          | 86.00          | 93.00          | 79.00          | 96.00          |
| [xlm-roberta-base-finetuned-wolof-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-wolof-finetuned-ner-swahili) | [wol](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-wolof) | swa                  | 87.80          | 86.50          | 89.14          | 86.00          | 90.00          | 78.00          | 93.00          |
| [xlm-roberta-base-finetuned-yoruba-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-yoruba-finetuned-ner-swahili) | [yor](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-yoruba) | swa                  | 87.73          | 86.67          | 88.80          | 85.00          | 91.00          | 75.00          | 93.00          |
| [xlm-roberta-base-finetuned-ner-swahili](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-swahili) | [base](https://huggingface.co/xlm-roberta-base) | swa                  | 88.71          | 86.84          | 90.67          | 83.00          | 91.00          | 79.00          | 95.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-wolof](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-wolof) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | wol                  | 69.01          | 73.25          | 65.23          | 27.00          | 85.00          | 52.00          | 67.00          |
| [xlm-roberta-base-finetuned-wolof-finetuned-ner-wolof](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-wolof-finetuned-ner-wolof) | [wol](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-wolof) | wol                  | 69.02          | 67.60          | 70.51          | 30.00          | 84.00          | 44.00          | 71.00          |
| [xlm-roberta-base-finetuned-ner-wolof](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-wolof) | [base](https://huggingface.co/xlm-roberta-base) | wol                  | 66.12          | 69.46          | 63.09          | 30.00          | 84.00          | 54.00          | 59.00          |
| [xlm-roberta-base-finetuned-swahili-finetuned-ner-yoruba](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-yoruba) | [swa](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili) | yor                  | 80.29          | 78.34          | 82.35          | 77.00          | 82.00          | 73.00          | 86.00          |
| [xlm-roberta-base-finetuned-yoruba-finetuned-ner-yoruba](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-yoruba-finetuned-ner-yoruba) | [yor](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-yoruba) | yor                  | 83.68          | 79.92          | 87.82          | 78.00          | 86.00          | 74.00          | 92.00          |
| [xlm-roberta-base-finetuned-ner-yoruba](https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-ner-yoruba) | [base](https://huggingface.co/xlm-roberta-base) | yor                  | 78.22          | 77.21          | 79.26          | 77.00          | 80.00          | 71.00          | 82.00          |

## Usage
To use these models, you can do the following, with just changing the model name ([source](https://huggingface.co/dslim/bert-base-NER)):

```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
model_name = 'mbeukman/xlm-roberta-base-finetuned-hausa-finetuned-ner-hausa'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Donald Trump Ya Rufe Gwamnatin Amurka"

ner_results = nlp(example)
print(ner_results)
# [{'entity': 'B-PER', 'score': 0.9998982, 'index': 1, 'word': '▁Donald', 'start': 0, 'end': 6}, {'entity': 'I-PER', 'score': 0.9999061, 'index': 2, 'word': '▁Trump', 'start': 6, 'end': 12}, {'entity': 'B-LOC', 'score': 0.9999413, 'index': 7, 'word': '▁Amurka', 'start': 30, 'end': 37}]
```