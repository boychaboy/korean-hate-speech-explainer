# Korean Hate Speech Classification and Explanation

This repository is a source code of a paper at KSC(한국정보과학회) 2021, 혐오 댓글 분류 모델 해석 및 편향성 제거 
- Authors: **Younghoon Jeong**, Jungyun Seo
- We used Sampling and Occlusion algorithm used in [Contextualizing Hate Speech Classifiers with Post-hoc Explanation](https://arxiv.org/pdf/2005.02439.pdf) 
- This code is reimplementation of [original code](https://inklab.usc.edu/contextualize-hate-speech/)

# Environment

`conda create -f environment.yaml`. 
`conda activate soc_env`

# How to run

## 1. Train

`./scripts/run_train.sh {gpu_id}` 
- Automatically trains classifier
- You can change model hyperparameters from script

## 2. Explain

`./scripts/run_explain.sh {gpu_id}`

runs explanation with the `test` data

- `--output_dir` : directory of classifier model
- `--lm_dir` : directory of language model (different with classifier)
- `--nb_range` : window size to sample n
- `--sample_n` : number of sentences to sample
- `--hiex` : hierarchical explanation (default : sequential explanation)
- `--output_filename` : name of the explanation file

explanation result is saved in `output_dir`

## 3. Visualize

`./scripts/run_visualize.sh`

- `--input_file` : file dir to visualize (*.txt, *.heix)
- `--hiex` : true if hierarchical explanation

---

# Final output

**Before debiasing** with Sampling and Occlusion algorithm

![fig_0](https://user-images.githubusercontent.com/48426904/138645281-76569acb-e795-4696-9079-6051cfa13a6f.png)

**After debiasing**

![fig_0](https://user-images.githubusercontent.com/48426904/138645358-2a130093-b3c8-4003-8959-aacdc5ef8547.png)

# How to modify (if you want to apply this method to other data)

## 1. Prepare data

```bash
|----data/beep
|----|----train.json
|----|----dev.json
```

- `Binary label`

## 2. Modify data processor

`loader/*.py`

- modfiy `class *Processor` 

---



## Reference

```
@inproceedings{kennedy2020contextualizing,
   author = {Kennedy*, Brendan and Jin*, Xisen and Mostafazadeh Davani, Aida and Dehghani, Morteza and Ren, Xiang},
   title = {Contextualizing {H}ate {S}peech {C}lassifiers with {P}ost-hoc {E}xplanation},
   year = {to appear},
   booktitle = {Proceedings of the 58th {A}nnual {M}eeting of the {A}ssociation for {C}omputational {L}inguistics}
} 
```
```
@inproceedings{moon-etal-2020-beep,
    title = "{BEEP}! {K}orean Corpus of Online News Comments for Toxic Speech Detection",
    author = "Moon, Jihyung  and
      Cho, Won Ik  and
      Lee, Junbum",
    booktitle = "Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.socialnlp-1.4",
    pages = "25--31",
    abstract = "Toxic comments in online platforms are an unavoidable social issue under the cloak of anonymity. Hate speech detection has been actively done for languages such as English, German, or Italian, where manually labeled corpus has been released. In this work, we first present 9.4K manually labeled entertainment news comments for identifying Korean toxic speech, collected from a widely used online news platform in Korea. The comments are annotated regarding social bias and hate speech since both aspects are correlated. The inter-annotator agreement Krippendorff{'}s alpha score is 0.492 and 0.496, respectively. We provide benchmarks using CharCNN, BiLSTM, and BERT, where BERT achieves the highest score on all tasks. The models generally display better performance on bias identification, since the hate speech detection is a more subjective issue. Additionally, when BERT is trained with bias label for hate speech detection, the prediction score increases, implying that bias and hate are intertwined. We make our dataset publicly available and open competitions with the corpus and benchmarks.",
}
```
