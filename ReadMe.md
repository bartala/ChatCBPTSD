# ChatCB-PTSD

## Overview
Free-text analysis using Machine Learning (ML)-based Natural Language Processing (NLP) shows promise for diagnosing psychiatric conditions. Chat Generative Pre-trained Transformer (ChatGPT) has demonstrated initial feasibility for this purpose; however, this work remains preliminary, and whether it can accurately assess mental illness remains to be determined. 
This study examines ChatGPT’s utility to identify post-traumatic stress disorder following childbirth (CB-PTSD), a maternal postpartum mental illness affecting millions of women annually, with no standard screening protocol. 
We explore ChatGPT’s potential to screen for CB-PTSD by analyzing maternal childbirth narratives as the sole data source. 
By developing an ML model that utilizes ChatGPT’s knowledge, we identify CB-PTSD via narrative classification. 
Our model outperformed  (F1 score: 0.82) ChatGPT and six previously published large language models trained on mental health or clinical domains data. 
Our results suggest that ChatGPT can be harnessed to identify CB-PTSD.
Our modeling approach can be generalized to other mental illness disorders.

# Repository Contents

## Papaer LateX Documents
`empty.eps`

`sn-article.tex`

`sn-bibliography.bib`

`sn-jnl.cls`

`images` - figures of the manuscript.


## Running the Code
`[harvard]_chatgpt.py` - implementation of Models \#1 to \#3 of the paper. These are OpenAI based models including ChatGPT.

`[harvard]_siamgenerateexamples_ptsd_.py` - Evaluates Model \#3 using various embeddings of LLMs previously trained within clinical and mental health realms.
This script also fine-tunes the selected LLMs on a classification task.

`requirements.txt` - required python packages to run the code.

`.env.example` - variable and credentials needed to run the code.

## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to alon.bartal@biu.ac.il.

## Citing our work
If you find this code useful for your research, please consider citing us:
```
@article{Bartal2023ChatCB-PTSD,
  title     = {Exploring ChatGPT’s Potential for Identifying Childbirth-Related Post-Traumatic Stress Disorder},
  author    = {Bartal, Alon and Jagodnik, Kathleen M. and Dekel, Sharon},
  journal   = {},
  volume    = {},
  number    = {},
  pages     = {from page– to page},
  year      = {2023}
}
