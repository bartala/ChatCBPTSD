# ChatCB-PTSD

## Overview
Maternal mental health is a critical concern during and after childbirth, affecting both mothers and their children. 
Untreated postpartum psychopathology can result in child neglect, leading to substantial pediatric health expenses and societal costs. 
Following childbirth, some women experience childbirth-related post-traumatic stress disorder (CB-PTSD). 
While postpartum depression is routinely screened, there is no standardized protocol for identifying those at risk of CB-PTSD. 
Recent computational advances in free text analysis show promise for diagnosing psychiatric conditions.
This study explores the potential of Chat Generative Pre-trained Transformer (ChatGPT), a large language model (LLM) developed by OpenAI, for the clinical task of screening for CB-PTSD by analyzing maternal narratives of childbirth experiences.
We utilize the power of ChatGPT's `gpt-3.5-turbo-16k` LLM to classify narratives using zero-shot and few-shot learning.
Additionally, we extract the numerical vector representation (embeddings) of narratives using the `text-embedding-ada-002` model via OpenAI’s API.
Using these embeddings, we trained a densely connected feedforward neural network (DFNN) to detect CB-PTSD via narrative classification.
Our DFNN model, trained using OpenAI's embeddings, outperformed ChatGPT's zero- and few-shot learning, and six previously published models trained on mental health and clinical domains.
Understanding the connection between narrative language and post-traumatic outcomes enhances support for at-risk mothers, benefiting both mother and child.

# Repository Contents
## Running the Code
`[harvard]_chatgpt.py` - implementation of Models \#1 to \#3 of the paper. These are OpenAI based models including ChatGPT.

`[harvard]_siamgenerateexamples_ptsd_.py` - Evaluates Model \#3 using various embeddings of LLMs previously trained within clinical and mental health realms.
This script also fine-tunes the selected LLMs on a classification task.

`requirements.txt` - required python packages to run the code.

`.env.example` - local variable and credentials needed to run the code.

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
