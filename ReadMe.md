# ChatCB-PTSD

## Overview
Maternal mental health is a critical concern during and after childbirth, affecting both mothers and their children. 
Untreated postpartum psychopathology can result in child neglect, leading to substantial pediatric health expenses and societal costs. 
Following childbirth, some women experience childbirth-related post-traumatic stress disorder (CB-PTSD). 
While postpartum depression is routinely screened, there is no standardized protocol for identifying those at risk of CB-PTSD. 
Recent computational advances in free text analysis show promise for diagnosing psychiatric conditions.
This study explores the potential of Chat Generative Pre-trained Transformer (ChatGPT), a large language model (LLM) developed by OpenAI, for the clinical task of screening for CB-PTSD by analyzing maternal narratives of childbirth experiences.
We utilize the power of ChatGPT's \texttt{gpt-3.5-turbo-16k} LLM to classify narratives using zero-shot and few-shot learning.
Additionally, we extract the numerical vector representation (embeddings) of narratives using the \texttt{text-embedding-ada-002} model via OpenAI’s API.
Using these embeddings, we trained a densely connected feedforward neural network (DFNN) to detect CB-PTSD via narrative classification.
Our DFNN model, trained using OpenAI's embeddings, outperformed ChatGPT's zero- and few-shot learning, and six previously published models trained on mental health and clinical domains.
Understanding the connection between narrative language and post-traumatic outcomes enhances support for at-risk mothers, benefiting both mother and child.

## Running the code


## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to alon.bartal@biu.ac.il.




## Citing
If you find this code useful for your research, please consider citing us:
```
@article{Bartal2023ChatCB-PTSD,
  title     = {Collective Memory of Dynamic Events: Analyzing Persistence of Attention to Bankruptcy Events},
  author    = {Bartal, Alon and Jagodnik, Kathleen M. and Dekel, Sharon},
  journal   = {},
  volume    = {},
  number    = {},
  pages     = {from page– to page},
  year      = {2023}
}
