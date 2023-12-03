

# Associated Publication:
# Bartal A, Jagodnik KM, Chan SD, Dekel S. (2023) ChatGPT Embeddings Can Be Used for Detecting Post-Traumatic Stress Following Childbirth Via Birth Stories. Scientific Reports.

# Project Description:
# Free-text analysis using Machine Learning (ML)-based Natural Language Processing (NLP) shows promise for diagnosing psychiatric conditions. Chat Generative Pre-trained Transformer (ChatGPT) has demonstrated initial feasibility for this purpose; however, this work remains preliminary, and whether it can accurately assess mental illness remains to be determined. This study examines ChatGPT’s utility to identify post-traumatic stress disorder following childbirth (CB-PTSD), a maternal postpartum mental illness affecting millions of women annually, with no standard screening protocol. Using a sample of 1,295 women who gave birth in the last six months and were 18+ years old, recruited through hospital announcements, social media, and professional organizations, Wwe explore ChatGPT’s potential to screen for CB-PTSD by analyzing maternal childbirth narratives as the sole data source. The PTSD Checklist for DSM-5 (PCL-5; cutoff 31) was used to assess CB-PTSD. By developing an ML model that utilizes ChatGPT’s knowledge, we identify CB-PTSD via narrative classification. Our model outperformed (F1 score: 0.82) ChatGPT and six previously published large language models (LLMs) trained on mental health or clinical domains data, suggesting that ChatGPT can be harnessed to identify CB-PTSD. Our modeling approach could be generalized to assess other mental health disorders.


openai.organization = "" # complete credentials
openai.api_key = "" # complete credentials

import openai
import tiktoken
import os
import pandas as pd
import time
import ast
import numpy as np

PTH = '...'

"""# Help functions for OpenAI"""

def get_embedding(text, model):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


# Classification
def get_completion(prompt, model="gpt-3.5-turbo-16k",temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message["content"]


# Create prompt
def return_few_shot_promt(text):
  prompt = f"""
              You are a Psychiatrists specialized in diagnosing and treating Post Traumatic Stress Disorder (PTSD).\
              I will provide you with a narrative written by a woman describing her birth experience.\
              Your task is to decide whether this woman is of high risk of PTSD (Label 1), or lower risk of PTSD (Label 0).\
              Your task is to determine whether this woman is at high risk of developing PTSD (Label 1) or lower risk of PTSD (Label 0).\
              Do not write anything but  '1' or '0'.
              Here are a few examples of text with their associated class lable as '1' (PTSD), or '0' (No-PTSD).\

              <Text>: '''{sick_narrative}'''
              <Label>: 1
              ###

              <Text>: '''{healthy_narrative}'''
              <Label>: 0
              ###

              <Text>: '''{text}'''
              """
  return(prompt)


# Create prompt
def return_zero_shot_promt(text):
  prompt = f"""
              You are a Psychiatrists specialized in diagnosing and treating Post Traumatic Stress Disorder (PTSD).\
              I will provide you with a narrative written by a woman describing her birth experience.\
              Your task is to decide whether this woman is of high risk of PTSD (Label 1), or lower risk of PTSD (Label 0).\
              Your task is to determine whether this woman is at high risk of developing PTSD (Label 1) or lower risk of PTSD (Label 0).\
              Do not write anything but  '1' or '0'.
              """
  return(prompt)


# Count number of tokens
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

"""# Load the data"""

# Load the data

# PCL-5 threshold
# Value selected based on: Kruger-Gottschalk, A. et al. The German version of the Posttraumatic Stress Disorder Checklist for DSM-5 (PCL-5): Psychometric properties and diagnostic utility. BMC Psychiatry. 17, 1–9 (2017).
threshold = 31

# COVID-19 Dataset: Recruitment from April 2020 to December 2020
df_covid = pd.read_csv(os.path.join(PTH,"covid.csv"))
df_covid['source'] = 'covid'

# CBEx Dataset: Recruitment from November 2016 to July 2018
df_cbex = pd.read_csv(os.path.join(PTH,"CBEx.csv"))
df_cbex['source'] = 'CBEx'


# A single dataframe of narratives of women with PCL-5 >=31 Only
all_df =  pd.concat([
                      df_covid[['record_id','spcl5_total', 'cb_delivery_narrative','pdi_total', 'source']],
                      df_cbex[['record_id','spcl5_total', 'cb_delivery_narrative','pdi_total','source']]
                    ])

# Remove sentences with fewer than 30 words
all_df['n_words'] = all_df['cb_delivery_narrative'].str.split().apply(len)
all_df = all_df[all_df['n_words'] >= 30]

# Set target variable
all_df['y']=0
all_df.loc[all_df['spcl5_total']>=threshold,'y'] = 1
all_df['y'].value_counts()

all_df

"""Count number of tokens and expected costs for OpenAI"""

tokens = 0
for text in all_df['cb_delivery_narrative'].tolist():
  tokens = tokens + (num_tokens_from_string("tiktoken is great!", "cl100k_base"))

print(tokens * 'replace with tolen price per used model')

"""# Classification with OpenAI"""

# Select rows with examples for few-shot learning
sick_narrative = all_df[all_df['spcl5_total'] > threshold].sample(n = 1)
healthy_narrative = all_df[all_df['spcl5_total'] < threshold].sample(n = 1)

# Remove few-shot examples from original database
remove_ids = [ sick_narrative['record_id'].iloc[0], healthy_narrative['record_id'].iloc[0] ]
all_df = all_df[~all_df['record_id'].isin(remove_ids)]

# Keep only narrative
sick_narrative = sick_narrative['cb_delivery_narrative'].tolist()[0]
healthy_narrative = healthy_narrative['cb_delivery_narrative'].tolist()[0]

responce_labels = []

"""## Zero-Shot Classification"""

i=0
text = all_df.iloc[i]['cb_delivery_narrative']
prompt = return_zero_shot_promt(text)
response = get_completion(prompt)

for i in range(0,len(all_df)):
  text = all_df.iloc[i]['cb_delivery_narrative']
  prompt = return_zero_shot_promt(text)
  response = get_completion(prompt)
  responce_labels.append([
                          all_df.iloc[i]['record_id'],
                          text,
                          all_df.iloc[i]['source'],
                          response
                          ])
  time.sleep(3)
  print(i)
  i = i + 1

df_zero_shot = pd.DataFrame(responce_labels)
df_zero_shot.columns = [['record_id','cb_delivery_narrative','source','y']]
df_zero_shot.to_csv('all_df_chatgpt.csv', index=False)


"""## Few-Shot Learning"""

responce_labels = []

for i in range(0,len(all_df)):
  text = all_df.iloc[i]['cb_delivery_narrative']
  prompt = return_few_shot_promt(text)
  response = get_completion(prompt)
  responce_labels.append([
                          all_df.iloc[i]['record_id'],
                          text,
                          all_df.iloc[i]['source'],
                          response
                          ])
  time.sleep(3)
  print(i)
  i = i + 1

df_few_shot = pd.DataFrame(responce_labels)
df_few_shot.columns = [['record_id','cb_delivery_narrative','source','y']]
df_few_shot.to_csv('all_df_chatgpt.csv', index=False)

"""# OpenAI: Get Embeddings"""

# Get embeddigns of narratives
i  = 1
ada_embedding = [] # length of embeddings is 1536
for text in all_df['cb_delivery_narrative'].tolist():
  print(i)
  embddings = get_embedding(text, model='text-embedding-ada-002') # text-embedding-ada-002 (https://openai.com/blog/new-and-improved-embedding-model)
  ada_embedding.append(embddings)
  i = i + 1

# Save embeddings
all_df['ada_embedding'] = ada_embedding
all_df.to_csv(os.path.join(PTH,'embedded_CBPTSD_text-embedding-ada-002.csv'), index=False)

"""# Model 3

## Help Functions
"""

import numpy as np
from sklearn.decomposition import PCA
from numpy import dot
from numpy.linalg import norm

# Hadamard Product
def vec_dist(v1,v2):
  v1 = np.array(v1)
  v2 = np.array(v2)
  res = v1*v2
  return(res)


# Normalize Data
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

def norma(data):
  # Fit and transform the data using the StandardScaler
  normalized_data = scaler.fit_transform(data)

  # Convert the normalized_data back to a DataFrame (optional)
  normalized_data_df = pd.DataFrame(normalized_data, columns=data.columns)

  return(normalized_data_df)

"""## Load Data"""

# load the data and set a threshold
all_df = pd.read_csv(os.path.join(PTH,'data/embedded_CBPTSD_text-embedding-ada-002.csv'))

# Specify PCL-5 Questionnaire Threshold = 31
threshold = 31

# Set target variable
all_df['y']=0
all_df.loc[all_df['spcl5_total']>=threshold,'y'] = 1
all_df['y'].value_counts()


"""## Extract Numeric Vectors of Narratives and Label Classes"""

all_embeddings = all_df['ada_embedding'].apply(ast.literal_eval).apply(lambda x: [float(val) for val in x])

all_embeddings = pd.DataFrame(all_embeddings.tolist())

# Normalize Data
all_embeddings = norma(all_embeddings)

all_embeddings['y'] = all_df['y'].tolist()


pos_embeddings = all_embeddings[all_embeddings['y'] == 1]

neg_embeddings = all_embeddings[all_embeddings['y'] == 0]

pos_embeddings.drop('y',axis=1, inplace = True)

neg_embeddings.drop('y',axis=1, inplace = True)

"""## Data Balancing"""

# Random sampling from the Negative class to get the same number of records as the Positive class
X_pos = pos_embeddings

X_neg =  neg_embeddings.sample(n = len(X_pos))

print(len(X_neg), len(X_pos))

# Reset Index
X_neg = X_neg.reset_index(drop=True)
X_pos = X_pos.reset_index(drop=True)

# Remove S sentences for testing later
s = 20

pos_test_sent = X_pos.sample(s)
neg_test_sent = X_neg.sample(s)

X_pos =  X_pos.drop(pos_test_sent.index, axis=0)
X_neg =  X_neg.drop(neg_test_sent.index, axis=0)

print(len(X_neg), len(X_pos))

"""## Create Pairs of Narratives [pos-pos; neg-neg; and pos-neg]"""

# Loop to create all <positive-positive> pairs as positive examples

pos_pos_vec = []

for i in range(0,len(X_pos) ):
  for j in range(i+1, len(X_pos) ):
    pos_pos_vec.append( vec_dist(X_pos.iloc[i,:], X_pos.iloc[j,:]) )

df_pos_pos = pd.DataFrame(pos_pos_vec)
df_pos_pos['y'] = 1

# Loop to create all <negative-negative> pairs as positive examples
neg_neg_vec = []

for i in range(0,len(X_neg) ):
  for j in range(i+1, len(X_neg) ):
    neg_neg_vec.append(  vec_dist(X_neg.iloc[i,:] , X_neg.iloc[j,:]) )

df_neg_neg = pd.DataFrame(neg_neg_vec)
df_neg_neg['y'] = 1

# Loop to create all <negative-positive> pairs as negative examples

pos_neg_vec = []

for i in range(0,len(X_pos) ):
  for j in range(0, len(X_neg) ):
    pos_neg_vec.append(  vec_dist(X_pos.iloc[i,:] , X_neg.iloc[j,:]) )

df_pos_neg = pd.DataFrame(pos_neg_vec)
df_pos_neg = df_pos_neg.sample(n = len(df_pos_pos)*2)
df_pos_neg['y'] = 0

X = pd.concat([df_pos_pos, df_neg_neg, df_pos_neg], axis = 0)

y = X['y'].astype("int")
print(y.value_counts())

X.drop('y',axis=1, inplace = True)

"""## Train a DNN of the Best Model"""

#https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

from tensorflow import keras
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense

def train_ksmodels(X,Y, epoc = 5):

  x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3,shuffle=True)

  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) # stop the training when there is no loss improvement for 3 consecutive epochs
  model = Sequential()
  model.add(Dense(400, input_dim=len(x_train.columns), activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  op = keras.optimizers.Adam(learning_rate=0.0004)
  model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
  model.fit(x_train, y_train, epochs = epoc, batch_size=32 , validation_data=(x_val,y_val), callbacks=[callback])
  #evaluate the keras model
  _, accuracy = model.evaluate(x_train, y_train)
  print('Accuracy: %.2f' % (accuracy*100))
  return(model)

nn_model = train_ksmodels(X,y, epoc = 50)

# Save the model in tensorflow format
from tensorflow import keras

nn_model.save(os.path.join(PTH,'model/MyModel_tf'),
              save_format='tf')

"""## Test the model on the Test set

### Predict class of a pair of vectors
"""

def predictions_siam(v, pos_v, neg_v, y_true):
    pos_embd_df = pd.DataFrame(vec_dist(v, pos_v))
    y_pos_pred = nn_model.predict(pos_embd_df.transpose().astype('float32'))[0][0]

    neg_embd_df = pd.DataFrame(vec_dist(v,neg_v))
    y_neg_pred = nn_model.predict(neg_embd_df.transpose().astype('float32'))[0][0]

    if y_pos_pred  >  y_neg_pred:
      return([1,y_true])
    return([0,y_true])

# Normalize the Test data
pos_test_sent['y'] = 1
neg_test_sent['y'] = 0
concatenated_df = pd.concat([pos_test_sent, neg_test_sent], axis=0)

y = concatenated_df['y'].astype("int")
print(y.value_counts())

concatenated_df.drop('y',axis=1, inplace = True)

for i in range(0,len(concatenated_df)):
  nn_model.predict(concatenated_df.iloc[i,0:])

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


y_predicted = y_pred['y_pred']
y_true = y_pred['y_true']

print(classification_report(y_true, y_predicted))

cm = confusion_matrix(y_true,y_predicted)
print('Confusion Matrix [Actual (rows) X Predicted(columns)]: \n', cm)
sensitivity1 = cm[1,1]/(cm[1,1]+cm[1,0])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Specificity : ', specificity1)
print("AUC:",roc_auc_score(y_true, y_predicted))
print("\n")
