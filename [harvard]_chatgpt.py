
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


# classification
def get_completion(prompt, model="gpt-3.5-turbo-16k",temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message["content"]


# create promt
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





# create promt
def return_zero_shot_promt(text):
  prompt = f"""
              You are a Psychiatrists specialized in diagnosing and treating Post Traumatic Stress Disorder (PTSD).\
              I will provide you with a narrative written by a woman describing her birth experience.\
              Your task is to decide whether this woman is of high risk of PTSD (Label 1), or lower risk of PTSD (Label 0).\
              Your task is to determine whether this woman is at high risk of developing PTSD (Label 1) or lower risk of PTSD (Label 0).\
              Do not write anything but  '1' or '0'.
              """
  return(prompt)


# count number of tokens
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

"""# Load the data"""

# Load the data

# pcl-5 threshold
threshold = 31

# COVID dataset
df_covid = pd.read_csv(os.path.join(PTH,"covid.csv"))
df_covid['source'] = 'covid'

# CBEx dataset
df_cbex = pd.read_csv(os.path.join(PTH,"CBEx.csv"))
df_cbex['source'] = 'CBEx'


# a single dataframe of narrtatives of women with PCL-5 >=32 Only!
all_df =  pd.concat([
                      df_covid[['record_id','spcl5_total', 'cb_delivery_narrative','pdi_total', 'source']],
                      df_cbex[['record_id','spcl5_total', 'cb_delivery_narrative','pdi_total','source']]
                    ])

# remove sentences with less than 30 words
all_df['n_words'] = all_df['cb_delivery_narrative'].str.split().apply(len)
all_df = all_df[all_df['n_words'] >= 30]

# set target variable
all_df['y']=0
all_df.loc[all_df['spcl5_total']>=threshold,'y'] = 1
all_df['y'].value_counts()

all_df

"""Count number of tokens and expected costs for OpenAI"""

tokens = 0
for text in all_df['cb_delivery_narrative'].tolist():
  tokens = tokens + (num_tokens_from_string("tiktoken is great!", "cl100k_base"))

print(tokens * 'replace with tolen price per used model')

"""# Classification with openai"""

# select rows with examples for few shot learning
sick_narrative = all_df[all_df['spcl5_total'] > threshold].sample(n = 1)
healthy_narrative = all_df[all_df['spcl5_total'] < threshold].sample(n = 1)

# remove few-shot examples from original database
remove_ids = [ sick_narrative['record_id'].iloc[0], healthy_narrative['record_id'].iloc[0] ]
all_df = all_df[~all_df['record_id'].isin(remove_ids)]

# keep only narrative
sick_narrative = sick_narrative['cb_delivery_narrative'].tolist()[0]
healthy_narrative = healthy_narrative['cb_delivery_narrative'].tolist()[0]

responce_labels = []

"""## Zero-shot classification"""

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


"""## Few-shot learning"""

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

"""# OpenAI: get embeddings"""

# get embeddigns of narratives
i  = 1
ada_embedding = [] # length of embeddings is 1536
for text in all_df['cb_delivery_narrative'].tolist():
  print(i)
  embddings = get_embedding(text, model='text-embedding-ada-002') # text-embedding-ada-002 (https://openai.com/blog/new-and-improved-embedding-model)
  ada_embedding.append(embddings)
  i = i + 1

# save embeddings
all_df['ada_embedding'] = ada_embedding
all_df.to_csv(os.path.join(PTH,'embedded_CBPTSD_text-embedding-ada-002.csv'), index=False)

"""# Model 3

## Help functions
"""

import numpy as np
from sklearn.decomposition import PCA
from numpy import dot
from numpy.linalg import norm

# hadamard product
def vec_dist(v1,v2):
  v1 = np.array(v1)
  v2 = np.array(v2)
  res = v1*v2
  return(res)


# normalize data
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

def norma(data):
  # Fit and transform the data using the StandardScaler
  normalized_data = scaler.fit_transform(data)

  # Convert the normalized_data back to a DataFrame (optional)
  normalized_data_df = pd.DataFrame(normalized_data, columns=data.columns)

  return(normalized_data_df)

"""## Load data"""

# load the data and set a threshold
all_df = pd.read_csv(os.path.join(PTH,'data/embedded_CBPTSD_text-embedding-ada-002.csv'))

threshold = 31

# set target variable
all_df['y']=0
all_df.loc[all_df['spcl5_total']>=threshold,'y'] = 1
all_df['y'].value_counts()


"""## Extrect numeric vectors of narratives and label classes"""

all_embeddings = all_df['ada_embedding'].apply(ast.literal_eval).apply(lambda x: [float(val) for val in x])

all_embeddings = pd.DataFrame(all_embeddings.tolist())

# normalize data
all_embeddings = norma(all_embeddings)

all_embeddings['y'] = all_df['y'].tolist()


pos_embeddings = all_embeddings[all_embeddings['y'] == 1]

neg_embeddings = all_embeddings[all_embeddings['y'] == 0]

pos_embeddings.drop('y',axis=1, inplace = True)

neg_embeddings.drop('y',axis=1, inplace = True)

"""## Data balanceing"""

# random sampling from the negative class to get the same number of records as the positive class
X_pos = pos_embeddings

X_neg =  neg_embeddings.sample(n = len(X_pos))

print(len(X_neg), len(X_pos))

# reset index
X_neg = X_neg.reset_index(drop=True)
X_pos = X_pos.reset_index(drop=True)

# remove S sentences for testing later
s = 20

pos_test_sent = X_pos.sample(s)
neg_test_sent = X_neg.sample(s)

X_pos =  X_pos.drop(pos_test_sent.index, axis=0)
X_neg =  X_neg.drop(neg_test_sent.index, axis=0)

print(len(X_neg), len(X_pos))

"""## Create pairs of narratives [pos-pos; neg-neg; and pos-neg]"""

# loop to create all <positive-positive> pairs as positive examples

pos_pos_vec = []

for i in range(0,len(X_pos) ):
  for j in range(i+1, len(X_pos) ):
    pos_pos_vec.append( vec_dist(X_pos.iloc[i,:], X_pos.iloc[j,:]) )

df_pos_pos = pd.DataFrame(pos_pos_vec)
df_pos_pos['y'] = 1

# loop to creat all <negative-negative> pairs as positive examples
neg_neg_vec = []

for i in range(0,len(X_neg) ):
  for j in range(i+1, len(X_neg) ):
    neg_neg_vec.append(  vec_dist(X_neg.iloc[i,:] , X_neg.iloc[j,:]) )

df_neg_neg = pd.DataFrame(neg_neg_vec)
df_neg_neg['y'] = 1

# loop to creat all <negative-positive> pairs as negative examples

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

"""## Train a DNN of the best model"""

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

# saving the model in tensorflow format
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

# normalize the Test data
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