


# Classify narratives using LIWC and Transformers-sentence embeddings

# The PCL-5 is a 20-item self-report measure that assesses the 20 DSM-5 symptoms of PTSD.


import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from numpy import loadtxt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

"""## Global variables"""

PTH = "..."

threshold = 31

# set random seed

seed_value = 567

import random
random.seed(seed_value)

np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

"""## Help functions"""

from sklearn.preprocessing import StandardScaler

def vec_dist(v1,v2):
  v1 = np.array(v1)
  v2 = np.array(v2)
  res = v1*v2  # --> Hadmar
  return(res)


def predictions_siam(v,pos_v,neg_v):
    pos_embd_df = pd.DataFrame(vec_dist(v, pos_v))
    y_pos_pred = nn_model.predict(pos_embd_df.transpose().astype('float32'))[0][0]

    neg_embd_df = pd.DataFrame(vec_dist(v,neg_v))
    y_neg_pred = nn_model.predict(neg_embd_df.transpose().astype('float32'))[0][0]


    if y_pos_pred > y_neg_pred:
      return(1)
    return(0)


def predictions_siam_v2(v,pos_v,neg_v, x_train_org):

    pos_embd_df = pd.DataFrame(vec_dist(v, pos_v))
    norm = pd.concat([pos_embd_df.transpose(), x_train_org],axis = 0)
    norm = norma(norm)
    pos_embd_df = norm.iloc[0,:]
    pos_embd_df = pd.DataFrame(pos_embd_df)
    y_pos_pred = nn_model.predict(pos_embd_df.transpose().astype('float32'))[0][0]


    neg_embd_df = pd.DataFrame(vec_dist(v,neg_v))
    norm = pd.concat([neg_embd_df.transpose(), x_train_org],axis = 0)
    norm = norma(norm)
    neg_embd_df = norm.iloc[0,:]
    neg_embd_df = pd.DataFrame(neg_embd_df)
    y_neg_pred = nn_model.predict(neg_embd_df.transpose().astype('float32'))[0][0]

    if y_pos_pred >  y_neg_pred:
      return([1,y_pos_pred])
    return([0,y_neg_pred])


# normalize by columns
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def standardization(df):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  return scaler.fit_transform(df)



def norma(data):

  # Create a StandardScaler object
  scaler = StandardScaler()

  # Fit and transform the data using the StandardScaler
  normalized_data = scaler.fit_transform(data)

  # Convert the normalized_data back to a DataFrame (optional)
  normalized_data_df = pd.DataFrame(normalized_data, columns=data.columns)

  return(normalized_data_df)

"""## Load data"""

# use with local server
url = os.path.join(PTH,"data.csv")
df = pd.read_csv(url)

# keep only narratives with more than 30 words
x = [len(x) for x in df['cb_delivery_narrative'].str.split()]
df['wc'] = x
df = df[df['wc']>=30]


"""## Get embeddings

### Transformers embeddings
"""

!huggingface-cli login

token = ''

embedding_model_path = [
                        'emilyalsentzer/Bio_ClinicalBERT',
                        'mental/mental-bert-base-uncased',
                        'microsoft/biogpt',
                        'AIMH/mental-xlnet-base-cased',
                        'mental/mental-roberta-base',
                      ]

model_id = 0

tokenizer = AutoTokenizer.from_pretrained(embedding_model_path[model_id],use_auth_token= token)
model = AutoModel.from_pretrained(embedding_model_path[model_id], use_auth_token=True)


def get_embeddings(text):
  # Step 1: Tokenize the text
  tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
  # Step 2: Get model embeddings
  with torch.no_grad():
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state  # Use last_hidden_state for embeddings
  return(embeddings)

embeddings = []

i = 1

for text in df['cb_delivery_narrative'].tolist():
  print(i)
  emb = get_embeddings(text)
  emb = emb.mean(dim=1)
  embeddings.append(emb)
  i = i+1

# convert list of tensors to list of numpy arrays
list_of_numpy_arrays = [tensor.numpy() for tensor in embeddings]

# Convert to numpy array and remove singleton dimensions
numpy_arrays = np.squeeze(np.array(list_of_numpy_arrays))
result = pd.DataFrame(numpy_arrays)

# create a dataframe of the embeddings

result.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

x = pd.concat([df, result], axis=1)

x.to_csv(os.path.join(PTH,'data',embedding_model_path[model_id].split("/")[-1]+'.csv'), index=False)

print("saved to: ", embedding_model_path[model_id].split("/")[-1]+'.csv' )

"""### Sentence embeddings (original data)"""

sbert_model = SentenceTransformer('all-mpnet-base-v2') # paraphrase-mpnet-base-v2

# Sentences are encoded by calling model.encode()
embeddings = sbert_model.encode(df['cb_delivery_narrative'].tolist(),show_progress_bar=True)

# create a dataframe of the embeddings
result = pd.DataFrame(embeddings)

result.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

x = pd.concat([df, result], axis=1)

x.to_csv(os.path.join(PTH,'data/mpnet_SentenceEmbeddings_PCL_COVID_LIWC.csv'), index=False)

"""## Data preperation"""

# read sentense embeddings

model_id = 0
print(embedding_model_path[model_id])

X_df = pd.read_csv(os.path.join(PTH, 'data', embedding_model_path[model_id].split("/")[-1]+'.csv')) # select embeddings

y = (X_df['spcl5_total']>=threshold).astype("int") # threshold

X = X_df.copy()
X = X.iloc[:,range(99,99+769)] # take only embeddings (start at 99); for LIWC start at 6

X['y'] = y.tolist()

if 'before' in X.columns:
  X.drop('before',axis=1, inplace = True)

# normalize data exluding the y variable

X_pos = X[X['y']==1].copy()
X_pos.drop('y', axis = 1, inplace = True)

X_neg = X[X['y']==0].copy()
X_neg.drop('y', axis = 1, inplace = True)

# random sampling from the negative class to get the same number of records as the positive class
X_neg = X_neg.sample(n = len(X_pos))

print(len(X_neg), len(X_pos))

# remove some sentences for testing later
s = 20

pos_test_sent = X_pos.sample(s)
neg_test_sent = X_neg.sample(s)

X_pos =  X_pos.drop(pos_test_sent.index, axis=0)
X_neg =  X_neg.drop(neg_test_sent.index, axis=0)

# loop to create all positive-positive pairs as positive examples

pos_pos_vec = []

for i in range(0,len(X_pos) ):
  for j in range(i+1, len(X_pos) ):
    pos_pos_vec.append( vec_dist(X_pos.iloc[i,:], X_pos.iloc[j,:]) )

df_pos_pos = pd.DataFrame(pos_pos_vec)
df_pos_pos['y'] = 1

# loop to creat all negative-negative pairs as positive examples
neg_neg_vec = []

for i in range(0,len(X_neg) ):
  for j in range(i+1, len(X_neg) ):
    neg_neg_vec.append(  vec_dist(X_neg.iloc[i,:] , X_neg.iloc[j,:]) )

df_neg_neg = pd.DataFrame(neg_neg_vec)
df_neg_neg['y'] = 1

# loop to creat all negative-positive pairs as negative examples

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

from tensorflow import keras
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

"""### test with the Test set"""

y_true = [1]*s + [0]*s

rpt = classification_report(y_true, y_pred, output_dict=True)
rpt = pd.DataFrame(rpt).transpose()

if not('report' in locals() or 'report' in globals()):
  report = pd.DataFrame()
  print("in")

cm = confusion_matrix(y_true,y_pred)
sensitivity1 = cm[1,1]/(cm[1,1]+cm[1,0])
specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])

rpt['sensitivity'] = sensitivity1
rpt['specificity'] = specificity1
rpt['auc'] = roc_auc_score(y_true, y_pred)
rpt['run'] = datetime.now().strftime("%H:%M:%S")

report = pd.concat([report, rpt], axis = 0)



print('Confusion Matrix [Actual (rows) X Predicted(columns)]: \n', cm)
print('Sensitivity : ', sensitivity1 )
print('Specificity : ', specificity1)
print("AUC:",roc_auc_score(y_true, y_pred))

print(embedding_model_path[model_id])
print(classification_report(y_true, y_pred))

averaged_df = report.loc['1',['precision',	'recall',	'f1-score',	'support',	'sensitivity',	'specificity',	'auc']].mean()
averaged_df

"""# Fine-tuning"""

model_id = 0
print(embedding_model_path[model_id])

# read sentense embeddings

X_df = pd.read_csv(os.path.join(PTH, 'data', embedding_model_path[model_id].split("/")[-1]+'.csv')) # select embeddings

y = (X_df['spcl5_total']>=threshold).astype("int") # threshold

X = X_df.copy()
X = X.iloc[:,range(99,99+769)] # take only embeddings (start at 99); for LIWC start at 6

if 'before' in X.columns:
  X.drop('before',axis=1, inplace = True)

tokenizer = AutoTokenizer.from_pretrained(embedding_model_path[model_id],use_auth_token = token)
model = AutoModel.from_pretrained(embedding_model_path[model_id], use_auth_token=True)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix

# Load and preprocess your dataset
# X contains your text data and y contains corresponding labels

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) # binary

# Prepare data loaders for training and testing
train_data = [(text, label) for text, label in zip(X_train, y_train)]
test_data = [(text, label) for text, label in zip(X_test, y_test)]

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()


num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in train_data_loader:
        texts, labels = batch
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(labels)

        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    average_loss = total_loss / total_samples

    # Validation and testing
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        predicted_labels_all = []
        true_labels_all = []
        for batch in test_data_loader:
            texts, labels = batch
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            labels = torch.tensor(labels)

            outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += len(labels)

            predicted_labels_all.extend(predicted_labels.cpu().numpy())
            true_labels_all.extend(labels.cpu().numpy())

        accuracy = correct_predictions / total_samples
        f1 = f1_score(true_labels_all, predicted_labels_all)
        precision = precision_score(true_labels_all, predicted_labels_all)
        recall = recall_score(true_labels_all, predicted_labels_all)
        auc = roc_auc_score(true_labels_all, predicted_labels_all)

        tn, fp, fn, tp = confusion_matrix(true_labels_all, predicted_labels_all).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {average_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"AUC: {auc:.4f}")