import transformers
import torch
import datasets
import json
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,BertModel,AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from bidi.algorithm import get_display
import arabic_reshaper
from pathlib import Path

class ExpoDataset(Dataset):
  def __init__(self,dataset_text):
    self.dataset=dataset_text
    self.tokenizer=BertTokenizer.from_pretrained(f'{Path(__file__).parent.parent}\\models\\MARBERT_Tokenizer')
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self,idx):
    item=self.tokenizer(str(self.dataset[idx]),
                                        add_special_tokens=True,
                                        max_length=44,
                                        padding='max_length',  
                                        truncation=True,
                                        return_tensors='pt')
    item['input_ids']=item['input_ids'][0]
    item['attention_mask']=item['attention_mask'][0]
    return item

class AttentionLayer(nn.Module):
    def __init__(self, rnn_size):
        super(AttentionLayer, self).__init__()
        self.U = nn.Linear(rnn_size, 1,bias=False) # for the learnable paramter u
        torch.nn.init.xavier_normal_(self.U.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        # α = softmax(u_t^T U)
        attention_weights = self.U(hidden_states) # (batch_size, seq_len ,1)
        attention_weights = self.softmax(attention_weights.squeeze(2))  # (batch_size, seq_len)

        # eq.11: v = sum( α * hidden_states )
        attention_output = hidden_states * attention_weights.unsqueeze(2)  # (batch_size, seq_len, rnn_size)
        attention_output = attention_output.sum(dim = 1)  # (batch_size, rnn_size)

        return attention_output, attention_weights
    

class AttentionRNN_AraBERT(nn.Module):
  def __init__(self,hidden_size1,hidden_size2,num_layers,dider=True,droprate=0.2,num_classes=3,model_type='rnn'):
    super().__init__()
    self.bert=BertModel.from_pretrained(f'{Path(__file__).parent.parent}\\models\\MARBERT_BASE')
    # freeze the bert weights
    for param in self.bert.parameters():
      param.requires_grad=False

    self.embedding_dim=self.bert.config.hidden_size
    self.rnn_hidden_size=hidden_size1
    if model_type=='gru':
      self.rnn=nn.GRU(self.embedding_dim,hidden_size1,num_layers,batch_first=True,bidirectional=dider)
    elif model_type=='lstm':
      self.rnn=nn.LSTM(self.embedding_dim,hidden_size1,num_layers,batch_first=True,bidirectional=dider)
    else:
      self.rnn=nn.RNN(self.embedding_dim,hidden_size1,num_layers,batch_first=True,bidirectional=dider)
    self.dropout=nn.Dropout(droprate)
    self.d=2 if dider else 1
    self.attention=AttentionLayer(hidden_size1*self.d)
    self.fc1=nn.Linear(hidden_size1*self.d,hidden_size2)
    self.fc2=nn.Linear(hidden_size2,num_classes)
    for name,param in self.rnn.named_parameters():
      if 'weight' in name:
        nn.init.xavier_normal_(param)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)
  def forward(self,input_ids,attention_mask):
    with torch.no_grad():
      embeddings=self.bert(input_ids=input_ids,attention_mask=attention_mask)
    x=embeddings[0]
    x=self.dropout(x)
    x,h=self.rnn(x)
    x,attention_weights=self.attention(x)
    x=self.fc1(x)
    x=F.relu(x)
    x=self.dropout(x)
    x=self.fc2(x)
    return x,attention_weights
  
class ensemble_model(nn.Module):
  def __init__(self,*models: nn.Module): #accept a variable number of models
    super().__init__()
    self.models=nn.ModuleList(models)
    self.n_classifiers=len(models)
  def forward(self,input_ids,attention_mask):
    outputs=[model(input_ids,attention_mask)[0] for model in self.models]
    average_output=sum(outputs)/self.n_classifiers
    return average_output
  
def get_dataloader(dataframe):
  dataset=ExpoDataset(dataframe)
  data_loader=DataLoader(dataset,batch_size=64,shuffle=False)
  return data_loader

def get_model(device,type=None):
    model_param=json.load(open(f'{Path(__file__).parent.parent}\\models\\models_parameters.json','r'))
    AttentionRNN_model=AttentionRNN_AraBERT(model_param['rnn']['hidden_size1'],
                                            model_param['rnn']['hidden_size2'],
                                            model_param['rnn']['num_layers'],
                                            model_param['rnn']['bidirectional'],
                                            model_param['rnn']['droprate'],
                                            model_type='rnn').to(device)
    AttentionRNN_model.load_state_dict(torch.load(f'{Path(__file__).parent.parent}\\models\\MARBERT+Attention-RNN.pth',map_location=device))
  
    AttentionGRU_model=AttentionRNN_AraBERT(model_param['gru']['hidden_size1'],
                                            model_param['gru']['hidden_size2'],
                                            model_param['gru']['num_layers'],
                                            model_param['gru']['bidirectional'],
                                            model_param['gru']['droprate'],
                                            model_type='gru').to(device)
    AttentionGRU_model.load_state_dict(torch.load(f'{Path(__file__).parent.parent}\\models\\MARBERT+Attention-GRU.pth',map_location=device))

    AttentionLSTM_model=AttentionRNN_AraBERT(model_param['lstm']['hidden_size1'],
                                             model_param['lstm']['hidden_size2'],
                                             model_param['lstm']['num_layers'],
                                             model_param['lstm']['bidirectional'],
                                             model_param['lstm']['droprate'],
                                             model_type='lstm').to(device)
    AttentionLSTM_model.load_state_dict(torch.load(f'{Path(__file__).parent.parent}\\models\\MARBERT+Attention-LSTM.pth',map_location=device))
    ensemble_classifer=ensemble_model(AttentionRNN_model,AttentionGRU_model,AttentionLSTM_model).to(device)
    if type =='lstm':
      return AttentionLSTM_model
    return ensemble_classifer
    


def infer_sentiments(model, dataloader,device,type=None):
  predictions=[]
  model.eval()
  for idx,batch in enumerate(dataloader):
    input_ids,attention_mask =batch['input_ids'].to(device), batch['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    if type =='lstm':
      output=output[0]
    predictions.extend(F.softmax(output,dim=1).argmax(dim=1).tolist())
  return predictions
