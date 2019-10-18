"""
              ┏┓      ┏┓ + +
             ┏┛┻━━━━━━┛┻┓ + +
             ┃          ┃
             ┃    ━     ┃ ++ + + +
            ██████━██████ +
             ┃          ┃ +
             ┃    ┻     ┃
             ┃          ┃ + +
             ┗━┓      ┏━┛
               ┃      ┃
               ┃      ┃ + + + +
               ┃      ┃   
               ┃      ┃ + 　　　　神兽保佑,loss->0
               ┃      ┃        
               ┃      ┃  +
               ┃      ┗━━━━━┓ + +
               ┃            ┣┓
               ┃            ┏┛
               ┗━┓┓┏━━━━┳┓┏━┛ + + + +
                 ┃┫┫    ┃┫┫
                 ┗┻┛    ┗┻┛ + + + +

    author: abtion
    email: abtion@outlook.com
 
"""
import torch.nn as nn
from config import FLAGS
import torch.nn.functional as F

from model.base_models import Highway


class CLF_Model(nn.Module):
    def __init__(self):
        super(CLF_Model, self).__init__()
        # self.high = Highway(2, size=FLAGS.bert_size)
        self.trans_layer = nn.TransformerEncoderLayer(FLAGS.hidden_size, 8)
        self.trans_encoder = nn.TransformerEncoder(self.trans_layer, 1)
        self.linear1 = nn.Linear(FLAGS.bert_size, FLAGS.hidden_size)
        self.linear2 = nn.Linear(FLAGS.hidden_size, 64)
        self.linear3 = nn.Linear(64, 2)

    def forward(self, x):
        # x = x.unsqueeze(2)
        # x = self.high(x)
        x = x.squeeze()
        x = self.linear1(x)
        x = F.gelu(x)
        x = x.unsqueeze(0)
        x = self.trans_encoder(x)
        x = self.trans_encoder(x)
        x = x.squeeze()
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.linear3(x)
        return x
