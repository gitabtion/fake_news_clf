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
import torch
import numpy as np
from bert_serving.client import BertClient
from torch.utils.data import Dataset
import emoji
import pandas as pd
from tqdm import tqdm
from config import FLAGS
from model.clf_model import CLF_Model
import torch.nn.functional as F


class NewsDataset(Dataset):
    def __init__(self, file):
        data = pd.read_csv(file)
        self.texts = [emoji.demojize(s) for s in list(data.iloc[:, 1])]
        self.labels = list(data.iloc[:, 2])
        self.num = len(self.labels)
        # print(np.sum(self.labels))
        # print(self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def get_loader(file):
    dataset = NewsDataset(file)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              FLAGS.batch_size,
                                              shuffle=True,
                                              num_workers=4, )
    return data_loader


def train(bc, model, dataset, optimizer):
    model = model.to(FLAGS.device)
    model.train()
    losses = []
    f1s = []
    summaries = []
    loss_func = torch.nn.CrossEntropyLoss()
    for i, (text, label) in enumerate(dataset):
        text = torch.tensor(bc.encode(list(text))).to(FLAGS.device)
        label = label.to(FLAGS.device).long()
        preds = model(text)
        loss = loss_func(preds, label)
        acc = 1 - torch.sum(torch.abs(torch.argmax(preds, dim=1) - label)).float() / label.shape[0]
        recall = torch.sum((torch.argmax(preds, dim=1) * 2 == label).long()).float() / (
                label.shape[0] - torch.sum(label))
        f1 = 2 * acc * recall / (acc + recall)
        summary = torch.sum(torch.argmax(preds, dim=1)).float() / label.shape[0]
        print(
            f'{i}/{len(dataset)}, '
            f'train loss: {loss.item():.4f}, '
            f'acc: {acc.item():.4f}, '
            f'f1: {f1.item():.4f}, '
            f'summary: {summary.cpu().item():.4f}/{label.shape[0]}')
        losses.append(loss.cpu().item())
        f1s.append(f1.cpu().item())
        summaries.append(summary.cpu().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(
        f'\ntrain mean loss: {np.mean(np.array(losses)):.4f}, '
        f'mean f1: {np.mean(np.array(f1s)):.4f}, '
        f'summary: {np.mean(summaries)}')
    model_test(bc, model)
    # eval(bc, model)
    torch.save(model, FLAGS.checkpiont_file)


def model_test(bc, model=None, dataset=None):
    if model is None:
        model = torch.load(FLAGS.checkpiont_file, map_location=FLAGS.device)
    if dataset is None:
        dataset = get_loader(FLAGS.test_file)
    model = model.to(FLAGS.device)
    model.eval()
    losses = []
    f1s = []
    for text, label in dataset:
        text = torch.tensor(bc.encode(list(text))).to(FLAGS.device)
        label = label.to(FLAGS.device).long()
        preds = model(text)
        loss = F.cross_entropy(preds, label)
        acc = torch.sum((torch.argmax(preds, dim=1) == label).long()).float() / label.shape[0]
        recall = torch.sum((torch.argmax(preds, dim=1) * 2 == label).long()).float() / (
                label.shape[0] - torch.sum(label))
        print(f'acc:{acc}, recall:{recall}')
        f1 = 2 * acc * recall / (acc + recall)
        # print(f'test loss: {loss.item():.4f} acc: {acc.item():.4f} f1: {f1.item():.4f}')
        losses.append(loss.item())
        f1s.append(f1.item())
    print(f'\ntest mean loss: {np.mean(np.array(losses)):.4f}, mean f1: {np.mean(np.array(f1s)):.4f}')


def train_entry():
    if FLAGS.model_from_file:
        model = torch.load(FLAGS.checkpiont_file)
    else:
        model = CLF_Model()
    bc = BertClient(check_length=False)
    optimizer = torch.optim.Adamax(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    train_data = get_loader(FLAGS.train_file)
    for epoch in tqdm(range(FLAGS.epochs)):
        train(bc, model, train_data, optimizer)


def eval(bc, model=None):
    data = pd.read_csv(FLAGS.eval_file)
    if model is None:
        model = torch.load(FLAGS.checkpiont_file)
    model = model.to(FLAGS.device)
    model.eval()
    texts = [emoji.demojize(_data) for _data in list(data.iloc[:, 1])]
    encoded_texts = torch.tensor(bc.encode(texts)).to(FLAGS.device)
    preds = model(encoded_texts)
    labels = torch.argmax(preds, dim=1).cpu().data.numpy()
    print(f'\neval: {np.sum(labels)}')
    data['label'] = pd.Series(labels)
    data = data.drop(['comment'], axis=1)
    data.to_csv(FLAGS.submit_file, index=False)


def main():
    # torch.cuda.set_device(3)
    train_entry()
    bc = BertClient()
    # model_test(bc)
    eval(bc)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    main()
