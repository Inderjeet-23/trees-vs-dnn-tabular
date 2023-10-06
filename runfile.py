import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm

def make_dataset(file, k=-1):
	data = pd.read_csv(file)
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].values

	X = StandardScaler().fit_transform(X)
	y = LabelEncoder().fit_transform(y)

	if k != -1:
		X = SelectKBest(f_classif, k=k).fit_transform(X, y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	X_train = torch.FloatTensor(X_train)
	X_test = torch.FloatTensor(X_test)
	y_train = torch.FloatTensor(y_train)
	y_test = torch.FloatTensor(y_test)

	return X_train, X_test, y_train, y_test


class TabularModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def model_train(model, X_train, y_train, loss_fn, optimizer, n_epochs=250):
    acc = []
    model.train()
    
    # for epoch in range(n_epochs):
    for _ in tqdm(range(n_epochs), desc="Training model for k = "+str(X_train.shape[1])):
        y_pred = model(X_train).squeeze()
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc.append((y_pred.round() == y_train).float().mean())
		
        
    return acc

def plot_all_accuracy(acc, save_path=None):
    for key, value in acc.items():
        plt.plot(range(len(value)), value, label=key)
    plt.legend()
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if save_path:
        plt.savefig(save_path)
    # plt.show()

k_val = [8, 6, 4, 2]
plt_dict = {}

for k in k_val:
	X_train, X_test, y_train, y_test = make_dataset("clf_cat/electricity.csv", k)

	# Define the model, loss function, and optimizer
	input_size = X_train.shape[1]
	hidden_size = 64
	num_classes = 1

	model = TabularModel(input_size, hidden_size, num_classes)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	plt_dict['MLP_'+str(k)] = model_train(model, X_train, y_train, criterion, optimizer, n_epochs=500)

	eval_set = [(X_test, y_test)]
	eval_metric = ["auc"]

	model = XGBClassifier(objective='binary:logistic', n_estimators=500, learning_rate=0.01, eval_metric=eval_metric)
	model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
	plt_dict['XGBoost_'+str(k)] = model.evals_result()['validation_0']['auc']

plot_all_accuracy(plt_dict, save_path="img/electricity.png")