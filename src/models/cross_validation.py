import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook

def split_training_set(X, y, **params):
	return train_test_split(X, y, **params)

def split_data(X, y, itr, ite):
	if isinstance(X, pd.DataFrame):
		Xtr = X.iloc[itr]
		Xte = X.iloc[ite]

		ytr = y.iloc[itr]
		yte = y.iloc[ite]

	else:

		Xtr = X[itr]
		Xte = X[ite]

		ytr = y[itr]
		yte = y[ite]

	return Xtr, Xte, ytr, yte


def cross_validate(X, y, model, SEED):
	cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
	ll = [] # log loss values
	
	for itr, ite in tqdm_notebook(cv.split(X, y)):
		Xtr, Xte, ytr, yte = split_data(X, y, itr, ite)
		
		print('Train model')
		model.fit(Xtr, ytr)
		fold_preds = model.predict_proba(Xte)
		fold_ll    = log_loss(yte, fold_preds)
		print('Log loss: {}'.format(fold_ll))
		
		ll.append(fold_ll)
		print('='*75)
		
	return ll