from sklearn.linear_model import LogisticRegression

def train_linear_model(X, y, **params):
	model = LogisticRegression(**params)
	model.fit(X, y)

	return model # fitted model