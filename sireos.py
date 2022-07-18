import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

datasets =  ['WDBC_withoutdupl_norm_v08']
for dataset in datasets:
	#Reading dataset
	data = np.genfromtxt('data/' + dataset)
	n_samples, n_features = np.shape(data)

	#Pairwise distance to set the parameter of the heat kernel
	D = pairwise_distances(data)
	t = np.quantile(D[np.nonzero(D)], 0.01)

	for i in range(1,11):
		#Reading solutions
		X = np.genfromtxt('solutions/' + dataset + '/' + str(i))
		X = X[:,None]
		#Normalization of scorings
		X = X/X.sum()

		#Computing the index
		score = 0
		for j in range(0, n_samples):
			score += np.mean(np.exp(-np.linalg.norm(data[None, j, :] - np.delete(data[None, :, :],j, axis=1), axis=-1)**2/(2*t*t))) * X[j]

		print(score)