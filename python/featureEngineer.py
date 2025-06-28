import numpy as np
import cvxpy as cp
from copy import deepcopy
from scipy.linalg import pinv
from scipy.optimize import minimize
# Mute all warnings
import warnings
warnings.filterwarnings("ignore")


def combine_kernels(weights, kernels):
	# length of weights should be equal to length of matrices
	n = len(weights)
	result = np.zeros_like(kernels[:, :, 0])

	for i in range(n):
		result = result + weights[i] * kernels[:, :, i]

	return result


def kernel_gip(adjmat, dim, gamma):
	y = adjmat
	if dim == 1:
		ga = np.dot(y, y.T)
	else:
		ga = np.dot(y.T, y)
	ga = gamma * ga / np.mean(np.diag(ga))
	d = np.exp(-kernel_to_distance(ga))
	return d


def kernel_to_distance(k):
	di = np.diag(k)
	d = np.tile(di, (len(k), 1)) + \
		np.tile(di.reshape(-1, 1), (1, len(k))) - 2 * k
	return d


# Compute Gaussian similarity
def gaussiansimilarity(interaction, nd, nm):
	gamad = nd/((np.linalg.norm(interaction))**2)
	C = deepcopy(interaction)
	kd = np.zeros([nd, nd])
	D = np.dot(C, C.T)
	for i in range(nd):
		for j in range(nd):
			kd[i, j] = np.exp(-gamad*(D[i, i]+D[j, j]-2*D[i, j]))
	

	gamam = nm/((np.linalg.norm(interaction))**2)
	
	km = np.zeros([nm, nm])
	E = np.dot(C.T, C)
	for i in range(nm):
		for j in range(nm):
			km[i, j] = np.exp(-gamam*(E[i, i]+E[j, j]-2*E[i, j]))

	return kd, km


def KNN(network, k):
	"""
	Construct the K Nearest Neighbors (KNN) network.

	Parameters:
	network : similarity matrix (2D numpy array)
	k       : number of nearest neighbors (int)
	
	Returns:
	knn_network : KNN network (2D numpy array)
	"""
	rows, cols = network.shape
	np.fill_diagonal(network, 0)
	knn_network = np.zeros((rows, cols))

	for i in range(rows):
		sort_network, idx = np.sort(network[i, :])[::-1], np.argsort(network[i, :])[::-1]
		knn_network[i, idx[:k]] = sort_network[:k]

	return knn_network


# Label propagation with local network similarity (LPLNS)
def calculate_labels(W, Y, alpha):
	return (1 - alpha) * pinv(np.eye(W.shape[0]) - alpha * W) @ Y


def optimization_similarity_matrix(
	feature_matrix, nearest_neighbor_matrix, tag, regulation
):
	row_num = feature_matrix.shape[0]
	# W = np.zeros((1, row_num))
	W = np.zeros_like(feature_matrix)
	if tag == 1:
		row_num = 1
	for i in range(row_num):
		nearest_neighbors = feature_matrix[
			np.array(nearest_neighbor_matrix[i, :]).astype(bool), :
		]
		neighbors_num = nearest_neighbors.shape[0]
		G1 = np.tile(feature_matrix[i, :], (neighbors_num, 1)) - nearest_neighbors
		G2 = np.tile(feature_matrix[i, :], (neighbors_num, 1)).T - nearest_neighbors.T
		if regulation == "regulation2":
			G_i = G1 @ G2 + np.eye(neighbors_num)
		if regulation == "regulation1":
			G_i = G1 @ G2
		H = 2 * G_i

		# Define the quadratic objective function
		def objective(w):
			return 0.5 * w.T @ H @ w

		# Linear equality constraint: sum(w) = 1
		def constraint_eq(w):
			return np.sum(w) - 1

		# Bounds for the weights: 0 <= w_i <= inf
		bounds = [(0, None) for _ in range(neighbors_num)]

		# Initial guess for the weights
		w0 = np.ones(neighbors_num) / neighbors_num

		# Constraints dictionary
		constraints = {"type": "eq", "fun": constraint_eq}

		# Use minimize from scipy.optimize to solve the quadratic program
		result = minimize(
			objective,
			w0,
			method="trust-constr",
			bounds=bounds,
			constraints=constraints,
			options={"disp": False},
		)

		# Extract the optimized weights
		w = result.x
		W[i, np.array(nearest_neighbor_matrix[i, :]).astype(bool)] = w
	return W


def Label_Propagation(feature_matrix, tag, neighbor_num, regulation):
	# Using the method of label propagation to predict the interaction
	distance_matrix = calculate_instances(feature_matrix)  # Calculate distance matrix
	nearst_neighbor_matrix = calculate_neighbors(
		distance_matrix, neighbor_num
	)  # Calculate nearest neighbors
	W = optimization_similarity_matrix(
		feature_matrix, nearst_neighbor_matrix, tag, regulation
	)  # Optimize similarity matrix
	return W


def calculate_instances(feature_matrix):
	# calculate the distance between each feature vector of lncRNAs or proteins.
	# row_num, col_num = feature_matrix.shape
	# distance_matrix = np.zeros((row_num, row_num))
	# for i in range(row_num):
	#     for j in range(i + 1, row_num):
	#         distance_matrix[i, j] = np.sqrt(np.sum((feature_matrix[i, :] - feature_matrix[j, :])**2))  # Euclidean distance
	#         distance_matrix[j, i] = distance_matrix[i, j]
	#     distance_matrix[i, i] = col_num
	# return distance_matrix

	## Calculate Euclidean distance matrix by row vectorization
	return np.sum((feature_matrix[:, np.newaxis] - feature_matrix) ** 2, axis=-1)


def calculate_neighbors(distance_matrix, neighbor_num):
	# calculate the nearest K neighbors
	sort_indices = np.argsort(distance_matrix, axis=1)
	row_num, col_num = distance_matrix.shape
	nearst_neighbor_matrix = np.zeros((row_num, col_num))
	for i in range(row_num):
		nearst_neighbor_matrix[i, sort_indices[i, :neighbor_num]] = distance_matrix[
			i, sort_indices[i, :neighbor_num]
		]
	return nearst_neighbor_matrix


def integrateSimilarity(FS, FSP, SS, SSP, interaction, featureEngineer=None):

	if featureEngineer == 'LNS':
		neighbor_num = 10
		nd, nm = interaction.shape
		kd, km = gaussiansimilarity(interaction, nd, nm)
		SS = np.where(SS > 0, SS, kd)
		FS = np.where(FS > 0, FS, km)
		SS_new = Label_Propagation(SS, 0, neighbor_num, "regulation2")
		FS_new = Label_Propagation(FS, 0, neighbor_num, "regulation2")
		return interaction, FS_new, SS_new 
	else:
		nd, nm = interaction.shape
		kd, km = gaussiansimilarity(interaction, nd, nm) # calculate gaussiansimilarity
		# Add kd to SS with weight SSP
		sm = FS * FSP + km * (1 - FSP)
		sd = SS * SSP + kd * (1 - SSP)
		return interaction, sm, sd


if __name__ == "__main__":
	np.random.seed(0)
	interaction = np.random.randint(0, 2, (30, 50))
	sd = np.random.uniform(0, 1, (30, 30))
	sm = np.random.uniform(0, 1, (50, 50))
	# sd = np.where(sd > 0.5, sd, 0)
	FSP = np.ones_like(sm)
	SSP = np.ones_like(sd)

	print(integrateSimilarity(sm, FSP, sd, SSP, interaction, featureEngineer='vae'))


