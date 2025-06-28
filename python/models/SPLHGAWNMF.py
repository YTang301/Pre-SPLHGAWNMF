from .ConstructHW import *
import math


class Hyper_Model(object):

    def __init__(self, name='SPLDHyperAWNTF'):
        super().__init__()
        self.name = name

    def SPLDHyperAWNTF(self, X, m_embeding, d_embeding, W, r=4, alpha=2, beta=2, \
                       lam_t = 0.001, lam_c = 0.3, max_iter=2000, tol=1e-5):

        nd, nm = X.shape

        np.random.seed(0)
        M = np.random.rand(nd, r)
        D = np.random.rand(nm, r)
        C = np.random.rand(nd, nm)
        Xbar = np.ones_like(X) - X

        '''Hypergraph Learning'''
        Dv_m, S_m = constructHW(m_embeding)
        Dv_d, S_d = constructHW(d_embeding)

        k = 1.2
        # k = 16
        k_end = 0.008
        gama = 1.2

        while k > k_end:
            for _ in range(max_iter):
                '''Updating the factor matrix M'''
                numerator = np.multiply(np.sqrt(W), X + np.multiply(Xbar, C)) @ D + alpha * S_d @ M
                denominator = np.multiply(np.sqrt(W), M @ D.T) @ D + alpha * Dv_d @ M + lam_t * (M)
                M = np.multiply(M, numerator / (denominator + 1e-8))


                '''Updating the factor matrix D'''
                numerator = np.multiply(np.sqrt(W), X + np.multiply(Xbar, C)).T @ M + beta * S_m @ D
                denominator = np.multiply(np.sqrt(W.T), D @ M.T) @ M + beta * Dv_m @ D + lam_t * (D)
                D = np.multiply(D, numerator / (denominator + 1e-8))


                '''Updating the adaptive weight tensor C'''
                Xstar = X - M @ D.T
                C = -1.0 * np.multiply(np.multiply(W, Xstar), Xbar) / (np.multiply(W, np.square(Xbar)) + lam_c)
                C = np.maximum(C, 0)

                output_X = M @ D.T
                err = np.linalg.norm(X - output_X) / np.linalg.norm(X)

                if err < tol:
                    break

            '''reconstructed tensor'''
            l = X - M @ D.T + np.multiply(Xbar, C)

            ### To run HGAWNMF, comment the following lines
            for i in range(nd):
                for j in range(nm):
                    if l[i, j] <= 1 / (k + 1 / gama)**2:
                        W[i, j] = 1
                    elif l[i, j] >= 1 / k**2:
                        W[i, j] = 0
                    else:
                        W[i, j] = gama * (1 / np.sqrt(l[i, j]) - k)

            k = k / 1.2

        predict_X = M @ D.T

        return predict_X

    def __call__(self):

        return getattr(self, self.name, None)
    

def vanillaSPLHyperModel(interaction, sd, sm):
    """"Interface to call for SPLDHyperAWNTF model"""
    train_tensor = interaction.copy().astype(float)
    train_tensor = imputationEM(interaction, sm)
    nd, nm = interaction.shape

    rd = np.zeros([nd, 1])
    rm = np.zeros([nm, 1])

    for i in range(nd):
        rd[i] = math.pow(np.linalg.norm(interaction[i, :]), 2)
    gamad = nd / rd.sum()


    for j in range(nm):
        rm[j] = math.pow(np.linalg.norm(interaction[:, j]), 2)
    gamam = nm / rm.sum()


    DGSM = np.zeros([nd, nd])
    for m in range(nd):
        for n in range(nd):
            DGSM[m, n] = np.exp(
                -gamad * math.pow(np.linalg.norm(interaction[m, :] - interaction[n, :]), 2))


    MGSM = np.zeros([nm, nm])
    for r in range(nm):
        for t in range(nm):
            MGSM[r, t] = np.exp(
                -gamam * math.pow(np.linalg.norm(interaction[:, r] - interaction[:, t]), 2))


    ID = np.zeros([nd, nd])

    for h1 in range(nd):
        for h2 in range(nd):
            if sd[h1, h2] == 0:
                ID[h1, h2] = DGSM[h1, h2]
            else:
                ID[h1, h2] = sd[h1, h2]


    IM = np.zeros([nm, nm])

    for q1 in range(nm):
        for q2 in range(nm):
            if sm[q1, q2] == 0:
                IM[q1, q2] = MGSM[q1, q2]
            else:
                IM[q1, q2] = sm[q1, q2]


    concat_v = np.hstack([interaction.T, IM])
    concat_d = np.hstack([interaction, ID])

    W = np.ones((nd, nm))

    predicted = Hyper_Model().SPLDHyperAWNTF(train_tensor, concat_v, concat_d, W)
    return predicted

def imputationEM(interaction, sm):
    """Impute the missing entries using EM algorithm"""
    # nd, n = interaction.shape
    # interaction = interaction.copy().astype(float)
    def kernel(sm, gamma=1):
        return np.exp(-gamma*np.sum((sm[:, np.newaxis] - sm) ** 2, axis=-1))

    # interaction = np.ones_like(interaction).astype(float)
    k = kernel(sm)
    for i in range(interaction.shape[0]):
        row = interaction[i, ]
        missing = np.where(row == 0)[0]
        observed = np.where(row != 0)[0]
        missingPre = k[missing][:, observed] @ np.linalg.pinv(k[observed][:, observed]) @ row[observed]
        interaction[i, missing] = missingPre
    
    # print(f"Interaction = {interaction}")

    return interaction 

