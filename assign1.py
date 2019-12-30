

import argparse
import pandas as pd
import numpy as np

class Reduce_Dim():
    def __init__(self, Tolerance, Max_Iterations, dim, alpha):
        '''
        @topic: Declare the initial parameters.
        @parameters:
            1. Tolerance: tolerance used as your stopping condition;
            2. Max_Iterations: maximum iterations that the program may run;
            3. dim: first "dim" eigenvectors to project datapoints in the "dim"-dimensional subspace.
            4. alpha: a rate between [0,1] that fraction of total variance is not less than this rate.
        '''
        self.Tolerance = Tolerance
        self.Max_Iterations = Max_Iterations
        self.dim = dim
        self.alpha = alpha

    def pass_parameters(self):
        '''
        @topic: Pass the arguments on terminal.
        '''
        parser = argparse.ArgumentParser(description='Arguments')
        parser.add_argument('dataset', help="Select a dataset(txt file) as input.")
        parser.add_argument('stdout_label', \
                            help="Input question#(e.g.'a') to get corresponding output; \
                            Input 'All' to get all outputs.", nargs='?', default='All')
        args = parser.parse_args()
        dataset_name = args.dataset # dataset_name = 'magic04.data'
        stdout_label = args.stdout_label
        if args.stdout_label != 'All':
            print("To input specific question number can get corresponding output.")
            print("Parameter 'All' is default value (Or leave it empty).")
        return dataset_name, stdout_label

    def inputData(self, dataset_name):
        '''
        @topic: Input and preprocess the dataset.
        @parameters:
            dataset_name: the name of dataset.
        '''
        df = pd.read_csv(dataset_name, header=None)
        train_data = df.values
        train_X = train_data[:, 0:-1]
        #train_Y = train_data[:, -1: ]
        return train_X.astype(float) #, train_Y

    def zScore(self, X):
        '''
        @topic: Apply z-normalization to input dataset.
        @parameters:
            X: dataset.
        @reference: textbook p.g.52
        '''
        return (X - np.mean(X, axis=0))/np.std(X, axis=0)

    def coVar(self, X):
        '''
        @topic: Compute the covariance of input dataset.
        @parameters:
            X: dataset.
        @formulas: Z = X - mu'; cov = (Z'*Z)/n
        '''
        Z = X - np.mean(X, axis=0)
        cov = np.dot(Z.T, Z)/len(X)
        return cov

    def powIter_DomEig(self, A):
        '''
        @topic: Implement power iteration method to compute the dominant eigenvector.
        @parameters:
            A: dataset.
        @reference: textbook p.g.106 ALGORITHM4.1.
        '''
        k = 0
        P = np.ones((A.shape[0],1))
        error = np.ones_like(P)
        while k < self.Max_Iterations and error.any() > self.Tolerance :
            k += 1
            i = 0
            P_new = np.dot(A.T, P) # eigenvector estimate
            i = P_new.tolist().index(np.max(P_new)) # maximum value index
            eig_val = P_new[i]/P[i] # eigenvalue estimate
            P_new /= P_new[i] # scale vector
            error = abs(P_new - P)
            P = P_new
        # Normalize the dominant eigenvector to be a unit vector by using frobenius norm.
        eig_vec = P_new/np.linalg.norm(P_new)
        return eig_val, eig_vec

    def totalVar(self, X):
        '''
        @topic: Compute the total variance
        @parameters:
            X: dataset.
        @reference: textbook p.g.193 and p.g.196 -- Total Projected Variance
        @comment: Way1: compute the coordinate(Nx2) and total variance.
        '''
        cov = self.coVar(X) # (10x10)
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_val_idx = np.argsort(eig_val)[::-1] # Sort the eigenvalues as ascending order and return its index as descending order.
        eig_vec = eig_vec[:,eig_val_idx] # Return the new eigenvectors as descending order.
        U_dim = eig_vec[:,0:int(self.dim)] # size = Num_features x Dim
        A = np.dot(U_dim.T, X.T) # Coordinate: (2x10)dot(10xN) = (2xN) 
        totalVar = sum(sum(np.dot(A, A.T)))/len(X) # Total Var = ∑|a-0|^2/N
        return totalVar

    def totalVar2(self, X):
        '''
        @topic: Compute the total variance
        @parameters:
            X: dataset.
        @reference: textbook p.g.184 Example 7.1. and p.g.196
        @comment: Way2: compute projected data and the trace of its covariance.
        '''
        cov = self.coVar(X) # (10x10)
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_val_idx = np.argsort(eig_val)[::-1] # Sort the eigenvalues as ascending order and return its index as descending order.
        eig_vec = eig_vec[:,eig_val_idx] # Return the new eigenvectors as descending order.
        U_dim = eig_vec[:,0:int(self.dim)] # size = Num_features x Dim
        A = np.dot(U_dim.T, X.T) # Coordinate: (2x10)dot(10xN) = (2xN)
        X_prj = np.dot(U_dim, A) # Projected datapoints: (10x2)dot(2xN) = (10xN)
        cov_prj = self.coVar(X_prj.T) # X_prj.T (Nx10)
        eig_val_prj, eig_vec_prj = np.linalg.eig(cov_prj)
        var_prj = sum(eig_val_prj[0: int(self.dim)]) # Trace of projected datapoints in first "dim" eigenvalues.
        return var_prj

    def eig_dcmp(self, X):
        '''
        @topic: Eigen-decomposition and return the diagonalized eigenvalue and the eigenvectors.
        @parameters:
            X: dataset.
        @reference: textbook p.g.200
        '''
        cov = self.coVar(X)
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_val_idx = np.argsort(eig_val)[::-1] # Sort the eigenvalues as ascending order and return its index as descending order.
        eig_vec = eig_vec[:,eig_val_idx] # Return the new eigenvectors as descending order.
        eig_val_dig = np.dot(np.dot(eig_vec.T, cov), eig_vec)
        return eig_val_dig, eig_vec

    def pca(self, X):
        '''
        @topic: Implement PCA algorithm with SVD to find the principle vectors Ur and its co-ordinate Ar.
        @parameters:
            X: dataset.
        @reference: textbook p.g.198 ALGORITHM 7.1 and p.g.209 7.4 SINGULAR VALUE DECOMPOSITION
        '''
        r = 0.
        fr = 0.
        cov = self.coVar(X) # compute covariance matrix
        u, s, vh = np.linalg.svd(cov)
        eig_val, eig_vec = s, vh.T
        while fr < self.alpha:
            r += 1 # choose dimensionality
            fr = sum(eig_val[0:int(r)])/sum(eig_val) # fraction of total variance
        trace = sum(eig_val[0:int(r)])
        Ur = eig_vec[:, 0:int(r)] # reduced basis (10x(r-1))
        Ar = np.dot(Ur.T, X.T) # reduced dimensionality data ((r-1)x10)dot(10xN)
        return Ar.T, trace, Ur, int(r) # here Ar.T (Nx10), Ur (10x10)

    def pca2(self, X):
        '''
        @topic: Implement PCA algorithm to find the principle vectors Ur and its co-ordinate Ar.
        @parameters:
            X: dataset.
        @reference: textbook p.g.198 ALGORITHM 7.1
        '''
        r = 0.
        fr = 0.
        cov = self.coVar(X) # compute covariance matrix
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_val_idx = np.argsort(eig_val) # Sort the eigenvalues as ascending order and return its index.
        eig_val_idx = eig_val_idx[::-1] # Return the index of eigenvalues as descending order.
        eig_val = eig_val[eig_val_idx] # Return the new eigenvalues as descending order.
        eig_vec = eig_vec[:,eig_val_idx] # Return the new eigenvectors as descending order.
        while fr < self.alpha:
            r += 1 # choose dimensionality
            fr = sum(eig_val[0:int(r)])/sum(eig_val) # fraction of total variance
        trace = sum(eig_val[0:int(r)])
        Ur = eig_vec[:, 0:int(r)] # reduced basis (10x(r-1))
        Ar = np.dot(Ur.T, X.T) # reduced dimensionality data ((r-1)x10)dot(10xN)
        return Ar.T, trace, Ur, int(r) # here Ar.T (Nx10), Ur (10x10)


if __name__ == '__main__':
    # Parameters initialization
    Tolerance = 1e-6
    Max_Iterations = 100
    dim = 2
    alpha=0.95

    # Class instantiation
    rd = Reduce_Dim(Tolerance, Max_Iterations, dim, alpha)

    # Pass Parameters
    #dataset_name, stdout_label = rd.pass_parameters()
    dataset_name = 'magic04.data'
    stdout_label = 'All'    

    # Input dataset
    train_X = rd.inputData(dataset_name)

    # a. Apply z-normalization (Textbook P.g.52) to input dataset
    train_X = rd.zScore(train_X)
    if stdout_label == 'a' or stdout_label == 'All':
        print("#"*50)
        print("a. Apply z-normalization to dataset.")
        print("train_X: ", train_X)

    # b. Compute the sample covariance matrix and verify with np.cov
    cov = rd.coVar(train_X)
    cov_np = np.cov(train_X,  bias=True, rowvar=False)
    if stdout_label == 'b' or stdout_label == 'All':    
        print("#"*50)
        print("b. Compute the sample covariance matrix and verify with 'np.cov'.")
        print("cov: ",cov, "\n", "cov_np: ", cov_np)
        error0 = abs(cov - cov_np)
        if error0.all() < 1e-6:
            print("Both results are equal.")
        else:
            print("Both results are different.")

    # c. Compute dominant eigenvalue and eigenvector and verify with np.linalg.eig
    eig_val_pi, eig_vec_pi = rd.powIter_DomEig(cov)
    eig_val_np, eig_vec_np = np.linalg.eig(cov)
    if stdout_label == 'c' or stdout_label == 'All':
        print("#"*50)
        print("c. Compute dominant eigenvalue and eigenvector and verify with 'np.linalg.eig'.")
        print("The dominant eigenvalue and eigenvector computed by power-iteration method.")
        print("eig_val_pi: ", eig_val_pi, "\n", "eig_vec_pi: ", eig_vec_pi)
        print("The dominant eigenvalue and eigenvector computed with 'np.linalg.eig'.")
        print("eig_val_np: ", eig_val_np[0:1], "\n", "eig_vec_np: ", eig_vec_np[:,0:1])
        error1 = abs(eig_val_pi - eig_val_np[0:1])
        error2 = abs(abs(eig_vec_pi) - abs(eig_vec_np[:,0:1]))
        if error1 < 1e-6 and all(error2[er] < 1e-6 for er in range(len(error2))):
            print("Both results are equal.")
        else:
            print("Both results are different.")

    # d. Compute the projected datapoints and its variance
    totalVar = rd.totalVar(train_X)
    #totalVar2 = self.totalVar2(train_X)
    if stdout_label == 'd' or stdout_label == 'All':
        print("#"*50)
        print("d. Compute the projected datapoints and its variance")
        print("Total Projected Variance: ", totalVar)
        #print("Total Projected Variance2: ", totalVar2)

    # e. Print the covariance matrix Σ in its eigen-decomposition form ∑=U*Λ*U.T
    eig_val_dig, eig_vec = rd.eig_dcmp(train_X)
    if stdout_label == 'e' or stdout_label == 'All':
        print("#"*50)
        print("e. Print the covariance matrix Σ in its eigen-decomposition form (∑=U*Λ*U.T).")
        print("Covariance matrix ∑: \n", cov)
        print("Eigenvectors U: \n", eig_vec)
        print("Diagonalized eigenvalues Λ: \n", eig_val_dig)

    # f. PCA Algorithm
    # f-1. Implement PCA Algorithm to find the principle vectors Ur that preserve 95% of variance.
    # f-2. Print the co-ordinate Ar of the first 10 data points by using the above set of vectors as the new basis vector.
    pick = 10 # As required, pick first 10 samples.
    Ar, trace, Ur, dimension = rd.pca(train_X) # A_r represents the coordinates of X in the new basis.
    if stdout_label == 'f' or stdout_label == 'All':
        print("#"*50)
        print("f. PCA Algorithm")
        print("{0}% of variance can be preserved in {1} dimension.".format(100*alpha, int(dimension)))
        print("The principle vectors Ur: \n {0}".format(Ur))
        print("The coordinate Ar of the first 10 data points: \n", Ar[0: int(pick)])

    # g. Compute the covariance of the projected data points and compare with the sum of eigenvalues corresponding to principal vectors.
    Xr = np.dot(Ur, Ar.T).T
    cov_Xr = rd.coVar(Xr)
    eig_val_Xr, eig_vec_Xr = np.linalg.eig(cov_Xr)
    trace_CovXr = sum(eig_val_Xr[0: int(dimension)])
    if stdout_label == 'g' or stdout_label == 'All':
        print("#"*50)
        print("g. Compute the covariance of the projected data points and compare with the trace corresponding to principal vectors.")
        print("The covariance of the projected data points COV_Xr: \n", cov_Xr)
        print("The trace of COV_Xr: ", trace_CovXr)
        print("The trace corresponding to principal vectors: ", trace)
        if abs(trace_CovXr - trace) < 1e-6:
            print("Both are matched!")
        else:
            print("Both are not matched!")

    '''
    # Save output to txtfile.
    title = "assign1_junz"
    with open("%s.txt"%title,"w") as f:
        f.write( )
    '''
