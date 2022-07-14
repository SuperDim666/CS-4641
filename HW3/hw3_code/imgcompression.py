import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images ((N,D) arrays) as well as color images ((N,D,3) arrays)
        In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
        each image (1 channel for black and white and 3 channels for RGB)
        Image is the matrix X.

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images

        Return:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V^T: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
        """
        if len(X.shape) == 3:
            U_chan_0, S_chan_0, V_chan_0 = np.linalg.svd(X[:, :, 0])
            U_chan_1, S_chan_1, V_chan_1 = np.linalg.svd(X[:, :, 1])
            U_chan_2, S_chan_2, V_chan_2 = np.linalg.svd(X[:, :, 2])
            
            U = np.array([U_chan_0,U_chan_1,U_chan_2]).transpose(1,2,0)
            minND = min(X.shape[0], X.shape[1])
            S = np.concatenate((S_chan_0.reshape(minND,1), S_chan_1.reshape(minND,1), S_chan_2.reshape(minND,1)), axis=1)
            V = np.array([V_chan_0, V_chan_1, V_chan_2]).transpose(1, 2, 0)
        else:
            U, S, V = np.linalg.svd(X)
        return U, S, V
        #raise NotImplementedError


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.

        Args:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            Xrebuild: (N,D) numpy array of reconstructed image / (N,D,3) numpy array for color images

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if len(U.shape) == 3:
            U_chan_0, S_chan_0, V_chan_0 = U[:, :, 0], S[:, 0], V[:, :, 0]
            U_chan_1, S_chan_1, V_chan_1 = U[:, :, 1], S[:, 1], V[:, :, 1]
            U_chan_2, S_chan_2, V_chan_2 = U[:, :, 2], S[:, 2], V[:, :, 2]
            
            X_chan_0 = np.matmul(U_chan_0[:, : k], np.matmul(np.diag(S_chan_0[: k]), V_chan_0[: k]))
            X_chan_1 = np.matmul(U_chan_1[:, : k], np.matmul(np.diag(S_chan_1[: k]), V_chan_1[: k]))
            X_chan_2 = np.matmul(U_chan_2[:, : k], np.matmul(np.diag(S_chan_2[: k]), V_chan_2[: k]))
            
            Xrebuild = np.array([X_chan_0, X_chan_1, X_chan_2]).transpose(1, 2, 0)
        else:
            Xrebuild = np.matmul(U[:, : k], np.matmul(np.diag(S[: k]), V[: k]))
        return Xrebuild
        #raise NotImplementedError
        

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        N = X.shape[0]
        D = X.shape[1]
        compressed_size = k * (N + D + 1)
        original_size = N * D
        return compressed_size / original_size
        #raise NotImplementedError


    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (min(N,D), ) numpy array black and white images / (min(N,D),3) numpy array for color images
           k: int, rank of approximation

        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        sigma_1 = sum(S[: k] ** 2)
        sigma_i = sum(S ** 2)
        return sigma_1 / sigma_i
        #raise NotImplementedError
