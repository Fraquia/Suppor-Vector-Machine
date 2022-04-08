class customSVMlight(object):
    
    # init funtion
    def __init__(self, x, y, C, gamma, indexes, alpha):
        self.x = x; self.y = y
        self.C = C; self.gamma = gamma
        self.I = indexes
        self.alpha = alpha
    
    # funtion to get all the parameters
    def get_par(self):
        return self.x, self.y, self.C, self.gamma, self.alpha
    
    # set function
    def set_par(self, x, y, C, gamma, alpha):
        self.x = x; self.y = y
        self.C = C; self.gamma = gamma
        self.alpha = alpha
        
    # linear kernel
    def lin_ker(self, x):
        return(np.power(x.dot(x.T) + 1, self.gamma))
    
    # calculating the alpha by minimizing the lagrangian for the soft SVM
    def opt_alpha(self):
        x, y, C, gamma, alpha = self.get_par()
        I = self.I; m = len(I)
        not_I = np.setdiff1d(np.indices(alpha.shape), I)
        y_I = y[I]; Q = np.outer(y, y) * self.lin_ker(x)
        val_b = -np.sum(alpha[not_I] * y[not_I])
        start_time = time()
        
        # costruction of the function to minimize
        P = matrix(Q[I, :][:, I])
        q = matrix(np.full(m, -1) + (Q[I, :][:, not_I]).dot(alpha[not_I]))
        
        # contruction of the matrix for the disequation part          
        G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))        
        h = matrix(np.hstack((np.zeros(m), np.full(m, C))))
        
        # contruction of the matrix for the equation part
        A = matrix(y_I.reshape(1, -1))
        b = matrix(np.array([val_b]))
        
        # updating with the final value of alpha
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        self.set_par(x, y, C, gamma, np.array(solution['x']).reshape(m))
        return solution['primal objective'], solution['iterations'], time() - start_time