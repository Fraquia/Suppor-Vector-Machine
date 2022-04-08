def R(alpha,c):
  Lp = np.where((alpha <= 1e-06)&(y==1))[0]                 #first condition
  Um = np.where((alpha >= C -(1e-06))&(y==-1))[0]           #second condition
  inside = np.where((alpha > 1e-06)&(alpha < C-(1e-06)))[0] #third condition
  r = np.concatenate((Lp,Um,inside)).astype(int)
  r = np.sort(r)
  return r

def S(alpha,c):
  Lp = np.where((alpha <= 1e-06)&(y==-1))[0]                #first condition
  Um = np.where((alpha >= C -(1e-06))&(y==1))[0]            #second condition
  inside = np.where((alpha > 1e-06)&(alpha < C-(1e-06)))[0] #third condition 
  s = np.concatenate((Lp,Um,inside)).astype(int)
  s = np.sort(r)
  return s

class customSVM(object):
    
    # init funtion
    def __init__(self, x, y, C, gamma):
        self.x = x; self.y = y
        self.C = C; self.gamma = gamma
        self.alpha = np.zeros(y.shape[0])
    
    # funtion to get all the parameters
    def get_par(self):
        return self.x, self.y, self.C, self.gamma, self.alpha
    
    # set function
    def set_par(self, x, y, C, gamma, alpha):
        self.x = x; self.y = y
        self.C = C; self.gamma = gamma
        self.alpha = alpha
        
    # linear kernel
    def lin_ker(self):
        return(np.power(self.x.dot(self.x.T) + 1, self.gamma))
    
    # calculating the alpha by minimizing the lagrangian for the soft SVM

    def opt_alpha(self):

      x, y, C, gamma, alpha = self.get_par()
      n_samples, n_features = x.shape
      start_time = time()
      
      # Compute the Gram matrix

      K = np.zeros((n_samples, n_samples))

      '''
      for i in range(n_samples):
        for j in range(n_samples):
          K[i,j] = self.lin_ker(x[i], x[j])
      '''

      # costruction of the function to minimize

      P = matrix(np.outer(y,y) * self.lin_ker())
      q = matrix(np.ones(n_samples) * -1)

      A = matrix(y, (1,n_samples))
      b = matrix(0.0)

      G = matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
      h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))

      # solve QP problem
      solvers.options['show_progress'] = False
      solution = solvers.qp(P, q, G, h, A, b)

      # updating with the final value of alpha
      self.set_par(x, y, C, gamma, np.array(solution['x']))
      #print(self.lin_ker()[0, 0])
      return solution['primal objective'], solution['iterations'], time() - start_time