# Let's use PCA

ytrain_pca, s, V = np.linalg.svd(enki_linear.Gstar.T)
ytrain_pca = ytrain_pca[:,0:enki_linear.n_obs]
ytrain_pca = ytrain_pca.T

def emulate(Y):
    gpmodels = []

    for ii, y in enumerate(Y):
        with gp.defer_build():
            k = gp.kernels.RBF(input_dim = p, ARD = True)
            # m0 = gp.mean_functions.Linear([[1.],[1.]])
            m = gp.models.GPR(enki_linear.Ustar.T, y[:,np.newaxis], k)

            # This prior is good for data in a 2 units scale.
            m.kern.lengthscales.prior = gp.priors.Gamma(1.4942, 1/5.66074)
            
        m.compile()
        gp.train.ScipyOptimizer().minimize(m);
        gpmodels.append(m)

    return gpmodels

gpmodels = emulate(ytrain_pca)

def predict_gps(gpmodels, X):
    gpmeans = np.empty(shape = (len(gpmodels), len(X)))
    gpvars = np.empty(shape = (len(gpmodels), len(X)))

    for ii, model in enumerate(gpmodels):
        mean_pred, var_pred = model.predict_y(X)
        gpmeans[ii,:] = mean_pred.flatten()
        gpvars[ii,:] = var_pred.flatten()

    return [gpmeans, gpvars]

gplot, varplot = predict_gps(gpmodels, np.vstack([xs.flatten(), ys.flatten()]).T)
gstar, varstar = predict_gps(gpmodels, enki_linear.Ustar.mean(axis = 1).reshape(1, -1))

Sigmastar = V.T.dot(np.diag((s**2) * varstar.flatten()).dot(V))

Hplot_pca_full = np.empty(shape = Hplot.shape)
Hplot_pca_mean = np.empty(shape = Hplot.shape)
Hplot_pca_semi = np.empty(shape = Hplot.shape)

# Sigmastar = DV.T.dot(np.diag(varstar.flatten()).dot(DV))
Sigmastar = V.T.dot(np.diag((s**2) * varstar.flatten()).dot(V))

for ii in range(len(varplot.T)):
    # Sigma = DV.T.dot(np.diag(varplot[:,ii]).dot(DV))
    Sigma = V.T.dot(np.diag((s**2) * varplot[:,ii]).dot(V))
    gmean = V.T.dot(s * gplot[:,ii])
    
    Hplot_pca_mean[ii] = ((gmean - y_obs) * np.linalg.solve(2 * (Gamma), gmean - y_obs)).sum()
    Hplot_pca_full[ii] = ((gmean - y_obs) * np.linalg.solve(2 * (Gamma + Sigma), gmean - y_obs)).sum()
    Hplot_pca_semi[ii] = ((gmean - y_obs) * np.linalg.solve(2 * (Gamma + Sigmastar), gmean - y_obs)).sum()

fig, axes = plt.subplots(1,4, figsize = (16, 4));

cmap = plt.get_cmap('pink');
axes[0].contourf(xs, ys, Hs.reshape(60, 60) + \
                  1.0 * ((xs-enki_linear.mu[0])**2 + (ys-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), 
                  np.percentile(Hs + 1.0 * ((xs.flatten()-enki_linear.mu[0])**2 + \
                                      (ys.flatten()-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), range(0,101,10)), 
                  cmap = cmap);
axes[0].set_title('True contours\n');

axes[1].contourf(xs, ys, Hplot_pca_mean.reshape(60, 60) + \
                  1.0 * ((xs-enki_linear.mu[0])**2 + (ys-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), 
                  np.percentile(Hplot_pca_mean + 1.0 * ((xs.flatten()-enki_linear.mu[0])**2 + \
                                        (ys.flatten()-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), 
                                range(0,101,10)), 
                  cmap = cmap);
axes[1].set_title('Emulated mean\n(PCA)');

axes[2].contourf(xs, ys, Hplot_pca_full.reshape(60, 60) + \
                  1.0 * ((xs-enki_linear.mu[0])**2 + (ys-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), 
                  np.percentile(Hplot_pca_full + 1.0 * ((xs.flatten()-enki_linear.mu[0])**2 + \
                                        (ys.flatten()-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), 
                                range(0,101,10)), 
                  cmap = cmap);
axes[2].set_title('Emulated mean\n and variance (PCA)');

axes[3].contourf(xs, ys, Hplot_pca_semi.reshape(60, 60) + \
                  1.0 * ((xs-enki_linear.mu[0])**2 + (ys-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), 
                  np.percentile(Hplot_pca_semi + 1.0 * ((xs.flatten()-enki_linear.mu[0])**2 + \
                                        (ys.flatten()-enki_linear.mu[1])**2)/(2*(enki_linear.sigma**2)), 
                                range(0,101,10)), 
                  cmap = cmap);
axes[3].set_title('Emulated mean\n and variance* (PCA)');