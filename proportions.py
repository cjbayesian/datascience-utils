import numpy as np
import scipy as sp

def prop_diff_ci(x1,n1,x2,n2,percent_ci=0.95):
    x1, n1, x2, n2 = float(x1), float(n1), float(x2), float(n2)
    p_hat1 = x1/n1
    p_hat2 = x2/n2
    term1 = (p_hat1*(1-p_hat1))/n1
    term2 = (p_hat2*(1-p_hat2))/n2
    sigma = np.sqrt(term1 + term2)
    z = sp.stats.norm.ppf(1-(1-percent_ci)/2)
    delta_hat = p_hat1 - p_hat2
    lower = delta_hat - z*sigma
    upper = delta_hat + z*sigma
    return delta_hat, lower, upper


def prop_diff_ci_boot(x1,n1,x2,n2,percent_ci=0.95,nboot=1000000):
    s1 = sp.stats.beta.rvs(x1,n1-x1,size=nboot)
    s2 = sp.stats.beta.rvs(x2,n2-x2,size=nboot)
    delta_s = (s1 - s2)
    delta_hat = np.mean(delta_s)
    lower, upper = np.percentile(delta_s,((1-percent_ci)/2*100,(1-(1-percent_ci)/2)*100))
    return delta_hat, lower, upper