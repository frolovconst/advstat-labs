import numpy as np
from sklearn import linear_model
import sklearn.model_selection as mdsl
from sklearn.model_selection import cross_validate

def bstrapped_param(p_func, sample, B):
    n = sample.size
    strapped_samples = np.random.choice(sample, (B,n))
#     print(np.apply_along_axis(p_func, 1, strapped_samples).shape)
#     print(np.apply_along_axis(p_func, 1, strapped_samples))
    strapped_params = np.apply_along_axis(p_func, 1, strapped_samples)
    strp_params_mean = strapped_params.mean()
    strp_params_std = strapped_params.std()
    print('Evaluated parameter is {:.3f}\nIts MSE is {:.3f}'.format(strp_params_mean, strp_params_std))
    
def create_B_sample_indices(dataset, B):
    m = dataset.shape[0]
    n = dataset.shape[1]
    result_indices = np.random.choice(m, (B,m))
    return result_indices


def train_B_regressions(dataset, B):
    indices_2d = create_B_sample_indices(dataset,B)
    m = dataset.shape[0]
    n = dataset.shape[1]-6+1
    result = np.empty((B,5,n))
#     sigmas  = np.empty(B)
    regr = linear_model.LinearRegression(normalize=True)
    i=0
    for line in indices_2d:
#         print(line)
#         print(dataset[line][:,1:-5].shape)
        regr.fit(dataset[line][:,1:-5], dataset[line][:,-5:])
        coefs = regr.coef_
#         print(coefs.shape)
        result[i] = np.insert(coefs,0, regr.intercept_, axis=1)
#         sigmas[i] = np.std(regr.predict(dataset[line][:,1:-5])[:,1]-dataset[line][:,-4])
#         result = np.append(result, coefs)
        i += 1
#     return result, sigmas
    return result


def analyze_coefs(coefs, B):
    for value in range(coefs.shape[1]):
        print('Target {}:'.format(value))
        for feature_num in range(coefs.shape[2]):
            print('\tFeature {:d}:\n\t\t95%-Confidence interval: {:.3f} +/- {:.3f}\n\t\tRange: {:.3f} - {:.3f}\n\t\t% of outliers: {:.3f}\n\t\tNumber of outliers: {}'.format(
                feature_num+1, 
                coefs[:,value,feature_num].mean(),
                coefs[:,value,feature_num].std(),
                coefs[:,value,feature_num].min(), 
                coefs[:,value,feature_num].max(), 
                coefs[:,value,feature_num][(coefs[:,value,feature_num]>coefs[:,value,feature_num].mean()+3*coefs[:,value,feature_num].std()) | (coefs[:,value,feature_num]<coefs[:,value,feature_num].mean()-3*coefs[:,value,feature_num].std())].size  /B ,coefs[:,value,feature_num][(coefs[:,value,feature_num]>coefs[:,value,feature_num].mean()+3*coefs[:,value,feature_num].std()) | (coefs[:,value,feature_num]<coefs[:,value,feature_num].mean()-3*coefs[:,value,feature_num].std())].size               ))