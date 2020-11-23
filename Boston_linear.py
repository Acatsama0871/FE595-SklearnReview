# Boston_linear.py
# Fit a linear model to the Boston Housing data and display the predictor importance

# modules
import numpy as np
from tabulate import tabulate
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def main():
    # load dataset
    data = load_boston()
    dataX = data.data
    dataY = data.target
    feature_names = data.feature_names

    # fit the linear model
    model_linear = LinearRegression()
    model_linear.fit(dataX, dataY)

    # output fit result
    pred =model_linear.predict(dataX)
    print('-'*80)
    print('Fit results:')
    print('Coefficients:', model_linear.coef_, '\n')
    print('The training MSE:', mean_squared_error(dataY, pred))
    print('The coefficients of determination:', r2_score(dataY, pred))
    print('-'*80, '\n\n')

    # importance of predictors
    coef_abs = np.abs(model_linear.coef_)
    temp = sorted(coef_abs, reverse=True)
    ordered_index = [np.where(coef_abs == i)[0][0] for i in temp]
    ordered_features = [feature_names[i] for i in ordered_index]
    ordered_coef_abs = [round(coef_abs[i], 4) for i in ordered_index]
    ordered_coef = [round(model_linear.coef_[i], 4) for i in ordered_index]

    # output
    print('Feature Importance: from greatest to least','\n')
    print(tabulate(np.array([ordered_features, ordered_coef_abs, ordered_coef]).T,
                   headers=['Feature Name', 'Abs(Coefficients)', 'Coefficients']))



if __name__ == '__main__':
    main()
