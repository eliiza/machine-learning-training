import pandas as pd
import numpy as np
def evaluate_model(model_fn):
    '''
    Consumes a function model_fn
    and evaluates its predictive accuracy against 
    the housing prices test set.
    '''
    test_data = pd.read_csv("housing_price_data/test_data.csv")
    actual_values = test_data['SalePrice']
    test_input = test_data.filter(regex='^(?!SalePrice$).*') #Pass in all columns except SalePrice
    predicted_saleprice = model_fn(test_input)
    mae = np.mean(np.abs(predicted_saleprice-actual_values))
    print("The model is inaccurate by $%.2f on average." % mae)
    return mae