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


def encode_features(data):
    features = data.copy()
    
    # Encode Central Air where Y is 1, and N is 0
    features['CentralAir'] = features['CentralAir'] == 'Y'

    # Encode basement condition using one-hot-encoding
    bsmt_cond_map = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
    features['BsmtQuality'] = features['BsmtCond'].map(bsmt_cond_map).fillna(0) # Some houses have no basement

    # Encode electrical
    features = pd.concat([features, encode_electrical(features['Electrical'])], axis=1)
    
    # Combine bed and bath    
    features['BedBath'] = features['FullBath'] * features['BedroomAbvGr']
    
    # Scale numeric values
    scaler = MinMaxScaler()
    scaled_columns = [
        'FullBath',
        'BedroomAbvGr', 
        'BedBath',
        'GrLivArea',
        'CentralAir', 
        'FuseA',
        'FuseF',
        'FuseP', 
        'Mix',
        'SBrkr', 
        'BsmtQuality'
    ]
    scaled_features = features[scaled_columns]
    return pd.DataFrame(scaler.fit_transform(scaled_features), columns = scaled_columns)

def encode_label(data):
    labels = data.copy()['SalePrice']
    scaler = MinMaxScaler()
    scaler.fit(labels)    
    labels = pd.DataFrame(scaler.transform(labels), columns = ['SalePrice'])
    return (labels['SalePrice'], scaler)

def decode_label(data, scaler):
    return scaler.inverse_transform(data)