import pandas as pd
import numpy as np

def load_datasets():
    train_data = pd.read_csv('./data/raw/train.csv')
    return train_data

def handle_missing_values(df):
    # Select relevant features (square footage, bedrooms, bathrooms) and target (SalePrice)
    features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
    target = df['SalePrice']
    
    # Fill missing values with the mean of the respective columns
    features = features.fillna(features.mean())
    target = target.fillna(target.mean())
    
    # Combine the features and target into a single dataframe
    df = pd.concat([features, target], axis=1)

    return df.dropna()


def main():
    train_data = load_datasets()
    train_data = handle_missing_values(train_data)
    train_data.to_csv('./data/processed/processed_train_data.csv', index=False)

if __name__ == "__main__":
    main()