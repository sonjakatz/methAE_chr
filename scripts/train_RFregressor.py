import numpy as np
import os
import pandas as pd
import pickle
import argparse

from sklearn.ensemble import RandomForestRegressor

def loadData(file):
    with open(file, "rb") as f: data = pickle.load(f) #
    X = data["beta"]
    y = data["pheno"]["Age"]

    ## remove NAs 
    df = pd.DataFrame(y)
    X = X[df.notna()["Age"]]
    y = df[df.notna()["Age"]].ravel()
    return X, y

def trainRegr(X, y, outFile):
    print("\nTraining model...")
    regr = RandomForestRegressor()
    regr.fit(X, y)
    # save model
    os.makedirs(f"logs/{outFile}", exist_ok=True)
    pickle.dump(regr, open(f'logs/{outFile}/RFregressorModel.sav', 'wb'))
    print(f"Saved model... \t logs/{outFile}")
    return regr

def score(model, X, y):
    r2 = model.score(X, y)
    print(r2)


def main():
    PATH_data = "data/"
    PATH_train = os.path.join(PATH_data, args.train_file)
    PATH_test = os.path.join(PATH_data, args.test_file)
    
    X_train, y_train = loadData(PATH_train)
    X_test, y_test = loadData(PATH_test)

    model = trainRegr(X_train, y_train, outFile=args.name)
    score(model, X_test, y_test)
    
if __name__ == "__main__":
    ## add argparse here
    parser = argparse.ArgumentParser(description='Train RandomForest Regressor')
    parser.add_argument('--name', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    args = parser.parse_args()
    main()