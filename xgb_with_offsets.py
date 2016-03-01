#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from ml_metrics import quadratic_weighted_kappa


SEED = 837

SKIP_CV = False
CV_N_FOLDS = 3

# I tried a lot of methods to calculate optimal values for the offsets
# (i.e. fmin() functions, random searches), but in the end the best values
# were obtained by manual fine-tuning.
OFFSETS = [0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1]


def to_categories(n):

    if n > 7.99:
        n = 7.99
    if n < 0.0:
        n = 0.0

    n = n + OFFSETS[int(n)]

    n = round(n)
    if n < 1:
        n = 1
    if n > 8:
        n = 8

    return int(n)


def cross_validate(model, df):
    
    scores = list()

    print("# Applying %d-fold cross-validation" % CV_N_FOLDS)
    kf = KFold(len(df), n_folds=CV_N_FOLDS)
    for index, (train, test) in enumerate(kf):

        ts_fold = time.time()

        df_train = df.ix[train]
        df_test = df.ix[test]

        model.fit(
            df_train.drop(["Response", "Id"], axis=1).values,
            df_train["Response"].values)

        score = quadratic_weighted_kappa(
            df_test["Response"].values,
            list(map(round, model.predict(df_test.drop(["Response", "Id"], axis=1).values)))
        )
        scores.append(score)

        print("Confusion matrix")
        print(confusion_matrix(
            df_test["Response"].values,
            list(map(round, model.predict(df_test.drop(["Response", "Id"], axis=1).values)))
        ))

        print("Fold %d score %0.5f (%d secs)" % (index + 1, score, (time.time() - ts_fold)))

        feature_importances = dict()
        tot = 0
        for i, v in enumerate(model._Booster.get_fscore()):
            n = int(v.strip('f'))
            tot = tot + n 
            feature_importances[df_test.drop(["Response", "Id"], axis=1).columns[i]] = n

        print("Top 10 features by importance:")
        for feature in sorted(feature_importances, key=feature_importances.get, reverse=True)[:10]:
            if feature_importances[feature] >= 1.0 / len(df_test.drop(["Response", "Id"], axis=1).columns):
                 print(" * %s %0.3f %%" % (feature, feature_importances[feature] / tot))

        
    print("Avg score: %0.5f (+/- %0.5f)" % (np.average(scores), np.std(scores)))


def train_model(df, dump):

    ts_main = time.time()

    model = XGBRegressor(
        seed=SEED,
        colsample_bytree=0.40,
        colsample_bylevel=1,
        subsample=0.90,
        silent=True,
        reg_lambda=1.0,
        reg_alpha=1.0,
        n_estimators=2100,
        min_child_weight=1,
        max_depth=10,
        objective="reg:linear",
        learning_rate=0.01)

    print("## Training Gradient Boosting Regressor with parameters: ")
    print(model.get_params())

    if not SKIP_CV:
        cross_validate(model, df)
        print("# Fitting the model with the complete dataset")

    model.fit(
        df.drop(["Response", "Id"], axis=1).values,
        df["Response"].values)

    print("Dumping model at %s" % dump)
    joblib.dump(model, dump)

    print("Total time %d secs" % (time.time() - ts_main))
    return model


def load_or_train(df, dump):
    try:
        model = joblib.load(dump)
        print("Model loaded from %s" % dump)
    except Exception:
        model = train_model(df, dump)
    finally:
        return model


def encode_features(df):
    df["Product_Info_1"] = df["Product_Info_1"].astype('category')
    df["Product_Info_1"].cat.set_categories([1, 2], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Product_Info_1'], prefix='Product_Info_1')], axis=1)
    df.drop("Product_Info_1", axis=1, inplace=True)

    df["Product_Info_2"] = df["Product_Info_2"].astype('category')
    df["Product_Info_2"].cat.set_categories([u'A1', u'A2', u'A3', u'A4', u'A5', u'A6', u'A7', u'A8', u'B1', u'B2', u'C1', u'C2', u'C3', u'C4', u'D1', u'D2', u'D3', u'D4', u'E1'], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Product_Info_2'], prefix='Product_Info_2')], axis=1)
    df.drop("Product_Info_2", axis=1, inplace=True)

    df["Product_Info_5"] = df["Product_Info_5"].astype('category')
    df["Product_Info_5"].cat.set_categories([2, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Product_Info_5'], prefix='Product_Info_5')], axis=1)
    df.drop("Product_Info_5", axis=1, inplace=True)

    df["Product_Info_6"] = df["Product_Info_6"].astype('category')
    df["Product_Info_6"].cat.set_categories([1, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Product_Info_6'], prefix='Product_Info_6')], axis=1)
    df.drop("Product_Info_6", axis=1, inplace=True)

    df["Product_Info_7"] = df["Product_Info_7"].astype('category')
    df["Product_Info_7"].cat.set_categories([1, 2, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Product_Info_7'], prefix='Product_Info_7')], axis=1)
    df.drop("Product_Info_7", axis=1, inplace=True)

    df["Employment_Info_3"] = df["Employment_Info_3"].astype('category')
    df["Employment_Info_3"].cat.set_categories([1, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Employment_Info_3'], prefix='Employment_Info_3')], axis=1)
    df.drop("Employment_Info_3", axis=1, inplace=True)

    df["Employment_Info_5"] = df["Employment_Info_5"].astype('category')
    df["Employment_Info_5"].cat.set_categories([2, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Employment_Info_5'], prefix='Employment_Info_5')], axis=1)
    df.drop("Employment_Info_5", axis=1, inplace=True)

    df["InsuredInfo_1"] = df["InsuredInfo_1"].astype('category')
    df["InsuredInfo_1"].cat.set_categories([1, 2, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['InsuredInfo_1'], prefix='InsuredInfo_1')], axis=1)
    df.drop("InsuredInfo_1", axis=1, inplace=True)

    df["InsuredInfo_2"] = df["InsuredInfo_2"].astype('category')
    df["InsuredInfo_2"].cat.set_categories([2, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['InsuredInfo_2'], prefix='InsuredInfo_2')], axis=1)
    df.drop("InsuredInfo_2", axis=1, inplace=True)

    df["InsuredInfo_4"] = df["InsuredInfo_4"].astype('category')
    df["InsuredInfo_4"].cat.set_categories([2, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['InsuredInfo_4'], prefix='InsuredInfo_4')], axis=1)
    df.drop("InsuredInfo_4", axis=1, inplace=True)

    df["InsuredInfo_5"] = df["InsuredInfo_5"].astype('category')
    df["InsuredInfo_5"].cat.set_categories([1, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['InsuredInfo_5'], prefix='InsuredInfo_5')], axis=1)
    df.drop("InsuredInfo_5", axis=1, inplace=True)

    df["InsuredInfo_6"] = df["InsuredInfo_6"].astype('category')
    df["InsuredInfo_6"].cat.set_categories([1, 2], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['InsuredInfo_6'], prefix='InsuredInfo_6')], axis=1)
    df.drop("InsuredInfo_6", axis=1, inplace=True)

    df["InsuredInfo_7"] = df["InsuredInfo_7"].astype('category')
    df["InsuredInfo_7"].cat.set_categories([1, 3], inplace=True)
    df = pd.concat([df, pd.get_dummies(df['InsuredInfo_7'], prefix='InsuredInfo_7')], axis=1)
    df.drop("InsuredInfo_7", axis=1, inplace=True)

    return df


def fillnan(df):

    df["Nulls"] = df.isnull().sum(axis=1)

    df.loc[df.Employment_Info_1.isnull(), "Employment_Info_1"] = -1
    df.loc[df.Employment_Info_4.isnull(), "Employment_Info_4"] = -1
    df.loc[df.Employment_Info_6.isnull(), "Employment_Info_6"] = -1
    df.loc[df.Insurance_History_5.isnull(), "Insurance_History_5"] = -1
    df.loc[df.Family_Hist_2.isnull(), "Family_Hist_2"] = -1
    df.loc[df.Family_Hist_3.isnull(), "Family_Hist_3"] = -1
    df.loc[df.Family_Hist_4.isnull(), "Family_Hist_4"] = -1
    df.loc[df.Family_Hist_5.isnull(), "Family_Hist_5"] = -1
    df.loc[df.Medical_History_1.isnull(), "Medical_History_1"] = -1
    df.loc[df.Medical_History_10.isnull(), "Medical_History_10"] = -1
    df.loc[df.Medical_History_15.isnull(), "Medical_History_15"] = -1
    df.loc[df.Medical_History_24.isnull(), "Medical_History_24"] = -1
    df.loc[df.Medical_History_32.isnull(), "Medical_History_32"] = -1

    df["BMI_Ins_Age"] = df.BMI * df.Ins_Age
    df["BMI_Product_Info_4"] = df.BMI * df.Product_Info_4
    df["BMI_Medical_History_23"] = df.BMI / df.Medical_History_23
    df["Medical_History_15_Exp"] = df.Medical_History_15.map(lambda x: x*x)
    df["Old"] = df.Ins_Age.map(lambda x: 1 if x >= 0.60 else 0)
    
    return df


def predict(model, df):
    
    return list(map(
        to_categories, model.predict(df.values)))


def main():
    df = pd.read_csv("./data/raw/train.csv", header=0, sep=",", encoding="utf-8")
    df = encode_features(df)
    df = fillnan(df)

    model = load_or_train(df, './data/models/xgb.pkl')
    predictions = predict(model, df.drop(["Response", "Id"], axis=1))

    kappa = quadratic_weighted_kappa(
        df.Response.values,
        predictions
    )

    print("Kappa: %0.5f" % kappa)

    cm = confusion_matrix(
        df.Response.values,
        predictions
    )

    print("Confusion matrix: \n%s" % cm)

    df_test = pd.read_csv("./data/raw/test.csv", header=0, sep=",", encoding="utf-8")
    df_test= encode_features(df_test)
    df_test = fillnan(df_test)
    predictions = predict(model, df_test.drop(["Id"], axis=1))

    df_test["Response"] = pd.Series(predictions)
    df_test.to_csv('./data/predictions/xgb.csv', columns=["Id", "Response"], index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
