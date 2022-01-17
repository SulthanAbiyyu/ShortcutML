from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor

# Classification model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

# Misc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BaselineModel:

    def __init__(self, type="regression"):
        self.type = type

    def regression_models(self):
        return [
            ("LinearRegression", LinearRegression()),
            ("RidgeRegression", Ridge()),
            ("LassoRegression", Lasso()),
            ("XGBRegressor", XGBRegressor()),
            ("LGBMRegressor", LGBMRegressor()),
            ("SupportVectorRegressor", SVR()),
            ("RandomForestRegressor", RandomForestRegressor()),
            ("GradientBoostingRegressor", GradientBoostingRegressor()),
            ("AdaBoostRegressor", AdaBoostRegressor()),
            ("BaggingRegressor", BaggingRegressor()),
            ("ExtraTreesRegressor", ExtraTreesRegressor()),
            ("VotingRegressor", VotingRegressor(estimators=[("LinearRegression", LinearRegression()),
                                                            ("Ridge", Ridge()),
                                                            ("Lasso", Lasso()),
                                                            ("XGBRegressor",
                                                             XGBRegressor()),
                                                            ("LGBMRegressor",
                                                             LGBMRegressor()),
                                                            ("SVR", SVR()),
                                                            ("RandomForestRegressor",
                                                             RandomForestRegressor()),
                                                            ("GradientBoostingRegressor",
                                                             GradientBoostingRegressor()),
                                                            ("AdaBoostRegressor",
                                                             AdaBoostRegressor()),
                                                            ("BaggingRegressor",
                                                             BaggingRegressor()),
                                                            ("ExtraTreesRegressor",
                                                             ExtraTreesRegressor())
                                                            ]
                                                )
             )
        ]

    def classification_models(self):
        return [
            ("LogisticRegression", LogisticRegression()),
            ("SGDClassifier", SGDClassifier()),
            ("SupportVectorClassifier", SVC()),
            ("RandomForestClassifier", RandomForestClassifier()),
            ("GradientBoostingClassifier", GradientBoostingClassifier()),
            ("AdaBoostClassifier", AdaBoostClassifier()),
            ("BaggingClassifier", BaggingClassifier()),
            ("ExtraTreesClassifier", ExtraTreesClassifier()),
            ("XGBClassifier", XGBClassifier(verbosity=0, silent=True)),
            ("LGBMClassifier", LGBMClassifier()),
            ("KNeighborsClassifier", KNeighborsClassifier()),
            ("VotingClassifier", VotingClassifier(estimators=[("LogisticRegression", LogisticRegression()),
                                                              ("SGDClassifier",
                                                               SGDClassifier()),
                                                              ("SVC", SVC()),
                                                              ("RandomForestClassifier",
                                                               RandomForestClassifier()),
                                                              ("GradientBoostingClassifier",
                                                               GradientBoostingClassifier()),
                                                              ("AdaBoostClassifier",
                                                               AdaBoostClassifier()),
                                                              ("BaggingClassifier",
                                                               BaggingClassifier()),
                                                              ("ExtraTreesClassifier",
                                                               ExtraTreesClassifier()),
                                                              ("XGBClassifier",
                                                               XGBClassifier(verbosity=0, silent=True)),
                                                              ("LGBMClassifier",
                                                               LGBMClassifier()),
                                                              ("KNeighborsClassifier",
                                                               KNeighborsClassifier())
                                                              ]
                                                  )
             )
        ]

    def get_models(self):
        if self.type == "regression":
            return self.regression_models()
        elif self.type == "classification":
            return self.classification_models()
        else:
            raise ValueError("Type must be regression or classification")

    def score(self, y_true, y_pred):
        if self.type == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.type == "classification":
            return f1_score(y_true, y_pred, average="macro")
        else:
            raise ValueError("Type must be regression or classification")

    def evaluate(self, X_train, X_test, y_train, y_test):
        self.test_result = []
        for model in self.get_models():
            model[1].fit(X_train, y_train)
            y_pred = model[1].predict(X_test)
            if self.type == "regression":
                self.test_result.append([model[0], self.score(y_test, y_pred), model[1]])

            elif self.type == "classification":
                self.test_result.append([model[0], self.score(y_test, y_pred), model[1]])

        if self.type == "regression":
            self.test_result = pd.DataFrame(
                self.test_result, columns=["model", "rmse_score", "model class"]).sort_values(by="rmse_score")

        elif self.type == "classification":
            self.test_result = pd.DataFrame(
                self.test_result, columns=["model", "f1_score", "model_class"]).sort_values(by="f1_score", ascending=False)
        
        self.best_model = self.test_result.iloc[0, 2]

    def plot_baseline(self):
        if self.type == "regression":
            plt.title("Regression Baseline Model")
            # plt.xticks(rotation=90)
            sns.barplot(y="model", x="rmse_score", data=self.test_result)
            plt.xlabel("RMSE Score")
            plt.ylabel("Models")
            plt.show()

        elif self.type == "classification":
            plt.title("Classification Baseline Model")
            # plt.xticks(rotation=90)
            sns.barplot(y="model", x="f1_score", data=self.test_result)
            plt.xlabel("F1 Score")
            plt.ylabel("Models")
            plt.show()
