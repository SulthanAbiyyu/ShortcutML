from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import r_regression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class UnivariateFeatureSelection:

    def __init__(self, X, y, type="regression"):
        self.X = X
        self.y = y
        self.type = type

    # classification

    def chi2__fit(self):
        K = SelectKBest(chi2, k="all")
        K.fit(self.X, self.y)
        self.chi2_score = K.scores_
        self.chi2_pvalues = K.pvalues_
        self.chi2_selected_features = K.get_support()
        self.chi2_selected_features_names = self.X.columns[self.chi2_selected_features]
        return self

    def f_classif__fit(self):
        K = SelectKBest(f_classif, k="all")
        K.fit(self.X, self.y)
        self.f_classif_score = K.scores_
        self.f_classif_pvalues = K.pvalues_
        self.f_classif_selected_features = K.get_support()
        self.f_classif_selected_features_names = self.X.columns[self.f_classif_selected_features]
        return self

    def mutual_info_classif__fit(self):
        K = SelectKBest(mutual_info_classif, k="all")
        K.fit(self.X, self.y)
        self.mutual_info_classif_score = K.scores_
        self.mutual_info_classif_pvalues = K.pvalues_
        self.mutual_info_classif_selected_features = K.get_support()
        self.mutual_info_classif_selected_features_names = self.X.columns[
            self.mutual_info_classif_selected_features]
        return self

    # regression

    def mutual_info_regression__fit(self):
        K = SelectKBest(mutual_info_regression, k="all")
        K.fit(self.X, self.y)
        self.mutual_info_regression_score = K.scores_
        self.mutual_info_regression_pvalues = K.pvalues_
        self.mutual_info_regression_selected_features = K.get_support()
        self.mutual_info_regression_selected_features_names = self.X.columns[
            self.mutual_info_regression_selected_features]
        return self

    def f_regression__fit(self):
        K = SelectKBest(f_regression, k="all")
        K.fit(self.X, self.y)
        self.f_regression_score = K.scores_
        self.f_regression_pvalues = K.pvalues_
        self.f_regression_selected_features = K.get_support()
        self.f_regression_selected_features_names = self.X.columns[
            self.f_regression_selected_features]
        return self

    def r_regression__fit(self):
        K = SelectKBest(r_regression, k="all")
        K.fit(self.X, self.y)
        self.r_regression_score = K.scores_
        self.r_regression_pvalues = K.pvalues_
        self.r_regression_selected_features = K.get_support()
        self.r_regression_selected_features_names = self.X.columns[
            self.r_regression_selected_features]
        return self

    def get_selected_features(self):
        if self.type == "classification":
            self.chi2__fit()
            self.chi2_df = pd.concat([pd.DataFrame(
                self.chi2_selected_features_names), pd.DataFrame(self.chi2_score)], axis=1)
            self.chi2_df.columns = ["Features", "Chi Squared Score"]

            self.f_classif__fit()
            self.f_classif_df = pd.concat([pd.DataFrame(
                self.f_classif_selected_features_names), pd.DataFrame(self.f_classif_score)], axis=1)
            self.f_classif_df.columns = ["Features", "F Score"]

            self.mutual_info_classif__fit()
            self.mutual_info_classif_df = pd.concat([pd.DataFrame(
                self.mutual_info_classif_selected_features_names), pd.DataFrame(self.mutual_info_classif_score)], axis=1)
            self.mutual_info_classif_df.columns = [
                "Features", "Mutual Information Score"]

        elif self.type == "regression":
            self.f_regression__fit()
            self.f_regression_df = pd.concat([pd.DataFrame(
                self.f_regression_selected_features_names), pd.DataFrame(self.f_regression_score)], axis=1)
            self.f_regression_df.columns = ["Features", "F Score"]

            self.r_regression__fit()
            self.r_regression_df = pd.concat([pd.DataFrame(
                self.r_regression_selected_features_names), pd.DataFrame(self.r_regression_score)], axis=1)
            self.r_regression_df.columns = ["Features", "R Score"]

            self.mutual_info_regression__fit()
            self.mutual_info_regression_df = pd.concat([pd.DataFrame(
                self.mutual_info_regression_selected_features_names), pd.DataFrame(self.mutual_info_regression_score)], axis=1)
            self.mutual_info_regression_df.columns = [
                "Features", "Mutual Information Score"]

        else:
            raise ValueError(
                "Type must be either classification or regression")

    def plot(self):
        if self.type == "classification":
            # multiple plots
            plt.figure(figsize=(10, 10))
            plt.subplot(3, 1, 1)
            plt.title("Chi Squared Score")
            sns.barplot(x="Chi Squared Score", y="Features",
                        data=self.chi2_df.sort_values(by="Chi Squared Score", ascending=False))
            plt.subplot(3, 1, 2)
            plt.title("F Score")
            sns.barplot(x="F Score", y="Features", data=self.f_classif_df.sort_values(
                by="F Score", ascending=False))
            plt.subplot(3, 1, 3)
            plt.title("Mutual Information Score")
            sns.barplot(x="Mutual Information Score", y="Features",
                        data=self.mutual_info_classif_df.sort_values(by="Mutual Information Score", ascending=False))
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
            plt.show()

        # sns x sama y ditukar
        elif self.type == "regression":
            # multiple plots
            plt.figure(figsize=(10, 10))
            plt.subplot(3, 1, 1)
            plt.title("F Score")
            sns.barplot(x="F Score", y="Features", data=self.f_regression_df.sort_values(
                by="F Score", ascending=False))
            plt.subplot(3, 1, 2)
            plt.title("R Score")
            sns.barplot(x="R Score", y="Features", data=self.r_regression_df.sort_values(
                by="R Score", ascending=False))
            plt.subplot(3, 1, 3)
            plt.title("Mutual Information Score")
            sns.barplot(x="Mutual Information Score", y="Features",
                        data=self.mutual_info_regression_df.sort_values(by="Mutual Information Score", ascending=False))
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
            plt.show()

        else:
            raise ValueError(
                "Type must be either classification or regression")
