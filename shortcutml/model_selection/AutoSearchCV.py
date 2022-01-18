from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class AutoSearchCV:

    def __init__(self, model, scoring=None, cv=5, n_jobs=None, verbose=1,type="grid"):
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.type = type


    # TODO: Review params combinations
    def params_dict(self):

        # Regression
        if str(self.model) == "LinearRegression()":
            return {
                    "normalize": [True, False]
                    }

        elif str(self.model) == "Ridge()":
            return {"alpha": [0.01, 0.1, 1, 10, 100, 1000],
                    "solver": ["sag", "saga", "lsqr", "svd", "cholesky"]
                    }

        elif str(self.model) == "Lasso()":
            return {"alpha": [0.01, 0.1, 1, 10, 100, 1000],
                    "solver": ["sag", "saga", "lsqr", "svd", "cholesky"]
                    }
        
        elif str(self.model) == "XGBRegressor()":
            return {"learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "gamma": [0, 0.1, 1, 10, 100, 1000],
                    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    }
        
        elif str(self.model) == "LGBMRegressor()":
            return {"learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "gamma": [0, 0.1, 1, 10, 100, 1000],
                    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    }
        
        elif str(self.model) == "SVR()":
            return {"C": [0.01, 0.1, 1, 10, 100, 1000],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "degree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "gamma": [0, 0.1, 1, 10, 100, 1000],
                    "coef0": [0, 0.1, 1, 10, 100, 1000],
                    }
        
        elif str(self.model) == "RandomForestRegressor()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "max_features": ["auto", "sqrt", "log2"],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
        
        elif str(self.model) == "GradientBoostingRegressor()":
            return {"learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "max_features": ["auto", "sqrt", "log2"],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
        
        elif str(self.model) == "AdaBoostRegressor()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "loss": ["linear", "square", "exponential"],
                    }
        
        elif str(self.model) == "BaggingRegressor()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_samples": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "max_features": ["auto", "sqrt", "log2"],
                    }
        
        elif str(self.model) == "ExtraTreesRegressor()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "max_features": ["auto", "sqrt", "log2"],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
        
        elif str(self.model) == "VotingRegressor()":
            raise ValueError("VotingRegressor is not supported")

        # Classification model
        elif str(self.model) == "LogisticRegression()":
            return {"penalty": ["l1", "l2"],
                    "C": [0.01, 0.1, 1, 10, 100, 1000],
                    "solver": ["liblinear", "sag", "saga"]
                    }
        
        elif str(self.model) == "SGDClassifier()":
            return {"loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "alpha": [0.01, 0.1, 1, 10, 100, 1000],
                    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
                    "eta0": [0.01, 0.1, 1, 10, 100, 1000],
                    "power_t": [0.01, 0.1, 1, 10, 100, 1000],
                    "tol": [0.01, 0.1, 1, 10, 100, 1000],
                    "warm_start": [True, False],
                    "class_weight": ["balanced", None],
                    }
        
        elif str(self.model) == "SVC()":
            return {"C": [0.01, 0.1, 1, 10, 100, 1000],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "degree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "gamma": [0, 0.1, 1, 10, 100, 1000],
                    "coef0": [0, 0.1, 1, 10, 100, 1000]
                    }
                
        elif str(self.model) == "RandomForestClassifier()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "max_features": ["auto", "sqrt", "log2"],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
        
        elif str(self.model) == "GradientBoostingClassifier()":
            return {"learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "max_features": ["auto", "sqrt", "log2"],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
        
        elif str(self.model) == "AdaBoostClassifier()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "loss": ["linear", "square", "exponential"],
                    }
        
        elif str(self.model) == "BaggingClassifier()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_samples": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "max_features": ["auto", "sqrt", "log2"],
                    }
        
        elif str(self.model) == "ExtraTreesClassifier()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "max_features": ["auto", "sqrt", "log2"],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
        
        elif str(self.model) == "VotingClassifier()":
            raise ValueError("VotingClassifier is not supported")
        
        elif str(self.model) == "XGBClassifier()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "gamma": [0, 0.1, 1, 10, 100, 1000],
                    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "max_delta_step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    }
        
        elif str(self.model) == "LGBMClassifier()":
            return {"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "learning_rate": [0.01, 0.1, 1, 10, 100, 1000],
                    "num_leaves": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "min_child_samples": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    }
        
        else:
            raise ValueError("Estimator not supported")


    def search(self, X, y):
        self.X = X
        self.y = y
        
        if self.type == "grid":
            self.grid = GridSearchCV(self.model, self.params_dict(), scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs,
                                verbose=self.verbose)
            self.grid.fit(self.X, self.y)
            self.best_estimator_ = self.grid.best_estimator_
            self.best_score_ = self.grid.best_score_
            self.best_params_ = self.grid.best_params_
            self.best_index_ = self.grid.best_index_
            self.cv_results_ = self.grid.cv_results_
        
        elif self.type == "random":
            self.random = RandomizedSearchCV(self.model, self.params_dict(), scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose)
            self.random.fit(self.X, self.y)
            self.best_estimator_ = self.random.best_estimator_
            self.best_score_ = self.random.best_score_
            self.best_params_ = self.random.best_params_
            self.best_index_ = self.random.best_index_
            self.cv_results_ = self.random.cv_results_

        else:
            raise ValueError("Type not supported: Please choose 'grid' or 'random'")
        
        print(f"""
        
        SEARCH SUMMARY
        
        Best estimator  : {self.best_estimator_}
        Best score      : {self.best_score_}
        Best params     : {self.best_params_}
        
        """)