import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, plot, iplot
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Imputer, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestRegressor, VotingClassifier, BaggingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from sklearn.utils import check_array
from scipy import sparse
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, mean_squared_error
from scipy.stats import expon, reciprocal
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectFromModel, f_classif
import category_encoders as ce


# Load Data

data = pd.read_csv("datasets/train.csv")
split = StratifiedShuffleSplit(n_splits= 1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['Survived']):
    train = data.loc[train_index]
    test = data.loc[test_index]
print(data.info())

print(data.describe())

print(data.head())

print(data.corr()['Age'])

def ticket_mod(ticket):
    return any(c.isalpha() for c in ticket)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributes]

class CustomAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X.is_copy=False
        X.loc[:, 'FamNo'] = X['SibSp'] + X['Parch']
        X.loc[:, 'HasFam'] = X['FamNo'].where(X['FamNo'] == 0, 1)
        X.loc[:, 'FamNo'] += 1
        X.loc[:, 'CabinCheck'] = ~X['Cabin'].isnull()
        X.loc[:, 'Title'] = X['Name'].map(lambda x: x.split(',')[1].split(' ')[1])
        X.loc[:, 'Fare'] = X['Fare'].where(X['Fare'] != 0, X['Fare'].mean())
        X.loc[:, 'FarePP'] = X['Fare'] / X['FamNo']
        X['SexByPclass'] = X.loc[:, 'Sex'].map(str) + X.loc[:, 'Pclass'].map(str)
        X.loc[:, 'Cabins'] = pd.Series(X['Cabin'].where(~X['Cabin'].isnull(), 'N'))
        X['Ability'] = X['SibSp'] - X['Parch']
        X['TitleByPclass'] = X['Title'].map(str) + X.loc[:, 'Pclass'].map(str)
        X.loc[:, 'Cabins'] = X['Cabins'].apply(lambda x: x[0])

        #X.loc[:, 'Age'] = self.imputer.transform(X.loc[:, 'Age'].values.reshape(1, -1))[0]

        #X.loc[:, 'CabinCheck'] = X['CabinCheck'].map({False: 'A', True: 'B'})


        #X['PclassGroupSize'], index = (X['Pclass']+X['FamNo']).factorize()
        #self.imputer.transform(X.loc[:, 'Age'].values.reshape(1, -1))


        #X['AgeByFamNo'] = X['Age'] / (X['FamNo']+1)
        #X['AgeBySibSp'] = X['Age'] / (X['SibSp']+1)
        #X['AgeByParch'] = X['Age'] / (X['Parch']+1)

        #X.loc[:, 'FareCat'] = pd.qcut(X['Fare'], 4, labels=["1", "2", "3", "4"])
#        X.loc[:, 'AgeCat'] = pd.qcut(X['Age'], 5, labels=["1", "2", "3", "4", "5"])




        #X['EmbarkedGroupSize'], index = (X['Embarked'] + X['FamNo'].map(str)).factorize()
        #X.loc[:, 'TicketCheck'] = ~X['Ticket'].apply(ticket_mod)

       # X['TitleByAgeCat'] = X['Title'] + X['AgeCat'].astype(str)
        return X


def top_indices(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances,  k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = top_indices(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
ohe_attribs = [ 'Pclass', 'Embarked']
ord_attribs = [ 'HasFam', 'Sex', 'Title', 'CabinCheck'  ]
num_attribs = [ 'Age', 'FarePP', 'FamNo', ]





def final_results(Name, clf, X_test):

    clf_test_pred = clf.predict(X_test)

    clf_test_results = pd.concat([test_data['PassengerId'], pd.Series(clf_test_pred).rename('Survived')], axis=1)


    clf_test_results.to_csv(Name+".csv", index=False)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

class CatEnc(BaseEstimator, TransformerMixin):
    def __init__(self, type):
        self.type = type
    def fit(self, X, y=None):
        if self.type == 'backdiff':
            self.encoder = ce.BackwardDifferenceEncoder(handle_unknown='ignore')
        if self.type == 'binenc':
            self.encoder = ce.BinaryEncoder(handle_unknown='impute')
        if self.type == 'hashenc':
            self.encoder = ce.HashingEncoder()
        if self.type == 'helmenc':
            self.encoder = ce.HelmertEncoder(handle_unknown='impute')
        if self.type == 'onehot':
            self.encoder = ce.OneHotEncoder(handle_unknown='ignore')
        if self.type == 'ordenc':
            self.encoder = ce.OrdinalEncoder(handle_unknown='impute')
        if self.type == 'sumenc':
            self.encoder = ce.SumEncoder(handle_unknown='ignore')
        if self.type == 'polyenc':
            self.encoder = ce.PolynomialEncoder(handle_unknown='impute')
        self.encoder.fit(X, y)
        return self
    def transform(self, X):
        X = self.encoder.transform(X)
        self.categories_ = list(X)

        return X

class DataFrameImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)

        return self

    def transform(self, X, y=None):
        X= X.fillna(self.fill)
        return X

class log_apply(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.apply_along_axis(np.log, 1, X )
        return X

cat_enc_ord = CatEnc(type='ordenc')
cat_enc_ohe = CatEnc(type='onehot')


ord_pipeline = Pipeline([
    ('Selector', DataFrameSelector(ord_attribs)),
    ('Imputer', DataFrameImputer()),
    ('CategoryEncoder', cat_enc_ord),
    ('StandardScaler', StandardScaler()),
])

ohe_pipeline = Pipeline([
    ('Selector', DataFrameSelector(ohe_attribs)),
    ('Imputer', DataFrameImputer()),
    ('CategoryEncoder', cat_enc_ohe),
])

num_pipeline = Pipeline([
    ('Selector', DataFrameSelector(num_attribs)),
    ('Imputer', Imputer(strategy='median')),
    ('Log', log_apply()),
    ('StandardScaler', StandardScaler()),

])
full_pipeline = FeatureUnion([
    ('NumPipeline', num_pipeline),
    ('OrdPipeline', ord_pipeline),
    ('OhePipeline', ohe_pipeline),

])
rf_clf = RandomForestClassifier(random_state=42)

class undo_sparse(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray()
Prepare = Pipeline([
    ('AttributesAdder', CustomAttributesAdder()),
    ('Scaler', full_pipeline),
    #('UndoSparse', undo_sparse())
])
Prepare.fit(train.drop('Survived', axis=1), train['Survived'].copy())
train_exp_data = Prepare.transform(train.drop('Survived', axis=1))
val_exp_data = Prepare.transform(test.drop('Survived', axis=1))
X_train = train.drop('Survived', axis=1)
y_train = train['Survived'].copy()
X_val = test.drop('Survived', axis=1).copy()
y_val = test['Survived'].copy()
print(X_train[:5])

test_data = pd.read_csv("datasets/test.csv")

X_test = test_data.copy()
test_exp_data = Prepare.transform(X_test.copy())


attribs = num_attribs.copy()
for cat in ord_attribs:
    attribs.append(cat)
print(attribs)
for label in cat_enc_ohe.categories_:
    attribs.append(label)
print(attribs)
print(type(train_exp_data))
print(train_exp_data)
print(train_exp_data.shape, len(attribs))
train_exp_data = pd.DataFrame(train_exp_data, columns=attribs)
test_exp_data = pd.DataFrame(test_exp_data, columns=attribs)
val_exp_data = pd.DataFrame(val_exp_data, columns=attribs)

rf_clf.fit(train_exp_data, y_train)

print(sorted(zip(rf_clf.feature_importances_, attribs), reverse=True))


#train_exp_data['Age'] = train['Age']

#print(train_exp_data.head())
#print(train)
#print(type(train['Age'].notnull()))
#print(train['Age'])
#print(train_exp_data[train['Age'].notnull().values])

X_age_train = train_exp_data[train.loc[:, 'Age'].notnull().values].drop('Age', axis=1)
y_age_train = train[train['Age'].notnull().values]['Age']




print("RANDOM FOREST REGRESSOR")
rf_r = RandomForestRegressor()
rf_r_param_grid = {
    'n_estimators': [5, 10, 50, 100, 250],
    'criterion': ['mae'],
    'max_features': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'bootstrap': ['True', 'False']
}
rf_r_gs = GridSearchCV(rf_r, rf_r_param_grid, scoring="neg_mean_squared_error", cv=5, verbose=1)
rf_r_gs.fit(X_age_train, y_age_train)
print(sorted(zip(rf_r_gs.best_estimator_.feature_importances_, list(X_age_train)), reverse=True))
rf_r_pred = rf_r_gs.predict(X_age_train)
print("Best Params: ", rf_r_gs.best_params_)
print("Average CV RMSE: ", np.sqrt(-1*rf_r_gs.best_score_))
print("Train RMSE: ", np.sqrt(mean_squared_error(y_age_train, rf_r_pred)))
print(X_train.head())
X_train['Age'] = X_train['Age'].where(~X_train['Age'].isnull(), rf_r_gs.predict(train_exp_data.drop('Age', axis=1))).values
X_val['Age'] = X_val['Age'].where(~X_val['Age'].isnull(), rf_r_gs.predict(val_exp_data.drop('Age', axis=1))).values
X_test['Age'] = X_test['Age'].where(~X_test['Age'].isnull(), rf_r_gs.predict(test_exp_data.drop('Age', axis=1))).values

train_exp_data['Age'] = X_train['Age'].values
print(X_train.head())
exp_data = train.copy()
exp_data['Age'] = X_train['Age'].values
exp_data.loc[:, 'FamNo'] = exp_data['SibSp'] + exp_data['Parch']
exp_data.loc[:, 'HasFam'] = exp_data['FamNo'].where(exp_data['FamNo'] == 0, 1)
exp_data.loc[:, 'FamNo'] += 1
exp_data.loc[:, 'CabinCheck'] = ~exp_data['Cabin'].isnull()
exp_data.loc[:, 'Title'] = exp_data['Name'].map(lambda x: x.split(',')[1].split(' ')[1])
exp_data.loc[:, 'Fare'] = exp_data['Fare'].where(exp_data['Fare'] != 0, exp_data['Fare'].mean())
exp_data.loc[:, 'FarePP'] = exp_data['Fare'] / exp_data['FamNo']
exp_data.loc[:, 'Cabins'] = pd.Series(exp_data['Cabin'].where(~exp_data['Cabin'].isnull(), 'N'))
exp_data.loc[:, 'Cabins'] = exp_data['Cabins'].apply(lambda x: x[0])

print(exp_data.head())

#exp_data.hist(figsize=(12, 8))
#plt.show()
#sns.countplot(x="Pclass", hue="Survived", data=exp_data)
#plt.show()
#sns.countplot(x="Sex", hue="Survived", data=exp_data)
#plt.show()
#sns.barplot(x="Sex", y="Survived", hue="Pclass", data=exp_data)
#plt.show()
#sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=exp_data)
#plt.show()
#sns.boxplot(x='Embarked', y='Fare', hue='Survived', data=exp_data)
#plt.show()
#sns.barplot(x='CabinCheck', y='Survived', hue='Sex', data=exp_data, estimator=sum)
#plt.show()
#sns.barplot(x='Cabins', y='Survived', hue='Sex', data=exp_data, estimator=sum)
#plt.show()
#sns.barplot(x='Pclass', y='Survived', hue='Sex', data=exp_data, estimator=sum)
#plt.show()
#sns.boxplot(x='Title', y='FarePP', hue='Survived', data=exp_data)
#plt.show()
#sns.countplot(x='Embarked', hue='Survived', data=exp_data)
#plt.show()
#sns.countplot(x='Pclass', hue='Survived', data=exp_data)
#plt.show()
#sns.distplot(exp_data['Age'])
#plt.show()
#sns.distplot(exp_data['Fare'])
#plt.show()

#print("LINEAR SVR")
#svr = LinearSVR()
#svr_param_grid = {
#    'C': [0.01, 0.05, 0.1, 0.5, 1, 10],
#    'tol': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
#    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
#}
#svr_gs = GridSearchCV(svr, svr_param_grid, scoring="neg_mean_squared_error", cv=5, verbose=1)
#svr_gs.fit(X_age_train, y_age_train)
##print(sorted(zip(svr_gs.best_estimator_.feature_importances_, list(X_age_train)), reverse=True))
#svr_pred = svr_gs.predict(X_age_train)
#print("Best Params: ", svr_gs.best_params_)
#print("Average CV RMSE: ", np.sqrt(-1*svr_gs.best_score_))
#print("Train RMSE: ", np.sqrt(mean_squared_error(y_age_train, svr_pred)))

#print("SGD REGRESSOR")
#sgd_r = SGDRegressor()
#sgd_r_param_grid = {
#    'penalty': ['l1', 'l2'],
#    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
#    'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
#    'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
#}
#sgd_r_gs = RandomizedSearchCV(sgd_r, sgd_r_param_grid, scoring="neg_mean_squared_error", cv=5, verbose=1, n_iter=100)
#sgd_r_gs.fit(X_age_train, y_age_train)
#print(sorted(zip(sgd_r_gs.best_estimator_.feature_importances_, list(X_age_train)), reverse=True))
#sgd_r_pred = sgd_r_gs.predict(X_age_train)
#print("Best Params: ", sgd_r_gs.best_params_)
#print("Average CV RMSE: ", np.sqrt(-1*sgd_r_gs.best_score_))
#print("Train RMSE: ", np.sqrt(mean_squared_error(y_age_train, sgd_r_pred)))


# param_grid = {
#            'Prepare__Scaler__NumPipeline__Imputer__strategy': ['median', 'mean', 'most_frequent'],
#            'TopFeatureSelector__n_components': [2, 3, 4, 5, 6],
#            'TopFeatureSelector__gamma': np.linspace(0.03, 0.05, 10),
#            'TopFeatureSelector__kernel': ['rbf', 'sigmoid'],
#            'Classifier__n_estimators': [5, 10, 50, 100, 250, 500],
#            #'Classifier__max_features': [4, 6, 8, 10, 12],
 #           'Classifier__min_impurity_decrease': [0.000001, 0.00025, 0.00004,  0.00005, 0.00006, 0.00075, 0.0001 ],
 #       }

rf_clf.fit(train_exp_data, y_train)
print(sorted(zip(rf_clf.feature_importances_, attribs), reverse=True))
# RFPredict = Pipeline([
#    ("Prepare", Prepare),
#    ("TopFeatureSelector", KernelPCA(n_components=3)),
#    ("Classifier", RandomForestClassifier()),
# ])


# rs = RandomizedSearchCV(RFPredict, param_grid, cv=5, n_jobs=1,
#                                    verbose=1, scoring="f1", refit=True, n_iter=200)
#    rs.fit(X_train, y_train)
#    print("Best Params: ", rs.best_params_)
#    print("Best Score: ", rs.best_score_)
#    val_test = Prepare.transform(X_val.copy())
#    print("Validation Results:")
#    y_val_pred = rf_clf.predict(val_test)
#    y_val_scores = cross_val_predict(rf_clf, val_test, y_val, cv=5, method='predict_proba')
#    print("Confusion Matrix: ", confusion_matrix(y_val, y_val_pred))
#    print("Acc: ", accuracy_score(y_val, y_val_pred))
#    print("F1: ", f1_score(y_val, y_val_pred))
#    print("Prec: ", precision_score(y_val, y_val_pred))
#    print("Recall: ", recall_score(y_val, y_val_pred))
#    print("ROC-AUC score: ", roc_auc_score(y_val, np.argmax(y_val_scores, axis=1)))







#tfs = TopFeatureSelector(rf_clf.feature_importances_, k= 9)
#PrepareSelect = Pipeline([
#
#    ('Prepare', Prepare),
#    ('TopFeatureSelector', tfs),
    #('ClassifierSelector', EstimatorSelectionHelper(models, params, tfs.X_val, tfs.y_val, tfs.X_test, attribs))

#])
# Data Exploration

#print(train_exp_data.corr())

#sns.countplot(data=train_exp_data, x="Pclass", hue="Survived")
#plt.show()

#sns.countplot(data=train_exp_data, x="FareCat", hue="Survived")
#plt.show()




models = {
#    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegressionCV(),
#    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(),
    'SVC': SVC(max_iter=2000)
}

params = {
#    'ExtraTreesClassifier': { 'n_estimators': [5, 10, 50, 100, 250, 500], 'max_depth': [3, 4, 5, 6, 7] },
    'RandomForestClassifier': { 'n_estimators': [5, 10, 50, 100, 250, 500], 'max_depth': [3, 4, 5, 6]},
    'AdaBoostClassifier':  { 'n_estimators': [5, 10, 50, 100, 250, 500]},
    'GradientBoostingClassifier': { 'loss': ['deviance', 'exponential'], 'learning_rate': [0.05, 0.065, 0.070, 0.075, 0.080, 0.085, 0.09, 0.095],
                        'max_depth': [2, 3, 4, 5, 6], 'n_estimators': [5, 10, 50, 100, 250, 500]},# 'min_samples_leaf' : [2]},#, 4], 'min_samples_leaf': [2, 3] },
    'LogisticRegression' : { 'penalty': ['l1', 'l2'], 'Cs': [1, 2, 3, 4, 5], 'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'solver': ['liblinear']},
#    'KNN' : { 'n_neighbors': [ 9, 10, 11, 12, 13, 14, 15], },
    'MLP' : { 'activation' : ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
                         'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], },
    'SVC': { 'kernel': ['poly', 'rbf'], 'C': reciprocal(20, 200000), 'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                 'gamma': expon(scale=1.0), 'coef0': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'degree': [2, 3, 4] }
}

def evaluate_classifier(clf, X_train, y_train, X_val, y_val, model, type='Randomized'):
    print("Train Length: ", len(X_train))
    print("Val Length: ", len(X_val))
    if 'Voting' not in model and 'Bagging' not in model:
        print(model+type+" Search Results: ")
        print("Best Score: ", clf.best_score_)
        print("Best Params: ", clf.best_params_)
        print("Best Estimator: ", clf.best_estimator_)

    print("Training Results:")
    y_train_pred = clf.predict(X_train)
    # y_train_scores = cross_val_predict(MLP_clf, X_train, y_train, cv=5, method='decision_function')
    print("Confusion Matrix: ", confusion_matrix(y_train, y_train_pred))
    print("Acc: ", accuracy_score(y_train, y_train_pred))
    print("F1: ", f1_score(y_train, y_train_pred))
    print("Prec: ", precision_score(y_train, y_train_pred))
    print("Recall: ", recall_score(y_train, y_train_pred))
    # print("ROC-AUC score: ", roc_auc_score(y_train, y_train_scores))

    print("Validation Results:")
    y_val_pred = clf.predict(X_val)
    # y_val_scores = cross_val_predict(MLP_clf, X_val, y_val, cv=5, method='decision_function')
    print("Confusion Matrix: ", confusion_matrix(y_val, y_val_pred))
    print("Acc: ", accuracy_score(y_val, y_val_pred))
    print("F1: ", f1_score(y_val, y_val_pred))
    print("Prec: ", precision_score(y_val, y_val_pred))
    print("Recall: ", recall_score(y_val, y_val_pred))
    # print("ROC-AUC score: ", roc_auc_score(y_val, y_val_scores))

def tune_clfs(models, params, X_train, y_train, X_val, y_val, X_test, cv=5, n_jobs=1, refit=True, scoring='f1', verbose=2, n_iter=600 ):
    clf = {}
    for key in models.keys():
        print(key)
        model = models[key]
        PrepareSelectPredict = Pipeline([
            ('Prepare', Prepare),
            ("TopFeatureSelector", KernelPCA(n_components=3)),
            ('Classifier', model)
        ])
        param_grid = {
            'Prepare__Scaler__NumPipeline__Imputer__strategy': ['median', 'mean', 'most_frequent'],
            'TopFeatureSelector__n_components': [2, 3, 4, 5, 6],
            'TopFeatureSelector__gamma': np.linspace(0.03, 0.05, 10),
            'TopFeatureSelector__kernel': ['rbf', 'sigmoid'],
        }


        for ix1, val1 in params.items():
            if ix1 == key:
                for ix2, val2 in val1.items():
                    param_grid['Classifier__' + ix2] = val2

        if 'Grid' in key:
            rs = GridSearchCV(PrepareSelectPredict, param_grid, cv=cv, n_jobs=n_jobs,
                                    verbose=verbose, scoring=scoring, refit=refit)
            type = 'Grid'
        else:
            rs = RandomizedSearchCV(PrepareSelectPredict, param_grid, cv=cv, n_jobs=n_jobs,
                                verbose=verbose, scoring=scoring, refit=refit, n_iter=n_iter)
            type='Rand'


        rs.fit(X_train.copy(), y_train)
        clf[key] = rs
        print("FIT COMPLETE")
        print("Test Length: ", len(X_test))
        evaluate_classifier(rs, X_train, y_train, X_val, y_val, model=key, type=type)
        final_results(key, rs, X_test)
    vc_soft = VotingClassifier([
        ('SVC', clf['SVC']),
        ('RandomForestClassifier', clf['RandomForestClassifier']),
        ('AdaBoostClassifier', clf['AdaBoostClassifier']),
        ('GradientBoostingClassifier', clf['GradientBoostingClassifier']),
        ('LogisticRegression', clf['LogisticRegression']),
        ('MLP', clf['MLP'])
    ], voting='soft')
    vc_soft.fit(X_train, y_train)
    evaluate_classifier(vc_soft, X_train, y_train, X_val, y_val, model="Voting Classifier", type="")
    final_results("Voting Classifier (Soft)", vc_soft, X_test)
    vc_hard = VotingClassifier([
        ('SVC', clf['SVC']),
        ('RandomForestClassifier', clf['RandomForestClassifier']),
        ('AdaBoostClassifier', clf['AdaBoostClassifier']),
        ('GradientBoostingClassifier', clf['GradientBoostingClassifier']),
        ('LogisticRegression', clf['LogisticRegression']),
        ('MLP', clf['MLP'])
    ], voting='hard')
    vc_hard.fit(X_train, y_train)
    evaluate_classifier(vc_hard, X_train, y_train, X_val, y_val, model="Voting Classifier", type="")
    final_results("Voting Classifier (Hard)", vc_hard, X_test)
    bc = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators = 500, max_samples=100, bootstrap=True, n_jobs=1
    )
    param_grid = {
        'Prepare__Scaler__NumPipeline__Imputer__strategy': ['median', 'mean', 'most_frequent'],
        'TopFeatureSelector__n_components': [2, 3, 4, 5, 6],
        'TopFeatureSelector__gamma': np.linspace(0.03, 0.05, 10),
        'TopFeatureSelector__kernel': ['rbf', 'sigmoid'],
        'Classifier__n_estimators': [5, 10, 50, 250, 500],
        'Classifier__max_samples': [0.1, 0.25, 0.5, 0.75, 1],
    }
    PrepareSelectPredict = Pipeline([
        ('Prepare', Prepare),
        ("TopFeatureSelector", KernelPCA(n_components=3)),
        ('Classifier', bc)
    ])
    rs = RandomizedSearchCV(PrepareSelectPredict, param_grid, cv=cv, n_jobs=n_jobs,
                            verbose=verbose, scoring=scoring, refit=refit, n_iter=n_iter)
    rs.fit(X_train, y_train)
    evaluate_classifier(rs, X_train, y_train, X_val, y_val, model="Bagging Classifier", type="")
    final_results("Bagging Classifier (RF)", rs, X_test)


tune_clfs(models, params, X_train, y_train, X_val, y_val, X_test)
