Supervised Learning Notes
================
Shivam Verma
20/01/2020

## Important Link

[Gini
Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)  
[Mathematical Formulation of a Decision
Tree](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)  
[Machine Learning Demo in
R](https://github.com/StatQuest?tab=repositories)  
[MDS, Stat-ML
Dictionary](https://ubc-mds.github.io/resources_pages/terminology/)

[Supervised Learning
Book](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)

[Scikit learn on linear
models](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

![Penalties By Solver](C:\\MyDisk\\MDS\\DSCI_571\\Solvers_Penalties.jpg)

``` python
# general plotting function for many models
    from model_plotting import plot_tree, plot_knn_regressor, plot_lowess
    plot_tree(X, y, model, predict_proba = True)
```

### Python code for Linear Regression

``` python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X,y)
    predicted = model.predict(z)
```

### Python code for Logistic Regression

``` python
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X,y)
    predicted = model.predict(z)
```

## Loess/Lowess

``` python
    from statsmodels.nonparametric.smoothers_lowess import lowess
    n = 30 # number of samples
    k = 1  # k for kNN
    np.random.seed(0) # fix seed for reproducibility
    X = np.linspace(-1,1,n)+np.random.randn(n)*0.01
    X = X[:, None]
    y = (np.random.randn(n,1) + X*5).ravel()
    X = X.ravel()
    z = lowess(y, X, frac=5/n) # frac is like "k/n", larger vales lead to smoother curves
    plot_lowess(X, y, z)
```

## Generative vs discriminative models

  - A visual metaphor from Jurafsky and Martin 2019: Imagine we are
    trying to distinguish dog images from cat images.  
  - A generative model would have the goal of understanding what dogs
    look like and what cats look like. You can ask a generative model to
    draw a dog. During predict, we ask which model fits the given image
    best.  
  - A discriminative model is only trying to learn to distinguish the
    classes. For instance, if all the dogs in the training data are
    wearing collars and cats aren???t, it would tell you that the
    difference between cats and dogs is that they do not wear collars.

## The fundamental Tradeoff, bias-variance Tradeoff

  - E\_test = (E\_test-E\_train) + (E\_train-E\_Best) + E\_Best  
  - (E\_test-E\_train): this difference relates to the Variance between
    the models, measures how sensitive we are to training data  
  - (E\_train-E\_Best): Bias, how low can we make the training error
    i.e., without increasing the variance term is the question  
  - E\_Best: noise, error of the best model. THis is also the
    irreducible error of the best model [Complete proof
    here](C:\\MyDisk\\MDS\\DSCI_571\\BiasVarianceTradeoff.pdf) ![Also
    look at this is
    important](C:\\MyDisk\\MDS\\DSCI_571\\How_to_decide_k_in_CV.png)
    [Why models with higher K have higher
    variance](https://stats.stackexchange.com/questions/61783/bias-and-variance-in-leave-one-out-vs-k-fold-cross-validation/357749#357749)

**Paraphrasing Yves Grandvalet the author of a 2004 paper on the topic I
would summarize the intuitive argument as follows:**  
\- If cross-validation were averaging independent estimates: then
leave-one-out CV one should see relatively lower variance between models
since we are only shifting one data point across folds and therefore the
training sets between folds overlap substantially.  
\- This is not true when training sets are highly correlated:
Correlation may increase with K and this increase is responsible for the
overall increase of variance in the second scenario. Intuitively, in
that situation, leave-one-out CV may be blind to instabilities that
exist, but may not be triggered by changing a single point in the
training data, which makes it highly variable to the realization of the
training set.  
\- [Too
much](http://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)

> My Intuition: Initially if you start increasing k then the variance
> should go down untill the data start to overlap majorly where the
> vairance start increasing if the data were independent it would
> continue to go down.

## Parametric vs Non-Parametric

  - **Parametric**: A learning model that summarizes data with a set of
    parameters of fixed size (independent of the number of training
    examples) is called a parametric model. No matter how much data you
    throw at a parametric model, it won???t change its mind about how many
    parameters it needs.
      - Examples:
          - Logistic Regression  
          - Linear Discriminant Analysis  
          - Perceptron  
          - Naive Bayes (I think as long as number of features remain
            the same its parametric but if vocabulary itself increases
            then it might be Non-Parametric)  
          - Simple Neural Networks
      - Benefits:
          - Simpler: These methods are easier to understand and
            interpret results.  
          - Speed: Parametric models are very fast to learn from data.  
          - Less Data: They do not require as much training data and can
            work well even if the fit to the data is not perfect.
      - Limitations:
          - Constrained: By choosing a functional form these methods are
            highly constrained to the specified form.  
          - Limited Complexity: The methods are more suited to simpler
            problems.  
          - Poor Fit: In practice the methods are unlikely to match the
            underlying mapping function.
  - **Non-Parametric**: Nonparametric methods are good when you have a
    lot of data and no prior knowledge, and when you don???t want to worry
    too much about choosing just the right features.
      - Examples:
          - k-Nearest Neighbors  
          - Decision Trees like CART and C4.5 (Decision trees are kind
            of ambiguous)  
          - Support Vector Machines  
      - Benefits:
          - Flexibility: Capable of fitting a large number of functional
            forms.  
          - Power: No assumptions (or weak assumptions) about the
            underlying function.  
          - Performance: Can result in higher performance models for
            prediction.  
      - Limitations:
          - More data: Require a lot more training data to estimate the
            mapping function.  
          - Slower: A lot slower to train as they often have far more
            parameters to train.  
          - Overfitting: More of a risk to overfit the training data and
            it is harder to explain why specific predictions are
made.

## Typical Hyperparameter optimization framework

``` python
    from sklearn.model_selection import train_test_split, cross_val_score

    # split into training/validation and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # calculation loop
    k_dict = {'k':[], 'train_error':[], 'cv_error': []}
    for k in np.arange(1, 51):
        model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train.to_numpy().ravel())
        k_dict['k'].append(k)    
        k_dict['train_error'].append(1 - model.score(X_train, y_train.to_numpy().ravel())) # this is a little wrong...
        k_dict['cv_error'].append(1 - cross_val_score(model, X_train, y_train.to_numpy().ravel(), cv=5).mean())
    k_df = pd.DataFrame(k_dict)
    k_df = k_df.melt(id_vars='k', value_name='Error', var_name='Data') # melt datadframe to work with altair

    alt.Chart(k_df).mark_line().encode(x="k", y="Error", color='Data').properties(title='Accuracy vs Depths')


################ 
# Grid Search 
################ 

  from sklearn import svm, datasets
  from sklearn.model_selection import GridSearchCV
  
  iris = datasets.load_iris()
  parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
  svc = svm.SVC()
  clf = GridSearchCV(svc, parameters)
  clf.fit(iris.data, iris.target)
  sorted(clf.cv_results_.keys())
```

## Feature normalization/standardization

``` python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Normalization
    # sets range to [0,1]
    X -= np.min(X,axis=0)
    X /= np.max(X,axis=0)
    MinMaxScaler()
    
    minmax = MinMaxScaler()
    minmax.fit(X_train2)
    X_train2_minmax = minmax.transform(X_train2)
    X_test2_minmax = minmax.transform(X_test2)
# standardization
    # sets sample mean to  0 , s.d. to  1
    X -= np.mean(X,axis=0)
    X /=  np.std(X,axis=0)
    StandardScaler()

    scaler = StandardScaler()
    scaler.fit(X_train2)
    X_train2_scaled = scaler.transform(X_train2)
    X_test2_scaled = scaler.transform(X_test2)
```

## Power transformers

  - Apply a power transform featurewise to make data more Gaussian-like.
  - Power transforms are a family of parametric, monotonic
    transformations that are applied to make data more Gaussian-like.
    This is useful for modeling issues related to heteroscedasticity
    (non-constant variance), or other situations where normality is
    desired.
  - Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing
    variance and minimizing skewness is estimated through maximum
    likelihood.
      - Box-Cox requires input data to be strictly positive, while
        Yeo-Johnson supports both positive or negative data.
      - By default, zero-mean, unit-variance normalization is applied to
        the transformed
data.

<!-- end list -->

``` python
    # PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
    import numpy as np
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer()
    data = [[1, 2], [3, 2], [4, 5]]
    print(pt.fit(data)) # The optimal lambda parameter for minimizing skewness is estimated on each feature independently using maximum likelihood.
    print(pt.lambdas_)
    print(pt.transform(data)) # Apply the power transform to each feature using the fitted lambdas.
```

## Encoding categorical variables

### Label encoding

  - Encode labels with value between 0 and n\_classes-1.
  - label encoding is only useful if there is ordinality in your
data

<!-- end list -->

``` python
        from sklearn import preprocessing #Also: from sklearn.preprocessing import LabelEncoder #
        le = preprocessing.LabelEncoder()
        le.fit([1, 2, 2, 6])
        le.classes_
        le.transform([1, 1, 2, 6]) 
            # le.fit_transform([1, 1, 2, 6]) # this fits the data and then transform it
        le.inverse_transform([0, 0, 1, 2])
```

### One-hot encoding

  - The features are encoded using a one-hot (aka ???one-of-K??? or ???dummy???)
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array.
  - we have n categories in our column
  - We create n new binary columns to represent those categories

<!-- end list -->

``` python
        from sklearn.preprocessing import OneHotEncoder
        df = pd.read_csv('data/depression_data.csv')
        X = df[['age', 'weight', 'treatment']]
        y = df['effect']

        ohe = OneHotEncoder(sparse=False, dtype=int)
        ohe.fit_transform(X[['treatment']])
        pd.concat((X, pd.DataFrame(ohe.fit_transform(X[['treatment']]), columns=ohe.categories_[0])), axis=1).head()

        ohe = OneHotEncoder(drop='first', sparse=False, dtype=int)
        X = pd.concat((X, pd.DataFrame(ohe.fit_transform(X[['treatment']]), columns=ohe.categories_[0][1:])), axis=1)
        # drop feature : ???first??? or a list/array of shape (n_features,), default=None.
            # None : retain all features (the default).
            # ???first??? : drop the first category in each feature. If only one category is present, the feature will be dropped entirely.
            # array : drop[i] is the category in feature X[:, i] that should be dropped.
```

## Preprocessing Both Numeric & Categorical

  - preprocess both numeric features (e.g., scaling) and categorical
    features (e.g., OHE) at the same time ColumnTransformer()
      - his estimator allows different columns or column subsets of the
        input to be transformed separately and the features generated by
        each transformer will be concatenated to form a single feature
        space. This is useful for heterogeneous or columnar data, to
        combine several feature extraction mechanisms or transformations
        into a single transformer.

<!-- end list -->

``` python
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

        df = pd.read_csv('data/depression_data.csv')
        X = df[['age', 'weight', 'treatment']]
        y = df['effect']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

        numeric_features = ['age', 'weight']
        categorical_features = ['treatment']

        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), numeric_features),
            ('ohe', OneHotEncoder(drop='first'), categorical_features)])

        # names the transformed variables
        # please note that `preprocessor.named_transformers_` doesn't work before fitting...once its fit this works
        col_names = (numeric_features + list(preprocessor.named_transformers_['ohe'].get_feature_names(categorical_features)))

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), index=X_train.index, columns=col_names)
        X_test = pd.DataFrame(preprocessor.transform(X_test), index=X_test.index, columns=X_train.columns)
```

## OVO vs OVR

  - Important Link:
    [Wiki](https://en.wikipedia.org/wiki/Multiclass_classification),
    [Link1](https://datastoriesweb.wordpress.com/2017/06/11/classification-one-vs-rest-and-one-vs-one/),
    [Link2](https://www.quora.com/Whats-an-intuitive-explanation-of-one-versus-one-classification-for-support-vector-machines),
    [Link3](https://en.etsmtl.ca/ETS/media/ImagesETS/Labo/LIVIA/Publications/2006/MILGRAM_IWFHR_2006.pdf)

### OVO

  - ovo is slower than ovr because it has to train more classifiers
  - ovo is trained only on a subset of the data (2 classes at a time)
  - local decision

### OVR

  - in ovr each classifier is trained on the full dataset but if you
    have a large number of classes and if one of the classes is very
    skewed then that might pose a problem.
  - There are ways to deal with class imbalance (e.g., assigning more
    weight to a rare class) but they don???t always work well.
  - It may create class imbalance when we are training individual
    classifiers and if for some class we happen to have only a small
    number instances, the classifier won???t be able to learn much about
    that class.
  - But remember that if this is a representative distribution of the
    test data, the class will be rare in the test data as well and you
    might not see a big difference in the test accuracy because poorly
    performing on rarely occurring class won???t matter much.
  - global decision

## Ensemble Models

  - The main advantage of ensembles of different classifiers is that it
    is unlikely that all classifiers will make the same mistake. In
    fact, as long as every error is made by a minority of the
    classifiers, you will achieve optimal classification\!
    Unfortunately, the inductive biases of different learning algorithms
    are highly correlated. This means that different algorithms are
    prone to similar types of errors. In particular, ensembles tend to
    reduce the variance of classifiers. So if you have a classification
    algorithm that tends to be very sensitive to small changes in the
    training data, ensembles are likely to be useful.

  - Note that the bootstrapped data sets will be similar. However, they
    will not be too similar. For example, if N is large then the number
    of examples that are not present in any particular bootstrapped
    sample is relatively large. The probability that the first training
    example is not selected once is (1 ??? 1/N). The probability that it
    is not selected at all is (1 ??? 1/N) N. As N ??? ???, this tends to 1/e ???
    0.3679. (Already for N = 1000 this is correct to four decimal
    points.) So only about 63% of the original training examples will be
    represented in any given bootstrapped set

  - Since bagging tends to reduce variance, it provides an alternative
    approach to regularization.

  - Key assumption: classifiers are independent in their predictions.  

  - If classifiers tend to make different kinds of errors ensembles tend
    to reduce the variance of classifiers.

  - General methods to combine different classifiers to build an
    ensemble model
    
      - Majority voting (VotingClassifier)

<!-- end list -->

``` python
    classifiers = {
        "Decision tree"         : DecisionTreeClassifier(max_depth=5),
        "KNN"                   : KNeighborsClassifier(),
        "Naive Bayes"           : GaussianNB(),
        "Logistic Regression"   : LogisticRegression(),
        "SVM"                   : SVC(probability=True)
    }
    ensemble = VotingClassifier(classifiers.items(), voting="soft")
    ensemble.fit(X_train, y_train);
    print('Ensemble performance: \n')
    show_scores(ensemble, X_train, y_train, X_test, y_test)
```

  - argument voting: If ???hard???, uses predicted class labels for majority
    rule voting. Else if ???soft???, predicts the class label based on the
    argmax of the sums of the predicted probabilities, which is
    recommended for an ensemble of well-calibrated classifiers.  
  - Bagging

## Pipelines

  - Pipelines are a more elegant and organized way to do things

<!-- end list -->

``` python
numeric_features = ['age', 'fnlwgt', 'education.num', 
                    'capital.gain', 'capital.loss', 
                    'hours.per.week']

categorical_features = ['workclass', 'education', 'marital.status', 
                        'occupation', 'relationship', 
                        'race', 'sex', 'native.country']

numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())
                                    ])


categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', 
                                                                    fill_value='missing')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                         ])

preprocessor = ColumnTransformer(
                                 transformers=[
                                    ('num', numeric_transformer, numeric_features),
                                    ('cat', categorical_transformer, categorical_features)
                                ])


# DummyClassifier is a classifier that makes predictions using simple rules. This classifier is useful as a simple baseline to compare with other (real) classifiers. Do not use it for real problems.

# Baseline model
from sklearn.dummy import DummyClassifier
print('Fitting baseline model: ')
dummy = DummyClassifier()

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', dummy)])

t = time.time()
clf.fit(X_train, y_train)
tr_err, valid_err = get_scores(clf, X_train, y_train, X_valid, y_valid)
elapsed_time = time.time() - t



# Hyperparameter optimization with Logistic Regression
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_trainvalid, y_trainvalid)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(X_valid, y_valid)))
```

# Algorithms

## Boosting

  - Important Links: [source book
    chapter](http://ciml.info/dl/v0_99/ciml-v0_99-ch13.pdf), [Not
    reading
    now](https://mitpress.mit.edu/sites/default/files/titles/content/boosting_foundations_algorithms/chapter001.html),
    [Bagging vs
    Boosting](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)

  - High-level Summary
    
      - AdaBoost  
      - Gradient boosting
          - Gradient boosting is a particular interpretation of boosting
            that allows generalizing it.  
          - Gradient boosting interprets predictions as parameters, then
            does gradient descent w.r.t. predictions. Each ???iteration???
            adds a base model (???learner???) to the ensemble.  
      - XGBoost (eXtreme Gradient Boosting)
          - XGBoost is a particular industrial-strength implementation
            of gradient boosted trees.  
      - Light GBM
          - Another tree-based gradient boosting framework  
          - Great for large-scale datasets  
          - Lower memory because of depth-first search like growth

  - Boosting is the process of taking a crummy learning algorithm
    (technically called a weak learner) and turning it into a great
    learning algorithm (technically, a strong learner).

  - Boosting is more of a ???framework??? than an algorithm.

  - The particular boosting algorithm discussed here is AdaBoost, short
    for ???adaptive boosting algorithm.??? AdaBoost is famous because it was
    one of the first practical boosting algorithms: it runs in
    polynomial time and does not require you to define a large number of
    hyperparameters. It gets its name from the latter benefit: it
    automatically adapts to the data that you give it.

  - One of the major attractions of boosting is that it is perhaps easy
    to design computationally efficient weak learners. A very popular
    type of weak learner is a shallow decision tree: a decision tree
    with a small depth limit.

![Image of algo](C:\\MyDisk\\MDS\\Adaboost.PNG)

  - **Strong learning algorithm**- a strong learning algorithm L as
    follows. When given a desired error rate e, a failure probability ??
    and access to ???enough??? labeled examples from some distribution D,
    then, with high probability (at least 1 ??? ??), L learns a classifier
    f that has error at most e

  - **weak learning algorithm**- a weak learning algorithm W that only
    has to achieve an error rate of 49%, rather than some arbitrary
    user-defined parameter e. (49% is arbitrary: anything strictly less
    than 50% would be fine.

> **Boosting analogy**(kind of what apoorv was saying)- The intuition
> behind AdaBoost is like studying for an exam by using a past exam. You
> take the past exam and grade yourself. The questions that you got
> right, you pay less attention to. Those that you got wrong, you study
> more. Then you take the exam again and repeat this process. You
> continually down-weight the importance of questions you routinely
> answer correctly and up-weight the importance of questions you
> routinely answer incorrectly. After going over the exam multiple
> times, you hope to have mastered everything.

  - adaptive parameter ??- The (weighted) error rate of
    ![f^{(k)}](https://latex.codecogs.com/png.latex?f%5E%7B%28k%29%7D
    "f^{(k)}") is used to determine the adaptive parameter ??, which
    controls how ???important???
    ![f^{(k)}](https://latex.codecogs.com/png.latex?f%5E%7B%28k%29%7D
    "f^{(k)}") is. As long as the weak learner does, indeed, achieve \<
    50% error, then ?? will be greater than zero. As the error drops to
    zero, ?? grows without bound.

### Understanding what alpha does-

![understanding what alpha
does](C:\\MyDisk\\MDS\\adaboost_adaptive_param.PNG)

## Boosting decision stumps-

![Image of algo](C:\\MyDisk\\MDS\\stump_adaboost_linear_classifier.PNG)
![Image of
algo](C:\\MyDisk\\MDS\\stump_adaboost_linear_classifier_v2.PNG)

  - In fact, a very popular weak learner is a decision decision stump: a
    decision tree that can only ask one question. Boosting algorithm
    takes weak learners like decisionstump and convert it into a linear
    classifier form (the math is pretty simple actually).

  - By concentrating on decision stumps, all weak functions must have
    the form   
    ![f(x) = s(2x\_d
    ??? 1)](https://latex.codecogs.com/png.latex?f%28x%29%20%3D%20s%282x_d%20%E2%88%92%201%29
    "f(x) = s(2x_d ??? 1)")  
    , where s ??? {??1} and d indexes some feature

  - When working with decision stumps, AdaBoost actually provides an
    algorithm for learning linear classifiers\! In fact, this connection
    has recently been strengthened: you can show that AdaBoost provides
    an algorithm for optimizing exponential loss.

  - You can notice that this is nothing but a two-layer neural network,
    with K-many hidden units\! Of course it???s not a classifically
    trained neural network (once you learn w(k) you never go back and
    update it), but the structure is identical.

## Random forests General ideal

  - fit a diverse set of classifiers by injecting randomness in the
    classifier construction
    
      - Randomness in Data:
          - Build each tree on a bootstrap sample (i.e., a sample drawn
            with replacement from the training set)  
      - Randomness in Features:
          - Consider a random subset of features at each split
            (RandomForestClassifier)  
          - Consider a random subset of features at each split and
            random threshold (ExtraTreesClassifier)  

  - At each node of the tree:
    
      - Randomly select a subset of features out of all features
        (independently for each node).  
      - Find the best split on the selected features.  
      - Grow the trees to maximum depth.  

  - predict by taking the average of predictions given by individual
    classifiers  

  - Random forests are usually more accurate compared to decision trees,
    in fact they are usually one of the best performing off-the-shelf
    classifiers.  

  - error rate depends upon the following:
    
      - The correlation between any two trees in the forest. Higher the
        correlation higher the error rate.  
      - The error rate of each individual tree in the forest. Lowering
        the error rate of the individual trees decreases the forest
        error rate.  

  - Can easily parallelize training because all trees are independent of
    each other  

  - Random forests are less likely to overfit  

  - Like decision trees, but unlike most ML models, the running time of
    calling predict for a random forest is independent of ????.  

  - Increasing the hyperparameter max\_features (the number of features
    to consider for a split) makes the model more complex and moves the
    fundamental tradeoff toward lower training error.
    
      - Intuitively, higher value for max\_features means we have higher
        number of features to be considered at each split, which means
        that the individual trees are more similar to what a decision
        tree algorithm would give you. But the ensemble would be less
        diverse and hence less effective now when it comes to
        generalization. (Recall that we also have randomness from
        bootstrap resampling and so it???s still likely to be better than
        a decision tree.)  

  - Increasing the hyperparameter n\_estimators (the number of trees)
    makes the model more complex and moves the fundamental tradeoff
    toward lower training error.
    
      - This is False. This is kind of ambiguous but I would say false
        here. In random forests we get rid of the variance by averaging,
        and larger the number of trees the better. But large number of
        trees means more computational time. According to
        [Breiman](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf),
        increasing `n_estimators` in a random forest ensemble should not
        increase generalization error and should not lead to
        overfitting. (That said, in practice, people have observed that
        increasing n\_estimators can lead to overfitting in some cases.)

  - One of the most computationally expensive aspects of ensembles of
    decision trees is training the decision trees. This is very fast for
    decision stumps, but for deeper trees it can be prohibitively
    expensive. The expensive part is choosing the tree structure. Once
    the tree structure is chosen, it is very cheap to fill in the leaves
    (i.e., the predictions of the trees) using the training data.

  - An efficient and surprisingly effective alternative is to use trees
    with fixed structures and random features. Collections of trees are
    called forests, and so classifiers built like this are called random
    forests

  - It takes three arguments: the data, a desired depth of the decision
    trees, and a number K of total decision trees to build. The
    algorithm generates each of the K trees independently, which makes
    it very easy to parallelize. For each trees, it constructs a full
    binary tree of depth depth. The features used at the branches of
    this tree are selected randomly, typically with replacement, meaning
    that the same feature can appear multiple times, even in one branch.
    The leaves of this tree, where predictions are made, are filled in
    based on the training data. This last step is the only point at
    which the training data is used. The resulting classifier is then
    just a voting of the Kmany random
trees.

### Python code for Random Forest, [Wiki](https://en.wikipedia.org/wiki/Random_forest)

``` python
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier()
  model.fit(X, y)
  predicted = model.predict(z)
```

## SVMs: General idea

[SVM
tutorial](https://www.microsoft.com/en-us/research/publication/a-tutorial-on-support-vector-machines-for-pattern-recognition/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F67119%2Fsvmtutorial.pdf)

  - Choose the hyperplane which is furthest away from the closest
    training points.  
  - In other words, choose the hyperplane which has the largest margin,
    where margin is the distance from the boundary to the nearest
    point(s).  
  - Intuitively, more margin is good because it leaves more ???room???
    before we make an error.  
  - Main idea: the decision boundary only depends on the support
    vectors.  
  - SVMs not only work with linearly-separable data. We can use the
    ???kernel??? trick to deal with non linearly separable surfaces
      - The general idea is mapping the non-linearly separable data into
        a higher dimensional space where you can find a separating
        hyperplane.  
  - Note that SVC multiclass mode is implemented using one-vs-one scheme
    whereas LinearSVC uses one-vs-rest scheme.
      - That said, we can use the OneVsRestClassifier wrapper to
        implement one-vs-rest with SVC.

### Support vectors

  - Each training example either is or isn???t a ???support vector???.  
  - This gets decided during fit.  
  - It does not apply to test examples.  
  - If we delete all non-support vector the decision boundary won???t
    change  
  - If we delete a support vector the decision boundary gets changed

### Key Hyperparameters of rbf SVM are

  - `gamma`
      - Larger gamma(more complex): more rapid fluctuations in the
        decision surface (can change from red to blue in a short
        distance).  
      - Smaller gamma smoother fluctuations.  
  - `C`
      - Large C, more complex  
      - You can think of larger C as insistence on getting low training
        error; smaller C forgives more errors.  
      - Larger C also leads to more confident
predictions.

### [The math of SVM explained well](https://stackoverflow.com/questions/9480605/what-is-the-relation-between-the-number-of-support-vectors-and-training-data-and)

  - Support Vector Machines are an optimization problem. They are
    attempting to find a hyperplane that divides the two classes with
    the largest margin. The support vectors are the points which fall
    within this margin. It???s easiest to understand if you build it up
    from simple to more complex.
  - Hard Margin Linear SVM
      - In an a training set where the data is linearly separable, and
        you are using a hard margin (no slack allowed), the support
        vectors are the points which lie along the supporting
        hyperplanes (the hyperplanes parallel to the dividing hyperplane
        at the edges of the margin)

### Python code for Support Vector Machine

``` python
    from sklearn.svm import SVC
    model = SVC(kernel=???rbf???)
    model.fit(X,y)
    predicted = model.predict(z)
```

## Scikit learning- KNN:

  - Source:
    [Link1](https://scikit-learn.org/stable/modules/neighbors.html),
    [Link2](https://github.ubc.ca/MDS-2019-20/DSCI_571_sup-learn-1_students/blob/master/lectures/lecture3.ipynb)

  - **Questions**:
    
      - Dont understand how scikit learn switches from tree based
        approach to brute force using the leaf size parameter of the
        tree
          - I think its kind of a backend optimization using the
            leafsize parameter  
      - Why cant KNN just predict by checking the shading color for a
        test example isntead of computing all those distances? **Tom???s
        Answer**
          - Firstly, if you look at the code I wrote, the ???shading
            colour??? is generated by creating an extremely fine mesh of
            points and then predicting every single one of them. That is
            computationally very expensive (but great for
            visualization\!). It???s not like kNN just output a nice
            boundary that we could use.  
          - But, you could use this method to generate ???decision
            boundaries??? for yourself if you wanted to. It???s just the to
            do this you have to make hundreds, thousands or even
            millions of predictions depending on your data size and
            number of features. So it doesn???t seem worth it. Also note
            that if you obtained more data, or more features in future,
            or changed your hyperparameters, you would then have to
            recalculate these decision boundaries\!

  - K-d trees & more on Knn:
    [Link1](https://machine-learning-course.readthedocs.io/en/latest/content/supervised/knn.html),
    [Link2](https://github.ubc.ca/MDS-2019-20/DSCI_512_alg-data-struct_students/blob/master/lectures/lecture4.ipynb)

### Classification:

  - Neighbors-based classification is a type of instance-based learning
    or non-generalizing learning: it does not attempt to construct a
    general internal model, but simply stores instances of the training
    data.

  - scikit-learn implements two different nearest neighbors classifiers:
    
      - `KNeighborsClassifier` (most commonly used) implements learning
        based on the `k` nearest neighbors of each query point, where
        `k` is an integer value specified by the user.  
      - `RadiusNeighborsClassifier` implements learning based on the
        number of neighbors within a fixed radius `r` of each training
        point, where `r` is a floating-point value specified by the
        user.
          - In cases where the data is not uniformly sampled,
            radius-based neighbors classification in
            RadiusNeighborsClassifier can be a better choice.

  - In general a larger k suppresses the effects of noise, but makes the
    classification boundaries less distinct.  

  - For high-dimensional parameter spaces, this method becomes less
    effective due to the so-called ???curse of dimensionality???.

  - `weights`
    
      - `weights = 'uniform'`, assigns uniform weights to each
        neighbor.  
      - `weights = 'distance'` assigns weights proportional to the
        inverse of the distance from the query point.  
      - Alternatively, a user-defined function of the distance can be
        supplied to compute the weights.

### Regression:

  - The label assigned to a query point is computed based on the mean of
    the labels of its nearest neighbors.  

  - scikit-learn implements two different neighbors regressors:
    
      - `KNeighborsRegressor` implements learning based on the k nearest
        neighbors of each query point, where k is an integer value
        specified by the user.  
      - `RadiusNeighborsRegressor` implements learning based on the
        neighbors within a fixed radius r of the query point, where r is
        a floating-point value specified by the user.  

  - `weights` follow a similar approach as in classification

  - The most naive neighbor search implementation involves the
    brute-force computation (`algorithm = 'brute'`) of distances between
    all pairs of points in the dataset: for N samples in D dimensions,
    this approach scales as
    ![O(D\*N^2)](https://latex.codecogs.com/png.latex?O%28D%2AN%5E2%29
    "O(D*N^2)").

  - To address the computational inefficiencies of the brute-force
    approach, a variety of tree-based data structures have been invented
    like KD-Tree(`algorithm = 'kd_tree'`). In this way, the
    computational cost of a nearest neighbors search can be reduced to
    ![O(D\*N\*log(N))](https://latex.codecogs.com/png.latex?O%28D%2AN%2Alog%28N%29%29
    "O(D*N*log(N))") or better.  

  - The construction of a KD tree is very fast: because partitioning is
    performed only along the data axes, no D-dimensional distances need
    to be computed. Once constructed, the nearest neighbor of a query
    point can be determined with only
    ![O(log(N))](https://latex.codecogs.com/png.latex?O%28log%28N%29%29
    "O(log(N))") distance computations.
    
      - Though the KD tree approach is very fast for low-dimensional
        (D\<20) neighbors searches, it becomes inefficient as D grows
        very large: this is one manifestation of the so-called ???curse of
        dimensionality???.

  - Where KD trees partition data along Cartesian axes, ball trees
    (`algorithm = 'ball_tree'`) partition data in a series of nesting
    hyper-spheres. This makes tree construction more costly than that of
    the KD tree, but results in a data structure which can be very
    efficient on highly structured data, even in very high dimensions.  

  - A ball tree (complexity:
    ![O(D\*log(N))](https://latex.codecogs.com/png.latex?O%28D%2Alog%28N%29%29
    "O(D*log(N))")) recursively divides the data into nodes defined by a
    centroid C and radius r, such that each point in the node lies
    within the hyper-sphere defined by r and C. The number of candidate
    points for a neighbor search is reduced through use of the triangle
    inequality: |x+y| \<= |x| + |y| (vector mathmatics used here. x+y is
    a vector sum of 2 vectors x & y)
    
      - A single distance calculation between a test point and the
        centroid is sufficient to determine a lower and upper bound on
        the distance to all points within the node.  

  - Because of the spherical geometry of the ball tree nodes, it can
    out-perform a KD-tree in high dimensions, though the actual
    performance is highly dependent on the structure of the training
    data.

### Choice of algorithm is dependent on following things:

  - number of samples N (i.e.??n\_samples) and dimensionality D
    (i.e.??n\_features).
      - KD tree query can be very efficient for D \<20. If you add a new
        dimension to a data then the amount of data needed to make the
        data similarly sparseed will be exponential because you are
        adding a whole new dimension.  
      - For N less than 30, brute force algorithms can be more efficient
        than a tree-based approach.  
      - Tree based algo uses leaf size parameter: this controls the
        number of samples at which a query switches to brute-force. This
        allows both algorithms to approach the efficiency of a
        brute-force computation for small N.  
  - data structure: intrinsic dimensionality of the data and/or sparsity
    of the data.
      - This is related to dimensionality reduction, i.e.??identifying
        Intrinsic dimensionality d\<=D  
  - number of neighbors k requested for a query point.
      - Brute force is unaffected by k.  
      - Ball tree and KD tree query time will become slower as k
        increases because a larger k leads to the necessity to search a
        larger portion of the parameter space.  
      - As k becomes large compared to N, the ability to prune branches
        in a tree-based query is reduced. In this situation, Brute force
        queries can be more efficient.  
      - A higher k value will ignore outliers to the data and a lower
        will give more weight to them. If the k value is too high it
        will not be able to classify the data, so k needs to be
        relatively small.  
  - number of query points.
      - Amortization- doing a lot of work up front to save time later.
        Building a tree that takes long time and then using it for
        querying(takes very less time) a lot of samples is better than
        brute force.  
      - The cost of ball-tree & KD Tree construction becomes negligible
        when amortized over many queries.

<!-- end list -->

``` python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

X = pd.DataFrame({'feature1': [5, 4, 2, 10, 9, 9],
                  'feature2': [2, 3, 2, 10, -1, 9]})
y = pd.DataFrame({'target': ['0', '0', '1', '1', '1', '2']})

knn = KNeighborsClassifier(n_neighbors=3).fit(X, y.to_numpy().ravel())
plot_tree(X, y, knn)
knn.predict(np.atleast_2d([0, 0]))

y = pd.DataFrame([0, 0, 1, 1, 1, 2])
knn = KNeighborsRegressor(n_neighbors=3).fit(X, y)
knn.predict(np.atleast_2d([0, 0]))

1 - knn.score(X, y)
```

## Naive Bayes

[Scikit learn Naive bayes short
documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)

  - [???bag-of-words??? representation refers to turning text into numerical
    feature
    vectors](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
      - We call vectorization the general process of turning a
        collection of text documents into numerical feature vectors This
        specific strategy (tokenization, counting and normalization) is
        called the Bag of Words or ???Bag of n-grams??? representation.
        Documents are described by word occurrences while completely
        ignoring the relative position information of the words in the
        document.
          - tokenizing strings and giving an integer id for each
            possible token, for instance by using white-spaces and
            punctuation as token separators.  
          - counting the occurrences of tokens in each document.  
          - normalizing and weighting with diminishing importance tokens
            that occur in the majority of samples / documents.
  - `CountVectorizer()`
      - Convert a collection of text documents to a matrix of token
        counts  
      - This implementation produces a sparse representation of the
        counts using scipy.sparse.csr\_matrix.  
      - If you do not provide an a-priori dictionary and you do not use
        an analyzer that does some kind of feature selection then the
        number of features will be equal to the vocabulary size found by
        analyzing the data.
  - `MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)`
      - alpha : float, optional (default=1.0)  
    
      - Additive (Laplace/Lidstone) smoothing parameter (0 for no
        smoothing).
        
          - How does the choice of alpha influence our results?
              - High ???? ???dilutes the data??? and gives an underfit model
                (one that tends towards predicting the priors)
              - Low ???? ???empowers the data??? and gives an overfit model
                (one that tends towards predicting the product of the
                conditional probabilities)  
    
      - fit\_prior : boolean, optional (default=True)  
    
      - Whether to learn class prior probabilities or not. If false, a
        uniform prior will be used.
    
      - class\_prior : array-like, size (n\_classes,), optional
        (default=None)  
    
      - Prior probabilities of the classes. If specified the priors are
        not adjusted according to the data.
  - `partial_fit(X, y, classes=None, sample_weight=None)`
      - Incremental fit on a batch of samples.  
      - This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.  
      - This is especially useful when the whole dataset is too big to
        fit in memory at once.  
      - This method has some performance overhead hence it is better to
        call partial\_fit on chunks of data that are as large as
        possible (as long as fitting in the memory budget) to hide the
        overhead.
          - More on [`out of core
            classification`](https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html#sphx-glr-auto-examples-applications-plot-out-of-core-classification-py)
  - `BernoulliNB`
      - this model doesn???t care about word counts, it just cares about
        whether a word exists or not in a document (binary), and how
        many documents there are.
  - `Gaussian`
      - Assume each feature is normally distributed

### Transform a count matrix to a normalized tf or tf-idf representation

  - Tf means term-frequency while tf-idf means term-frequency times
    inverse document-frequency. This is a common term weighting scheme
    in information retrieval, that has also found good use in document
    classification.

  - The goal of using tf-idf instead of the raw frequencies of
    occurrence of a token in a given document is to scale down the
    impact of tokens that occur very frequently in a given corpus and
    that are hence empirically less informative than features that occur
    in a small fraction of the training corpus.

  - The formula that is used to compute the tf-idf for a term t of a
    document d in a document set is tf-idf(t, d) = tf(t, d) \* idf(t),
    and the idf is computed as idf(t) = log \[ n / df(t) \] + 1 (if
    smooth\_idf=False), where n is the total number of documents in the
    document set and df(t) is the document frequency of t; the document
    frequency is the number of documents in the document set that
    contain the term t. The effect of adding ???1??? to the idf in the
    equation above is that terms with zero idf, i.e., terms that occur
    in all documents in a training set, will not be entirely ignored.
    (Note that the idf formula above differs from the standard textbook
    notation that defines the idf as idf(t) = log \[ n / (df(t) + 1)
    \]).

  - If smooth\_idf=True (the default), the constant ???1??? is added to the
    numerator and denominator of the idf as if an extra document was
    seen containing every term in the collection exactly once, which
    prevents zero divisions: idf(d, t) = log \[ (1 + n) / (1 + df(d, t))
    \] + 1.
    
      - More on
        <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer>

<!-- end list -->

``` python
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
X = cv.fit_transform(df['sentence']) # CSR sparse matrix
bow = pd.DataFrame(X.toarray(), columns=sorted(cv.vocabulary_), index=df['sentence']) # works only on small data set
bow

# Multinomial Naive Bayes from lab2
from sklearn.naive_bayes import MultinomialNB
cv = CountVectorizer()
X_train_counts = cv.fit_transform(X_train['review'])
X_test_counts = cv.transform(X_test['review'])

nb = MultinomialNB()
nb.fit(X_train_counts, np.squeeze(y_train))

print("The accuracy score of the Multinomial NB model using CountVectorizer on train data is:", nb.score(X_train_counts, np.squeeze(y_train)), "and on test data is: ", nb.score(X_test_counts, np.squeeze(y_test)))

print(nb.classes_)
nb.predict_proba(X_test_counts)
nb.predict(X_test_counts)


# Using tf-idf representation
cv2 = TfidfVectorizer()
X_train_tfidf = cv2.fit_transform(X_train['review'])
X_test_tfidf = cv2.transform(X_test['review'])

nb2 = MultinomialNB()
nb2.fit(X_train_tfidf, np.squeeze(y_train))

print("The accuracy score of the Multinomial NB model using CountVectorizer on train data is:", nb2.score(X_train_tfidf, np.squeeze(y_train)), "and on test data is: ", nb2.score(X_test_tfidf, np.squeeze(y_test)))
# Multinomial Naive Bayes from lab2

# works the same way
nb = BernoulliNB() # Bernouli
nb = GaussianNB() # Gaussian


##### Scikit
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
##### Scikit
```

## Gradient Boosting

**Great source:
[souce](http://www.chengli.io/tutorials/gradient_boosting.pdf)**

  - `Gradient boosting = Gradient Descent + Boosting`  

  - Gradient Boosting is an extension of adaboost to handle a variety of
    loss functions  

  - Difficulty in using Gradient Boosting for Different Problems:
    `regression ===> classification ===> ranking`

  - You are given ![(x1, y1),(x2, y2), ...,(xn,
    yn)](https://latex.codecogs.com/png.latex?%28x1%2C%20y1%29%2C%28x2%2C%20y2%29%2C%20...%2C%28xn%2C%20yn%29
    "(x1, y1),(x2, y2), ...,(xn, yn)"), and the task is to fit a model
    ![F(x)](https://latex.codecogs.com/png.latex?F%28x%29 "F(x)") to
    minimize square loss. Suppose your friend wants to help you and
    gives you a model F. You check his model and find the model is good
    but not perfect. There are some mistakes: ![F(x1)
    = 0.8](https://latex.codecogs.com/png.latex?F%28x1%29%20%3D%200.8
    "F(x1) = 0.8"), while ![y1
    = 0.9](https://latex.codecogs.com/png.latex?y1%20%3D%200.9
    "y1 = 0.9"), and ![F(x2)
    = 1.4](https://latex.codecogs.com/png.latex?F%28x2%29%20%3D%201.4
    "F(x2) = 1.4") while ![y2
    = 1.3](https://latex.codecogs.com/png.latex?y2%20%3D%201.3
    "y2 = 1.3")??? How can you improve this model?

  - You can add an additional model (regression tree) h to F, so the new
    prediction will be ![F(x) +
    h(x)](https://latex.codecogs.com/png.latex?F%28x%29%20%2B%20h%28x%29
    "F(x) + h(x)").

  - Or, equivalently,  

<!-- end list -->

    h(x1) = y1 ??? F(x1)
    h(x2) = y2 ??? F(x2)
    ...
    h(xn) = yn ??? F(xn)

  - where h(x) are called residuals and you just fit a regression tree h
    to data ![(x1, y1 ??? F(x1)),(x2, y2 ??? F(x2)), ...,(xn, yn ???
    F(xn))](https://latex.codecogs.com/png.latex?%28x1%2C%20y1%20%E2%88%92%20F%28x1%29%29%2C%28x2%2C%20y2%20%E2%88%92%20F%28x2%29%29%2C%20...%2C%28xn%2C%20yn%20%E2%88%92%20F%28xn%29%29
    "(x1, y1 ??? F(x1)),(x2, y2 ??? F(x2)), ...,(xn, yn ??? F(xn))")

  - If the new model F + h is still not satisfactory, we can add another
    regression tree???

### Gradient Descent

  - You have a loss function J which is sometimes sum of squared errors.
    You determine the slope of loss function and using that information
    you move in the direction where the error decreases that is why you
    multiply negative gradient with some value like
    ![\\rho](https://latex.codecogs.com/png.latex?%5Crho "\\rho") which
    can change value depending on the slope or remain constant.

  - Relation between Gradient Descent and Gradient Boosting-  
    ![Relation between Gradient Descent and Gradient
    Boosting](C:\\MyDisk\\MDS\\relation_grad_desc_grad_boost.PNG)

  - For regression with square loss,
    
      - residual ??? negative gradient  
      - fit h to residual ??? fit h to negative gradient  
      - update F based on residual ??? update F based on negative gradient

<!-- end list -->

``` 
    start with an initial model, say, F(x) = mean of all yi
    iterate until converge:
        calculate negative gradients ???g(xi)
        fit a regression tree h to negative gradients ???g(xi)
        F := F + ??h, where ?? = 1
```

**So we are actually updating our model using gradient descent\!**

  - The benefit of formulating this algorithm using gradients is that it
    allows us to consider other loss functions and derive the
    corresponding algorithms in the same way.  
  - Squared loss function not robust to outliers but easy to deal with
    mathematically. Outliers are heavily punished because the error is
    squared.
      - absolute loss & huber loss are more robust to outliers

<!-- end list -->

``` python
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0).fit(X_train, y_train)
```

## Naive Bayes vs.??Logistic regression

  - Both are probabilistic classifiers but Naive Bayes is ???generative???
    and Logistic regression is ???discriminative???  
  - Naive Bayes is a generative model because it???s modeling the joint
    distribution over the features
    ![x](https://latex.codecogs.com/png.latex?x "x") and labels
    ![y](https://latex.codecogs.com/png.latex?y "y").  
  - Logistic regression it directly models the probability
    ![p(y|x)](https://latex.codecogs.com/png.latex?p%28y%7Cx%29
    "p(y|x)")

## Choosing a classifier: Logistic regression vs.??Naive Bayes

  - Naive Bayes has overly strong conditional independence assumptions.
    So not great when features are correlated.
      - If two features are strongly correlated, Naive Bayes will
        overestimating the evidence of that feature.  
  - Logistic regression is much more robust to correlated features
      - If two features are correlated, regression will assign part of
        the weight to one and part to another.  
  - For smaller datasets Naive Bayes is a good choice. Logistic
    regression generally works better on larger datasets and is a common
    default.

## Why people use linear classifiers?

  - Logistic regression is used EVERYWHERE\!  
  - Fast training and testing.
      - Training on huge datasets.  
      - Testing is just computing
        ![w^Tx\_i](https://latex.codecogs.com/png.latex?w%5ETx_i
        "w^Tx_i").  
  - Weights ![w\_j](https://latex.codecogs.com/png.latex?w_j "w_j") are
    easy to understand.
      - It???s how much ![x\_j](https://latex.codecogs.com/png.latex?x_j
        "x_j") changes the prediction and in what direction.  
  - Are somewhat related to neural networks (can be thought of as a
    1-layer neural network)  
  - When I carry out research, logistic regression is the first model I
    try as a baseline.  
  - By default LogisticRegression uses one-vs-rest strategy to deal with
    multi-class
