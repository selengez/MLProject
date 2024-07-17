#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split,RandomizedSearchCV,learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter




# In[ ]:


df = pd.read_csv('secondary_data.csv', delimiter= ';')


# In[ ]:


df.head()


# In[ ]:


######Data Preprocessing####


# In[ ]:


#1- defining null values
null = df.isnull().sum().sum()


# In[ ]:


null


# In[ ]:


missing_values = df.isnull().sum()
missing_values


# In[ ]:


columns_with_missing_values = missing_values[missing_values > 0]
columns_with_missing_values


# In[ ]:


#calculating rate of missing values
percentage = df.isnull().mean()*100
print(percentage)


# In[ ]:


# drop columns have null values more than 40
df = df.drop(columns=percentage[percentage > 40].index)


# In[ ]:


#drop na rows from gill attachment bcs it has high null values
df = df.dropna(subset=['gill-attachment'])
df.isnull().sum()


# In[ ]:


print(df.dtypes, percentage)


# In[ ]:



numerical_cols = ['cap-diameter', 'stem-height', 'stem-width']
categorical_cols = ['gill-attachment', 'cap-surface', 'ring-type']

missing_numerical_cols = [col for col in numerical_cols if col not in df.columns]
missing_categorical_cols = [col for col in categorical_cols if col not in df.columns]

if missing_numerical_cols:
    print(f"Missing numerical columns: {missing_numerical_cols}")
else:
    
    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

if missing_categorical_cols:
    print(f"Missing categorical columns: {missing_categorical_cols}")
else:
   
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(df.head())


# In[ ]:


df.head()


# In[ ]:



row_count, column_count = df.shape

print(f"Number of rows: {row_count}")
print(f"Number of columns: {column_count}")


# In[ ]:


#######EDA######


# In[ ]:


##pie chart for distribution of target value
class_counts = df['class'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['tomato','skyblue'], startangle=140)
plt.title('Distribution of Poisonous and Edible Mushrooms')
plt.show()


# In[ ]:


# distribution for numerical columns
summary_stats = df[numerical_cols].describe()


class_distribution = df['class'].value_counts()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()



# In[ ]:


# distributions for categorical features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(categorical_cols):
    sns.countplot(x=df[col], ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[ ]:


# Correlation matrix for numerical features
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

summary_stats, class_distribution
#after encoding recreate- didnt add report bcs it doesnt contains target value


# In[ ]:


#box plot for num values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    sns.boxplot(x='class', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'Box Plot of {col} by Class')
plt.tight_layout()
plt.show()



# In[ ]:


# bar plot for cat columns
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='class', data=df, ax=axes[i])
    axes[i].set_title(f'Bar Plot of {col} by Class')
plt.tight_layout()
plt.show()


# In[ ]:


#### Data Encoding


# In[ ]:


categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns


# In[ ]:


###to create balanced target variable values the 50-50 split
poisonous = df[df['class'] == 'p']
edible = df[df['class'] == 'e']

#Determine the size of the smaller class
min_size = min(len(poisonous), len(edible))

# the larger class to match the size of the smaller class
poisonous_downsampled = poisonous.sample(min_size, random_state=42)
edible_downsampled = edible.sample(min_size, random_state=42)

# Concatenate the balanced subsets
balanced_df = pd.concat([poisonous_downsampled, edible_downsampled])


balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(balanced_df['class'].value_counts())


print(balanced_df.head())
print(balanced_df.info())
plt.figure(figsize=(6, 4))
sns.countplot(data=balanced_df, x='class')
plt.title('Distribution of Mushroom Classes')
plt.show()


# In[ ]:



X = df.drop(['class'], axis=1)
y = df['class']

# Convert categorical variables to dummy variables (one-hot encoding)
X = pd.get_dummies(X)
X.head()

# Encode the target variable using LabelEncoder
label_encoder = LabelEncoder()
boolean_columns = X.select_dtypes(include=['bool']).columns
X[boolean_columns] = X[boolean_columns].astype(int)
y = label_encoder.fit_transform(y)

#scaling
scaler = MinMaxScaler()
categorical_col = ['cap-diameter', 'stem-height', 'stem-width']


X[categorical_col] = scaler.fit_transform(X[categorical_col])

X[categorical_col].head()


# Split the data as training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.head())


# In[ ]:


#### Classes###


# In[ ]:


class TreeNode:
    def __init__(self, data=None, feature_idx=None, feature_val=None, prediction_probs=None, impurity=None, is_leaf=False):
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.impurity = impurity
        self.is_leaf = is_leaf
        self.left = None
        self.right = None

    def decision_criterion(self, data_point):
        if self.is_leaf:
            return None
        return data_point[self.feature_idx] < self.feature_val

    def node_def(self):
        return f'Feature Index: {self.feature_idx}, Feature Value: {self.feature_val}, Impurity: {self.impurity}, Prediction Probs: {self.prediction_probs}, Is Leaf: {self.is_leaf}'


# In[ ]:



class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=4, min_samples_leaf=1, min_information_gain=0.0, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.criterion = criterion
        self.feature_importances_ = None

    def entropy(self, class_probabilities):
        return sum([-p * np.log2(p) for p in class_probabilities if p > 0])

    def gini_index(self, class_probabilities):
        return 1 - sum([p ** 2 for p in class_probabilities])

    def variance_reduction(self, subsets):
        total_count = sum([len(subset) for subset in subsets])
        if total_count == 0:
            return 0
        total_variance = np.var(np.concatenate(subsets)) if total_count > 0 else 0
        weighted_variances = sum([np.var(subset) * len(subset) / total_count for subset in subsets if len(subset) > 0])
        return total_variance - weighted_variances

    def class_probabilities(self, labels):
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def data_impurity(self, labels):
        class_probs = self.class_probabilities(labels)
        if self.criterion == 'entropy':
            return self.entropy(class_probs)
        elif self.criterion == 'gini':
            return self.gini_index(class_probs)
        elif self.criterion == 'variance_reduction':
            return np.var(labels) if len(labels) > 0 else 0
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def partition_impurity(self, subsets):
        total_count = sum([len(subset) for subset in subsets])
        if self.criterion == 'variance_reduction':
            return self.variance_reduction(subsets)
        return sum([self.data_impurity(subset) * (len(subset) / total_count) for subset in subsets])

    def split(self, data, feature_idx, feature_val):
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2

    def find_best_split(self, data):
        min_part_impurity = float('inf')
        min_impurity_feature_idx = None
        min_impurity_feature_val = None

        for idx in range(data.shape[1] - 1):
            feature_val = np.median(data[:, idx])
            g1, g2 = self.split(data, idx, feature_val)
            part_impurity = self.partition_impurity([g1[:, -1], g2[:, -1]])
            if part_impurity < min_part_impurity:
                min_part_impurity = part_impurity
                min_impurity_feature_idx = idx
                min_impurity_feature_val = feature_val
                g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_impurity_feature_idx, min_impurity_feature_val, min_part_impurity

    def find_label_probs(self, data):
        labels_as_integers = data[:, -1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == label)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def create_tree(self, data, current_depth):
        if current_depth >= self.max_depth or len(np.unique(data[:, -1])) == 1:
            is_leaf = True
            label_probabilities = self.find_label_probs(data)
            node_impurity = self.data_impurity(data[:, -1])
            return TreeNode(data, prediction_probs=label_probabilities, impurity=node_impurity, is_leaf=is_leaf)

        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_impurity = self.find_best_split(data)
        label_probabilities = self.find_label_probs(data)
        node_impurity = self.data_impurity(data[:, -1])
        information_gain = node_impurity - split_impurity

        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(data.shape[1] - 1)
        self.feature_importances_[split_feature_idx] += information_gain
        print(f"Feature {split_feature_idx} importance updated by {information_gain} (total: {self.feature_importances_[split_feature_idx]})")


        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, split_impurity)

        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            node.is_leaf = True
            return node
        elif information_gain < self.min_information_gain:
            node.is_leaf = True
            return node

        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)

        return node

    def predict_one_sample(self, X):
        node = self.tree

        while node and not node.is_leaf:
            if node.decision_criterion(X):
                node = node.left
            else:
                node = node.right

        return node.prediction_probs if node else None

    def fit(self, X, y):
        self.labels_in_train = np.unique(y)
        train_data = np.concatenate((X, np.reshape(y, (-1, 1))), axis=1)
        self.tree = self.create_tree(data=train_data, current_depth=0)
        self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def predict_proba(self, X):
        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X)
        return pred_probs

    def predict(self, X):
        pred_probs = self.predict_proba(X)
        preds = np.argmax(pred_probs, axis=1)
        return preds

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf, "min_information_gain": self.min_information_gain, "criterion": self.criterion}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def print_recursive(self, node, level=0):
        if node is not None:
            self.print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self.print_recursive(node.right, level + 1)

    def print_tree(self):
        self.print_recursive(node=self.tree)


# In[ ]:




def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

dt_entropy = DecisionTree(max_depth=13, min_samples_leaf=4, min_information_gain=0.01, criterion='entropy')
dt_entropy.fit(X_train, y_train)
print("Entropy Criterion:")
evaluate_model(dt_entropy, X_test, y_test)

dt_gini = DecisionTree(max_depth=13, min_samples_leaf=4, min_information_gain=0.01, criterion='gini')
dt_gini.fit(X_train, y_train)
print("\nGini Criterion:")
evaluate_model(dt_gini, X_test, y_test)


# In[ ]:


decision_tree = DecisionTree()

param_grid = {
    'max_depth': [7, 10, 13, 15],
    'min_samples_leaf': [1,2,4,6],
    'min_information_gain': [0, 0.01, 0.1],
    'criterion': ['entropy', 'gini']
}


#  RandomizedSearchCV
random_search = RandomizedSearchCV(decision_tree, param_distributions=param_grid, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=48)

random_search.fit(X_train, y_train)

#  the best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_

# Use the best parameters to train the final model
best_dtc = random_search.best_estimator_
best_dtc.fit(X_train, y_train)


test_accuracy = best_dtc.score(X_test, y_test)
train_accuracy = best_dtc.score(X_train, y_train)
test_accuracy = best_dtc.score(X_test, y_test)


cv_results = random_search.cv_results_

train_sizes, train_scores, val_scores = learning_curve(
    best_dtc, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.figure()
plt.title('Learning Curves')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.legend(loc='best')
plt.show()


# In[ ]:


print(f'Best Parameters: {best_params}')
print(f'Best Cross-validation Accuracy: {best_score * 100:.2f}%')
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print("Cross-Validation Mean Scores: ", cv_results['mean_test_score'])
print("Cross-Validation Std Dev: ", cv_results['std_test_score'])


# In[ ]:





param_grid = {
    'max_depth': [3,5,7,10,12,13],
    'min_samples_leaf': [4,5,6,8,10],
    'min_information_gain': [0.005,0.01,0.05, 0.1],
    'criterion': ['entropy', 'gini']
}


random_search = RandomizedSearchCV(decision_tree, param_distributions=param_grid, n_iter=96, cv=5, scoring='accuracy', n_jobs=-1, random_state=48)


random_search.fit(X_train, y_train)


best_params = random_search.best_params_
best_score = random_search.best_score_


best_dtc = random_search.best_estimator_
best_dtc.fit(X_train, y_train)


train_accuracy = best_dtc.score(X_train, y_train)
test_accuracy = best_dtc.score(X_test, y_test)
cv_results = random_search.cv_results_






train_sizes, train_scores, val_scores = learning_curve(
    best_dtc, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.figure()
plt.title('Learning Curves')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.legend(loc='best')
plt.show()


# In[ ]:



print(f'Best Parameters: {best_params}')
print(f'Best Cross-validation Accuracy: {best_score * 100:.2f}%')
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print("Cross-Validation Mean Scores: ", cv_results['mean_test_score'])
print("Cross-Validation Std Dev: ", cv_results['std_test_score'])


# In[ ]:


decision_tree = DecisionTree()
param_grid = {
    'max_depth': [7, 10, 12, 15],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_information_gain': [0.005, 0.01, 0.05],
    'criterion': ['entropy', 'gini']
}

random_search = RandomizedSearchCV(decision_tree, param_distributions=param_grid, n_iter=92, cv=5, scoring='accuracy', n_jobs=-1, random_state=48)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_score = random_search.best_score_


best_dtc = random_search.best_estimator_
best_dtc.fit(X_train, y_train)

test_accuracy = best_dtc.score(X_test, y_test)

train_accuracy = best_dtc.score(X_train, y_train)
test_accuracy = best_dtc.score(X_test, y_test)



cv_results = random_search.cv_results_


train_sizes, train_scores, val_scores = learning_curve(
    best_dtc, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.figure()
plt.title('Learning Curves')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.legend(loc='best')
plt.show()


# In[ ]:


print(f'Best Parameters: {best_params}')
print(f'Best Cross-validation Accuracy: {best_score * 100:.2f}%')
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print("Cross-Validation Mean Scores: ", cv_results['mean_test_score'])
print("Cross-Validation Std Dev: ", cv_results['std_test_score'])


# In[ ]:


importances = best_dtc.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns


plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[ ]:




