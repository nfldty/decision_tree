import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
#Currently, This model only supports Pandas Dataframe


def entropy(p1):
    if p1 == 0 or p1 == 1:
        return 0
    else:
        return -p1 * np.log2(p1) - (1 - p1) * np.log2((1-p1))


def fraction(y):
    return y.value_counts()[0] / len(y) if len(y.value_counts()) == 2 else 0

    
def split(X, y, feature):

    left_X = pd.DataFrame(columns=X.columns)
    right_X= pd.DataFrame(columns=X.columns)
    left_y = pd.Series(dtype=int)
    right_y = pd.Series(dtype=int)
    
    for i in range(len(X)):
        print(i)
        row = X.iloc[[i]]
        print(row[feature])
        print(row[feature].iloc[0])
        if row[feature].iloc[0] == 1:
            left_X = pd.concat([left_X, row])
            left_y = pd.concat([left_y, pd.Series([y.iloc[i]])])
        else:
            right_X = pd.concat([right_X, row])
            right_y = pd.concat([right_y, pd.Series([y.iloc[i]])])

    left_X.reset_index(inplace=True, drop=True)
    right_X.reset_index(inplace=True, drop=True)
    print(left_X)
    print(right_X)
    left_X.drop(columns=feature, inplace=True)
    left_y.drop(columns=feature, inplace=True)
    left_y.reset_index(inplace=True, drop=True)
    right_y.reset_index(inplace=True, drop=True)
   
    return left_X, left_y, right_X, right_y
    
class DecisionTree:
    
    def __init__(self, impurity_function='entropy', max_depth=10, gain_threshold= 0.01, num_of_examples_threshold=1):
        
        self.impurity_function = impurity_function
        self.max_depth = max_depth
        self.gain_threshold = gain_threshold
        self.num_of_examples_threshold = num_of_examples_threshold
        self.impurity = None
        self.left = None
        self.right = None
        self.major_class = None
        self.split_feature = None
        self.accuracy = None
    
    def fit(self, X, y):

        y_val = y.value_counts()
        if 0 in y_val:
            neg_val = y_val[0]
        else:
            neg_val = 0
        self.major_class = 0 if neg_val > len(y) - neg_val else 1

        m, n = X.shape
        if n <= self.num_of_examples_threshold:
            return

        if self.impurity_function == 'entropy':
            self.impurity = entropy(fraction(y))

        max_info_gain = (float('-inf'), '', None, None)

        for feature in X.columns:
            left_true = 0
            left_total = 0
            right_true = 0
            right_total = 0

            for i in range(m):
                row = X.iloc[i, :]
                if row.loc[feature] == 1:
                    left_true += y.iloc[i]
                    left_total += 1
                else:

                    right_true += y.iloc[i]
                    right_total += 1

            frac_left = left_true / left_total if left_total != 0 else 0
            frac_right = right_true / right_total if right_total != 0 else 0

            impurity_left = entropy(frac_left)
            impurity_right = entropy(frac_right)
            info_gain = self.impurity - (frac_left * impurity_left + frac_right * impurity_right)
            max_info_gain = max(max_info_gain, (info_gain, feature))

        if max_info_gain[0] <= self.gain_threshold:
            return

        self.split_feature = max_info_gain[1]

        left_X, left_y, right_X, right_y = split(X, y, self.split_feature)

        self.left = DecisionTree()
        self.left.fit(left_X, left_y)
        self.right = DecisionTree()
        self.right.fit(right_X, right_y)

    def is_leaf(self):
        return self.split_feature is None

    def predict(self, X):
        result = [0 for i in range(len(X))]
        for i in range(len(X)):
            row = X.iloc[i, :]       
            result[i] = self.predict_row(row)
        return result

    def predict_row(self, x):

        if self.is_leaf():
            return self.major_class
        elif x.loc[self.split_feature] == 0:
            return self.left.predict_row(x)
        else:
            return self.right.predict_row(x)
    
    def __repr__(self):
        return self.toString()

    def toString(self, space=0):
        str = ' ' * space + f'({self.split_feature} , {self.major_class})'
        
        if self.is_leaf():
            return str
        else:
            str += '\n' + self.left.toString(space + 5)
            str += '\n' + self.right.toString(space + 5)
            return str

    def score(self, X, y):
        correct = 0
        result = self.predict(X)
        for i in range(len(result)):
            if result[i] == y[i]:
                correct += 1
        return correct / len(result)


if __name__ == "__main__":
    example_data = pd.DataFrame([[1, 1, 1, 1],
                                 [0, 0, 1, 1],
                                 [0, 1, 0, 0],
                                 [1, 0, 1, 0],
                                 [1, 1, 1, 1],
                                 [1, 1, 0, 1],
                                 [0, 0, 0, 0],
                                 [1, 1, 0, 1],
                                 [0, 1, 0, 0],
                                 [0, 1, 0, 0]], columns=['ear_shape', 'face_shape', 'whisker', 'cat'])

    tree = DecisionTree()
    X = example_data.drop(columns='cat')
    print(X)
    y = example_data['cat']
    tree.fit(X, y)
    print(tree)
    print(tree.predict(example_data))
    print(tree.score(X, y))
    
    
   