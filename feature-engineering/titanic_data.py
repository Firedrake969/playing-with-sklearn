import re
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, grid_search, ensemble, linear_model, preprocessing, cross_validation, feature_selection

# Thanks to dataquest.io

train = pd.read_csv('./training.csv')
test = pd.read_csv('./testing.csv')

# Set missing ages and fares to median - not always best practice
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())


# Encode categorical variables into numbers
# Can also use preprocessing.LabelEncoder()
train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
test.loc[test['Sex'] == 'male', 'Sex'] = 0
test.loc[test['Sex'] == 'female', 'Sex'] = 1

train['Embarked'] = train['Embarked'].fillna('S')
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2

test['Embarked'] = test['Embarked'].fillna('S')
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'C', 'Embarked'] = 1
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 2

# Generating a familysize column
train['FamilySize'] = train['SibSp'] + train['Parch']

# The .apply method generates a new series
train['NameLength'] = train['Name'].apply(lambda x: len(x))


# Generating a familysize column
test['FamilySize'] = test['SibSp'] + test['Parch']

# The .apply method generates a new series
test['NameLength'] = test['Name'].apply(lambda x: len(x))

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

# Get all the titles and print how often each one occurs.
titles = train["Name"].apply(get_title)
for k,v in title_mapping.items():
    titles[titles == k] = v
# Add in the title column.
train["Title"] = titles

# Get all the titles and print how often each one occurs.
titles = test["Name"].apply(get_title)
for k,v in title_mapping.items():
    titles[titles == k] = v
# Add in the title column.
test["Title"] = titles

family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

family_ids = train.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[train["FamilySize"] < 3] = -1

train["FamilyId"] = family_ids

family_ids = test.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[test["FamilySize"] < 3] = -1

test["FamilyId"] = family_ids