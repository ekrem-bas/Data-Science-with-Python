########################################
# Pandas
########################################
# Pandas Series
# Reading Data
# Quick Look at Data
# Selection in Pandas
# Aggregation & Grouping
# Apply & Lambda
# Join
########################################
# Pandas Series
########################################
import numpy as np
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5]) # Gives array with indexes
type(s) # pandas.core.series.Series
s.index # RangeIndex(start=0, stop=5, step=1)
s.dtype # dtype('int64')
s.size # 5
s.ndim # 1
s.values # array([10, 77, 12,  4,  5])
type(s.values) # <class 'numpy.ndarray'>
s.head(3) # first 3 elements and their indexes and the type of them (dtype: int64)
s.tail(3) # last 3 elements and their indexes and the type of them (dtype: int64)

########################################
# Reading Data
########################################
df = pd.read_csv("Datasets/advertising.csv")
df.head()

########################################
# Quick Look at Data
########################################
import seaborn as sns

df = sns.load_dataset(name= "titanic")
df.head(n=5) # first n rows
df.tail() # last 5 rows
df.shape # shape of dataframe
df.info() # information about dataframe such as type of variables, columns, non-null count etc.
df.columns # names of columns
df.index # index information
df.describe().T # some statistical data about dataframe (mean, std, min, max, ...) - T for transpose (good appearance)
df.isnull().values.any() # is there any null values on dataframe (T or F)
df.isnull().sum() # show number of null values of each column of dataframe
df['sex'].value_counts() # show how many people of each gender are in the dataframe

########################################
# Selection in Pandas (Rows)
########################################
import seaborn as sns
import pandas as pd

df = sns.load_dataset(name= "titanic")
df.head()
df.index # indexes of dataframe
df[0:13] # slicing on dataframe show 0 to 13 (13th row not included) rows
df.drop(index=0, axis='rows').head() # delete first row of the dataframe (not saved)

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis='rows').head() # delete given rows

# how to save it
# 1. df = df.drop(delete_indexes, axis='rows')
# 2. use inplace | df.drop(delete_indexes, axis=0, inplace=True)

########################################
# Convert Column Into an Index
########################################
df['age'].head()
df.age.head() # same thing

df.index = df['age'] # set indexes to age values
df.drop('age', axis='columns', inplace=True) # save it
df.head()

########################################
# Convert Index Into a Column
########################################
# add new column to dataframe
df['age'] = df.index
df.head()

# delete the age column to show second way
df.drop('age', axis='columns', inplace=True)
# 2nd way use reset_index() (it will reset the indexes and insert it those values as a column to dataframe)
df = df.reset_index()
df.head()

########################################
# Selection in Pandas (Columns)
########################################
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None) # to see all columns of the dataframe when using head(), tail() etc.
df = sns.load_dataset(name= "titanic")
df.head()

'age' in df # is there any column related to the given name (T or F)
df['age'].head() # get given column of dataframe
df.age.head() # same thing

type(df['age'].head()) # type of the column (pandas.Series)
type(df[['age']].head()) # use this for general purposes like functions etc. when dataframe is needed

df[['age', 'alive']] # get more than one column of the dataframe

col_names = ['age', 'adult_male', 'alive']
df[col_names]

df['age2'] = df['age'] * 2 # add new column to dataframe
df['age3'] = df['age'] / 3

df.drop('age3', axis='columns', inplace=True) # delete the given column from dataframe and save it with inplace command

df.drop(col_names, axis='columns') # delete multiple columns from dataframe (not saved)

df.loc[:, df.columns.str.contains('age')].head() # get columns that contain 'age' in their names
df.loc[:, ~df.columns.str.contains('age')].head() # get columns that does not contain 'age' in their names

########################################
# iloc & loc
########################################
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)
df = sns.load_dataset(name= "titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3] # get 0 to 3 rows (3 not included)
df.iloc[0, 0] # get the item in the first row and column

# loc: label based selection
df.loc[0:3] # searches given labels and gets them (search 0, 1, 2, 3 in index)
df.iloc[0:3, 'age'] # wrong syntax (cannot use label on iloc)
df.iloc[0:3, 0:3] # correct one (gives first 3 rows and first 3 columns)
df.loc[0:3, 'age'] # get 0 to 3 rows (total 4 rows) and age column
df.loc[0:3, ['age', 'alive', 'embarked']]  # get multiple columns with label based selection

########################################
# Conditional Selection
########################################
import seaborn as sns
import pandas as pd
df = sns.load_dataset(name= "titanic")
df.head()

df[df['age'] > 50].head() # get dataframe with age > 50 conditional
df[df['age'] > 50]['age'].count() # 64 people over 50 years old

df.loc[df['age'] > 50, ['class', 'age']].head() # get people over 50 years old with their classes

df.loc[(df['age'] > 50) & (df['sex'] == 'male'), ['class', 'age']].head() # multiple condition use ()

df_new = df.loc[(df['age'] > 50) # or operation for some column
       & (df['sex'] == 'male')
       & ((df['embark_town'] == 'Southampton') | (df['embark_town'] == 'Cherbourg')),
        ['class', 'age', 'embark_town']]

df_new.head() # (cherbourg or southampton) and (male) and (age > 50)

########################################
# Aggregation & Grouping (count, first, last, mean, median, min, max, std, var, sum, pivot table)
########################################
import seaborn as sns
import pandas as pd
df = sns.load_dataset(name= "titanic")
df.head()

df.groupby('sex')['age'].mean() # get mean of ages for each gender

df.groupby('sex').agg({'age' : 'mean'}) # get mean of ages by agg

df.groupby('sex').agg({'age' : ['mean', 'sum']}) # get mean and summation of ages by agg

df.groupby('sex').agg({'age' : ['mean', 'sum'],  # group by 'sex' and get mean, summation of ages and mean of survived
                       'survived': 'mean'})

df.groupby(['sex', 'embark_town']).agg({'age' : ['mean'], # group by 'sex' and 'embark_town'
                       'survived': 'mean'})

df.groupby(['sex', 'embark_town', 'class']).agg({'age' : ['mean'], # group by 'sex', 'embark_town', 'class'
                                        'survived': 'mean'})

# group by 'sex', 'embark_town', 'class' and get mean of ages, rate of survives and gender count
df.groupby(['sex', 'embark_town', 'class']).agg({
    'age' : 'mean',
    'survived': 'mean',
    'sex' : 'count'
})

########################################
# Pivot Table
########################################
import seaborn as sns
import pandas as pd
df = sns.load_dataset(name= "titanic")
df.head()

df.pivot_table('survived', "sex", "embarked", aggfunc='count')

df.pivot_table('survived', "sex", ["embarked", "class"])

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.head()

df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex", ["new_age", "class"])

########################################
# Apply & Lambda
########################################
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset(name= "titanic")
df.head()

df['age2'] = df['age'] * 2
df['age3'] = df['age'] * 5
df.head()

df[["age", "age2", "age3"]].apply(lambda x: x / 10).head() # divide all columns by 10 that contains 'age'

df.loc[:, df.columns.str.contains('age')].apply(lambda x: x / 10).head() # same thing but more useful

df.loc[:, df.columns.str.contains('age')].apply(lambda x: (x - x.mean()) / x.std()).head() # complex mathematical operation

def standard_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains('age')].apply(standard_scaler) # same thing with function defining

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains('age')].apply(standard_scaler)
df.head()

########################################
# Join Operations (Concat)
########################################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)

########################################
# Join Operations (Merge)
########################################
df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group": ["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees": ["mark", "john", "dennis", "maria"],
                    "start_date" : [2010, 2009, 2014, 2019]})
pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Purpose: We Want to access manager information for each employee
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"group" : ["accounting", "engineering", "hr"],
                    "manager" : ["Caner", "Mustafa", "Berkcan"]})

pd.merge(df3, df4)