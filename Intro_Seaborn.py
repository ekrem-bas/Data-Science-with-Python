###############################
# Seaborn
###############################

###############################
# Categorical Data Visualization
###############################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df['sex'].value_counts().plot(kind='bar') # pandas
plt.show()

###############################
# Numerical Data Visualization
###############################
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist() # pandas
plt.show()