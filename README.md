# Ex-06-Feature-Transformation

DATE:

GITHUB LINK: https://github.com/saran7d/Ex-06-Feature-Transformation.git

COLAB LINK: https://colab.research.google.com/drive/1tReShgYYCUREBJTE8TJbLLu_p6HKVPv6?usp=sharing
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the features of the data set

STEP 4
Save the data to the file

# CODE

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```

# OUTPUT

# Dataset:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/A.png)

# Head:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/B.png)

# Null data:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/C.png)

# Information:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/D.png)

# Description:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/E.png)

# Highly Positive Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/F.png)

# Highly Negative Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/G.png)

# Moderate Positive Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/H.png)

# Moderate Negative Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/I.png)

# Log of Highly Positive Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/J.png)

# Log of Moderate Positive Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/K.png)

# Reciprocal of Highly Positive Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/L.png)

# Square root tranformation:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/M.png)

# Power transformation of Moderate Positive Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/N.png)

# Power transformation of Moderate Negative Skew:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/O.png)

# Quantile transformation:

![](https://github.com/saran7d/Ex-06-Feature-Transformation/blob/main/P.png)

# Result

Thus, Feature transformation is performed and executed successfully for the given dataset.
