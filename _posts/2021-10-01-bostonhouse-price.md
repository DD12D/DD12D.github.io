---
layout: post
title: House price
subtitle: 마크다운 문법을 이용하여 글을 작성
categories: markdown
tags: [test]
---

```
from math import ceil, sqrt as root
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNetCV, LassoCV 
from sklearn.ensemble import RandomForestRegressor
import warnings
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv
/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
/kaggle/input/house-prices-advanced-regression-techniques/train.csv
/kaggle/input/house-prices-advanced-regression-techniques/test.csv
/kaggle/input/submission/submission.csv
```

# Exploratory Data Analysis

### We start by loading and viewing the data available to us.

In [2]:

```
train_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_set.head()
```

Out[2]:

|      | Id   | MSSubClass | MSZoning | LotFrontage | LotArea | Street | Alley | LotShape | LandContour | Utilities | ...  | PoolArea | PoolQC | Fence | MiscFeature | MiscVal | MoSold | YrSold | SaleType | SaleCondition | SalePrice |
| :--- | :--- | :--------- | :------- | :---------- | :------ | :----- | :---- | :------- | :---------- | :-------- | :--- | :------- | :----- | :---- | :---------- | :------ | :----- | :----- | :------- | :------------ | :-------- |
| 0    | 1    | 60         | RL       | 65.0        | 8450    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ...  | 0        | NaN    | NaN   | NaN         | 0       | 2      | 2008   | WD       | Normal        | 208500    |
| 1    | 2    | 20         | RL       | 80.0        | 9600    | Pave   | NaN   | Reg      | Lvl         | AllPub    | ...  | 0        | NaN    | NaN   | NaN         | 0       | 5      | 2007   | WD       | Normal        | 181500    |
| 2    | 3    | 60         | RL       | 68.0        | 11250   | Pave   | NaN   | IR1      | Lvl         | AllPub    | ...  | 0        | NaN    | NaN   | NaN         | 0       | 9      | 2008   | WD       | Normal        | 223500    |
| 3    | 4    | 70         | RL       | 60.0        | 9550    | Pave   | NaN   | IR1      | Lvl         | AllPub    | ...  | 0        | NaN    | NaN   | NaN         | 0       | 2      | 2006   | WD       | Abnorml       | 140000    |
| 4    | 5    | 60         | RL       | 84.0        | 14260   | Pave   | NaN   | IR1      | Lvl         | AllPub    | ...  | 0        | NaN    | NaN   | NaN         | 0       | 12     | 2008   | WD       | Normal        | 250000    |

5 rows × 81 columns

In [3]:

```
test_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_set.head()
```

Out[3]:

|      | Id   | MSSubClass | MSZoning | LotFrontage | LotArea | Street | Alley | LotShape | LandContour | Utilities | ...  | ScreenPorch | PoolArea | PoolQC | Fence | MiscFeature | MiscVal | MoSold | YrSold | SaleType | SaleCondition |
| :--- | :--- | :--------- | :------- | :---------- | :------ | :----- | :---- | :------- | :---------- | :-------- | :--- | :---------- | :------- | :----- | :---- | :---------- | :------ | :----- | :----- | :------- | :------------ |
| 0    | 1461 | 20         | RH       | 80.0        | 11622   | Pave   | NaN   | Reg      | Lvl         | AllPub    | ...  | 120         | 0        | NaN    | MnPrv | NaN         | 0       | 6      | 2010   | WD       | Normal        |
| 1    | 1462 | 20         | RL       | 81.0        | 14267   | Pave   | NaN   | IR1      | Lvl         | AllPub    | ...  | 0           | 0        | NaN    | NaN   | Gar2        | 12500   | 6      | 2010   | WD       | Normal        |
| 2    | 1463 | 60         | RL       | 74.0        | 13830   | Pave   | NaN   | IR1      | Lvl         | AllPub    | ...  | 0           | 0        | NaN    | MnPrv | NaN         | 0       | 3      | 2010   | WD       | Normal        |
| 3    | 1464 | 60         | RL       | 78.0        | 9978    | Pave   | NaN   | IR1      | Lvl         | AllPub    | ...  | 0           | 0        | NaN    | NaN   | NaN         | 0       | 6      | 2010   | WD       | Normal        |
| 4    | 1465 | 120        | RL       | 43.0        | 5005    | Pave   | NaN   | IR1      | HLS         | AllPub    | ...  | 144         | 0        | NaN    | NaN   | NaN         | 0       | 1      | 2010   | WD       | Normal        |

5 rows × 80 columns

#### The train set consists of 1460 rows and 81 features while the test set consists of 1459 rows and 80 features. What feature is missing in the test set?

In [4]:

```
set(train_set.columns) - set(test_set.columns)
```

Out[4]:

```
{'SalePrice'}
```

In [5]:

```
sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
```

#### Apparently, the `SalePrice` for the test_set is what we are to predict.

#### How about we take a closer look into the type of prices available in the dataset?

In [6]:

```
train_set["SalePrice"].describe()
```

Out[6]:

```
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64
```

In [7]:

```
# Let's take a look at the distribution of the SalePrice sns.distplot(train_set['SalePrice'] , fit=norm);(mu, sigma) = norm.fit(train_set['SalePrice'])plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],            loc='best')plt.ylabel('Frequency')plt.title('SalePrice distribution')
```

Out[7]:

```
Text(0.5, 1.0, 'SalePrice distribution')
```

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___11_1.png)

#### Looks like the price is left skewed so we work with the log of the prices.

In [8]:

```
price = np.log1p(train_set["SalePrice"]) sns.distplot(price , fit=norm);(mu, sigma) = norm.fit(price)plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],            loc='best')plt.ylabel('Frequency')plt.title('SalePrice distribution')
```

Out[8]:

```
Text(0.5, 1.0, 'SalePrice distribution')
```

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___13_1.png)

#### Now that looks more like it...

#### What year did we have the most sales?

In [9]:

```
plt.figure(figsize=(18,10))plots = train_set["YrSold"].value_counts().plot(kind="bar")for bar in plots.patches:    plots.annotate(format(bar.get_height(), '.0f'),                   (bar.get_x() + bar.get_width() / 2,                    bar.get_height()), ha='center', va='center',                   size=15, xytext=(0, 8),                   textcoords='offset points')plt.title("Houses Sold over the Years")plt.ylabel("Number")plt.show()
```

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___16_0.png)

#### Looks like we had more house sales in 2009 and the least sales in 2010.

#### What type of sale was the most purchased?

In [10]:

```
plt.figure(figsize=(18,10))plots = train_set["SaleType"].value_counts().plot(kind="bar")for bar in plots.patches:    plots.annotate(format(bar.get_height(), '.0f'),                   (bar.get_x() + bar.get_width() / 2,                    bar.get_height()), ha='center', va='center',                   size=15, xytext=(0, 8),                   textcoords='offset points')plt.title("Most purchased Sale Type")plt.ylabel("Frequency")plt.xlabel("Sale Type")plt.show()
```

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___18_0.png)

#### How about another look at the data

In [11]:

```
train_set.groupby(["YrSold", "MoSold"]).count()
```

Out[11]:

|        |        | Id   | MSSubClass | MSZoning | LotFrontage | LotArea | Street | Alley | LotShape | LandContour | Utilities | ...  | 3SsnPorch | ScreenPorch | PoolArea | PoolQC | Fence | MiscFeature | MiscVal | SaleType | SaleCondition | SalePrice |
| :----- | :----- | :--- | :--------- | :------- | :---------- | :------ | :----- | :---- | :------- | :---------- | :-------- | :--- | :-------- | :---------- | :------- | :----- | :---- | :---------- | :------ | :------- | :------------ | :-------- |
| YrSold | MoSold |      |            |          |             |         |        |       |          |             |           |      |           |             |          |        |       |             |         |          |               |           |
| 2006   | 1      | 10   | 10         | 10       | 7           | 10      | 10     | 1     | 10       | 10          | 10        | ...  | 10        | 10          | 10       | 1      | 3     | 0           | 10      | 10       | 10            | 10        |
| 2      | 9      | 9    | 9          | 7        | 9           | 9       | 0      | 9     | 9        | 9           | ...       | 9    | 9         | 9           | 0        | 2      | 0     | 9           | 9       | 9        | 9             |           |
| 3      | 25     | 25   | 25         | 22       | 25          | 25      | 1      | 25    | 25       | 25          | ...       | 25   | 25        | 25          | 1        | 4      | 1     | 25          | 25      | 25       | 25            |           |
| 4      | 27     | 27   | 27         | 23       | 27          | 27      | 6      | 27    | 27       | 27          | ...       | 27   | 27        | 27          | 0        | 2      | 1     | 27          | 27      | 27       | 27            |           |
| 5      | 38     | 38   | 38         | 31       | 38          | 38      | 2      | 38    | 38       | 38          | ...       | 38   | 38        | 38          | 0        | 5      | 1     | 38          | 38      | 38       | 38            |           |
| 6      | 48     | 48   | 48         | 40       | 48          | 48      | 3      | 48    | 48       | 48          | ...       | 48   | 48        | 48          | 0        | 10     | 0     | 48          | 48      | 48       | 48            |           |
| 7      | 67     | 67   | 67         | 59       | 67          | 67      | 2      | 67    | 67       | 67          | ...       | 67   | 67        | 67          | 1        | 12     | 3     | 67          | 67      | 67       | 67            |           |
| 8      | 23     | 23   | 23         | 20       | 23          | 23      | 2      | 23    | 23       | 23          | ...       | 23   | 23        | 23          | 1        | 4      | 0     | 23          | 23      | 23       | 23            |           |
| 9      | 15     | 15   | 15         | 15       | 15          | 15      | 1      | 15    | 15       | 15          | ...       | 15   | 15        | 15          | 0        | 2      | 1     | 15          | 15      | 15       | 15            |           |
| 10     | 24     | 24   | 24         | 22       | 24          | 24      | 1      | 24    | 24       | 24          | ...       | 24   | 24        | 24          | 0        | 2      | 1     | 24          | 24      | 24       | 24            |           |
| 11     | 16     | 16   | 16         | 11       | 16          | 16      | 0      | 16    | 16       | 16          | ...       | 16   | 16        | 16          | 0        | 3      | 0     | 16          | 16      | 16       | 16            |           |
| 12     | 12     | 12   | 12         | 8        | 12          | 12      | 0      | 12    | 12       | 12          | ...       | 12   | 12        | 12          | 0        | 3      | 0     | 12          | 12      | 12       | 12            |           |
| 2007   | 1      | 13   | 13         | 13       | 12          | 13      | 13     | 0     | 13       | 13          | 13        | ...  | 13        | 13          | 13       | 0      | 4     | 0           | 13      | 13       | 13            | 13        |
| 2      | 8      | 8    | 8          | 7        | 8           | 8       | 0      | 8     | 8        | 8           | ...       | 8    | 8         | 8           | 0        | 1      | 0     | 8           | 8       | 8        | 8             |           |
| 3      | 23     | 23   | 23         | 15       | 23          | 23      | 1      | 23    | 23       | 23          | ...       | 23   | 23        | 23          | 0        | 4      | 0     | 23          | 23      | 23       | 23            |           |
| 4      | 23     | 23   | 23         | 20       | 23          | 23      | 1      | 23    | 23       | 23          | ...       | 23   | 23        | 23          | 0        | 6      | 3     | 23          | 23      | 23       | 23            |           |
| 5      | 43     | 43   | 43         | 34       | 43          | 43      | 1      | 43    | 43       | 43          | ...       | 43   | 43        | 43          | 0        | 7      | 1     | 43          | 43      | 43       | 43            |           |
| 6      | 59     | 59   | 59         | 43       | 59          | 59      | 5      | 59    | 59       | 59          | ...       | 59   | 59        | 59          | 0        | 16     | 2     | 59          | 59      | 59       | 59            |           |
| 7      | 51     | 51   | 51         | 43       | 51          | 51      | 3      | 51    | 51       | 51          | ...       | 51   | 51        | 51          | 1        | 11     | 3     | 51          | 51      | 51       | 51            |           |
| 8      | 40     | 40   | 40         | 32       | 40          | 40      | 4      | 40    | 40       | 40          | ...       | 40   | 40        | 40          | 0        | 6      | 3     | 40          | 40      | 40       | 40            |           |
| 9      | 11     | 11   | 11         | 10       | 11          | 11      | 1      | 11    | 11       | 11          | ...       | 11   | 11        | 11          | 0        | 1      | 0     | 11          | 11      | 11       | 11            |           |
| 10     | 16     | 16   | 16         | 12       | 16          | 16      | 1      | 16    | 16       | 16          | ...       | 16   | 16        | 16          | 0        | 2      | 0     | 16          | 16      | 16       | 16            |           |
| 11     | 24     | 24   | 24         | 21       | 24          | 24      | 2      | 24    | 24       | 24          | ...       | 24   | 24        | 24          | 0        | 2      | 0     | 24          | 24      | 24       | 24            |           |
| 12     | 18     | 18   | 18         | 17       | 18          | 18      | 0      | 18    | 18       | 18          | ...       | 18   | 18        | 18          | 0        | 4      | 0     | 18          | 18      | 18       | 18            |           |
| 2008   | 1      | 13   | 13         | 13       | 12          | 13      | 13     | 0     | 13       | 13          | 13        | ...  | 13        | 13          | 13       | 1      | 3     | 0           | 13      | 13       | 13            | 13        |
| 2      | 10     | 10   | 10         | 10       | 10          | 10      | 1      | 10    | 10       | 10          | ...       | 10   | 10        | 10          | 0        | 2      | 0     | 10          | 10      | 10       | 10            |           |
| 3      | 18     | 18   | 18         | 13       | 18          | 18      | 4      | 18    | 18       | 18          | ...       | 18   | 18        | 18          | 0        | 1      | 0     | 18          | 18      | 18       | 18            |           |
| 4      | 26     | 26   | 26         | 22       | 26          | 26      | 4      | 26    | 26       | 26          | ...       | 26   | 26        | 26          | 0        | 6      | 0     | 26          | 26      | 26       | 26            |           |
| 5      | 38     | 38   | 38         | 29       | 38          | 38      | 1      | 38    | 38       | 38          | ...       | 38   | 38        | 38          | 0        | 11     | 1     | 38          | 38      | 38       | 38            |           |
| 6      | 51     | 51   | 51         | 41       | 51          | 51      | 9      | 51    | 51       | 51          | ...       | 51   | 51        | 51          | 0        | 10     | 1     | 51          | 51      | 51       | 51            |           |
| 7      | 49     | 49   | 49         | 39       | 49          | 49      | 3      | 49    | 49       | 49          | ...       | 49   | 49        | 49          | 1        | 6      | 1     | 49          | 49      | 49       | 49            |           |
| 8      | 29     | 29   | 29         | 27       | 29          | 29      | 2      | 29    | 29       | 29          | ...       | 29   | 29        | 29          | 0        | 3      | 3     | 29          | 29      | 29       | 29            |           |
| 9      | 17     | 17   | 17         | 14       | 17          | 17      | 1      | 17    | 17       | 17          | ...       | 17   | 17        | 17          | 0        | 4      | 1     | 17          | 17      | 17       | 17            |           |
| 10     | 22     | 22   | 22         | 19       | 22          | 22      | 0      | 22    | 22       | 22          | ...       | 22   | 22        | 22          | 0        | 7      | 1     | 22          | 22      | 22       | 22            |           |
| 11     | 17     | 17   | 17         | 15       | 17          | 17      | 1      | 17    | 17       | 17          | ...       | 17   | 17        | 17          | 0        | 2      | 1     | 17          | 17      | 17       | 17            |           |
| 12     | 14     | 14   | 14         | 13       | 14          | 14      | 1      | 14    | 14       | 14          | ...       | 14   | 14        | 14          | 0        | 3      | 0     | 14          | 14      | 14       | 14            |           |
| 2009   | 1      | 12   | 12         | 12       | 9           | 12      | 12     | 1     | 12       | 12          | 12        | ...  | 12        | 12          | 12       | 0      | 0     | 0           | 12      | 12       | 12            | 12        |
| 2      | 10     | 10   | 10         | 8        | 10          | 10      | 1      | 10    | 10       | 10          | ...       | 10   | 10        | 10          | 0        | 1      | 0     | 10          | 10      | 10       | 10            |           |
| 3      | 19     | 19   | 19         | 15       | 19          | 19      | 1      | 19    | 19       | 19          | ...       | 19   | 19        | 19          | 0        | 4      | 1     | 19          | 19      | 19       | 19            |           |
| 4      | 26     | 26   | 26         | 22       | 26          | 26      | 1      | 26    | 26       | 26          | ...       | 26   | 26        | 26          | 0        | 5      | 1     | 26          | 26      | 26       | 26            |           |
| 5      | 37     | 37   | 37         | 30       | 37          | 37      | 2      | 37    | 37       | 37          | ...       | 37   | 37        | 37          | 0        | 7      | 1     | 37          | 37      | 37       | 37            |           |
| 6      | 59     | 59   | 59         | 45       | 59          | 59      | 3      | 59    | 59       | 59          | ...       | 59   | 59        | 59          | 0        | 7      | 1     | 59          | 59      | 59       | 59            |           |
| 7      | 61     | 61   | 61         | 54       | 61          | 61      | 4      | 61    | 61       | 61          | ...       | 61   | 61        | 61          | 0        | 11     | 0     | 61          | 61      | 61       | 61            |           |
| 8      | 30     | 30   | 30         | 24       | 30          | 30      | 3      | 30    | 30       | 30          | ...       | 30   | 30        | 30          | 0        | 9      | 1     | 30          | 30      | 30       | 30            |           |
| 9      | 20     | 20   | 20         | 18       | 20          | 20      | 1      | 20    | 20       | 20          | ...       | 20   | 20        | 20          | 0        | 2      | 1     | 20          | 20      | 20       | 20            |           |
| 10     | 27     | 27   | 27         | 23       | 27          | 27      | 1      | 27    | 27       | 27          | ...       | 27   | 27        | 27          | 0        | 8      | 2     | 27          | 27      | 27       | 27            |           |
| 11     | 22     | 22   | 22         | 16       | 22          | 22      | 1      | 22    | 22       | 22          | ...       | 22   | 22        | 22          | 0        | 10     | 3     | 22          | 22      | 22       | 22            |           |
| 12     | 15     | 15   | 15         | 9        | 15          | 15      | 1      | 15    | 15       | 15          | ...       | 15   | 15        | 15          | 0        | 3      | 0     | 15          | 15      | 15       | 15            |           |
| 2010   | 1      | 10   | 10         | 10       | 9           | 10      | 10     | 1     | 10       | 10          | 10        | ...  | 10        | 10          | 10       | 0      | 2     | 1           | 10      | 10       | 10            | 10        |
| 2      | 15     | 15   | 15         | 12       | 15          | 15      | 0      | 15    | 15       | 15          | ...       | 15   | 15        | 15          | 0        | 6      | 0     | 15          | 15      | 15       | 15            |           |
| 3      | 21     | 21   | 21         | 17       | 21          | 21      | 1      | 21    | 21       | 21          | ...       | 21   | 21        | 21          | 0        | 5      | 3     | 21          | 21      | 21       | 21            |           |
| 4      | 39     | 39   | 39         | 32       | 39          | 39      | 0      | 39    | 39       | 39          | ...       | 39   | 39        | 39          | 0        | 6      | 0     | 39          | 39      | 39       | 39            |           |
| 5      | 48     | 48   | 48         | 38       | 48          | 48      | 1      | 48    | 48       | 48          | ...       | 48   | 48        | 48          | 0        | 14     | 5     | 48          | 48      | 48       | 48            |           |
| 6      | 36     | 36   | 36         | 30       | 36          | 36      | 3      | 36    | 36       | 36          | ...       | 36   | 36        | 36          | 0        | 6      | 4     | 36          | 36      | 36       | 36            |           |
| 7      | 6      | 6    | 6          | 5        | 6           | 6       | 0      | 6     | 6        | 6           | ...       | 6    | 6         | 6           | 0        | 1      | 1     | 6           | 6       | 6        | 6             |           |

55 rows × 79 columns

#### Well, a closer look at the charts and data just shows that nothing spectacular happened in 2009. The number of purchases increase by the year and it is safe to say that the data available to us at the moment does not have complete records for the whole year 2010, only records up to the month of July in 2010 are available.

`Alley`, `PoolQC`, `Fence`, `MiscFeature` have the highest number of NaN values, we can fill this for EDA as nan means that feature is unavailable for the apartment. Other columns can be filled with their means if numerical and modes if categorical. However, I'll drop `MiscFeature`.

# Features Processing

#### In order to determine the features that play important roles in the determination of the sales price, we can start by checking the features that have strong positive or negative correlations with the sales price.

In [12]:

```
#correlation matrixcorr = train_set.corr()f, ax = plt.subplots(figsize=(16, 10))sns.heatmap(corr)
```

Out[12]:

```
<AxesSubplot:>
```

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___25_1.png)

#### From the above matrix, we can see that `OverallQual` has the highest positive correlation with `SalesPrice` followed closely by `GrLivarea` while `OverallCond`, `KitchenAbvGr`, `EnclosedPorch` have the lowest correlations with `SalesPrice`. This simply means that the features with high correlation can be very useful in predicting prices and we should pay close attention to them. We could drop the ones with negative correlation or no correlation at all to the sales price but i choose not to as information, no matter how little can still be gotten from those.

In [13]:

```
train_set.isnull().sum()[train_set.isnull().sum() > 0]
```

Out[13]:

```
LotFrontage      259Alley           1369MasVnrType         8MasVnrArea         8BsmtQual          37BsmtCond          37BsmtExposure      38BsmtFinType1      37BsmtFinType2      38Electrical         1FireplaceQu      690GarageType        81GarageYrBlt       81GarageFinish      81GarageQual        81GarageCond        81PoolQC          1453Fence           1179MiscFeature     1406dtype: int64
```

In [14]:

```
train_set.drop(["PoolQC", "Fence", "MiscFeature", "Alley"], axis=1)
```

Out[14]:

|      | Id   | MSSubClass | MSZoning | LotFrontage | LotArea | Street | LotShape | LandContour | Utilities | LotConfig | ...  | EnclosedPorch | 3SsnPorch | ScreenPorch | PoolArea | MiscVal | MoSold | YrSold | SaleType | SaleCondition | SalePrice |
| :--- | :--- | :--------- | :------- | :---------- | :------ | :----- | :------- | :---------- | :-------- | :-------- | :--- | :------------ | :-------- | :---------- | :------- | :------ | :----- | :----- | :------- | :------------ | :-------- |
| 0    | 1    | 60         | RL       | 65.0        | 8450    | Pave   | Reg      | Lvl         | AllPub    | Inside    | ...  | 0             | 0         | 0           | 0        | 0       | 2      | 2008   | WD       | Normal        | 208500    |
| 1    | 2    | 20         | RL       | 80.0        | 9600    | Pave   | Reg      | Lvl         | AllPub    | FR2       | ...  | 0             | 0         | 0           | 0        | 0       | 5      | 2007   | WD       | Normal        | 181500    |
| 2    | 3    | 60         | RL       | 68.0        | 11250   | Pave   | IR1      | Lvl         | AllPub    | Inside    | ...  | 0             | 0         | 0           | 0        | 0       | 9      | 2008   | WD       | Normal        | 223500    |
| 3    | 4    | 70         | RL       | 60.0        | 9550    | Pave   | IR1      | Lvl         | AllPub    | Corner    | ...  | 272           | 0         | 0           | 0        | 0       | 2      | 2006   | WD       | Abnorml       | 140000    |
| 4    | 5    | 60         | RL       | 84.0        | 14260   | Pave   | IR1      | Lvl         | AllPub    | FR2       | ...  | 0             | 0         | 0           | 0        | 0       | 12     | 2008   | WD       | Normal        | 250000    |
| ...  | ...  | ...        | ...      | ...         | ...     | ...    | ...      | ...         | ...       | ...       | ...  | ...           | ...       | ...         | ...      | ...     | ...    | ...    | ...      | ...           | ...       |
| 1455 | 1456 | 60         | RL       | 62.0        | 7917    | Pave   | Reg      | Lvl         | AllPub    | Inside    | ...  | 0             | 0         | 0           | 0        | 0       | 8      | 2007   | WD       | Normal        | 175000    |
| 1456 | 1457 | 20         | RL       | 85.0        | 13175   | Pave   | Reg      | Lvl         | AllPub    | Inside    | ...  | 0             | 0         | 0           | 0        | 0       | 2      | 2010   | WD       | Normal        | 210000    |
| 1457 | 1458 | 70         | RL       | 66.0        | 9042    | Pave   | Reg      | Lvl         | AllPub    | Inside    | ...  | 0             | 0         | 0           | 0        | 2500    | 5      | 2010   | WD       | Normal        | 266500    |
| 1458 | 1459 | 20         | RL       | 68.0        | 9717    | Pave   | Reg      | Lvl         | AllPub    | Inside    | ...  | 112           | 0         | 0           | 0        | 0       | 4      | 2010   | WD       | Normal        | 142125    |
| 1459 | 1460 | 20         | RL       | 75.0        | 9937    | Pave   | Reg      | Lvl         | AllPub    | Inside    | ...  | 0             | 0         | 0           | 0        | 0       | 6      | 2008   | WD       | Normal        | 147500    |

1460 rows × 77 columns

In [15]:

```
date_features = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']def change_dates(df):    data = df.copy()        for feature in date_features:        data[feature] = data['YrSold'] - data[feature]        return data
```

In [16]:

```
df = change_dates(train_set.copy().drop(["SalePrice", "Id"], axis=1))numeric_feats = df.dtypes[df.dtypes != "object"].index
```

In [17]:

```
numeric_feats
```

Out[17]:

```
Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',       'MoSold', 'YrSold'],      dtype='object')
```

In [18]:

```
# Let's take a look at the distribution of the numeric features sns.set_theme(rc = {'figure.dpi': 300, 'axes.labelsize' : 8,                     'axes.facecolor': '#E5E5E5', 'grid.color': '#faf5f5'},                     font_scale = 0.55)for i, feature in enumerate(numeric_feats):    plt.figure(i)    sns.distplot(df[feature], fit=norm)
```

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_0.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_1.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_2.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_3.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_4.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_5.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_6.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_7.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_8.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_9.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_10.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_11.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_12.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_13.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_14.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_15.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_16.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_17.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_18.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_19.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_20.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_21.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_22.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_23.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_24.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_25.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_26.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_27.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_28.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_29.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_30.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_31.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_32.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_33.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_34.png)

![img](https://www.kaggleusercontent.com/kf/76067515/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NZR-oQIsvXGL0P7IetDLvg.4sObTVIS86V6iSOGP1-0dCgWP5jtYn_iV3cmjKSqh2ZQSxIUh09eduFVGQB6jfHG4W8qYDpVjuF3o3pXTyimTIY9PXlZFwwLem0wVsPMqMW2cBOk-LJoCGpW5wFBpe77hOgCFeRhdYn7_0goRE9K0YORGmCsaOz1T8y73Dbajld7aMsGrp_eWlMcQjnwyHgs2fQWzgGlp8gl0drpvP2hb-8t8GmJiSSSsnkT6trgmVrTSSnaReKuTsvBmvIQDpxAvBsKjuXI4XgxssaJhNsdpEFfMzS0pmA2wAhg5mKjppO--TGR1ig6GXGu6e-Qej8dV0ekrf_0I2QrPYuPE7Z_EbLklnPTDeC9LtWTd25PbMPo9GIU9EdmR2aWUgweBgkKTJj7F040MzNt-bk1o4v6Sp-kV3QAHXGNcj7fbnmNY3THWQEiuaMtIM-Lg206BGnhqw7otfOhDvYD2x_bq9z_0WSbFZ58tl7z20Izka3RaeAtrWuLHjBVPEInbnDOPNmibFdJxkMwvcEO8C6vAblT7l4h2c26egMz2bYmqw-kPkiUHPA6pREv_cjZhOg_SaFQgRem05EjIKVua3jqzSuyUP-GE3jnnmJXbYKyli35M-u63aiUScyxaM93k3DqI5uBbx6v3J1KNDoq0HfXGeEJ0PmZ4CjpoauQpb7mROxKOMg.q5kNhziK0b3wE_P9ZKXvHQ/__results___files/__results___32_35.png)

In [19]:

```
skewed_feats = df[numeric_feats].apply(lambda x: skew(x)) #compute skewnessskewed_feats = skewed_feats[skewed_feats > 0.75]skewed_feats = skewed_feats.index
```

In [20]:

```
df[numeric_feats] = df[numeric_feats].fillna(df[numeric_feats].median())df = df.fillna("Unavailable")df[skewed_feats] = np.log1p(df[skewed_feats])scaler = MinMaxScaler()df[numeric_feats] = scaler.fit_transform(df[numeric_feats])encoder = OneHotEncoder(handle_unknown="ignore")train_data = encoder.fit_transform(df)# poly_transformer = PolynomialFeatures()# train_data = poly_transformer.fit_transform(train_data)
```

# Modelling

In [21]:

```
#For cross validationdef rmsle_cv(model):    kf = KFold(        n_splits=5,        shuffle=True,        random_state=42).get_n_splits(train_data)    rmse = np.sqrt(-cross_val_score(        model,        train_data,        price,        scoring="neg_mean_squared_error",        cv = kf)    )    return rmse
```

In [22]:

```
# Prepare test settest_set.drop(["PoolQC", "Fence", "MiscFeature", "Alley"], axis=1)test_df = change_dates(test_set.copy().drop("Id", axis=1))test_df[numeric_feats] = test_df[numeric_feats].fillna(test_df[numeric_feats].median())test_df = test_df.fillna("Unavailable")test_df[skewed_feats] = np.log1p(test_df[skewed_feats])test_df[numeric_feats] = scaler.transform(test_df[numeric_feats])test_data = encoder.transform(test_df)# test_data = poly_transformer.fit_transform(test_data)
```

## Linear Models

### Linear Regression

In [23]:

```
seed = 0
```

In [24]:

```
lin_selector = SelectFromModel(    estimator=LinearRegression(),    threshold="median")lin_selector.fit(train_data, price)
```

Out[24]:

```
SelectFromModel(estimator=LinearRegression(), threshold='median')
```

In [25]:

```
linear_reg = LinearRegression()linear_reg.fit(train_data, price)
```

Out[25]:

```
LinearRegression()
```

In [26]:

```
rmsle_cv(linear_reg)
```

Out[26]:

```
array([0.14283714, 0.16413119, 0.15630111, 0.13979296, 0.16391921])
```

In [27]:

```
# Predict the test data using the Linear Model.predicted_linear_reg = linear_reg.predict(test_data)pd.DataFrame({'id': test_set['Id'], 'SalePrice': np.expm1(predicted_linear_reg)}).head(10)
```

Out[27]:

|      | id   | SalePrice     |
| :--- | :--- | :------------ |
| 0    | 1461 | 117929.611422 |
| 1    | 1462 | 153707.156014 |
| 2    | 1463 | 184852.114689 |
| 3    | 1464 | 204275.936189 |
| 4    | 1465 | 201719.679791 |
| 5    | 1466 | 166374.921059 |
| 6    | 1467 | 171321.778882 |
| 7    | 1468 | 182958.791049 |
| 8    | 1469 | 162406.750372 |
| 9    | 1470 | 138135.593656 |

In [28]:

```
print("Mean squared Log Error using Linear Regression on Test Set:", root(mean_squared_log_error(np.log1p(sample["SalePrice"]), predicted_linear_reg)))Mean squared Log Error using Linear Regression on Test Set: 0.0302655851568672
```

### Lasso

In [29]:

```
lasso = LassoCV(alphas = [0.0003], random_state=seed)lasso.fit(train_data, price)
```

Out[29]:

```
LassoCV(alphas=[0.0003], random_state=0)
```

In [30]:

```
rmsle_cv(lasso)
```

Out[30]:

```
array([0.13283362, 0.15724149, 0.14706245, 0.13479949, 0.14886391])
```

In [31]:

```
# Predict the test data using the Lasso Model.predicted_lasso = lasso.predict(test_data)pd.DataFrame({'id': test_set['Id'], 'SalePrice': np.expm1(predicted_lasso)}).head(10)
```

Out[31]:

|      | id   | SalePrice     |
| :--- | :--- | :------------ |
| 0    | 1461 | 122267.556496 |
| 1    | 1462 | 152593.167000 |
| 2    | 1463 | 171640.430993 |
| 3    | 1464 | 199036.172633 |
| 4    | 1465 | 199164.012054 |
| 5    | 1466 | 172347.264904 |
| 6    | 1467 | 175994.711204 |
| 7    | 1468 | 172265.203156 |
| 8    | 1469 | 177011.513522 |
| 9    | 1470 | 129778.948251 |

## Ridge

In [32]:

```
ridge = Ridge(alpha=10, solver="auto", random_state=seed)ridge.fit(train_data, price) 
```

Out[32]:

```
Ridge(alpha=10, random_state=0)
```

In [33]:

```
rmsle_cv(ridge)
```

Out[33]:

```
array([0.13212357, 0.15877068, 0.14961271, 0.13754792, 0.15368556])
```

In [34]:

```
# Predict the test data using the Ridge.predicted_ridge = ridge.predict(test_data)pd.DataFrame({'id': test_set['Id'], 'SalePrice': np.expm1(predicted_ridge)}).head(10)
```

Out[34]:

|      | id   | SalePrice     |
| :--- | :--- | :------------ |
| 0    | 1461 | 120948.340728 |
| 1    | 1462 | 153905.287203 |
| 2    | 1463 | 179770.053955 |
| 3    | 1464 | 202928.001220 |
| 4    | 1465 | 193334.531370 |
| 5    | 1466 | 166077.981425 |
| 6    | 1467 | 170316.206789 |
| 7    | 1468 | 176677.052401 |
| 8    | 1469 | 175094.271806 |
| 9    | 1470 | 134107.889144 |

## Elastic Net

In [35]:

```
# elastic_net = ElasticNet(random_state=0, alpha=0.0005, max_iter=5000, selection="random")elastic_net = ElasticNetCV(cv=5, alphas=[1e-3, 1e-2, 1e-1, 1, 0.0005, 0.0003, 0.005], random_state=seed)elastic_net.fit(train_data, price)
```

Out[35]:

```
ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 0.0005, 0.0003, 0.005], cv=5,             random_state=0)
```

In [36]:

```
rmsle_cv(elastic_net)
```

Out[36]:

```
array([0.13195711, 0.15708488, 0.14788423, 0.13455578, 0.14471272])
```

In [37]:

```
# Predict the test data using the Elastic Net.predicted_elastic = elastic_net.predict(test_data)pd.DataFrame({'id': test_set['Id'], 'SalePrice': np.expm1(predicted_elastic)}).head(10)
```

Out[37]:

|      | id   | SalePrice     |
| :--- | :--- | :------------ |
| 0    | 1461 | 121692.293591 |
| 1    | 1462 | 150387.050850 |
| 2    | 1463 | 174894.356078 |
| 3    | 1464 | 200038.850919 |
| 4    | 1465 | 204543.049842 |
| 5    | 1466 | 174967.637715 |
| 6    | 1467 | 179833.825604 |
| 7    | 1468 | 173455.527464 |
| 8    | 1469 | 178773.631034 |
| 9    | 1470 | 129034.498315 |

## Random Forest

In [38]:

```
## making predictions using the Random Forest algorithm forest_model = RandomForestRegressor(max_features=0.5, n_estimators=1000, random_state=seed)forest_model.fit(train_data, price)
```

Out[38]:

```
RandomForestRegressor(max_features=0.5, n_estimators=1000, random_state=0)
```

In [39]:

```
rmsle_cv(forest_model)
```

Out[39]:

```
array([0.14597971, 0.18769035, 0.17925757, 0.16764085, 0.16527499])
```

In [40]:

```
# Predict the test data using the Random Forest Model.predicted_rf = forest_model.predict(test_data)pd.DataFrame({'id': test_set['Id'], 'SalePrice': np.expm1(predicted_rf)}).head(10)
```

Out[40]:

|      | id   | SalePrice     |
| :--- | :--- | :------------ |
| 0    | 1461 | 119616.308107 |
| 1    | 1462 | 148077.784927 |
| 2    | 1463 | 176940.088432 |
| 3    | 1464 | 187439.277364 |
| 4    | 1465 | 201387.746891 |
| 5    | 1466 | 177074.885014 |
| 6    | 1467 | 173468.929012 |
| 7    | 1468 | 176335.887790 |
| 8    | 1469 | 177463.321214 |
| 9    | 1470 | 129152.082269 |

# Submission

In [41]:

```
submission = pd.DataFrame({'id': test_set['Id'], 'SalePrice': np.expm1(predicted_rf)})
submission.to_csv("submission.csv",index = False)
print("predictions successfully submitted")
predictions successfully submitted
```

