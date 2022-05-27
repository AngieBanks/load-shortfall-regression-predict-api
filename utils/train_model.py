"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
pip install plotly
Requirement already satisfied: plotly in c:\users\angel\anaconda3\lib\site-packages (5.8.0)
Requirement already satisfied: tenacity>=6.2.0 in c:\users\angel\anaconda3\lib\site-packages (from plotly) (8.0.1)
Note: you may need to restart the kernel to use updated packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
from statsmodels.graphics.correlation import plot_corr


# Libraries for data preparation and model building
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


import warnings

warnings.filterwarnings("ignore")


import pickle
# Setting global constants to ensure notebook results are reproducible
#PARAMETER_CONSTANT = ###
2. Loading the Data
df_train = pd.read_csv('df_train.csv') # load the train data
df_test = pd.read_csv('df_test.csv')  # load the test data

#to have a copy of our dataframes to be able to save our pickle file
df_test_copy = df_test.copy()
df_train_copy =df_train.copy()
#overview dataset
df_train.head()
Unnamed: 0	time	Madrid_wind_speed	Valencia_wind_deg	Bilbao_rain_1h	Valencia_wind_speed	Seville_humidity	Madrid_humidity	Bilbao_clouds_all	Bilbao_wind_speed	...	Madrid_temp_max	Barcelona_temp	Bilbao_temp_min	Bilbao_temp	Barcelona_temp_min	Bilbao_temp_max	Seville_temp_min	Madrid_temp	Madrid_temp_min	load_shortfall_3h
0	0	2015-01-01 03:00:00	0.666667	level_5	0.0	0.666667	74.333333	64.000000	0.0	1.000000	...	265.938000	281.013000	269.338615	269.338615	281.013000	269.338615	274.254667	265.938000	265.938000	6715.666667
1	1	2015-01-01 06:00:00	0.333333	level_10	0.0	1.666667	78.333333	64.666667	0.0	1.000000	...	266.386667	280.561667	270.376000	270.376000	280.561667	270.376000	274.945000	266.386667	266.386667	4171.666667
2	2	2015-01-01 09:00:00	1.000000	level_9	0.0	1.000000	71.333333	64.333333	0.0	1.000000	...	272.708667	281.583667	275.027229	275.027229	281.583667	275.027229	278.792000	272.708667	272.708667	4274.666667
3	3	2015-01-01 12:00:00	1.000000	level_8	0.0	1.000000	65.333333	56.333333	0.0	1.000000	...	281.895219	283.434104	281.135063	281.135063	283.434104	281.135063	285.394000	281.895219	281.895219	5075.666667
4	4	2015-01-01 15:00:00	1.000000	level_7	0.0	1.000000	59.000000	57.000000	2.0	0.333333	...	280.678437	284.213167	282.252063	282.252063	284.213167	282.252063	285.513719	280.678437	280.678437	6620.666667
5 rows × 49 columns

3. Exploratory Data Analysis (EDA)
Looking at our data, we observe that because of the shape of the data, it is not possbile to view ALL 49 columns thus, we cannot determine the hidden features. Therefore, we will use the transpose method (.T)

#using the transpose method, let us view our data again
df_train.head(10).T 
0	1	2	3	4	5	6	7	8	9
Unnamed: 0	0	1	2	3	4	5	6	7	8	9
time	2015-01-01 03:00:00	2015-01-01 06:00:00	2015-01-01 09:00:00	2015-01-01 12:00:00	2015-01-01 15:00:00	2015-01-01 18:00:00	2015-01-01 21:00:00	2015-01-02 00:00:00	2015-01-02 03:00:00	2015-01-02 06:00:00
Madrid_wind_speed	0.666667	0.333333	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0
Valencia_wind_deg	level_5	level_10	level_9	level_8	level_7	level_7	level_8	level_9	level_9	level_9
Bilbao_rain_1h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Valencia_wind_speed	0.666667	1.666667	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.333333
Seville_humidity	74.333333	78.333333	71.333333	65.333333	59.0	69.666667	83.666667	83.666667	86.0	87.0
Madrid_humidity	64.0	64.666667	64.333333	56.333333	57.0	67.333333	63.333333	64.0	63.333333	63.666667
Bilbao_clouds_all	0.0	0.0	0.0	0.0	2.0	12.333333	16.333333	8.666667	5.333333	15.333333
Bilbao_wind_speed	1.0	1.0	1.0	1.0	0.333333	0.666667	1.0	1.333333	1.0	1.0
Seville_clouds_all	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Bilbao_wind_deg	223.333333	221.0	214.333333	199.666667	185.0	191.333333	210.333333	238.666667	258.333333	255.333333
Barcelona_wind_speed	6.333333	4.0	2.0	2.333333	4.333333	4.666667	3.333333	2.0	1.666667	1.0
Barcelona_wind_deg	42.666667	139.0	326.0	273.0	260.0	254.666667	276.0	310.0	338.0	329.0
Madrid_clouds_all	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Seville_wind_speed	3.333333	3.333333	2.666667	4.0	3.0	2.666667	2.0	2.333333	3.0	3.0
Barcelona_rain_1h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Seville_pressure	sp25	sp25	sp25	sp25	sp25	sp25	sp25	sp25	sp25	sp25
Seville_rain_1h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Bilbao_snow_3h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Barcelona_pressure	1036.333333	1037.333333	1038.0	1037.0	1035.0	1035.666667	1038.0	1037.666667	1037.333333	1038.333333
Seville_rain_3h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Madrid_rain_1h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Barcelona_rain_3h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Valencia_snow_3h	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Madrid_weather_id	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0
Barcelona_weather_id	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0
Bilbao_pressure	1035.0	1035.666667	1036.0	1036.0	1035.333333	1035.666667	1036.0	1036.0	1036.666667	1037.0
Seville_weather_id	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0	800.0
Valencia_pressure	1002.666667	1004.333333	1005.333333	1009.0	NaN	NaN	1006.0	1005.666667	1005.333333	1006.333333
Seville_temp_max	274.254667	274.945	278.792	285.394	285.513719	282.077844	277.637	276.457333	276.151667	276.453
Madrid_pressure	971.333333	972.666667	974.0	994.666667	1035.333333	1013.333333	974.333333	974.666667	974.333333	975.333333
Valencia_temp_max	269.888	271.728333	278.008667	284.899552	283.015115	277.622563	272.270333	271.040667	270.535	271.661333
Valencia_temp	269.888	271.728333	278.008667	284.899552	283.015115	277.622563	272.270333	271.040667	270.535	271.661333
Bilbao_weather_id	800.0	800.0	800.0	800.0	800.0	800.666667	800.666667	800.333333	800.0	801.0
Seville_temp	274.254667	274.945	278.792	285.394	285.513719	282.077844	277.637	276.457333	276.151667	276.453
Valencia_humidity	75.666667	71.0	65.666667	54.0	58.333333	72.666667	83.333333	82.0	80.666667	75.333333
Valencia_temp_min	269.888	271.728333	278.008667	284.899552	283.015115	277.622563	272.270333	271.040667	270.535	271.661333
Barcelona_temp_max	281.013	280.561667	281.583667	283.434104	284.213167	284.165625	283.420333	282.474	281.726667	281.803
Madrid_temp_max	265.938	266.386667	272.708667	281.895219	280.678437	274.639229	268.287	266.882333	266.226667	266.878
Barcelona_temp	281.013	280.561667	281.583667	283.434104	284.213167	284.165625	283.420333	282.474	281.726667	281.803
Bilbao_temp_min	269.338615	270.376	275.027229	281.135063	282.252063	277.919	274.295437	272.903167	271.780115	271.673667
Bilbao_temp	269.338615	270.376	275.027229	281.135063	282.252063	277.919	274.295437	272.903167	271.780115	271.673667
Barcelona_temp_min	281.013	280.561667	281.583667	283.434104	284.213167	284.165625	283.420333	282.474	281.726667	281.803
Bilbao_temp_max	269.338615	270.376	275.027229	281.135063	282.252063	277.919	274.295437	272.903167	271.780115	271.673667
Seville_temp_min	274.254667	274.945	278.792	285.394	285.513719	282.077844	277.637	276.457333	276.151667	276.453
Madrid_temp	265.938	266.386667	272.708667	281.895219	280.678437	274.639229	268.287	266.882333	266.226667	266.878
Madrid_temp_min	265.938	266.386667	272.708667	281.895219	280.678437	274.639229	268.287	266.882333	266.226667	266.878
load_shortfall_3h	6715.666667	4171.666667	4274.666667	5075.666667	6620.666667	6842.0	10760.333333	10866.0	-1850.333333	-4002.333333
#checking the shape of the data
df_train.shape
(8763, 49)
There are 49 columns and 8763 rows in our dataset. We have an unnamed column (Unnamed:0) which has the same index value as seen above. This column is redundant and may have a negative impact on our training model. Valencia_wind_deg and Seville_pressure columns display categorical values. However, we need only numerical values in training our model thus, we will convert the values in those two columns to numerical values.

#Let us view the data types of the values of each column of our data set
df_train.info() 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8763 entries, 0 to 8762
Data columns (total 49 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   Unnamed: 0            8763 non-null   int64  
 1   time                  8763 non-null   object 
 2   Madrid_wind_speed     8763 non-null   float64
 3   Valencia_wind_deg     8763 non-null   object 
 4   Bilbao_rain_1h        8763 non-null   float64
 5   Valencia_wind_speed   8763 non-null   float64
 6   Seville_humidity      8763 non-null   float64
 7   Madrid_humidity       8763 non-null   float64
 8   Bilbao_clouds_all     8763 non-null   float64
 9   Bilbao_wind_speed     8763 non-null   float64
 10  Seville_clouds_all    8763 non-null   float64
 11  Bilbao_wind_deg       8763 non-null   float64
 12  Barcelona_wind_speed  8763 non-null   float64
 13  Barcelona_wind_deg    8763 non-null   float64
 14  Madrid_clouds_all     8763 non-null   float64
 15  Seville_wind_speed    8763 non-null   float64
 16  Barcelona_rain_1h     8763 non-null   float64
 17  Seville_pressure      8763 non-null   object 
 18  Seville_rain_1h       8763 non-null   float64
 19  Bilbao_snow_3h        8763 non-null   float64
 20  Barcelona_pressure    8763 non-null   float64
 21  Seville_rain_3h       8763 non-null   float64
 22  Madrid_rain_1h        8763 non-null   float64
 23  Barcelona_rain_3h     8763 non-null   float64
 24  Valencia_snow_3h      8763 non-null   float64
 25  Madrid_weather_id     8763 non-null   float64
 26  Barcelona_weather_id  8763 non-null   float64
 27  Bilbao_pressure       8763 non-null   float64
 28  Seville_weather_id    8763 non-null   float64
 29  Valencia_pressure     6695 non-null   float64
 30  Seville_temp_max      8763 non-null   float64
 31  Madrid_pressure       8763 non-null   float64
 32  Valencia_temp_max     8763 non-null   float64
 33  Valencia_temp         8763 non-null   float64
 34  Bilbao_weather_id     8763 non-null   float64
 35  Seville_temp          8763 non-null   float64
 36  Valencia_humidity     8763 non-null   float64
 37  Valencia_temp_min     8763 non-null   float64
 38  Barcelona_temp_max    8763 non-null   float64
 39  Madrid_temp_max       8763 non-null   float64
 40  Barcelona_temp        8763 non-null   float64
 41  Bilbao_temp_min       8763 non-null   float64
 42  Bilbao_temp           8763 non-null   float64
 43  Barcelona_temp_min    8763 non-null   float64
 44  Bilbao_temp_max       8763 non-null   float64
 45  Seville_temp_min      8763 non-null   float64
 46  Madrid_temp           8763 non-null   float64
 47  Madrid_temp_min       8763 non-null   float64
 48  load_shortfall_3h     8763 non-null   float64
dtypes: float64(45), int64(1), object(3)
memory usage: 3.3+ MB
From the info above, our entire data set is composed of three different data types: object, float, and int

df_train.isnull().sum() #checking for possible missing values
Unnamed: 0                 0
time                       0
Madrid_wind_speed          0
Valencia_wind_deg          0
Bilbao_rain_1h             0
Valencia_wind_speed        0
Seville_humidity           0
Madrid_humidity            0
Bilbao_clouds_all          0
Bilbao_wind_speed          0
Seville_clouds_all         0
Bilbao_wind_deg            0
Barcelona_wind_speed       0
Barcelona_wind_deg         0
Madrid_clouds_all          0
Seville_wind_speed         0
Barcelona_rain_1h          0
Seville_pressure           0
Seville_rain_1h            0
Bilbao_snow_3h             0
Barcelona_pressure         0
Seville_rain_3h            0
Madrid_rain_1h             0
Barcelona_rain_3h          0
Valencia_snow_3h           0
Madrid_weather_id          0
Barcelona_weather_id       0
Bilbao_pressure            0
Seville_weather_id         0
Valencia_pressure       2068
Seville_temp_max           0
Madrid_pressure            0
Valencia_temp_max          0
Valencia_temp              0
Bilbao_weather_id          0
Seville_temp               0
Valencia_humidity          0
Valencia_temp_min          0
Barcelona_temp_max         0
Madrid_temp_max            0
Barcelona_temp             0
Bilbao_temp_min            0
Bilbao_temp                0
Barcelona_temp_min         0
Bilbao_temp_max            0
Seville_temp_min           0
Madrid_temp                0
Madrid_temp_min            0
load_shortfall_3h          0
dtype: int64
It can be deduced that Valencia_pressure has missing values. We will fill up these missing values appropriately in the feature engineering section of this notebook.

# Let us take a look at the descriptive statistics of our data set in the transposed format
df_train.describe().T
count	mean	std	min	25%	50%	75%	max
Unnamed: 0	8763.0	4381.000000	2529.804538	0.000000	2190.500000	4381.000000	6571.500000	8.762000e+03
Madrid_wind_speed	8763.0	2.425729	1.850371	0.000000	1.000000	2.000000	3.333333	1.300000e+01
Bilbao_rain_1h	8763.0	0.135753	0.374901	0.000000	0.000000	0.000000	0.100000	3.000000e+00
Valencia_wind_speed	8763.0	2.586272	2.411190	0.000000	1.000000	1.666667	3.666667	5.200000e+01
Seville_humidity	8763.0	62.658793	22.621226	8.333333	44.333333	65.666667	82.000000	1.000000e+02
Madrid_humidity	8763.0	57.414717	24.335396	6.333333	36.333333	58.000000	78.666667	1.000000e+02
Bilbao_clouds_all	8763.0	43.469132	32.551044	0.000000	10.000000	45.000000	75.000000	1.000000e+02
Bilbao_wind_speed	8763.0	1.850356	1.695888	0.000000	0.666667	1.000000	2.666667	1.266667e+01
Seville_clouds_all	8763.0	13.714748	24.272482	0.000000	0.000000	0.000000	20.000000	9.733333e+01
Bilbao_wind_deg	8763.0	158.957511	102.056299	0.000000	73.333333	147.000000	234.000000	3.593333e+02
Barcelona_wind_speed	8763.0	2.870497	1.792197	0.000000	1.666667	2.666667	4.000000	1.266667e+01
Barcelona_wind_deg	8763.0	190.544848	89.077337	0.000000	118.166667	200.000000	260.000000	3.600000e+02
Madrid_clouds_all	8763.0	19.473392	28.053660	0.000000	0.000000	0.000000	33.333333	1.000000e+02
Seville_wind_speed	8763.0	2.425045	1.672895	0.000000	1.000000	2.000000	3.333333	1.166667e+01
Barcelona_rain_1h	8763.0	0.128906	0.634730	0.000000	0.000000	0.000000	0.000000	1.200000e+01
Seville_rain_1h	8763.0	0.039439	0.175857	0.000000	0.000000	0.000000	0.000000	3.000000e+00
Bilbao_snow_3h	8763.0	0.031912	0.557264	0.000000	0.000000	0.000000	0.000000	2.130000e+01
Barcelona_pressure	8763.0	1377.964605	14073.140990	670.666667	1014.000000	1018.000000	1022.000000	1.001411e+06
Seville_rain_3h	8763.0	0.000243	0.003660	0.000000	0.000000	0.000000	0.000000	9.333333e-02
Madrid_rain_1h	8763.0	0.037818	0.152639	0.000000	0.000000	0.000000	0.000000	3.000000e+00
Barcelona_rain_3h	8763.0	0.000439	0.003994	0.000000	0.000000	0.000000	0.000000	9.300000e-02
Valencia_snow_3h	8763.0	0.000205	0.011866	0.000000	0.000000	0.000000	0.000000	7.916667e-01
Madrid_weather_id	8763.0	773.527594	77.313315	211.000000	800.000000	800.000000	800.666667	8.040000e+02
Barcelona_weather_id	8763.0	765.979687	88.142235	200.666667	800.000000	800.333333	801.000000	8.040000e+02
Bilbao_pressure	8763.0	1017.739549	10.046124	971.333333	1013.000000	1019.000000	1024.000000	1.042000e+03
Seville_weather_id	8763.0	774.658818	71.940009	200.000000	800.000000	800.000000	800.000000	8.040000e+02
Valencia_pressure	6695.0	1012.051407	9.506214	972.666667	1010.333333	1015.000000	1018.000000	1.021667e+03
Seville_temp_max	8763.0	297.479527	8.875812	272.063000	291.312750	297.101667	304.150000	3.204833e+02
Madrid_pressure	8763.0	1010.316920	22.198555	927.666667	1012.333333	1017.333333	1022.000000	1.038000e+03
Valencia_temp_max	8763.0	291.337233	7.565692	269.888000	285.550167	291.037000	297.248333	3.142633e+02
Valencia_temp	8763.0	290.592152	7.162274	269.888000	285.150000	290.176667	296.056667	3.104267e+02
Bilbao_weather_id	8763.0	724.722362	115.846537	207.333333	700.333333	800.000000	801.666667	8.040000e+02
Seville_temp	8763.0	293.978903	7.920986	272.063000	288.282917	293.323333	299.620333	3.149767e+02
Valencia_humidity	8763.0	65.247727	19.262322	10.333333	51.333333	67.000000	81.333333	1.000000e+02
Valencia_temp_min	8763.0	289.867648	6.907402	269.888000	284.783333	289.550000	294.820000	3.102720e+02
Barcelona_temp_max	8763.0	291.157644	7.273538	272.150000	285.483333	290.150000	296.855000	3.140767e+02
Madrid_temp_max	8763.0	289.540309	9.752047	264.983333	282.150000	288.116177	296.816667	3.144833e+02
Barcelona_temp	8763.0	289.855459	6.528111	270.816667	284.973443	289.416667	294.909000	3.073167e+02
Bilbao_temp_min	8763.0	285.017973	6.705672	264.483333	280.085167	284.816667	289.816667	3.098167e+02
Bilbao_temp	8763.0	286.422929	6.818682	267.483333	281.374167	286.158333	291.034167	3.107100e+02
Barcelona_temp_min	8763.0	288.447422	6.102593	269.483333	284.150000	288.150000	292.966667	3.048167e+02
Bilbao_temp_max	8763.0	287.966027	7.105590	269.063000	282.836776	287.630000	292.483333	3.179667e+02
Seville_temp_min	8763.0	291.633356	8.178220	270.150000	285.816667	290.816667	297.150000	3.148167e+02
Madrid_temp	8763.0	288.419439	9.346796	264.983333	281.404281	287.053333	295.154667	3.131333e+02
Madrid_temp_min	8763.0	287.202203	9.206237	264.983333	280.299167	286.083333	293.884500	3.103833e+02
load_shortfall_3h	8763.0	10673.857612	5218.046404	-6618.000000	7390.333333	11114.666667	14498.166667	3.190400e+04
From the mean of some of the values above, there is a possibility of outliers present in our dataset.

#Checking for possible outliers
#High kurtosis (>3) indicates a large number of outliers and low kurtosis (<3) a lack of outliers
df_train.kurtosis()
Unnamed: 0                -1.200000
Madrid_wind_speed          2.036462
Bilbao_rain_1h            32.904656
Valencia_wind_speed       35.645426
Seville_humidity          -1.017983
Madrid_humidity           -1.167537
Bilbao_clouds_all         -1.533417
Bilbao_wind_speed          3.631565
Seville_clouds_all         2.155921
Bilbao_wind_deg           -1.083530
Barcelona_wind_speed       1.493635
Barcelona_wind_deg        -0.959160
Madrid_clouds_all          0.142079
Seville_wind_speed         1.398580
Barcelona_rain_1h        101.578931
Seville_rain_1h           93.840746
Bilbao_snow_3h           806.128471
Barcelona_pressure      3687.564230
Seville_rain_3h          413.136592
Madrid_rain_1h            76.584491
Barcelona_rain_3h        187.800460
Valencia_snow_3h        4089.323165
Madrid_weather_id          9.259047
Barcelona_weather_id       5.701882
Bilbao_pressure            1.825323
Seville_weather_id        10.710308
Valencia_pressure          2.211823
Seville_temp_max          -0.515989
Madrid_pressure            2.216199
Valencia_temp_max         -0.613755
Valencia_temp             -0.643793
Bilbao_weather_id          0.067814
Seville_temp              -0.504132
Valencia_humidity         -0.734345
Valencia_temp_min         -0.599551
Barcelona_temp_max        -0.728757
Madrid_temp_max           -0.662861
Barcelona_temp            -0.696555
Bilbao_temp_min           -0.230342
Bilbao_temp               -0.086363
Barcelona_temp_min        -0.474890
Bilbao_temp_max            0.283366
Seville_temp_min          -0.475564
Madrid_temp               -0.612299
Madrid_temp_min           -0.666646
load_shortfall_3h         -0.118999
dtype: float64
#Let us view our target variable Load_shortfall_3h against time on a line chart
fig = px.line(df_train, y = df_train['load_shortfall_3h'], x =df_train['time'], width =900, height=400 )
fig.show()
We can tell from the image the seasonality in the time axis on their load_shortfall_3h values, We will need to desample (break them into bits) this image to get a better understanding of the graph

To do this we will have to desample the time into:

Year, Months, Weeks, Days, Hours

df_train.groupby([df_train['time'].astype('datetime64').dt.hour])['load_shortfall_3h'].sum().plot(legend = True)
<AxesSubplot:xlabel='time'>

px.line(df_train.groupby([df_train['time'].astype('datetime64').dt.year])['load_shortfall_3h'].mean(),
        title = 'Load_shortfall_3h grouped by Year',
        y='load_shortfall_3h',width =800, height=400 )
The yearly Load_short_fall plots indicates an increase in load short fall from 2016 down to 2017 surpassing the previous years

px.line(df_train.groupby([df_train['time'].astype('datetime64').dt.month])['load_shortfall_3h'].mean(),
        title = 'Load_shortfall_3h grouped by Month of Year',
        y='load_shortfall_3h', width =800, height=400)
Also the plot above, indicates a higher 'load short fall' from middle of June down to December

px.line(df_train.groupby([df_train['time'].astype('datetime64').dt.weekofyear])['load_shortfall_3h'].mean(), 
        title = 'Load_shortfall_3h grouped by Week of the Year', y='load_shortfall_3h', width =700, height=400)
No much information can be deduced from the the week of the year Load_short_fall as shown above

px.line(df_train.groupby([df_train['time'].astype('datetime64').dt.dayofyear])['load_shortfall_3h'].mean(), 
        title = 'Load_shortfall_3h grouped by Day of the Year', y='load_shortfall_3h', width =700, height=400)
The minimum load_short_fall_3h recorded is 1,862k while the maximum is 17,306k as seen from the Day of the year plots

px.line(df_train.groupby([df_train['time'].astype('datetime64').dt.day])['load_shortfall_3h'].mean(), 
        title = 'Load_shortfall_3h grouped by Day of the Month', y='load_shortfall_3h', width =800, height=400 )
The plots above shows 10k to 12k consistent recorded values from middle of each to the end of the month

px.line(df_train.groupby([df_train['time'].astype('datetime64').dt.dayofweek])['load_shortfall_3h'].mean(), 
        title = 'Load_shortfall_3h grouped by Day of the Week', y='load_shortfall_3h', width =800, height=400 )
There seems to be a decrease in the Load_short_fall_3h Day of the week plots on Fridays and Saturdays, we can not account for the reasons

px.line(df_train.groupby([df_train['time'].astype('datetime64').dt.hour])['load_shortfall_3h'].mean(), 
        title = 'Load_shortfall_3h grouped by Hour of Day', y='load_shortfall_3h', width =800, height=400 )
There seems to be an increase in the Load_short_fall_3h hourly plots each day, mostly from 10hours and above

# Let us check the Distribution of our Data
plt.hist(df_train['load_shortfall_3h'])
(array([7.000e+01, 3.720e+02, 8.370e+02, 1.641e+03, 2.494e+03, 2.301e+03,
        9.000e+02, 1.400e+02, 7.000e+00, 1.000e+00]),
 array([-6618. , -2765.8,  1086.4,  4938.6,  8790.8, 12643. , 16495.2,
        20347.4, 24199.6, 28051.8, 31904. ]),
 <BarContainer object of 10 artists>)

# # Let's have a look at the correlation between the numeric variables.
fig = plt.figure(figsize=(10,8));
ax = fig.add_subplot(111);
plot_corr(df_train.corr(), xnames = df_train.corr().columns, ax = ax, );

First, we can easily tell the presence of high correlation (in red) between features on the heatmap at the bottom right corner of our graph. A breakdown of handling such occurence will be discussed in the feature engineering section of this notebook. It is important to perform this step when choosing the best features which in turn would result to an improvement of our model.

4. Data Engineering
We will be carrying out Feature Engineering in this section of our notebook to improve the performance of the model.

As we saw in the previous section (EDA), we highlighted some columns to be dropped as well as columns with categorical values.

We will now do the following:

Drop the Unnamed Column. Convert both Seville_pressure and Valencia_wind_degree columns from categorical to numerical values. Also, we will be converting our Time column values to date/time format as follows; Year. Month of the Year. Week of the Year. Day of the Year. Day of the Month. Day of the Week. Hour of the Week. Hour of the Day. This will enable us have a better and larger expression of our data during modeling

print(df_train['Valencia_pressure'].mean())
1012.0514065222828
#Filling in the missing values with the mean
df_train['Valencia_pressure'].fillna(df_train['Valencia_pressure'].mean(), inplace = True)
#converting Valencia_wind_deg and Seville_pressure columns from categorical to numerical datatypes.

df_train['Valencia_wind_deg'] = df_train['Valencia_wind_deg'].str.extract('(\d+)').astype('int64')
df_train['Seville_pressure'] = df_train['Seville_pressure'].str.extract('(\d+)').astype('int64')
The next step is to engineer new features from the time column

#Engineering New Features ( i.e Desampling the Time) to further expand our training data set

df_train['Year']  = df_train['time'].astype('datetime64').dt.year
df_train['Month_of_year']  = df_train['time'].astype('datetime64').dt.month
df_train['Week_of_year'] = df_train['time'].astype('datetime64').dt.weekofyear
df_train['Day_of_year']  = df_train['time'].astype('datetime64').dt.dayofyear
df_train['Day_of_month']  = df_train['time'].astype('datetime64').dt.day
df_train['Day_of_week'] = df_train['time'].astype('datetime64').dt.dayofweek
df_train['Hour_of_week'] = ((df_train['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - df_train['time'].astype('datetime64').dt.hour)
df_train['Hour_of_day']  = df_train['time'].astype('datetime64').dt.hour
Let us have a look at the correlation(s) between our newly created temporal features

Time_df = df_train.iloc[:,[-8,-7,-6,-5,-4,-3,-2,-1]]
plt.figure(figsize=[10,6])
sns.heatmap(Time_df.corr(),annot=True )
<AxesSubplot:>

Looking at our heatmap tells us that we have high Multicollinearity present in our new features. The features involved are -

Week of the year. Day of the year. Month of the year. Day of the week. Hour of the week. We would have to drop one of the features that have high correlation with each other.

Alongside dropping these features mentioned above, we would also be dropping the time and Unnamed column.

df_train = df_train.drop(columns=['Week_of_year','Day_of_year','Hour_of_week', 'Unnamed: 0','time'])
plt.figure(figsize=[35,15])
sns.heatmap(df_train.corr(),annot=True )
<AxesSubplot:>

Just as we mentioned in our EDA, we noticed the presence of high correlations between the predictor columns and also possible outliers.

Here, we would have to drop these columns to improve the performance of our model and reduce any possibility of overfitting in our model. Let us check if this approach corresponds with our feature selection. Using SelectKBest and Chi2 to perform Feature Selection.

## Splitting our data into dependent Variable and Independent Variable
X = df_train.drop(columns = 'load_shortfall_3h')
y = df_train['load_shortfall_3h'].astype('int')
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Features', 'Score']
new_X = featureScores.sort_values('Score',ascending=False).head(40)
new_X.head(40) #To get the most important features based on their score 
Features	Score
18	Barcelona_pressure	1.189344e+09
9	Bilbao_wind_deg	4.574064e+05
8	Seville_clouds_all	3.049398e+05
11	Barcelona_wind_deg	2.920143e+05
12	Madrid_clouds_all	2.862344e+05
6	Bilbao_clouds_all	1.705834e+05
32	Bilbao_weather_id	1.307308e+05
24	Barcelona_weather_id	7.121392e+04
5	Madrid_humidity	7.087652e+04
17	Bilbao_snow_3h	6.812971e+04
4	Seville_humidity	5.699050e+04
23	Madrid_weather_id	5.445955e+04
26	Seville_weather_id	4.703123e+04
34	Valencia_humidity	3.980066e+04
48	Day_of_month	3.443358e+04
50	Hour_of_day	3.167767e+04
15	Seville_pressure	2.687804e+04
14	Barcelona_rain_1h	2.171411e+04
3	Valencia_wind_speed	1.601889e+04
47	Month_of_year	1.293213e+04
1	Valencia_wind_deg	1.104442e+04
7	Bilbao_wind_speed	1.092892e+04
0	Madrid_wind_speed	1.017244e+04
49	Day_of_week	9.265478e+03
13	Seville_wind_speed	8.132635e+03
10	Barcelona_wind_speed	8.016649e+03
2	Bilbao_rain_1h	7.544582e+03
16	Seville_rain_1h	5.397681e+03
20	Madrid_rain_1h	4.226512e+03
29	Madrid_pressure	3.436256e+03
22	Valencia_snow_3h	3.110384e+03
37	Madrid_temp_max	2.281817e+03
44	Madrid_temp	2.106589e+03
45	Madrid_temp_min	2.054920e+03
28	Seville_temp_max	1.847097e+03
43	Seville_temp_min	1.589866e+03
33	Seville_temp	1.483057e+03
30	Valencia_temp_max	1.365686e+03
36	Barcelona_temp_max	1.260724e+03
31	Valencia_temp	1.229799e+03
This result backups our claim, were we saw in the heatmap multicollinearity between features, and from our feature selection, we can see those features as having the lowest significance in our data.

Dropping Outliers We have one more thing to do, which is to remove possible outliers. Also, we will select the important features for our model thus dropping others having multicollinearity

X = X[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Bilbao_weather_id', 
        'Valencia_humidity', 'Year', 'Month_of_year', 'Day_of_month', 'Day_of_week', 'Hour_of_day']]
plt.figure(figsize=[20,10])
sns.heatmap(X.corr(),annot=True )
<AxesSubplot:>

We have been able to remove the collinearity seen in previous heatmaps and also selected specific features to train our model with

Feature Scaling Lastly, before we carry out modeling, it is important to scale our data. As we saw during the EDA, we noticed how some columns(features) had values that were out of range when we compared their mean, max and standard deviation. This can result to bias in the model during decision making, thus it is important to convert all the column values to a certain range/scale.

What is Feature Scaling? Feature scaling is the process of normalising the range of features in a dataset. Real-world datasets often contain features that are varying in degrees of magnitude, range and units. Therefore, in order for machine learning models to interpret these features on the same scale, we need to perform feature scaling.

In this project, we will be carrying out Standard Scaling, becasue of it's robustness to outliers

# Create standardization object
scaler = StandardScaler()
# Save standardized features into new variable
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
X_scaled.head()
Madrid_wind_speed	Valencia_wind_deg	Bilbao_rain_1h	Valencia_wind_speed	Seville_humidity	Madrid_humidity	Bilbao_clouds_all	Bilbao_wind_speed	Seville_clouds_all	Bilbao_wind_deg	...	Bilbao_pressure	Seville_weather_id	Valencia_pressure	Bilbao_weather_id	Valencia_humidity	Year	Month_of_year	Day_of_month	Day_of_week	Hour_of_day
0	-0.950708	-0.096053	-0.362123	-0.796169	0.516117	0.270621	-1.335491	-0.501451	-0.565065	0.630823	...	1.718219	0.352274	-1.129531e+00	0.649842	0.540928	-1.226179	-1.602429	-1.675368	-0.00274	-1.090901
1	-1.130863	1.641580	-0.362123	-0.381412	0.692953	0.298017	-1.335491	-0.501451	-0.565065	0.607959	...	1.784583	0.352274	-9.289340e-01	0.649842	0.298645	-1.226179	-1.602429	-1.675368	-0.00274	-0.654451
2	-0.770554	1.294054	-0.362123	-0.657917	0.383491	0.284319	-1.335491	-0.501451	-0.565065	0.542632	...	1.817765	0.352274	-8.085757e-01	0.649842	0.021750	-1.226179	-1.602429	-1.675368	-0.00274	-0.218001
3	-0.770554	0.946527	-0.362123	-0.657917	0.118238	-0.044439	-1.335491	-0.501451	-0.565065	0.398912	...	1.817765	0.352274	-3.672620e-01	0.649842	-0.583957	-1.226179	-1.602429	-1.675368	-0.00274	0.218449
4	-0.770554	0.599000	-0.362123	-0.657917	-0.161751	-0.017043	-1.274045	-0.894581	-0.565065	0.255192	...	1.751401	0.352274	2.736630e-13	0.649842	-0.358980	-1.226179	-1.602429	-1.675368	-0.00274	0.654899
5 rows × 34 columns

y.head()
0    6715
1    4171
2    4274
3    5075
4    6620
Name: load_shortfall_3h, dtype: int32
5. Modelling
Model Building We'll split the data into train and test, to be able to evaluate the model that we build on the train data. Build a Linear Regression model which would serve as our base model using the train data. Try and improve the linear model by employing Lasso and Ridge Try out other models like decision trees, Random Forest and SVR

#Separating our models into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state = 42)
#checking the shape of the training and testing data

print('Training predictor:', X_train.shape)
print('Training target:', y_train.shape)
print('Testing predictor:', X_test.shape)
print('Testing target:', y_test.shape)
Training predictor: (7010, 34)
Training target: (7010,)
Testing predictor: (1753, 34)
Testing target: (1753,)
Multiple linear regression model
As our basline, we would first make use of Linear Model. The term linear model implies that the model is specified as a linear combination of features. Based on training data, the learning process computes one weight for each feature to form a model that can predict or estimate the target value.

#Instantiate the model
lm = LinearRegression()
#Fit the model into training set
lm.fit(X_train, y_train)

#predict on unseen data
predict = lm.predict(X_test)
train_predict = lm.predict(X_train) #predicting on the same training set
Lasso Regression (L1 Norm)
# Create LASSO model object, setting alpha to 0.01
""" when alpha is 0, Lasso regression produces the same coefficients as a linear regression. When alpha is very very large, all coefficients are zero."""
lasso = Lasso(alpha=0.01)
# Train the LASSO model
lasso.fit(X_train, y_train)
# Get predictions
lasso_pred = lasso.predict(X_test)
Ridge Regression (L2 Norm)
# Creating Ridge model
Ridge = Ridge()
# Train the model
Ridge.fit(X_train, y_train)
# Get predictions
Ridge_pred = Ridge.predict(X_test)
Support Vector Regressor
# Instantiate support vector regression model
Sv_reg = SVR(kernel='rbf', gamma='auto')
# Train the model
Sv_reg.fit(X_train,y_train)
# Get predictions
SV_pred = Sv_reg.predict(X_test)
Decision Tree Model
# Instantiate regression tree model
Reg_tree = DecisionTreeRegressor(random_state=6)
# Fitting the model
Reg_tree.fit(X_train,y_train)
Tree_pred = Reg_tree.predict(X_test)
Random Forest
# Our forest consists of 200 trees with a max depth of 8 
RF = RandomForestRegressor(n_estimators=200, max_depth=8)
# Fitting the model
RF.fit(X_train,y_train)
RF_predict = RF.predict(X_test)
In this section of our notebook, we will evaluate the performance of SIX MODELS we trained using metrics such as-

Root Mean Squared Error (RMSE), Mean Squared Error (MSE), Mean Absolute Error (MAE), Residual Sum of Squares Error (RSS)

#Comparing the True value and the Predicted Value of our models
Linear = pd.DataFrame({'Actual': y_test, 'Predicted': predict})
Lass_ = pd.DataFrame({'Actual': y_test, 'Predicted': lasso_pred})
Ridge_ = pd.DataFrame({'Actual': y_test, 'Predicted': Ridge_pred})
Sv_ = pd.DataFrame({'Actual': y_test, 'Predicted': SV_pred})
Des_ = pd.DataFrame({'Actual': y_test, 'Predicted': Tree_pred})
Rand_ = pd.DataFrame({'Actual': y_test, 'Predicted': RF_predict})
print(Linear.head()) #Linear Model 
print('\n')
print(Lass_.head()) # Lasso Model
print('\n')
print(Ridge_.head()) # Ridge Model
print('\n')
print(Sv_.head()) #SVR Model
print('\n')
print(Des_.head()) #Decision Tree Model
print('\n')
print(Rand_.head()) # Random Forest Model
      Actual     Predicted
1226   11450  12408.082166
7903   13693  12192.280096
1559   18337  12069.449711
3621   -1221   9420.766014
7552    8515  14081.831992


      Actual     Predicted
1226   11450  12408.040162
7903   13693  12192.285141
1559   18337  12069.426889
3621   -1221   9420.777827
7552    8515  14081.743368


      Actual     Predicted
1226   11450  12407.695156
7903   13693  12191.992218
1559   18337  12069.465054
3621   -1221   9420.780876
7552    8515  14081.276309


      Actual     Predicted
1226   11450  11114.379089
7903   13693  11104.075061
1559   18337  11221.728281
3621   -1221  11016.194928
7552    8515  11218.252090


      Actual  Predicted
1226   11450     8409.0
7903   13693    11016.0
1559   18337    18497.0
3621   -1221      -67.0
7552    8515     9642.0


      Actual     Predicted
1226   11450   7575.661241
7903   13693  16835.506721
1559   18337  14236.826501
3621   -1221   4467.041073
7552    8515  10814.817123
From the Predicted values above, we can see some models have values very close to the actual label, let us not get carried away as it doesn't tell the whole story.

Some of these results might be attributed to overfitting and also exposed to a lot of noise/outliers.

We will therefore test our model's performance based on the Metrics aforementioned in the previous cell.

Comparing the Root Mean Square Error across Models
Model_Performance = { 
    
                      'Test RMSE':
                    
                        {"Linear model": np.sqrt(metrics.mean_squared_error(y_test,predict)),
                        "Ridge": np.sqrt(metrics.mean_squared_error(y_test,Ridge_pred)),
                        "Lasso" : np.sqrt(metrics.mean_squared_error(y_test,lasso_pred)),
                         "SVR" : np.sqrt(metrics.mean_squared_error(y_test,SV_pred)),
                        "Decision Tree" : np.sqrt(metrics.mean_squared_error(y_test,Tree_pred)),
                        "Random Forest" : np.sqrt(metrics.mean_squared_error(y_test,RF_predict))}
                        
                    }

# create dataframe from dictionary
Model_Performance = pd.DataFrame(data=Model_Performance)
Model_Performance
Test RMSE
Decision Tree	3899.568966
Lasso	4848.114436
Linear model	4848.113470
Random Forest	3348.934482
Ridge	4848.110776
SVR	5294.338527
px.bar(Model_Performance, y =Model_Performance['Test RMSE'],
       color = Model_Performance.index, width =700, height=400)
From the graph above, we can confirm that the Random Forest model performs better than others in terms of RMSE

Comparing the Mean Square Error across Models
Model_Performance2 = { 
    
                      'Test MSE':
                    
                        {"Linear model": (metrics.mean_squared_error(y_test,predict)),
                        "Ridge": (metrics.mean_squared_error(y_test,Ridge_pred)),
                        "Lasso" : (metrics.mean_squared_error(y_test,lasso_pred)),
                         "SVR" : (metrics.mean_squared_error(y_test,SV_pred)),
                        "Decision Tree" : (metrics.mean_squared_error(y_test,Tree_pred)),
                        "Random Forest" : (metrics.mean_squared_error(y_test,RF_predict))}
                        
                    }

# create dataframe from dictionary
Model_Performance2 = pd.DataFrame(data=Model_Performance2)
Model_Performance2
Test MSE
Decision Tree	1.520664e+07
Lasso	2.350421e+07
Linear model	2.350420e+07
Random Forest	1.121536e+07
Ridge	2.350418e+07
SVR	2.803002e+07
px.bar(Model_Performance2, y =Model_Performance2['Test MSE'],
       color = Model_Performance2.index, width =700, height=400)
From the graph above, we can confirm that the Random Forest model performs better than others in terms of MSE

Comparing the Mean Absolute Error across Models
Model_Performance3= { 
    
                      'Test MAE':
                    
                        {"Linear model": (metrics.mean_absolute_error(y_test,predict)),
                        "Ridge": (metrics.mean_absolute_error(y_test,Ridge_pred)),
                        "Lasso" : (metrics.mean_absolute_error(y_test,lasso_pred)),
                         "SVR" : (metrics.mean_absolute_error(y_test,SV_pred)),
                        "Decision Tree" : (metrics.mean_absolute_error(y_test,Tree_pred)),
                        "Random Forest" : (metrics.mean_absolute_error(y_test,RF_predict))}
                        
                    }

# create dataframe from dictionary
Model_Performance3 = pd.DataFrame(data=Model_Performance3)
Model_Performance3
Test MAE
Decision Tree	2812.318882
Lasso	3861.387922
Linear model	3861.385356
Random Forest	2650.434940
Ridge	3861.391165
SVR	4226.823332
px.bar(Model_Performance3, y =Model_Performance3['Test MAE'],
       color = Model_Performance3.index, width =700, height=400)
From the graph above, we can confirm that the Random Forest model performs better than others in terms of MSE

Model_Performance4= { 
    
                      'Test R^2':
                    
                        {"Linear model": (metrics.r2_score(y_test,predict)),
                        "Ridge": (metrics.r2_score(y_test,Ridge_pred)),
                        "Lasso" : (metrics.r2_score(y_test,lasso_pred)),
                         "SVR" : (metrics.r2_score(y_test,SV_pred)),
                        "Decision Tree" : (metrics.r2_score(y_test,Tree_pred)),
                        "Random Forest" : (metrics.r2_score(y_test,RF_predict))}
                        
                    }

# create dataframe from dictionary
Model_Performance4 = pd.DataFrame(data=Model_Performance4)
Model_Performance4
Test R^2
Decision Tree	0.460034
Lasso	0.165399
Linear model	0.165399
Random Forest	0.601758
Ridge	0.165400
SVR	0.004693
px.bar(Model_Performance4, y =Model_Performance4['Test R^2'],
       color = Model_Performance4.index, width =700, height=400)
We have chosen Random Forest model as our most prefered model of choice.
#en_pred = v_reg.fit(X_train,y_train)
#RF.fit(X_train,y_train)
RF.fit(X_train,y_train)
print (RF, "\n")
RandomForestRegressor(max_depth=8, n_estimators=200) 

#saving the pickle file
RF_save_path = "RF.pkl"
with open(RF_save_path,'wb') as file:
    pickle.dump(RF,file)
#en_load_path = "v_reg.pkl"
RF_load_path = "RF.pkl"
with open(RF_load_path,'rb') as file:
    unpickled_RF = pickle.load(file)
RF_predict =  unpickled_RF.predict(X)
df_new = pd.DataFrame(RF_predict, columns=['load_shortfall_3h'])
df_test.head()
Unnamed: 0	time	Madrid_wind_speed	Valencia_wind_deg	Bilbao_rain_1h	Valencia_wind_speed	Seville_humidity	Madrid_humidity	Bilbao_clouds_all	Bilbao_wind_speed	...	Barcelona_temp_max	Madrid_temp_max	Barcelona_temp	Bilbao_temp_min	Bilbao_temp	Barcelona_temp_min	Bilbao_temp_max	Seville_temp_min	Madrid_temp	Madrid_temp_min
0	8763	2018-01-01 00:00:00	5.000000	level_8	0.0	5.000000	87.000000	71.333333	20.000000	3.000000	...	287.816667	280.816667	287.356667	276.150000	280.380000	286.816667	285.150000	283.150000	279.866667	279.150000
1	8764	2018-01-01 03:00:00	4.666667	level_8	0.0	5.333333	89.000000	78.000000	0.000000	3.666667	...	284.816667	280.483333	284.190000	277.816667	281.010000	283.483333	284.150000	281.150000	279.193333	278.150000
2	8765	2018-01-01 06:00:00	2.333333	level_7	0.0	5.000000	89.000000	89.666667	0.000000	2.333333	...	284.483333	276.483333	283.150000	276.816667	279.196667	281.816667	282.150000	280.483333	276.340000	276.150000
3	8766	2018-01-01 09:00:00	2.666667	level_7	0.0	5.333333	93.333333	82.666667	26.666667	5.666667	...	284.150000	277.150000	283.190000	279.150000	281.740000	282.150000	284.483333	279.150000	275.953333	274.483333
4	8767	2018-01-01 12:00:00	4.000000	level_7	0.0	8.666667	65.333333	64.000000	26.666667	10.666667	...	287.483333	281.150000	286.816667	281.816667	284.116667	286.150000	286.816667	284.483333	280.686667	280.150000
5 rows × 48 columns

output_en_df = pd.DataFrame({"time": df_test_copy['time'].reset_index(drop=True)})
en_file = output_en_df.join(df_new)
en_file['load_shortfall_3h'] = df_new
en_file.to_csv("RF_model_file.csv", index=False)
print(en_file)
                     time  load_shortfall_3h
0     2018-01-01 00:00:00        9454.853081
1     2018-01-01 03:00:00        9334.159023
2     2018-01-01 06:00:00        9352.296410
3     2018-01-01 09:00:00        9352.296410
4     2018-01-01 12:00:00        9451.513864
...                   ...                ...
2915  2018-12-31 09:00:00        4975.373938
2916  2018-12-31 12:00:00        8648.271280
2917  2018-12-31 15:00:00        8447.310486
2918  2018-12-31 18:00:00        8109.908496
2919  2018-12-31 21:00:00        7488.794840

[2920 rows x 2 columns]
output = pd.DataFrame({"time": df_test_copy['time']})
submission = output.join(df_new)
submission.to_csv('new_submission.csv',index = False)
submission.head()
