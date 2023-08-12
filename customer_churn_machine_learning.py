# Kütüphane importları ve pd-set_option ayarları

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load():
    data = pd.read_csv("Telco-Customer-Churn.csv")
    return data

df = load()
df.shape
df.head()
df.info()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Genel resim

def check_df(dataframe, head=5):
    print("########################### SHAPE ###########################")
    print(dataframe.shape)
    print("########################### DTYPES ###########################")
    print(dataframe.dtypes)
    print("########################### HEAD ###########################")
    print(dataframe.head(head))
    print("########################### TAIL ###########################")
    print(dataframe.tail(head))
    print("########################### NA ###########################")
    print(dataframe.isnull().sum())
    print("########################### QUANTİLES ###########################")
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_df(df)


# Nümerik ve kategorik değişkenlerin yakalanması

def grab_col_names(dataframe, cat_th=10, car_th=20):
    #kategorik değişkenler
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                   dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                   dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #nümerik değişkenler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    #çıktı ekranı
    print(f"Observation : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Kategorik değişkenlerin analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : dataframe[col_name].value_counts() / len(dataframe) * 100}))
    print("#####################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=False)



# Nümerik değişkenlerin analizi

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)

    if plot:
        dataframe[num_col].hist(bins=20)
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)

df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show(block=True)

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show(block=True)

df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show(block=True)

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two Year")
plt.show(block=True)

# Nümerik değişkenlerin hedef değişkene göre analizi

def target_summary_w_num(dataframe, target, num_col):
    print(dataframe.groupby(target)[num_col].mean(), end="\n\n\n")

for col in num_cols:
    target_summary_w_num(df, "Churn", col)


# Kategorik değişkenlerin hedef değişkene göre analizi

def target_summary_w_cat(dataframe, target, cat_col):
    print(cat_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_col)[target].mean(),
                        "Count": dataframe[cat_col].value_counts(),
                        "Ratio": 100 * dataframe[cat_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_w_cat(df, "Churn", cat_cols)


# Korelasyon

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte

df.corrwith(df["Churn"]).sort_values(ascending=False)


# Özellik Mühendisliği

def outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    quartile1= dataframe[variable].quantile(q1)
    quartile3= dataframe[variable].quantile(q3)
    interquantile = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquantile
    up_limit = quartile3 + 1.5 * interquantile
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

# herhangi bir aykırı değer bulunmamaktadır.


# Eksik değer analizi

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n\n\n")

    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)



#Özellik Çıkarımı

# tenure değikeninden yıllık kategorik değişken oluşturulması
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"


# kontratı 1 veya 2 yıllık müşterilerin Engaged olarak belirtilmesi.
df["NEW_ENGAGED"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# herhangi bir destek veya koruma almayan müşteri değişkeninin oluşturulması.
df["NEW_NOPROTECTION"] = df.apply(lambda x:1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or(x["TechSupport"] != "Yes") else 0, axis=1)

# müşterinin toplam aldığı servis sayısı
df["NEW_TOTALSERVICES"] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies'] ] == "Yes").sum(axis=1)


# streaming hizmeti alan müşteriler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# otomatik ödeme yapan müşteriler
df["NEW_FLAG_AUTOPAYMENT"] = df["PaymentMethod"].apply(lambda x:1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0 )

# müşterilerin ortalama aylık ödemesi
df["NEW_AVARAGE_CHARGES"] = df["TotalCharges"] / (df["tenure"] + 1)

# güncel fiyatın ortalama fiyata göre artışı
df["NEW_INCRESE"] = df["NEW_AVARAGE_CHARGES"] / df["MonthlyCharges"]

# servis başına ücret
df["NEW_AVARAGE_SERVICE_FEE"] = df["MonthlyCharges"] / (df["NEW_TOTALSERVICES"] + 1)

df.head()
df.shape


#Encoding işlemleri

cat_cols, num_cols, cat_but_car = grab_col_names(df)

## label encoder

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

## one-hot encoder

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TOTALSERVICES"]]

def one_hot_encoder(dataframe, cat_col, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns= cat_col, drop_first=False)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape


# Modelleme
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)


## Random forests

rf_model = RandomForestClassifier(random_state=17)

cv_results_1 = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_1["test_accuracy"].mean()
cv_results_1["test_f1"].mean()
cv_results_1["test_roc_auc"].mean()

# accuracy= 0.7892, f1= 0.5509, roc_auc= 0.8222

#Hiperparametre optimizasyonu
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_1_1 = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_1_1["test_accuracy"].mean()
cv_results_1_1["test_f1"].mean()
cv_results_1_1["test_roc_auc"].mean()

# rf_final = accuracy= 0.8007, f1= 0.5736, roc_auc= 0.8452 / rf_model = accuracy= 0.7892, f1= 0.5509, roc_auc= 0.8222


#GBM

gbm_model = GradientBoostingClassifier(random_state=17)

cv_results_2 = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_2["test_accuracy"].mean()
cv_results_2["test_f1"].mean()
cv_results_2["test_roc_auc"].mean()

# accuracy= 0.8030, f1= 0.5848, roc_auc= 0.8450


#Hiperparametre optimizasyonu
gbm_model.get_params()

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_2_1 = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_2_1["test_accuracy"].mean()
cv_results_2_1["test_f1"].mean()
cv_results_2_1["test_roc_auc"].mean()

#gbm_final = accuracy= 0.8046, f1= 0.5842, roc_auc= 0.8477 / gbm_model = accuracy= 0.8030, f1= 0.5848, roc_auc= 0.8450


#XGBOOST

xgboost_model = XGBClassifier(random_state=17)

cv_results_3 = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_3["test_accuracy"].mean()
cv_results_3["test_f1"].mean()
cv_results_3["test_roc_auc"].mean()

# accuracy= 0.7877, f1= 0.5628, roc_auc= 0.8239

#Hiperparametre optimizasyonu
xgboost_model.get_params()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_3_1 = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_3_1["test_accuracy"].mean()
cv_results_3_1["test_f1"].mean()
cv_results_3_1["test_roc_auc"].mean()

# xgboost_final = accuracy= 0.8015, f1= 0.5840, roc_auc= 0.8454 / xgboost_model = accuracy= 0.7877, f1= 0.5628, roc_auc= 0.8239


#LightGBM

lgbm_model = LGBMClassifier(random_state=17)

cv_results_4 = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_4["test_accuracy"].mean()
cv_results_4["test_f1"].mean()
cv_results_4["test_roc_auc"].mean()

# accuracy= 0.7924, f1= 0.5726, roc_auc= 0.8357

#Hiperparametre optimizasyonu

lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_4_1 = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_4_1["test_accuracy"].mean()
cv_results_4_1["test_f1"].mean()
cv_results_4_1["test_roc_auc"].mean()

# lgbm_final = accuracy= 0.8023, f1= 0.5860, roc_auc= 0.8444 / lgbm_model = accuracy= 0.7924, f1= 0.5726, roc_auc= 0.8357


#CatBOOST

catboost_model = CatBoostClassifier(random_state=17)

cv_results_5 = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_5["test_accuracy"].mean()
cv_results_5["test_f1"].mean()
cv_results_5["test_roc_auc"].mean()

# accuracy= 0.7972, f1= 0.5722, roc_auc= 0.8395


#Hiperparametre optimizasyonu
catboost_model.get_params()

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_5_1 = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_5_1["test_accuracy"].mean()
cv_results_5_1["test_f1"].mean()
cv_results_5_1["test_roc_auc"].mean()

# catboost_final =  accuracy= 0.8039, f1= 0.5772, roc_auc= 0.8474 / catboost_model = accuracy= 0.7972, f1= 0.5722, roc_auc= 0.8395



#CART

cart_model = DecisionTreeClassifier(random_state=17)

cv_results_6 = cross_validate(cart_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_6["test_accuracy"].mean()
cv_results_6["test_f1"].mean()
cv_results_6["test_roc_auc"].mean()

# accuracy= 0.7295, f1= 0.4930, roc_auc= 0.6556


#Hiperparametre optimizasyonu

cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_final = cart_model.set_params(**cart_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_6_1 = cross_validate(cart_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_6_1["test_accuracy"].mean()
cv_results_6_1["test_f1"].mean()
cv_results_6_1["test_roc_auc"].mean()

# cart_final = accuracy= 0.7861, f1= 0.5383, roc_auc= 0.8188 / cart_model = accuracy= 0.7295, f1= 0.4930, roc_auc= 0.6556


#KNN

knn_model = KNeighborsClassifier()

cv_results_7 = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_7["test_accuracy"].mean()
cv_results_7["test_f1"].mean()
cv_results_7["test_roc_auc"].mean()

# accuracy= 0.7651, f1= 0.5086, roc_auc= 0.7514

#Hiperparametre optimizasyonu

knn_model.get_params()

knn_params = {"n_neighbors" : range(2, 50)}

knn_best_grid = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_final = knn_model.set_params(**knn_best_grid.best_params_).fit(X, y)

cv_results_7_1 = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_7_1["test_accuracy"].mean()
cv_results_7_1["test_f1"].mean()
cv_results_7_1["test_roc_auc"].mean()

# knn_final = accuracy= 0.7883, f1= 0.5092, roc_auc= 0.8027 / knn_model = # accuracy= 0.7651, f1= 0.5086, roc_auc= 0.7514



#Logistic Regression

log_model = LogisticRegression(random_state=17)

cv_results_8 = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_8["test_accuracy"].mean()
cv_results_8["test_f1"].mean()
cv_results_8["test_roc_auc"].mean()

# log_model = accuracy= 0.8024, f1= 0.5804, roc_auc= 0.8427


# Final modeller cv_results sonuçları

# rf_final = accuracy= 0.8007, f1= 0.5736, roc_auc= 0.8452
# gbm_final = accuracy= 0.8046, f1= 0.5842, roc_auc= 0.8477
# xgboost_final = accuracy= 0.8015, f1= 0.5840, roc_auc= 0.8454
# lgbm_final = accuracy= 0.8023, f1= 0.5860, roc_auc= 0.8444
# catboost_final =  accuracy= 0.8039, f1= 0.5772, roc_auc= 0.8474
# cart_final = accuracy= 0.7861, f1= 0.5383, roc_auc= 0.8188
# knn_final = accuracy= 0.7883, f1= 0.5092, roc_auc= 0.8027
# log_model = accuracy= 0.8024, f1= 0.5804, roc_auc= 0.8427



#Feature importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
plot_importance(gbm_final, X)
plot_importance(cart_final, X)















