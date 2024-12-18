import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
import xgboost as xgb

# 讀取訓練數據
train_df = pd.read_csv(r'C:/Users/nonohuang/OneDrive/桌面/kaggle/kaggle/task4/introml_2024_task4_train.csv')

# 分離特徵和標籤
X = train_df.drop(columns=['id', 'class'])
y = train_df['class']

# 將類別標籤轉換為數值類型
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 處理缺失值
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_num = X.select_dtypes(include=['int64', 'float64'])
X_cat = X.select_dtypes(include=['object'])

if not X_num.empty:
    X_num_imputed = pd.DataFrame(num_imputer.fit_transform(X_num), columns=X_num.columns)
else:
    X_num_imputed = X_num

if not X_cat.empty:
    X_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_cat), columns=X_cat.columns)
else:
    X_cat_imputed = X_cat

X = pd.concat([X_num_imputed, X_cat_imputed], axis=1)

# 將非數值型特徵轉換為數值型
X = pd.get_dummies(X, columns=X.columns[X.dtypes == 'object'])

# 分割訓練和測試數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# 使用隨機搜索調整超參數
param_dist = {
    'n_estimators': randint(10, 50),  # 減少範圍
    'max_depth': randint(3, 6),        # 減少範圍
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42)  # 減少迭代次數
random_search.fit(X_train, y_train)

# 最佳參數
best_model = random_search.best_estimator_

# 評估模型
y_pred = best_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# 讀取測試數據
test_df = pd.read_csv(r'C:/Users/nonohuang/OneDrive/桌面/kaggle/kaggle/task4/introml_2024_task4_test_NO_answers_shuffled.csv')

# 處理缺失值
X_test_num = test_df.select_dtypes(include=['int64', 'float64'])
X_test_cat = test_df.select_dtypes(include=['object'])

if not X_test_num.empty:
    X_test_num_imputed = pd.DataFrame(num_imputer.transform(X_test_num), columns=X_test_num.columns)
else:
    X_test_num_imputed = X_test_num

if not X_test_cat.empty:
    X_test_cat_imputed = pd.DataFrame(cat_imputer.transform(X_test_cat), columns=X_test_cat.columns)
else:
    X_test_cat_imputed = X_test_cat

X_test = pd.concat([X_test_num_imputed, X_test_cat_imputed], axis=1)

# 將非數值型特徵轉換為數值型
X_test = pd.get_dummies(X_test, columns=X_test.columns[X_test.dtypes == 'object'])

# 確保測試數據和訓練數據有相同的特徵
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 預測
predictions = best_model.predict(X_test)

# 將數值類別標籤轉換回原始類別標籤
predictions = label_encoder.inverse_transform(predictions)

# 保存結果
output = pd.DataFrame({'id': test_df['id'], 'class': predictions})
output.to_csv('predictions.csv', index=False)