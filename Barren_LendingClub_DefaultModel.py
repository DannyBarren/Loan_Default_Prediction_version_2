import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Section 1: Data Loading and Feature Transformation
df = pd.read_csv('loan_data.csv')
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())
encoder = OneHotEncoder(sparse_output=False, drop='first')
purpose_encoded = encoder.fit_transform(df[['purpose']])
purpose_columns = encoder.get_feature_names_out(['purpose'])
df_encoded = pd.DataFrame(purpose_encoded, columns=purpose_columns)
df = pd.concat([df.drop('purpose', axis=1), df_encoded], axis=1)
df.to_csv('transformed_loan_data.csv', index=False)
print("\nTransformed First 5 Rows:\n", df.head())

# Section 2: Exploratory Data Analysis (EDA) & Handling Imbalance
print("\nDescriptive Statistics:\n", df.describe())
print("Target Class Distribution:\n", df['not.fully.paid'].value_counts(normalize=True))
numerical_cols = ['installment', 'log.annual.inc', 'dti', 'fico', 
                  'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 
                  'delinq.2yrs', 'pub.rec']
df[numerical_cols].hist(bins=20, figsize=(14, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()
sns.boxplot(x='not.fully.paid', y='fico', data=df)
plt.title('FICO Score vs. Default')
plt.show()

# Section 3: Additional Feature Engineering
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
columns_to_drop = ['int.rate']  # Optional: Add 'revol.util' if correlated >0.7 with 'revol.bal'
if columns_to_drop:
    df = df.drop(columns_to_drop, axis=1)
    print(f"Dropped columns: {columns_to_drop}")
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
print("Train Shape:", X_train.shape, "Test Shape:", X_test.shape)

# Apply SMOTE with adjusted neighbors
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Resampled Train Shape:", X_train_res.shape)

# Section 4: Modeling with Deep Learning
model = Sequential()
model.add(Dense(64, input_dim=X_train_res.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train_res, y_train_res, epochs=150, batch_size=64, validation_split=0.2, 
                    callbacks=[early_stop])
y_pred_prob = model.predict(X_test)
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx_pr = np.argmax(f1_scores)
optimal_threshold_pr = thresholds_pr[optimal_idx_pr]
y_pred_opt_pr = (y_pred_prob > optimal_threshold_pr).astype(int)
print("Optimal PR Threshold:", optimal_threshold_pr)
print("Classification Report with PR Optimal Threshold:\n", classification_report(y_test, y_pred_opt_pr))
print('ROC-AUC:', roc_auc_score(y_test, y_pred_prob))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training History')
plt.show()
pd.DataFrame(y_pred_opt_pr, columns=['predicted_default']).to_csv('output.csv', index=False)
print("Predictions saved to output.csv")