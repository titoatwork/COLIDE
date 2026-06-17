import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib, os, yaml

df = pd.read_csv('data/raw/toniot/train_test_network.csv')

NUMERIC = ['duration','src_bytes','dst_bytes','src_pkts','dst_pkts',
           'src_ip_bytes','dst_ip_bytes','src_port','dst_port','missed_bytes']
CATEGORICAL = ['proto','service','conn_state']

X = df[NUMERIC].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

for col in CATEGORICAL:
    le = LabelEncoder()
    X[col] = le.fit_transform(df[col].astype(str))

FEATURE_COLS = NUMERIC + CATEGORICAL
print(f'Features ({len(FEATURE_COLS)}): {FEATURE_COLS}')

le_label = LabelEncoder()
y = le_label.fit_transform(df['type'])
print(f'Classes: {list(le_label.classes_)}')

X_temp, X_test, y_temp, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

indices = []
for cls in np.unique(y_train):
    cls_idx = np.where(y_train == cls)[0]
    if len(cls_idx) > 10000:
        chosen = np.random.RandomState(42).choice(cls_idx, 10000, replace=False)
        indices.extend(chosen)
    else:
        indices.extend(cls_idx)
X_train = X_train[indices]
y_train = y_train[indices]

smote_strategy = {cls: 5000 for cls in np.unique(y_train) if (y_train==cls).sum() < 5000}
if smote_strategy:
    X_train, y_train = SMOTE(sampling_strategy=smote_strategy, random_state=42).fit_resample(X_train, y_train)

print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

OUT = 'data/processed_toniot'
np.save(f'{OUT}/X_train.npy', X_train)
np.save(f'{OUT}/y_train.npy', y_train)
np.save(f'{OUT}/X_val.npy', X_val)
np.save(f'{OUT}/y_val.npy', y_val)
np.save(f'{OUT}/X_test.npy', X_test)
np.save(f'{OUT}/y_test.npy', y_test)
joblib.dump(le_label, f'{OUT}/label_encoder.pkl')
joblib.dump(scaler, f'{OUT}/scaler.pkl')

config = yaml.safe_load(open(f'{OUT}/config_toniot.yaml'))
config['model']['input_features'] = 13
config['data']['feature_columns'] = FEATURE_COLS
with open(f'{OUT}/config_toniot.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Done. input_features=13')
