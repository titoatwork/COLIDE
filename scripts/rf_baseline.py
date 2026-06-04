import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
classification_report,
confusion_matrix,
precision_recall_fscore_support
)

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ============================================================

# Paths

# ============================================================

PROJECT_ROOT = (
Path(__file__)
.resolve()
.parent
.parent
)

TRAIN_PATH = (
PROJECT_ROOT /
"data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
)

TEST_PATH = (
PROJECT_ROOT /
"data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"
)

# ============================================================

# Features

# ============================================================

FEATURE_COLS = [

"N_IN_Conn_P_DstIP",
"N_IN_Conn_P_SrcIP",
"drate",
"max",
"mean",
"min",
"seq",
"srate",
"state_number",
"stddev"

]

LABEL_COL = "category"

# ============================================================

# Load Data

# ============================================================

print("\nLoading data...")

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

X = df_train[FEATURE_COLS]
y = df_train[LABEL_COL]

X_test = df_test[FEATURE_COLS]
y_test_raw = df_test[LABEL_COL]

# ============================================================

# Label Encoding

# ============================================================

le = LabelEncoder()

y = le.fit_transform(y)
y_test = le.transform(y_test_raw)

print("\nClasses:")
for idx, name in enumerate(le.classes_):
    print(idx, name)

# ============================================================

# Train / Validation Split

# ============================================================

X_train, X_val, y_train, y_val = train_test_split(

X,
y,

test_size=0.10,

stratify=y,

random_state=SEED

)

# ============================================================

# Undersample DDoS / DoS

# ============================================================

class_map = {
name: idx
for idx, name in enumerate(le.classes_)
}

undersample_strategy = {

class_map["DDoS"]: 100000,
class_map["DoS"]: 100000

}

rus = RandomUnderSampler(

sampling_strategy=undersample_strategy,

random_state=SEED

)

X_train, y_train = rus.fit_resample(
X_train,
y_train
)

print("\nAfter undersampling:")
print(pd.Series(y_train).value_counts().sort_index())

# ============================================================

# Class-Specific SMOTE

# ============================================================

normal_id = class_map["Normal"]
theft_id = class_map["Theft"]

X_df = pd.DataFrame(X_train)
y_series = pd.Series(y_train)

# --------------------------

# Normal

# --------------------------

smote_normal = SMOTE(

sampling_strategy={
    normal_id: 2000
},

k_neighbors=3,

random_state=SEED

)

X_train, y_train = smote_normal.fit_resample(
X_df,
y_series
)

# --------------------------

# Theft

# --------------------------

smote_theft = SMOTE(

sampling_strategy={
    theft_id: 1000
},

k_neighbors=2,

random_state=SEED

)

X_train, y_train = smote_theft.fit_resample(
X_train,
y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts().sort_index())

# ============================================================

# Scaling

# ============================================================

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)

# ============================================================

# Random Forest

# ============================================================

print("\nTraining RandomForest...")

rf = RandomForestClassifier(

n_estimators=100,

random_state=SEED,

n_jobs=-1

)

rf.fit(
X_train,
y_train
)

# ============================================================

# Evaluation Helper

# ============================================================

def evaluate(name, X_data, y_true):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    y_pred = rf.predict(X_data)

    print(

        classification_report(

            y_true,
            y_pred,

            target_names=le.classes_,

            labels=list(
                range(
                    len(le.classes_)
                )
            ),

            digits=4,

            zero_division=0

        )

    )

    macro = precision_recall_fscore_support(

        y_true,
        y_pred,

        average="macro",

        zero_division=0

    )

    weighted = precision_recall_fscore_support(

        y_true,
        y_pred,

        average="weighted",

        zero_division=0

    )

    print(
        f"Macro-F1: {macro[2]:.4f}"
    )

    print(
        f"Weighted-F1: {weighted[2]:.4f}"
    )

    print("\nConfusion Matrix:")

    print(

        confusion_matrix(

            y_true,
            y_pred,

            labels=list(
                range(
                    len(le.classes_)
                )
            )

        )

    )

# ============================================================

# Validation

# ============================================================

evaluate(

"VALIDATION RESULTS",

X_val,

y_val

)

# ============================================================

# Test

# ============================================================

evaluate(

"TEST RESULTS",

X_test,

y_test

)
