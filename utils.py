import numpy as np
import pandas as pd

from sklearn import model_selection

def create_folds(data, task_name, num_splits):
        data["kfold"] = -1
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data.loc[:, "bins"] = pd.cut(data[task_name], bins=num_bins, labels=False)
        kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
            data.loc[v_, 'kfold'] = f
        data = data.drop("bins", axis=1)
        return data

def create_csv_mask(output, input, task_name="5k"):
    df = pd.read_csv(input)
    df = df[df[task_name].notna()]
    df = df.reset_index(drop=True)
    df_5 = create_folds(df, task_name, num_splits=5)
    df_5.to_csv(output, index=False)
    return len(df)