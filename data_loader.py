import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(train_path, test_path, num_clients, batch_size, exclude_labels=[]):
    df_train = pd.read_feather(train_path)
    df_test = pd.read_feather(test_path)

    # 라벨 전처리
    df_train['Label'] = df_train['Label'].str.lower().str.replace(' ', '_')
    df_test['Label'] = df_test['Label'].str.lower().str.replace(' ', '_')

    # 불필요한 컬럼 제거
    df_train = df_train.drop(columns=['index'], errors='ignore')
    df_test = df_test.drop(columns=['index'], errors='ignore')

    # LabelBinary (이진 탐지용, 현재는 사용하지 않음)
    df_train['LabelBinary'] = (df_train['Label'] != 'normal').astype(int)
    df_test['LabelBinary'] = (df_test['Label'] != 'normal').astype(int)

    # Feature 선택
    numeric_features_train = df_train.select_dtypes(include=['float64', 'int64']).copy()
    if 'LabelBinary' not in numeric_features_train.columns:
        numeric_features_train['LabelBinary'] = df_train['LabelBinary']

    correlation_matrix = numeric_features_train.corr()
    correlation_with_label = correlation_matrix['LabelBinary'].abs().sort_values(ascending=False)
    selected_features = correlation_with_label.index[:21].drop('LabelBinary', errors='ignore')

    # 정규화
    scaler = StandardScaler()
    data_scaled_train = scaler.fit_transform(df_train[selected_features])
    data_scaled_test = scaler.transform(df_test[selected_features])

    # 라벨 수동 매핑 (exclude_labels 맨 뒤로 정렬)
    all_labels = sorted(df_train['Label'].unique())
    excluded = [label for label in all_labels if label in exclude_labels]
    non_excluded = [label for label in all_labels if label not in exclude_labels]
    reordered_labels = non_excluded + excluded

    label_to_code = {label: i for i, label in enumerate(reordered_labels)}
    code_to_label = {i: label for label, i in label_to_code.items()}

    df_train['LabelCode'] = df_train['Label'].map(label_to_code)
    df_test['Label'] = df_test['Label'].where(df_test['Label'].isin(reordered_labels))
    df_test['LabelCode'] = df_test['Label'].map(label_to_code)

    input_dim = data_scaled_train.shape[1]
    output_dim = len(label_to_code)

    # 클라이언트별 데이터 분배
    train_loaders_all = []
    train_loaders_excluded = []

    for _ in range(num_clients):
        # 전체 데이터 로더
        X_all = torch.tensor(data_scaled_train, dtype=torch.float32)
        y_all = torch.tensor(df_train['LabelCode'].values, dtype=torch.long)
        dataset_all = TensorDataset(X_all, y_all)
        loader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=True, drop_last=True)
        train_loaders_all.append(loader_all)

        # 제외된 데이터 로더
        df_train_excluded = df_train[~df_train['Label'].isin(exclude_labels)]
        data_scaled_train_excluded = scaler.transform(df_train_excluded[selected_features])
        X_excluded = torch.tensor(data_scaled_train_excluded, dtype=torch.float32)
        y_excluded = torch.tensor(df_train_excluded['LabelCode'].values, dtype=torch.long)
        dataset_excluded = TensorDataset(X_excluded, y_excluded)
        loader_excluded = DataLoader(dataset_excluded, batch_size=batch_size, shuffle=True, drop_last=True)
        train_loaders_excluded.append(loader_excluded)

    # 테스트 데이터
    data_scaled_test = scaler.transform(df_test[selected_features])
    X_test = torch.tensor(data_scaled_test, dtype=torch.float32)
    y_test = torch.tensor(df_test['LabelCode'].values, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders_all, train_loaders_excluded, test_loader, input_dim, output_dim, code_to_label
