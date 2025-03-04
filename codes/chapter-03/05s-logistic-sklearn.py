from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def load_dataset(data_path, test_per=0.2):
    # 读取数据
    raw_data = pd.read_csv(data_path)
    raw_data = raw_data.sample(frac=1)
    data = raw_data.iloc[:, 0:4].to_numpy(dtype=np.float32)

    # 标签处理
    label = raw_data['label'].to_numpy()
    label_set = np.unique(label).tolist()
    label = np.array([[label_set.index(l)] for l in label])

    # 数据预处理
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean)/std

    test_start = round(len(data) * (1 - test_per))

    # 数据划分
    test_data = data[test_start:]
    test_label = label[test_start:]
    data = data[0:test_start]
    label = label[0:test_start]
    return data, label, test_data, test_label


train_data, train_label, test_data, test_label = load_dataset("../dataset/iris.data-bi_class.txt")


model = LogisticRegression()
model.fit(train_data, train_label[:,0])

test_pred = model.predict(test_data)
print(f'正确率{np.mean(test_pred == test_label[:,0])}')
