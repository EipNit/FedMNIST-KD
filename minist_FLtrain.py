import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化

# 调整数据形状
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 联邦学习参数
num_clients = 10  # 客户端数量
num_rounds = 100  # 训练轮数
batch_size = 32
learning_rate = 0.0005  # 调整后的学习率
kd_temperature = 5.0    # 调整后的蒸馏温度
dirichlet_alpha = 0.5   # Dirichlet 分布的 alpha 参数

# 创建结果保存文件夹
os.makedirs("FL_result", exist_ok=True)
loss_acc_file = os.path.join("FL_result", "loss_accuracy_data.csv")

# 使用 Dirichlet 分布进行 Non-IID 数据划分
def generate_non_iid_data(x, y, num_clients, alpha=dirichlet_alpha):
    num_classes = y.shape[1]
    y_labels = np.argmax(y, axis=1)
    client_data_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        indices = np.where(y_labels == c)[0]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, proportions)
        for client_idx, split in enumerate(splits):
            client_data_indices[client_idx].extend(split)

    client_datasets = [(x[indices], y[indices]) for indices in client_data_indices]
    return client_datasets

# 生成客户端数据
client_datasets = generate_non_iid_data(x_train, y_train, num_clients, alpha=dirichlet_alpha)

# 定义 MLP 模型
def build_mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 定义 CNN 模型
def build_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 知识蒸馏函数
def knowledge_distillation(teacher, student, x_train, temperature=kd_temperature):
    teacher_logits = teacher.predict(x_train) / temperature
    soft_targets = tf.nn.softmax(teacher_logits, axis=1)
    student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.KLDivergence(), metrics=['accuracy'])
    student.fit(x_train, soft_targets, epochs=3, batch_size=batch_size, verbose=0)

# 客户端模型
client_models = [build_mlp() if i % 2 == 0 else build_cnn() for i in range(num_clients)]

# 初始化全局教师模型
global_teacher = build_mlp()

# 跟踪损失和准确率
loss_list, acc_list = [], []

# 联邦学习训练
for round_num in range(num_rounds):
    print(f"Round {round_num + 1}/{num_rounds}")

    for client_id in range(num_clients):
        x_client, y_client = client_datasets[client_id]
        client_models[client_id].fit(x_client, y_client, epochs=1, batch_size=batch_size, verbose=0)
        knowledge_distillation(client_models[client_id], global_teacher, x_client)

    # 评估全局教师模型
    loss, acc = global_teacher.evaluate(x_test, y_test, verbose=0)
    loss_list.append(loss)
    acc_list.append(acc)
    print(f"Test Accuracy after Round {round_num + 1}: {acc * 100:.2f}%")

# 保存损失和准确率数据
pd.DataFrame({"Round": range(1, num_rounds + 1), "Loss": loss_list, "Accuracy": acc_list}).to_csv(loss_acc_file, index=False)

print(f"最终准确率: {acc_list[-1] * 100:.2f}%")
