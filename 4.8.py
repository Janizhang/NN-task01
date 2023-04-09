#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:04:32 2023

@author: ynzhang
"""

import numpy as np

def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid 函数的导数"""
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def relu(Z):
    return np.maximum(0, Z)

# def softmax(Z):
#     t = np.exp(Z)
#     return t / np.sum(t, axis=0, keepdims=True)


def initialize_parameters(input_size, hidden_size, output_size):
    """初始化模型参数"""
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_propagation(X, parameters):
    """前向传播"""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 第一层
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    
    # 第二层
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache

def backward_propagation(X, Y, cache, parameters, lambd):
    """反向传播"""
    m = X.shape[0]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # 计算输出层误差
    dZ2 = A2 - Y.T
    
    # 计算输出层权重和偏置的梯度
    dW2 = np.dot(dZ2, A1.T) / m # + lambd * W2 / m 这里有必要加正则化吗
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    # 计算隐藏层误差
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(cache["Z1"])
    
    # 计算隐藏层权重和偏置的梯度
    dW1 = np.dot(dZ1, X) / m # + lambd * W1 / m  还有这里
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return gradients


def compute_loss(A2, Y, parameters, lambd):
    """计算损失"""
    n = Y.shape[0]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    cross_entropy_loss = -np.sum(Y.T * np.log(A2)) / n
    L2_regularization = (np.sum(np.square(W1)) + np.sum(np.square(W2))) * lambd / 2
    
    loss = cross_entropy_loss + L2_regularization
    
    return loss


def update_parameters(parameters, gradients, learning_rate):
    """更新模型参数"""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]
    
    # 更新权重和偏置
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    np.savez("model.npz", **parameters)
    
    return parameters

def train(X, Y, input_size, hidden_size, output_size, learning_rate, num_iterations, lambd, gamma):
    """训练模型"""
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    lr = 0
    max_lr = learning_rate
    base_lr = max_lr/4
    for i in range(num_iterations):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = X[batch_mask]
        y_batch = Y[batch_mask]
        
        A2, cache = forward_propagation(x_batch, parameters)
        #loss = compute_loss(A2, y_batch, parameters, lambd)
        gradients = backward_propagation(x_batch, y_batch, cache, parameters, lambd)
        parameters = update_parameters(parameters, gradients, lr)
        
        if i % 2000 == 0:
            #lr *= gamma
            cycle = np.floor(1+i/(2*step_size))
            x = np.abs(i/step_size - 2*cycle + 1)
            lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))*gamma**(i/2000)
            
    return parameters


def grid_search(X_train, Y_train, X_val, Y_val, input_size, output_size, learning_rates, hidden_sizes, lambdas, num_iterations):
    """网格搜索参数"""
    best_accuracy = -1
    best_params = {}
    for lr in learning_rates:
        for hs in hidden_sizes:
                for ld in lambdas:
                    print(f"学习率: {lr}，隐藏层大小: {hs}，正则化强度: {ld}")
                    k = 0.002/lr
                    num_iterations += int(k*10000)
                    parameters, accuracy = train_pro(X_train, Y_train, input_size, hs, output_size, lr, num_iterations, ld, gamma)
                    # accuracy = test(X_val, Y_val, parameters)
                    print(f"分类精度: {accuracy}")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        r, h, d = lr, hs, ld
    best_params = {"learning_rate": r, "hidden_size": h, "lambda": d}
    print(f"最佳参数: {best_params}，分类精度: {best_accuracy}")
    return best_params

def predict(X, parameters):
    """模型预测"""
    A2, _ = forward_propagation(X, parameters)
    predictions = np.argmax(A2, axis=0)
    return predictions

def test(X, Y, parameters):
    """测试模型"""
    predictions = predict(X, parameters)
    accuracy = np.mean(predictions == np.argmax(Y.T, axis=0))
    print(f"分类精度: {accuracy}")
    return accuracy

# 迭代训练
def train_pro(X, Y, input_size, hidden_size, output_size, learning_rate, num_iterations, lambd, gamma):
    """训练模型"""
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    lr = 0
    max_lr = learning_rate
    base_lr = max_lr/4
    # 在训练过程中定义一个变量来记录最佳的验证集分类精度
    best_val_acc = 0.0

    # 定义一个变量来记录连续验证集分类精度不再提升的次数
    no_improvement_count = 0
    max_no_improvement = 1000
    # 定义一个阈值，用于判断验证集分类精度是否提升
    val_acc_threshold = 0.001

    for i in range(num_iterations):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = X[batch_mask]
        y_batch = Y[batch_mask]
        
        A2, cache = forward_propagation(x_batch, parameters)
        #loss = compute_loss(A2, y_batch, parameters, lambd)
        gradients = backward_propagation(x_batch, y_batch, cache, parameters, lambd)
        parameters = update_parameters(parameters, gradients, lr)
        
        step_size = 500
        if i % step_size == 0:
            cycle = np.floor(1+i/(2*step_size))
            x = np.abs(i/step_size - 2*cycle + 1)
            lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))*gamma**(i/2000)
        
        predictions = predict(X_val, parameters)
        val_acc = np.mean(predictions == np.argmax(Y_val.T, axis=0))
        # 判断验证集分类精度是否提升
        if val_acc > best_val_acc + val_acc_threshold:
            best_val_acc = val_acc
            no_improvement_count = 0
        elif val_acc > 0.9:
            no_improvement_count += 1
        else:
            continue
        # 如果连续no_improvement_count次验证集分类精度未提升，则停止迭代
        if no_improvement_count >= max_no_improvement:
            #print("Validation accuracy did not improve for {} epochs. Stopping training.".format(no_improvement_count))
            break
    #print(f"分类精度: {val_acc}")
    return parameters, val_acc


"""第三部分"""

# 导入训练好的模型
def load_model(filename):
    model_params = np.load(filename)
    return model_params

model_params = load_model('model.npz')

# 加载数据集
import gzip
import pickle
import numpy as np

def one_hot(y):
    """将类别向量转换为独热编码"""
    one_hot_y = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        one_hot_y[i, y[i]] = 1
    return one_hot_y


def load_data():
    with gzip.open('/Users/ynzhang/Documents/fudan university/神经网络与机器学习/作业一/mnist.pkl.gz', 'rb') as f:
        train_data, val_data, test_data = pickle.load(f, encoding='latin1')
        X_train, Y_train = train_data[0], one_hot(train_data[1])
        X_val, Y_val = val_data[0], one_hot(val_data[1])
        X_test, Y_test = test_data[0], one_hot(test_data[1])
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
# 加载数据集


# 参数查找

input_size = 784
output_size = 10
learning_rates = [0.1, 0.01, 0.001]
hidden_sizes = [50, 100, 200]
lambdas = [0, 0.2, 0.5]
num_iterations = 10000
train_size = X_train.shape[0]
batch_size = 100
gamma = 0.9

best_params = grid_search(X_train, Y_train, X_val, Y_val, input_size, output_size, learning_rates, hidden_sizes, lambdas, num_iterations)

parameters = train(X_train, Y_train, input_size, best_params["hidden_size"], output_size, best_params["learning_rate"], num_iterations, best_params["lambda"],gamma)
test(X_test, Y_test, parameters)



import matplotlib.pyplot as plt


# 假设已经训练好了一个两层神经网络，权重保存在W1和W2中，偏置保存在b1和b2中

# 获取网络参数
W1 = parameters['W1']# 第一层权重
W2 = parameters['W2']# 第二层权重
b1 = parameters['b1']# 第一层偏置
b2 = parameters['b2']# 第二层偏置


# 绘制第一层权重的热力图
plt.imshow(W1, cmap='coolwarm', interpolation='nearest',aspect=10)

# 添加颜色条
plt.colorbar()

# 设置标题和坐标轴标签
plt.title("Layer 1 Weights Heatmap")
plt.xlabel("hidden")
plt.ylabel("input")

# 显示图像
plt.show()

# 绘制第一层偏置的热力图
plt.imshow(b1, cmap='Pastel1', interpolation='nearest', aspect=0.04)

# 添加颜色条
plt.colorbar()

# 设置标题和坐标轴标签
plt.title("Layer 1 Biases Heatmap")
plt.xlabel("biases")
plt.ylabel("hidden")

# 显示图像
plt.show()

# 绘制第二层权重的热力图
plt.imshow(W2, cmap='coolwarm', interpolation='nearest',aspect=4)

# 添加颜色条
plt.colorbar()

# 设置标题和坐标轴标签
plt.title("Layer 2 Weight Heatmap")
plt.xlabel("output")
plt.ylabel("hidden")

# 显示图像
plt.show()

# 绘制第二层偏置的热力图
plt.imshow(b2, cmap='Pastel1', interpolation='nearest', aspect=0.07)

# 添加颜色条
plt.colorbar()

# 设置标题和坐标轴标签
plt.title("Layer 2 Biases Heatmap")
plt.xlabel("biases")
plt.ylabel("output")

# 显示图像
plt.show()

def plot(best_params, X, Y):
    lr = best_params['learning_rate']
    hs = best_params['hidden_size']
    lm = best_params['lambda']
    acc = []
    acc_val = []
    loss_list = []
    loss_val =  []
    lr_list = []
    parameters = initialize_parameters(input_size, hs, output_size)
    for i in range(num_iterations):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = X[batch_mask]
        y_batch = Y[batch_mask]
        
        A2, cache = forward_propagation(x_batch, parameters)
        loss = compute_loss(A2, y_batch, parameters, lm)
        gradients = backward_propagation(x_batch, y_batch, cache, parameters, lm)
        parameters = update_parameters(parameters, gradients, lr)
        
        step_size = 2000
        if i % 2000 == 0:
            #lr *= gamma
            cycle = np.floor(1+i/(2*step_size))
            x = np.abs(i/step_size - 2*cycle + 1)
            lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))*gamma**(i/2000)
            lr_list.append(lr)
            #print(f"迭代次数: {i}，损失: {loss}")
            accuracy1 = test(X_train, Y_train, parameters)
            accuracy2 = test(X_val, Y_val, parameters)
            acc.append(accuracy1)
            acc_val.append(accuracy2)
            
            A2, cache = forward_propagation(X_train, parameters)
            loss1 = compute_loss(A2, Y_train, parameters, lm)
            loss_list.append(loss1)
            
            A2, cache = forward_propagation(X_val, parameters)
            loss2 = compute_loss(A2, Y_val, parameters, lm)
            loss_val.append(loss2)
            print(f"学习率为{lr:.4f}时，训练集分类精度: {accuracy1}, 验证集分类精度:{accuracy2}")
    plt.figure()
     
    
    epochs = range(len(acc))
     
    plt.plot(epochs, acc, 'r', label='Training acc') # 'bo'为画蓝色圆点，不连线
    plt.plot(epochs, acc_val, 'b', label='Validation acc') 
    plt.title('Training and validation accuracy')
    plt.legend() # 绘制图例，默认在右上角
     
    plt.figure()
     
    plt.plot(epochs, loss_list, 'bo', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.figure()
     
    plt.plot(epochs, lr_list, 'b', label='learning rate')
    plt.show()

plot(best_params, X_train, Y_train)    

def plot1(best_params, X, Y):
    lr = 0
    max_lr = 0.1
    base_lr = 0.005
    step_size = 2000
    hs = best_params['hidden_size']
    lm = best_params['lambda']
    acc = []
    acc_val = []
    loss_list = []
    loss_val =  []
    lr_list = []
    parameters = initialize_parameters(input_size, hs, output_size)
    for i in range(num_iterations):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = X[batch_mask]
        y_batch = Y[batch_mask]
        
        A2, cache = forward_propagation(x_batch, parameters)
        loss = compute_loss(A2, y_batch, parameters, lm)
        gradients = backward_propagation(x_batch, y_batch, cache, parameters, lm)
        parameters = update_parameters(parameters, gradients, lr)
        
        if i % 500 == 0 and lr < max_lr and i<5000:  
    
            lr_list.append(lr)
            #print(f"迭代次数: {i}，损失: {loss}")
            accuracy1 = test(X_train, Y_train, parameters)
            accuracy2 = test(X_val, Y_val, parameters)
            acc.append(accuracy1)
            acc_val.append(accuracy2)
            
            A2, cache = forward_propagation(X_train, parameters)
            loss1 = compute_loss(A2, Y_train, parameters, lm)
            loss_list.append(loss1)
            
            A2, cache = forward_propagation(X_val, parameters)
            loss2 = compute_loss(A2, Y_val, parameters, lm)
            loss_val.append(loss2)
            print(f"学习率为{lr:.4f}时，训练集分类精度: {accuracy1}, 验证集分类精度:{accuracy2}")
            lr += 0.01
            
        elif i % 2000 == 0:
            #lr *= gamma
            cycle = np.floor(1+i/(2*step_size))
            x = np.abs(i/step_size - 2*cycle + 1)
            lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))*gamma**(i/2000)
            lr_list.append(lr)
            #print(f"迭代次数: {i}，损失: {loss}")
            accuracy1 = test(X_train, Y_train, parameters)
            accuracy2 = test(X_val, Y_val, parameters)
            acc.append(accuracy1)
            acc_val.append(accuracy2)
            
            A2, cache = forward_propagation(X_train, parameters)
            loss1 = compute_loss(A2, Y_train, parameters, lm)
            loss_list.append(loss1)
            
            A2, cache = forward_propagation(X_val, parameters)
            loss2 = compute_loss(A2, Y_val, parameters, lm)
            loss_val.append(loss2)
            print(f"学习率为{lr:.4f}时，训练集分类精度: {accuracy1}, 验证集分类精度:{accuracy2}")
            
        else:
            continue
        
    plt.figure()
     
    
    epochs = range(len(acc))
     
    plt.plot(epochs, acc, 'r', label='Training acc') # 'bo'为画蓝色圆点，不连线
    plt.plot(epochs, acc_val, 'b', label='Validation acc') 
    plt.title('Training and validation accuracy')
    plt.legend() # 绘制图例，默认在右上角
     
    plt.figure()
     
    plt.plot(epochs, loss_list, 'bo', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.figure()
     
    plt.plot(epochs, lr_list, 'b', label='learning rate')
    plt.show()

plot1(best_params, X_train, Y_train)    

for i in range(num_iterations):
    # 获取mini-batch
    max_lr = 0.1
    base_lr = 0.001
    step_size = 2000
    if i % 2000 == 0:
        #lr *= gamma
        cycle = np.floor(1+i/(2*step_size))
        x = np.abs(i/step_size - 2*cycle + 1)
        lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))*gamma**(i/2000)
        print(lr)

def find(best_params, X, Y):
    lr = 0
    max_lr = 0.1
    base_lr = 0.001
    step_size = 2000
    hs = best_params['hidden_size']
    lm = best_params['lambda']
    acc = []
    acc_val = []
    loss_list = []
    loss_val =  []
    lr_list = []
    parameters = initialize_parameters(input_size, hs, output_size)
    for i in range(10000):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = X[batch_mask]
        y_batch = Y[batch_mask]
        
        A2, cache = forward_propagation(x_batch, parameters)
        loss = compute_loss(A2, y_batch, parameters, lm)
        gradients = backward_propagation(x_batch, y_batch, cache, parameters, lm)
        parameters = update_parameters(parameters, gradients, lr)
        
        if i % 100 == 0 and lr < max_lr:  
    
            lr_list.append(lr)
            #print(f"迭代次数: {i}，损失: {loss}")
            accuracy1 = test(X_train, Y_train, parameters)
            accuracy2 = test(X_val, Y_val, parameters)
            acc.append(accuracy1)
            acc_val.append(accuracy2)
            
            A2, cache = forward_propagation(X_train, parameters)
            loss1 = compute_loss(A2, Y_train, parameters, lm)
            loss_list.append(loss1)
            
            A2, cache = forward_propagation(X_val, parameters)
            loss2 = compute_loss(A2, Y_val, parameters, lm)
            loss_val.append(loss2)
            print(f"学习率为{lr:.4f}时，训练集分类精度: {accuracy1}, 验证集分类精度:{accuracy2}")
            lr += base_lr
            
    plt.figure()
     
    
    epochs = range(len(acc))
     
    plt.plot(epochs, acc, 'r', label='Training acc') # 'bo'为画蓝色圆点，不连线
    plt.plot(epochs, acc_val, 'b', label='Validation acc') 
    plt.title('Training and validation accuracy')
    plt.legend() # 绘制图例，默认在右上角
     
    plt.figure()
     
    plt.plot(epochs, loss_list, 'bo', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.figure()
     
    plt.plot(epochs, lr_list, 'b', label='learning rate')
    plt.show()

find(best_params, X_train, Y_train)   