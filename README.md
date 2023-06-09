## 构建两层神经网络对数据集MNIST进行分类
### 1、训练
* 定义激活函数sigmoid、sigmoid函数的导数以及softmax函数
* 初始化模型参数，反向传播函数返回梯度
* 在损失计算函数中，加上 $L_2$ 正则化项 
* train函数中，采用学习率下降策略和优化器SGD
  * 虽然采用学习率衰减的方法能让模型收敛的更好，但是如果遇到鞍点的时候，模型就没法继续收敛，如果学习率此时很小，那将永远    无法走出鞍点；为了解决这一问题，我采用Cyclical Learning Rates(CRL)的方法，设置max_lr与base_lr这两个参数，让      学习率在这两个数之间变化，且max_lr会随时间衰减，从而达到学习率下降的目的
  * 在优化器SGD中，我采用的是mini-batch方法，每次随机获取小批量数据，大小由batch_size来指定，利用这些数据进行梯度更新

### 2、参数查找：学习率，隐藏层大小，正则化强度
* 定义网格搜索函数，在我们给出的包含多个学习率的列表learning_rates, 隐藏层大小列表hidden_sizes以及正则化强度列表lambdas中进行网格搜索，通过比较验证集的分类精度，选出最优的参数，最后输出最优参数及相应的分类精度
  * 学习率learning_rates为 $[ 0.1, 0.05, 0.01, 0.005, 0.001 ]$
  * 隐藏层大小hidden_sizes 为 $[ 50, 100, 200, 300 ]$
  * 正则化强度lambdas 为 $[ 0, 0.1, 0.2, 0.5 ]$

### 3、测试：用经过参数查找后的模型进行测试，输出分类精度
* predict函数用于在给定参数下做出预测
* test函数输出给定模型下数据集的分类精度
* 定义load_data()函数将mnist数据集导入，并分成三部分：训练集(X_train, Y_train)，验证集(X_val, Y_val)和测试集(X_test, Y_test)，数据集详情见[mnist](http://yann.lecun.com/exdb/mnist/)
* 给出确定的超参数以及待搜索的超参数列表：学习率，隐藏层大小和正则化强度
* 调用grid_search模型，得到最佳参数组合best_params
输出结果为：
```
learning_rates, hidden_sizes, lambdas, num_iterations)
学习率: 0.1，隐藏层大小: 50，正则化强度: 0
分类精度: 0.9343
学习率: 0.1，隐藏层大小: 50，正则化强度: 0.2
分类精度: 0.9346
学习率: 0.1，隐藏层大小: 50，正则化强度: 0.5
分类精度: 0.9369
学习率: 0.1，隐藏层大小: 100，正则化强度: 0
分类精度: 0.9337
学习率: 0.1，隐藏层大小: 100，正则化强度: 0.2
分类精度: 0.9345
学习率: 0.1，隐藏层大小: 100，正则化强度: 0.5
分类精度: 0.9349
学习率: 0.1，隐藏层大小: 200，正则化强度: 0
分类精度: 0.9312
学习率: 0.1，隐藏层大小: 200，正则化强度: 0.2
分类精度: 0.932
学习率: 0.1，隐藏层大小: 200，正则化强度: 0.5
分类精度: 0.9316
学习率: 0.01，隐藏层大小: 50，正则化强度: 0
分类精度: 0.8434
学习率: 0.01，隐藏层大小: 50，正则化强度: 0.2
分类精度: 0.8494
学习率: 0.01，隐藏层大小: 50，正则化强度: 0.5
分类精度: 0.8619
学习率: 0.01，隐藏层大小: 100，正则化强度: 0
分类精度: 0.8785
学习率: 0.01，隐藏层大小: 100，正则化强度: 0.2
分类精度: 0.8871
学习率: 0.01，隐藏层大小: 100，正则化强度: 0.5
分类精度: 0.8913
......
最佳参数：{学习率: 0.1，隐藏层大小: 50，正则化强度: 0.5}，分类精度：0.9369
```
### 4、将参数查找后得到的最佳模型中每层网络参数可视化
![W1](https://github.com/Janizhang/NN-task01/blob/main/img/L1.W.png)
![W2](https://github.com/Janizhang/NN-task01/blob/main/img/L2.W.png)
![b1,b2](https://github.com/Janizhang/NN-task01/blob/main/img/Biases.png)
### 5、可视化训练和测试的loss曲线，测试的accuracy曲线
![loss](https://github.com/Janizhang/NN-task01/blob/main/img/loss.png)
![acc](https://github.com/Janizhang/NN-task01/blob/main/img/acc.png)
![lr](https://github.com/Janizhang/NN-task01/blob/main/img/lr.png)

#### img文件夹中是代码运行产生的图片
#### 4.8.py文件中包含训练、参数查找、测试三个部分的代码
#### ipynb文件是对py文件的解释，导出成pdf报告
