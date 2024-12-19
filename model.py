import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, adaptive_lr=False, random_seed=None, patience=5, lr_decay_factor=0.5):
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 初始化权重和偏置
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        
        # 自适应学习率相关参数
        if adaptive_lr:
            self.prev_loss = float('inf')
            self.patience = patience                # 容忍多少轮loss没有改善
            self.lr_decay_factor = lr_decay_factor  # 学习率衰减因子
            self.patience_counter = 0
            self.min_lr = 1e-6
        
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        
    def adjust_learning_rate(self, current_loss):
        if not self.adaptive_lr:
            return
            
        if current_loss >= self.prev_loss:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.learning_rate = max(self.learning_rate * self.lr_decay_factor, self.min_lr)
                self.patience_counter = 0
                print(f"Learning rate adjusted to: {self.learning_rate}")
        else:
            self.patience_counter = 0
            
        self.prev_loss = current_loss
    
    def forward(self, X):
        # 隐含层
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self.tanh(self.hidden_input)
        
        # 输出层
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.output = self.sigmoid(self.output_input)
        
        return self.output
    
    def backward(self, X, y, output):
        self.output_error = output - y
        self.output_delta = self.output_error * self.sigmoid_derivative(self.output_input)
        
        self.hidden_error = np.dot(self.output_delta, self.weights2.T)
        self.hidden_delta = self.hidden_error * self.tanh_derivative(self.hidden_input)
        
        # 计算梯度
        dw2 = np.dot(self.hidden_output.T, self.output_delta)
        db2 = np.sum(self.output_delta, axis=0, keepdims=True)
        dw1 = np.dot(X.T, self.hidden_delta)
        db1 = np.sum(self.hidden_delta, axis=0, keepdims=True)
        
        return dw1, db1, dw2, db2
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def compute_accuracy(self, y_true, y_pred):
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        
        return np.mean(pred_labels == true_labels)

class BatchNeuralNetwork(NeuralNetwork):
    def train(self, X, y, epochs):
        loss_history = []
        
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y, output)
            loss_history.append(loss)
            
            # 调整学习率
            self.adjust_learning_rate(loss)
            
            # 反向传播
            dw1, db1, dw2, db2 = self.backward(X, y, output)
            
            self.weights1 -= self.learning_rate * dw1
            self.bias1 -= self.learning_rate * db1
            self.weights2 -= self.learning_rate * dw2
            self.bias2 -= self.learning_rate * db2
            
        final_output = self.forward(X)
        final_accuracy = self.compute_accuracy(y, final_output)
        
        return loss_history, final_accuracy

class StochasticNeuralNetwork(NeuralNetwork):
    def train(self, X, y, epochs):
        loss_history = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_i = X[idx:idx+1]
                y_i = y[idx:idx+1]
                
                # 前向传播
                output = self.forward(x_i)
                
                # 计算损失
                loss = self.compute_loss(y_i, output)
                total_loss += loss
                
                # 调整学习率
                if idx == indices[-1]:  # 一个epoch结束时调整学习率
                    self.adjust_learning_rate(total_loss / n_samples)
                
                # 反向传播
                dw1, db1, dw2, db2 = self.backward(x_i, y_i, output)
                
                self.weights1 -= self.learning_rate * dw1
                self.bias1 -= self.learning_rate * db1
                self.weights2 -= self.learning_rate * dw2
                self.bias2 -= self.learning_rate * db2
            
            loss_history.append(total_loss / n_samples)
            
        final_output = self.forward(X)
        final_accuracy = self.compute_accuracy(y, final_output)
        
        return loss_history, final_accuracy