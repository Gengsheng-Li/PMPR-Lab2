class ExperimentConfig:
    def __init__(self):
        # 随机种子配置
        self.random_seed = None
        
        # 隐含层节点数实验配置
        self.exp1 = True
        self.hidden_size_config = {
            'hidden_sizes': [5, 10, 20],    # 可以尝试的隐含层节点数列表
            'epochs': 2000,                 # 训练轮数
            'learning_rate': 0.1,           # 学习率
        }
        
        # 学习率实验配置
        self.exp2 = True
        self.learning_rate_config = {
            'learning_rates': [0.001, 0.01, 0.1],  # 可以尝试的学习率列表
            'hidden_size': 10,                     # 固定的隐含层节点数
            'epochs': 2000,                        # 训练轮数
        }
        
        # 自适应学习率实验配置
        self.exp3 = True
        self.adaptive_lr_config = {
            'hidden_size': 10,          # 使用的隐含层节点数
            'initial_lr': 0.1,          # 初始学习率
            'epochs': 2000,             # 训练轮数
            'patience': 3,              # 学习率调整的耐心值
            'lr_decay_factor': 0.9,     # 学习率衰减因子
            'min_lr': 1e-6,             # 最小学习率
        }

        # 指定超参的实验配置
        self.exp4 = True
        self.specific_config = {
            'hidden_size': 10,           # 隐含层节点数
            'epochs': 2000,              # 训练轮数
            'learning_rate': 0.1,        # 学习率
        }
        
        # 通用配置
        self.general_config = {
            'input_size': 3,       # 输入层节点数
            'output_size': 3,      # 输出层节点数
            'save_interval': 100,  # 每隔多少轮保存一次模型
        }