import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from model import BatchNeuralNetwork, StochasticNeuralNetwork
from config import ExperimentConfig

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_experiment_dirname(base_name, params_dict):
    timestamp = get_timestamp()
    params_str = '_'.join([f"{k}-{v}" for k, v in params_dict.items()])
    return f"{base_name}_{params_str}_{timestamp}"

def prepare_data():
    # 第一类数据
    class1_data = np.array([
        [ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63],
        [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
        [ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],
        [-0.76, 0.84, -1.96]
    ])

    # 第二类数据
    class2_data = np.array([
        [ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16],
        [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
        [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],
        [ 0.46, 1.49, 0.68]
    ])

    # 第三类数据
    class3_data = np.array([
        [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69],
        [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
        [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],
        [ 0.66, -0.45, 0.08]
    ])

    # 合并数据并创建标签
    X = np.vstack((class1_data, class2_data, class3_data))
    y = np.zeros((30, 3))
    y[0:10, 0] = 1   # 第一类
    y[10:20, 1] = 1  # 第二类
    y[20:30, 2] = 1  # 第三类

    return X, y

def save_config(config_dict, save_dir):
    config_path = os.path.join(save_dir, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

def train_and_compare_hidden_sizes(X, y, config):
    hidden_sizes = config.hidden_size_config['hidden_sizes']
    epochs = config.hidden_size_config['epochs']
    learning_rate = config.hidden_size_config['learning_rate']
    
    experiment_params = {
        'hidden_sizes': f"{min(hidden_sizes)}to{max(hidden_sizes)}",
        'epochs': epochs,
        'lr': learning_rate
    }
    
    experiment_dir = create_experiment_dirname('hidden_size_exp', experiment_params)
    full_exp_dir = os.path.join('results', experiment_dir)
    os.makedirs(full_exp_dir)
    
    # 保存完整配置
    save_config({
        'hidden_size_config': config.hidden_size_config,
        'general_config': config.general_config
    }, full_exp_dir)
    
    results = []
    for hidden_size in hidden_sizes:
        print(f"Training with hidden size: {hidden_size}")
        
        # 批量训练
        batch_nn = BatchNeuralNetwork(
            config.general_config['input_size'], 
            hidden_size,
            config.general_config['output_size'], 
            learning_rate,
            random_seed=config.random_seed
        )
        batch_loss, batch_accuracy  = batch_nn.train(X, y, epochs)
        
        # 随机训练
        stochastic_nn = StochasticNeuralNetwork(
            config.general_config['input_size'],
            hidden_size,
            config.general_config['output_size'],
            learning_rate,
            random_seed=config.random_seed
        )
        stochastic_loss, stochastic_accuracy = stochastic_nn.train(X, y, epochs)
        
        results.append({
            'hidden_size': hidden_size,
            'batch_final_loss': batch_loss[-1],
            'stochastic_final_loss': stochastic_loss[-1],
            'batch_loss_history': batch_loss,
            'stochastic_loss_history': stochastic_loss,
            'batch_accuracy': batch_accuracy,
            'stochastic_accuracy': stochastic_accuracy
        })
    
    return results, full_exp_dir

def train_and_compare_learning_rates(X, y, config):
    learning_rates = config.learning_rate_config['learning_rates']
    hidden_size = config.learning_rate_config['hidden_size']
    epochs = config.learning_rate_config['epochs']
    
    experiment_params = {
        'lr_range': f"{min(learning_rates)}to{max(learning_rates)}",
        'hidden_size': hidden_size,
        'epochs': epochs
    }
    
    experiment_dir = create_experiment_dirname('learning_rate_exp', experiment_params)
    full_exp_dir = os.path.join('results', experiment_dir)
    os.makedirs(full_exp_dir)
    
    # 保存完整配置
    save_config({
        'learning_rate_config': config.learning_rate_config,
        'general_config': config.general_config
    }, full_exp_dir)
    
    results = []
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        
        # 批量训练
        batch_nn = BatchNeuralNetwork(
            config.general_config['input_size'],
            hidden_size,
            config.general_config['output_size'],
            lr,
            random_seed=config.random_seed
        )
        batch_loss, batch_accuracy = batch_nn.train(X, y, epochs)
        
        # 随机训练
        stochastic_nn = StochasticNeuralNetwork(
            config.general_config['input_size'],
            hidden_size,
            config.general_config['output_size'],
            lr,
            random_seed=config.random_seed
        )
        stochastic_loss, stochastic_accuracy = stochastic_nn.train(X, y, epochs)
        
        results.append({
            'learning_rate': lr,
            'batch_final_loss': batch_loss[-1],
            'stochastic_final_loss': stochastic_loss[-1],
            'batch_loss_history': batch_loss,
            'stochastic_loss_history': stochastic_loss,
            'batch_accuracy': batch_accuracy,
            'stochastic_accuracy': stochastic_accuracy
        })
    
    return results, full_exp_dir

def plot_loss_curves(results, title, x_label, save_dir, exp_type):
    plt.figure(figsize=(12, 6))
    
    for result in results:
        label_value = result[x_label]
        plt.plot(result['batch_loss_history'], 
                label=f'Batch ({x_label}={label_value})')
        plt.plot(result['stochastic_loss_history'], 
                label=f'Stochastic ({x_label}={label_value})')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f'{exp_type}_loss_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_comparison(results, title, x_label, save_dir, exp_type):
    plt.figure(figsize=(12, 6))
    
    x_values = [str(result[x_label]) for result in results]
    batch_acc = [result['batch_accuracy'] for result in results]
    stochastic_acc = [result['stochastic_accuracy'] for result in results]
    
    x = np.arange(len(x_values))
    width = 0.35
    
    plt.bar(x - width/2, batch_acc, width, label='Batch')
    plt.bar(x + width/2, stochastic_acc, width, label='Stochastic')
    
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.title(f'Final Accuracy Comparison ({title})')
    plt.xticks(x, x_values)
    plt.legend()
    plt.grid(True, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(batch_acc):
        plt.text(i - width/2, v, f'{v:.2%}', ha='center', va='bottom')
    for i, v in enumerate(stochastic_acc):
        plt.text(i + width/2, v, f'{v:.2%}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(save_dir, f'{exp_type}_accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_numerical_results(results, save_dir, exp_type):
    with open(os.path.join(save_dir, 'numerical_results.txt'), 'w', encoding='utf-8') as f:
        if exp_type == 'hidden_size':
            f.write("隐含层节点数实验结果:\n")
            for result in results:
                f.write(f"\n隐含层节点数: {result['hidden_size']}\n")
                f.write(f"批量方式最终损失: {result['batch_final_loss']:.6f}\n")
                f.write(f"随机方式最终损失: {result['stochastic_final_loss']:.6f}\n")
        else:
            f.write("学习率实验结果:\n")
            for result in results:
                f.write(f"\n学习率: {result['learning_rate']}\n")
                f.write(f"批量方式最终损失: {result['batch_final_loss']:.6f}\n")
                f.write(f"随机方式最终损失: {result['stochastic_final_loss']:.6f}\n")

def train_and_compare_adaptive_lr(X, y, config):
    hidden_size = config.adaptive_lr_config['hidden_size']
    initial_lr = config.adaptive_lr_config['initial_lr']
    epochs = config.adaptive_lr_config['epochs']
    
    experiment_params = {
        'hidden_size': hidden_size,
        'initial_lr': initial_lr,
        'epochs': epochs
    }
    
    experiment_dir = create_experiment_dirname('adaptive_lr_exp', experiment_params)
    full_exp_dir = os.path.join('results', experiment_dir)
    os.makedirs(full_exp_dir)
    
    # 保存完整配置
    save_config({
        'adaptive_lr_config': config.adaptive_lr_config,
        'general_config': config.general_config
    }, full_exp_dir)
    
    results = []
    
    # 固定学习率训练
    print("Training with fixed learning rate...")
    fixed_batch_nn = BatchNeuralNetwork(
        config.general_config['input_size'],
        hidden_size,
        config.general_config['output_size'],
        initial_lr,
        adaptive_lr=False,
        random_seed=config.random_seed
    )
    fixed_batch_loss, fixed_batch_accuracy  = fixed_batch_nn.train(X, y, epochs)
    
    fixed_stochastic_nn = StochasticNeuralNetwork(
        config.general_config['input_size'],
        hidden_size,
        config.general_config['output_size'],
        initial_lr,
        adaptive_lr=False,
        random_seed=config.random_seed
    )
    fixed_stochastic_loss, fixed_stochastic_accuracy  = fixed_stochastic_nn.train(X, y, epochs)
    
    # 自适应学习率训练
    print("Training with adaptive learning rate...")
    print(f"patience: {config.adaptive_lr_config['patience']}")
    print(f"lr_decay_factor: {config.adaptive_lr_config['lr_decay_factor']}")
    adaptive_batch_nn = BatchNeuralNetwork(
        config.general_config['input_size'],
        hidden_size,
        config.general_config['output_size'],
        initial_lr,
        adaptive_lr=True,
        patience=config.adaptive_lr_config['patience'],
        lr_decay_factor=config.adaptive_lr_config['lr_decay_factor'],
        random_seed=config.random_seed
    )
    adaptive_batch_loss, adaptive_batch_accuracy  = adaptive_batch_nn.train(X, y, epochs)
    
    adaptive_stochastic_nn = StochasticNeuralNetwork(
        config.general_config['input_size'],
        hidden_size,
        config.general_config['output_size'],
        initial_lr,
        adaptive_lr=True,
        patience=config.adaptive_lr_config['patience'],
        lr_decay_factor=config.adaptive_lr_config['lr_decay_factor'],
        random_seed=config.random_seed
    )
    adaptive_stochastic_loss, adaptive_stochastic_accuracy  = adaptive_stochastic_nn.train(X, y, epochs)
    
    results = {
        'fixed_batch_loss': fixed_batch_loss,
        'fixed_stochastic_loss': fixed_stochastic_loss,
        'adaptive_batch_loss': adaptive_batch_loss,
        'adaptive_stochastic_loss': adaptive_stochastic_loss,
        'fixed_batch_accuracy': fixed_batch_accuracy,
        'fixed_stochastic_accuracy': fixed_stochastic_accuracy,
        'adaptive_batch_accuracy': adaptive_batch_accuracy,
        'adaptive_stochastic_accuracy': adaptive_stochastic_accuracy
    }
    
    return results, full_exp_dir

def plot_adaptive_lr_curves(results, save_dir):
    plt.figure(figsize=(12, 6))
    
    plt.plot(results['fixed_batch_loss'], 
            label='Fixed LR (Batch)')
    plt.plot(results['fixed_stochastic_loss'], 
            label='Fixed LR (Stochastic)')
    plt.plot(results['adaptive_batch_loss'], 
            label='Adaptive LR (Batch)')
    plt.plot(results['adaptive_stochastic_loss'], 
            label='Adaptive LR (Stochastic)')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs: Fixed vs Adaptive Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'adaptive_lr_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_adaptive_lr_accuracy(results, save_dir):
    plt.figure(figsize=(10, 6))
    
    methods = ['Fixed (Batch)', 'Fixed (Stochastic)', 
              'Adaptive (Batch)', 'Adaptive (Stochastic)']
    accuracies = [
        results['fixed_batch_accuracy'],
        results['fixed_stochastic_accuracy'],
        results['adaptive_batch_accuracy'],
        results['adaptive_stochastic_accuracy']
    ]
    
    x = np.arange(len(methods))
    plt.bar(x, accuracies)
    
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Final Accuracy: Fixed vs Adaptive Learning Rate')
    plt.xticks(x, methods, rotation=45)
    plt.grid(True, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(accuracies):
        plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adaptive_lr_accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_adaptive_lr_results(results, save_dir):
    with open(os.path.join(save_dir, 'numerical_results.txt'), 'w', encoding='utf-8') as f:
        f.write("自适应学习率实验结果:\n\n")
        f.write(f"固定学习率 (Batch) 最终损失: {results['fixed_batch_loss'][-1]:.6f}\n")
        f.write(f"固定学习率 (Stochastic) 最终损失: {results['fixed_stochastic_loss'][-1]:.6f}\n")
        f.write(f"自适应学习率 (Batch) 最终损失: {results['adaptive_batch_loss'][-1]:.6f}\n")
        f.write(f"自适应学习率 (Stochastic) 最终损失: {results['adaptive_stochastic_loss'][-1]:.6f}\n")

def train_with_specific_config(X, y, config):
    hidden_size = config.specific_config['hidden_size']
    epochs = config.specific_config['epochs']
    learning_rate = config.specific_config['learning_rate']
    
    experiment_params = {
        'hidden_size': hidden_size,
        'epochs': epochs,
        'lr': learning_rate
    }
    
    experiment_dir = create_experiment_dirname('specific_config_exp', experiment_params)
    full_exp_dir = os.path.join('results', experiment_dir)
    os.makedirs(full_exp_dir)
    
    # 保存完整配置
    save_config({
        'specific_config': config.specific_config,
        'general_config': config.general_config
    }, full_exp_dir)
    
    results = []
    
    print(f"Training with hidden size: {hidden_size}")
    print(f"Training with lr: {learning_rate}")
    
    # 批量训练
    batch_nn = BatchNeuralNetwork(
        config.general_config['input_size'], 
        hidden_size,
        config.general_config['output_size'], 
        learning_rate,
        random_seed=config.random_seed
    )
    batch_loss, batch_accuracy  = batch_nn.train(X, y, epochs)
    
    # 随机训练
    stochastic_nn = StochasticNeuralNetwork(
        config.general_config['input_size'],
        hidden_size,
        config.general_config['output_size'],
        learning_rate,
        random_seed=config.random_seed
    )
    stochastic_loss, stochastic_accuracy = stochastic_nn.train(X, y, epochs)
    
    results.append({
        'hidden_size': hidden_size,
        'batch_final_loss': batch_loss[-1],
        'stochastic_final_loss': stochastic_loss[-1],
        'batch_loss_history': batch_loss,
        'stochastic_loss_history': stochastic_loss,
        'batch_accuracy': batch_accuracy,
        'stochastic_accuracy': stochastic_accuracy
    })
    
    return results, full_exp_dir

def run_experiments(config):
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        print(f"Using random seed: {config.random_seed}")
    
    if not os.path.exists('results'):
        os.makedirs('results')

    X, y = prepare_data()
    
    # 实验1：比较不同隐含层节点数
    if config.exp1:
        print("开始实验1：比较不同隐含层节点数的影响...")
        hidden_size_results, hidden_size_dir = train_and_compare_hidden_sizes(X, y, config)
        plot_loss_curves(
            hidden_size_results, 
            'Loss vs Epochs for Different Hidden Layer Sizes',
            'hidden_size',
            hidden_size_dir,
            'hidden_size'
        )
        plot_accuracy_comparison(
            hidden_size_results,
            'Different Hidden Layer Sizes',
            'hidden_size',
            hidden_size_dir,
            'hidden_size'
        )
        save_numerical_results(hidden_size_results, hidden_size_dir, 'hidden_size')
    
    # 实验2：比较不同学习率
    if config.exp2:
        print("开始实验2：比较不同学习率的影响...")
        learning_rate_results, learning_rate_dir = train_and_compare_learning_rates(X, y, config)
        plot_loss_curves(
            learning_rate_results,
            'Loss vs Epochs for Different Learning Rates',
            'learning_rate',
            learning_rate_dir,
            'learning_rate'
        )
        plot_accuracy_comparison(
            learning_rate_results,
            'Different Learning Rates',
            'learning_rate',
            learning_rate_dir,
            'learning_rate'
        )
        save_numerical_results(learning_rate_results, learning_rate_dir, 'learning_rate')
    
    # 实验3：比较固定学习率和自适应学习率
    if config.exp3:
        print("开始实验3：比较固定学习率和自适应学习率的效果...")
        adaptive_lr_results, adaptive_lr_dir = train_and_compare_adaptive_lr(X, y, config)
        plot_adaptive_lr_curves(adaptive_lr_results, adaptive_lr_dir)
        plot_adaptive_lr_accuracy(adaptive_lr_results, adaptive_lr_dir)
        save_adaptive_lr_results(adaptive_lr_results, adaptive_lr_dir)
        
    # 实验4：绘制指定config下的训练loss曲线
    if config.exp4:
        print("开始实验4：绘制指定config下的训练loss曲线...")
        specific_config_results, specific_config_dir = train_with_specific_config(X, y, config)
        plot_loss_curves(
            specific_config_results, 
            f'Loss curves under specific configuration (hidden_size={config.specific_config['hidden_size']}, lr={config.specific_config['learning_rate']})',
            'hidden_size',
            specific_config_dir,
            'hidden_size'
        )

if __name__ == "__main__":
    config = ExperimentConfig()
    run_experiments(config)