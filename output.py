import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.metrics import confusion_matrix

# [修改点] 统一在头部导入需要的类
from mlp_model import MLP, SGD, train, predict, LinearLayer
from data_load import EuroSATLoader
from sklearn.model_selection import train_test_split

# ==========================
# 1. 超参数搜索模块 (网格搜索)
# ==========================
def hyperparameter_search(X_train, y_train, X_val, y_val, input_dim, output_dim):
    # [修改点] 调整搜索空间：增大网络容量、提高纯SGD的学习率、调小修复Bug后的正则化强度
    search_space = {
        'lr': [5e-3, 1e-3, 5e-4],
        'hidden_dims': [(512, 256), (1024, 512)],
        'reg_lambda': [1e-4, 1e-5, 1e-6],
        'activation': ['relu', 'sigmoid']
    }
    
    keys = search_space.keys()
    values = search_space.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    best_val_acc = 0.0
    best_config = None
    best_history = None
    
    print(f"=== 开始超参数搜索，共 {len(combinations)} 种组合 ===")
    
    for i, config in enumerate(combinations):
        print(f"\n--- 训练组合 {i+1}/{len(combinations)} ---")
        print(f"Config: {config}")
        
        hd = config['hidden_dims']
        model = MLP(input_dim=input_dim, hidden1=hd[0], hidden2=hd[1], 
                    output_dim=output_dim, activation=config['activation'])
        optimizer = SGD(model, lr=config['lr'], lr_decay=0.95, decay_epoch=10, momentum=0.9) 
        history = train(
            model, optimizer, X_train, y_train, X_val, y_val,
            epochs=20, batch_size=128, reg_lambda=config['reg_lambda'], 
            save_dir="tmp_search_weights"
        )
        
        current_best_val_acc = max(history['val_acc'])
        if current_best_val_acc > best_val_acc:
            best_val_acc = current_best_val_acc
            best_config = config
            best_history = history
            print(f"*** 发现更优配置！Val Acc: {best_val_acc:.4f} ***")
            
    print("\n=== 超参数搜索结束 ===")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Config: {best_config}")
    
    return best_config, best_history


# ==========================
# 2. 测试与评估模块
# ==========================
def evaluate_model(model, weight_path, X_test, y_test, class_names):
    model.load_weights(weight_path)
    
    preds = predict(model, X_test)
    acc = np.mean(preds == y_test)
    print(f"\n==> 独立测试集准确率: {acc:.4f} <==")
    
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix on Test Set', fontsize=15)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("混淆矩阵已保存为 confusion_matrix.png")
    
    return acc, preds


# ==========================
# 3. 可视化模块
# ==========================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_acc_curves.png')
    plt.show()
    print("Loss/Acc 曲线已保存为 loss_acc_curves.png")


def visualize_weights(model, class_names, num_neurons=12):
    first_linear_layer = None
    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            first_linear_layer = layer
            break
            
    if first_linear_layer is None:
        print("未找到线性层！")
        return
        
    W1 = first_linear_layer.W  # 形状: (12288, hidden1)
    
    plt.figure(figsize=(15, 6))
    for i in range(min(num_neurons, W1.shape[1])):
        w_img = W1[:, i].reshape(64, 64, 3)
        
        # 归一化到 0-255 以便显示
        w_min, w_max = w_img.min(), w_img.max()
        if w_max - w_min > 0:
            w_img = (w_img - w_min) / (w_max - w_min) * 255
        else:
            w_img = np.zeros_like(w_img)
        w_img = w_img.astype(np.uint8)
        
        plt.subplot(2, 6, i + 1)
        plt.imshow(w_img)
        plt.title(f'Neuron {i+1}')
        plt.axis('off')
        
    plt.suptitle('First Hidden Layer Weights Visualization', fontsize=15)
    plt.tight_layout()
    plt.savefig('weights_visualization.png')
    plt.show()
    print("权重可视化已保存为 weights_visualization.png")


def error_analysis(model, X_test, y_test, preds, class_names, num_samples=6):
    misclassified_indices = np.where(preds != y_test)[0]
    
    if len(misclassified_indices) == 0:
        print("没有分类错误的样本！")
        return
        
    sample_indices = np.random.choice(misclassified_indices, min(num_samples, len(misclassified_indices)), replace=False)
    
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(sample_indices):
        # [注意] 因为输入模型的数据已经标准化，还原显示时需要反标准化或直接用原始数据
        # 这里简单起见，直接用反标准化近似还原图像用于显示
        img = X_test[idx].reshape(64, 64, 3)
        # 由于标准化后数值不在0-255，直接clip显示可能偏暗，但足够观察错例
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        true_label = class_names[y_test[idx]]
        pred_label = class_names[preds[idx]]
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color='red', fontsize=12)
        plt.axis('off')
        
    plt.suptitle('Error Analysis: Misclassified Samples', fontsize=15)
    plt.tight_layout()
    plt.savefig('error_analysis.png')
    plt.show()
    print("错例分析已保存为 error_analysis.png")


# ==========================
# 4. 主程序执行流
# ==========================
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    data_dir = "EuroSAT_RGB"
    
    if not os.path.exists(data_dir):
        print(f"错误: 未找到数据集目录 '{data_dir}'")
        exit()
        
    loader = EuroSATLoader(data_dir)
    X, y = loader.load_data()
    class_names = loader.class_names

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # [修改点] 新增：零均值化和标准化 (极其关键！)
    print("正在对数据进行零均值化和标准化...")
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8 # 加上极小值防止除以0
    
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    print("数据标准化完成！")

    input_dim = X.shape[1]
    output_dim = len(class_names)

    # -------------------------
    # 阶段 2: 超参数搜索
    # -------------------------
    # 提示：网格搜索组合数较多，纯Numpy跑可能需要数小时。
    # 如果只是为了快速验证流程，可以先注释掉搜索，手动写一个 best_config 字典。
    best_config, search_history = hyperparameter_search(
        X_train, y_train, X_val, y_val, input_dim, output_dim
    )
    
    # -------------------------
    # 阶段 3: 用最佳超参数完整训练最终模型
    # -------------------------
    print("\n=== 开始使用最佳超参数训练最终模型 ===")
    hd = best_config['hidden_dims']
    final_model = MLP(input_dim=input_dim, hidden1=hd[0], hidden2=hd[1], 
                      output_dim=output_dim, activation=best_config['activation'])
    final_optimizer = SGD(final_model, lr=best_config['lr'], lr_decay=0.95, decay_epoch=10,momentum=0.9)
    
    final_history = train(
        final_model, final_optimizer, X_train, y_train, X_val, y_val,
        epochs=50, batch_size=128, reg_lambda=best_config['reg_lambda'],
        save_dir="final_model_weights"
    )
    
    best_weight_path = os.path.join("final_model_weights", "best_model")

    # -------------------------
    # 阶段 4: 评估与可视化
    # -------------------------
    plot_history(final_history)
    test_acc, test_preds = evaluate_model(final_model, best_weight_path, X_test, y_test, class_names)
    
    visualize_weights(final_model, class_names, num_neurons=12)
    
    # [修改点] 将标准化后的 X_test 传入，确保前向传播正确；显示逻辑已在函数内部处理
    error_analysis(final_model, X_test, y_test, test_preds, class_names, num_samples=6)