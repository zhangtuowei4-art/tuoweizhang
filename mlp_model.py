import os
import numpy as np

# ==========================
# 1. 模块化层设计 (面向对象)
# ==========================
class LinearLayer:
    # [修改点] 增加 init_method 参数，支持 he 和 xavier 初始化
    def __init__(self, input_dim, output_dim, init_method='he'):
        if init_method == 'he':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        elif init_method == 'xavier':
            # Xavier 初始化更适合 Sigmoid/Tanh
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01
            
        self.b = np.zeros((1, output_dim))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout, reg_lambda, N):
        # [修改点] 移除 / N，修复 L2 正则化被 Batch Size 削弱的 Bug
        self.dW = np.dot(self.x.T, dout) + reg_lambda * self.W
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.W.T)
        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, z):
        self.mask = (z > 0).astype(np.float32)
        return z * self.mask

    def backward(self, dout, reg_lambda=None, N=None):
        return dout * self.mask

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, z):
        z = np.clip(z, -500, 500) # 数值稳定
        self.out = 1.0 / (1.0 + np.exp(-z))
        return self.out

    def backward(self, dout, reg_lambda=None, N=None):
        return dout * self.out * (1.0 - self.out)

class SoftmaxCrossEntropy:
    """将 Softmax 和 CrossEntropy 合并，数值更稳定，梯度推导更简单"""
    def __init__(self):
        self.probs = None
        self.y = None

    def forward(self, z, y):
        self.y = y
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        N = z.shape[0]
        log_likelihood = -np.log(self.probs[np.arange(N), y] + 1e-12)
        loss = np.mean(log_likelihood)
        return loss

    def backward(self, reg_lambda=None, N=None):
        N = self.y.shape[0]
        dz = self.probs.copy()
        dz[np.arange(N), self.y] -= 1
        dz /= N  # 因为 loss 用了 mean，梯度要除以 N
        return dz

# ==========================
# 2. 灵活的 MLP 模型
# ==========================
class MLP:
    def __init__(self, input_dim, hidden1, hidden2, output_dim, activation='relu'):
        act_func = ReLU if activation == 'relu' else Sigmoid
        
        # [修改点] 根据激活函数动态选择初始化策略
        init_method = 'he' if activation == 'relu' else 'xavier'
        
        # 动态构建网络层 (最后一层通常推荐用 xavier)
        self.layers = [
            LinearLayer(input_dim, hidden1, init_method=init_method),
            act_func(),
            LinearLayer(hidden1, hidden2, init_method=init_method),
            act_func(),
            LinearLayer(hidden2, output_dim, init_method='xavier')
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss(self, logits, y, reg_lambda):
        N = logits.shape[0]
        data_loss = self.loss_fn.forward(logits, y)
        
        # [修改点] 移除 / N，修复 L2 正则化被 Batch Size 削弱的 Bug
        reg_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                reg_loss += np.sum(layer.W ** 2)
        reg_loss = 0.5 * reg_lambda * reg_loss
        
        return data_loss + reg_loss

    def backward(self, reg_lambda):
        N = self.loss_fn.y.shape[0]
        # 反向传播梯度
        dout = self.loss_fn.backward(N=N)
        
        for layer in reversed(self.layers):
            if isinstance(layer, LinearLayer):
                dout = layer.backward(dout, reg_lambda, N)
            else:
                dout = layer.backward(dout)
        
        return dout

    def save_weights(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                params[f'W{i}'] = layer.W
                params[f'b{i}'] = layer.b
        np.savez(path, **params)
        print(f"Weights saved to {path}.npz")

    def load_weights(self, path):
        if path.endswith('.npz'):
            path = path[:-4]
        if not os.path.exists(path + '.npz') and not os.path.exists(path):
            raise FileNotFoundError(f"Weight file {path}.npz not found.")
            
        data = np.load(path + '.npz' if not path.endswith('.npz') else path)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                layer.W = data[f'W{i}']
                layer.b = data[f'b{i}']
        print(f"Weights loaded from {path}.npz")


# ==========================
# 3. SGD 优化器 (含学习率衰减)
# ==========================
class SGD:
    def __init__(self, model, lr=0.01, lr_decay=0.95, decay_epoch=5, momentum=0.9):
        self.model = model
        self.lr = lr
        self.init_lr = lr
        self.lr_decay = lr_decay
        self.decay_epoch = decay_epoch
        self.momentum = momentum  # 新增：动量系数，通常设为 0.9
        
        # 新增：初始化速度字典，用于保存历史梯度
        self.velocities = {}
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LinearLayer):
                self.velocities[f'vW{i}'] = np.zeros_like(layer.W)
                self.velocities[f'vb{i}'] = np.zeros_like(layer.b)
        
    def step(self):
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LinearLayer):
                vW_key = f'vW{i}'
                vb_key = f'vb{i}'
                
                # 动量更新公式: v = momentum * v - lr * grad
                self.velocities[vW_key] = self.momentum * self.velocities[vW_key] - self.lr * layer.dW
                self.velocities[vb_key] = self.momentum * self.velocities[vb_key] - self.lr * layer.db
                
                # 参数更新公式: W = W + v
                layer.W += self.velocities[vW_key]
                layer.b += self.velocities[vb_key]
        
    def decay_lr(self, epoch):
        if (epoch + 1) % self.decay_epoch == 0 and epoch > 0:
            self.lr *= self.lr_decay
            print(f"--- Learning rate decayed to {self.lr:.6f} ---")


# ==========================
# 4. 评估与预测工具
# ==========================
def predict(model, X, batch_size=1024):
    preds = []
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        logits = model.forward(X[start:end])
        preds.append(np.argmax(logits, axis=1))
    return np.concatenate(preds)

def calc_accuracy(model, X, y):
    preds = predict(model, X)
    return np.mean(preds == y)


# ==========================
# 5. 训练主循环
# ==========================
def train(model, optimizer, X_train, y_train, X_val, y_val, 
          epochs=50, batch_size=64, reg_lambda=1e-3, save_dir="weights"):
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    N = X_train.shape[0]
    
    for epoch in range(epochs):
        optimizer.decay_lr(epoch)
        
        # Mini-batch 打乱数据
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_train_loss = 0.0
        num_batches = 0
        
        for start in range(0, N, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            logits = model.forward(X_batch)
            loss = model.compute_loss(logits, y_batch, reg_lambda)
            epoch_train_loss += loss
            
            model.backward(reg_lambda)
            optimizer.step()
            num_batches += 1
            
        # ---- Epoch 结束，评估模型 ----
        avg_train_loss = epoch_train_loss / num_batches
        train_acc = calc_accuracy(model, X_train, y_train)
        
        # [修改点] 分批次计算验证集 Loss，且不包含正则化项 (更客观反映模型真实误差)
        val_loss_sum = 0.0
        val_batches = 0
        for start in range(0, X_val.shape[0], batch_size):
            end = start + batch_size
            val_logits = model.forward(X_val[start:end])
            val_loss_sum += model.loss_fn.forward(val_logits, y_val[start:end])
            val_batches += 1
        val_loss = val_loss_sum / val_batches
        
        val_acc = calc_accuracy(model, X_val, y_val)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(save_dir, "best_model")
            model.save_weights(best_path)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}")

    return history