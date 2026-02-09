import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import sys
import logging
import os

# 配置日志记录器：同时输出到文件和控制台
log_filename = "training_log_att.txt"
if os.path.exists(log_filename):
    os.remove(log_filename) # 每次运行前删除旧日志，如需追加请注释此行

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # 写入文件
        logging.StreamHandler(sys.stdout)                    # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)

logger.info("代码开始运行...")
logger.info(f"日志将保存在: {log_filename}")

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 数据预处理与加载
# ==========================================
def load_and_process_data(filepath):
    logger.info(f"正在加载数据: {filepath}")
    df = pd.read_csv(filepath)
    
    # 处理缺失值：Rainfall存在NaN，用0填充
    df['Rainfall'] = df['Rainfall'].fillna(0)
    
    # 变量分类
    continuous_cols = ['Humidity', 'Temp', 'Apparent_Temp', 'Hour_Sin', 'Hour_Cos', 'Rainfall']
    binary_cols = ['Is_Weekend', 'Is_Holiday', 'Has_Rainfall']
    target_col = 'Load'
    feature_cols = continuous_cols + binary_cols
    
    # 归一化 (StandardScaler)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # 先进行数据集切分 (8:2)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # 3. 仅在训练集上 FIT，然后同时 TRANSFORM
    # 处理特征 (X)
    train_df[feature_cols] = scaler_x.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler_x.transform(test_df[feature_cols]) # 注意：这里是 transform

    # 处理目标值 (Y)
    train_df[[target_col]] = scaler_y.fit_transform(train_df[[target_col]])
    test_df[[target_col]] = scaler_y.transform(test_df[[target_col]]) # 注意：这里是 transform

    # 转换为 numpy
    train_data = train_df[feature_cols + [target_col]].values.astype(np.float32)
    test_data = test_df[feature_cols + [target_col]].values.astype(np.float32)
    
    logger.info("数据加载与预处理完成。")
    return train_data, test_data, scaler_y, len(feature_cols)

class LoadForecastDataset(Dataset):
    def __init__(self, data, seq_len=96, pred_len=96):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(data) - seq_len - pred_len + 1
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 输入 Load: [seq_len, 1] (只取最后一列)
        x_load = self.data[idx : idx + self.seq_len, -1:]
        
        # 输入 Covariates: [seq_len + pred_len, n_features] (取除最后一列外的所有列)
        # TiDE需要历史+未来的外生变量
        x_cov = self.data[idx : idx + self.seq_len + self.pred_len, :-1]
        
        # 标签 Load: [pred_len, 1]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, -1:]
        
        return torch.FloatTensor(x_load), torch.FloatTensor(x_cov), torch.FloatTensor(y)

# ==========================================
# 2. 模型定义 (DLinear分解 + TiDE编码)
# ==========================================

# 残差块定义
class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ResBlock, self).__init__()
        # 主路径
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 残差路径：如果输入输出维度不一致，需要一个线性投影
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()
            
        # 层归一化
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        # 先相加再归一化 (Add & Norm)
        return self.norm(x + res)


# 注意力块    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, query, key_value):
        # query: [Batch, query_dim] -> 历史编码
        # key_value: [Batch, Pred_Len, key_dim] -> 未来外生变量
        
        # 1. 投影与维度转换
        # query: [Batch, 1, hidden_dim]
        q = self.query_proj(query).unsqueeze(1)
        # k, v: [Batch, Pred_Len, hidden_dim]
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)

        # 2. 计算注意力权重 (Dot-product Attention)
        # scores: [Batch, 1, Pred_Len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 3. 加权求和
        # context: [Batch, 1, hidden_dim]
        context = torch.matmul(attn_weights, v)
        
        # 4. 映射回原维度并进行残差连接
        # out: [Batch, query_dim]
        out = self.out_proj(context.squeeze(1))
        return out + query # 残差结构
    

class DLinearTiDE(nn.Module):
    def __init__(self, seq_len, pred_len, n_cov, hidden_dim, dropout, cnn_kernel):
        super(DLinearTiDE, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. 趋势分解 (Trend Decomposition via CNN)
        # 使用Conv1d替代MA，kernel_size需为奇数以方便padding保持维度
        padding = (cnn_kernel - 1) // 2
        self.trend_conv = nn.Conv1d(in_channels=1, out_channels=1, 
                                    kernel_size=cnn_kernel, padding=padding, padding_mode='replicate')
        
        # 2. 修改：Encoder 仅负责编码历史信息 (历史负载 + 历史外生变量)
        hist_input_dim = seq_len * 1 + seq_len * n_cov
        self.hist_encoder = nn.Sequential(
            ResBlock(hist_input_dim, hidden_dim, hidden_dim, dropout),
            ResBlock(hidden_dim, hidden_dim, hidden_dim, dropout)
        )

        # 3. 新增：跨维度注意力层
        # Query 是历史编码 (hidden_dim)，Key/Value 是未来外生变量 (n_cov)
        self.cross_attn = CrossAttention(hidden_dim, n_cov, hidden_dim)
        
        # 4. TiDE 解码器 (使用残差块)
        # 将 hidden_dim 映射回 pred_len，内部包含非线性变换
        self.decoder = ResBlock(hidden_dim, hidden_dim, pred_len, dropout)
        
        # 4. Trend Component Predictor (Linear)
        # 简单的线性映射：历史趋势 -> 未来趋势
        self.trend_linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x_load, x_cov):
        # x_load: [Batch, Seq_Len, 1]
        # x_cov: [Batch, Seq_Len + Pred_Len, n_cov]
        batch_size = x_load.shape[0]
        
        # --- 1. Decomposition（趋势分解） ---
        # 调整维度适配Conv1d: [Batch, Channels, Seq_Len]
        x_load_perm = x_load.permute(0, 2, 1)
        trend = self.trend_conv(x_load_perm).permute(0, 2, 1) # [Batch, Seq_Len, 1]
        seasonal = x_load - trend
        
        # --- 2. Feature Splitting (拆分历史与未来外生变量) ---
        # x_cov 维度为 [Batch, Seq_Len + Pred_Len, n_cov]
        hist_cov = x_cov[:, :self.seq_len, :]    # [Batch, Seq_Len, n_cov]
        future_cov = x_cov[:, self.seq_len:, :]  # [Batch, Pred_Len, n_cov]
        
        # --- 3. History Encoding (历史编码) ---
        seasonal_flat = seasonal.reshape(batch_size, -1)
        hist_cov_flat = hist_cov.reshape(batch_size, -1)
        # 仅将历史信息输入 Encoder
        hist_input = torch.cat([seasonal_flat, hist_cov_flat], dim=1)
        hist_encoded = self.hist_encoder(hist_input) # [Batch, hidden_dim]
        
        # --- 4. Cross-Attention (跨维度注意力融合) ---
        # 用历史状态去 Query 未来的外生变量特征
        # attn_out 将包含“受未来天气/节假日调整后”的历史上下文
        attn_out = self.cross_attn(hist_encoded, future_cov) # [Batch, hidden_dim]
        
        # --- 5. Decoding & Recomposition (解码与重组) ---
        pred_seasonal = self.decoder(attn_out).unsqueeze(-1)
        
        trend_flat = trend.reshape(batch_size, -1)
        pred_trend = self.trend_linear(trend_flat).unsqueeze(-1)
        
        return pred_seasonal + pred_trend

# ==========================================
# 3. 训练与评估工具函数
# ==========================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    for x_load, x_cov, y in loader:
        x_load, x_cov, y = x_load.to(device), x_cov.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x_load, x_cov)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate_epoch(model, loader, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for x_load, x_cov, y in loader:
            x_load, x_cov, y = x_load.to(device), x_cov.to(device), y.to(device)
            output = model(x_load, x_cov)
            loss = criterion(output, y)
            losses.append(loss.item())
    return np.mean(losses)

def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {'MAPE': mape, 'WAPE': wape, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# ==========================================
# 4. 主程序：数据加载、贝叶斯优化、训练绘图
# ==========================================
if __name__ == "__main__":
    # 参数配置
    FILE_PATH = 'Area2_Data.csv'
    SEQ_LEN = 192  # 过去48小时
    PRED_LEN = 96 # 预测未来24小时
    BATCH_SIZE = 32

    # 加载数据
    train_data, test_data, scaler_y, n_features = load_and_process_data(FILE_PATH)

    # HPO用的验证集 (从训练集中分出最后10%)
    val_size_hpo = int(len(train_data) * 0.1)
    train_data_hpo = train_data[:-val_size_hpo]
    val_data_hpo = train_data[-val_size_hpo:]

    # 贝叶斯优化目标函数
    def objective(trial):
        # 搜索空间
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        cnn_kernel = trial.suggest_categorical('cnn_kernel', [3, 5, 7, 15, 25])
        
        # 构建数据集与Loader
        train_ds = LoadForecastDataset(train_data_hpo, SEQ_LEN, PRED_LEN)
        val_ds = LoadForecastDataset(val_data_hpo, SEQ_LEN, PRED_LEN)
        
        # 随机采样加速搜索 (Optional, 可去掉Sampler使用全量)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        model = DLinearTiDE(SEQ_LEN, PRED_LEN, n_features, hidden_dim, dropout, cnn_kernel).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 快速训练几轮以评估
        for epoch in range(3): 
            train_epoch(model, train_loader, criterion, optimizer)
        
        # 验证集Loss
        val_loss = validate_epoch(model, val_loader, criterion)
        
        # 计算MAPE作为优化目标
        preds, actuals = [], []
        model.eval()
        with torch.no_grad():
            for x_l, x_c, y in val_loader:
                x_l, x_c, y = x_l.to(device), x_c.to(device), y.to(device)
                out = model(x_l, x_c)
                preds.append(out.cpu().numpy())
                actuals.append(y.cpu().numpy())
        
        preds = scaler_y.inverse_transform(np.concatenate(preds).squeeze(-1))
        actuals = scaler_y.inverse_transform(np.concatenate(actuals).squeeze(-1))
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-5))) * 100
        
        return mape

    # 执行贝叶斯优化
    logger.info("开始进行贝叶斯超参数优化...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # 建议设置为10-20次
    logger.info(f"最佳参数组合: {study.best_params}")
    logger.info(f"最佳验证集 MAPE: {study.best_value:.4f}%")

    # 使用最佳参数进行最终全量训练
    best_params = study.best_params
    full_train_ds = LoadForecastDataset(train_data, SEQ_LEN, PRED_LEN)
    test_ds = LoadForecastDataset(test_data, SEQ_LEN, PRED_LEN)

    train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    final_model = DLinearTiDE(SEQ_LEN, PRED_LEN, n_features, 
                            best_params['hidden_dim'], 
                            best_params['dropout'], 
                            best_params['cnn_kernel']).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    EPOCHS = 15 # 最终训练轮数

    logger.info("开始进行最终全量训练...")
    for epoch in range(EPOCHS):
        t_loss = train_epoch(final_model, train_loader, criterion, optimizer)
        v_loss = validate_epoch(final_model, test_loader, criterion)
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}: Train Loss {t_loss:.4f}, Test Loss {v_loss:.4f}")

    # 5. 绘图与评估
    # 绘制Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig('loss_curve_att.png')
    logger.info("损失曲线已保存为 loss_curve_att.png")

    # 计算最终指标
    preds_list, actuals_list = [], []
    final_model.eval()
    with torch.no_grad():
        for x_l, x_c, y in test_loader:
            x_l, x_c, y = x_l.to(device), x_c.to(device), y.to(device)
            out = final_model(x_l, x_c)
            preds_list.append(out.cpu().numpy())
            actuals_list.append(y.cpu().numpy())

    # 逆归一化
    preds_inv = scaler_y.inverse_transform(np.concatenate(preds_list, axis=0).squeeze(-1))
    actuals_inv = scaler_y.inverse_transform(np.concatenate(actuals_list, axis=0).squeeze(-1))

    metrics = calculate_metrics(actuals_inv, preds_inv)
    logger.info("=== 最终测试集评估指标 ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    # 选择一天进行可视化 (随机选择一个样本)
    sample_idx = 0
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_inv[sample_idx], label='Ground Truth')
    plt.plot(preds_inv[sample_idx], label='Prediction', linestyle='--')
    plt.title(f'Load Forecast for One Day (Sample {sample_idx})')
    plt.xlabel('Time Steps (15 min)')
    plt.ylabel('Load')
    plt.legend()
    plt.savefig(f'forecast_sample_att.png')
    logger.info(f"可视化结果已保存为 forecast_sample_att.png")
