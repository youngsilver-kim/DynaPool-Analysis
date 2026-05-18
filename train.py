# ==============================================================================
# Upgraded DynaPool Benchmark: Adaptive Pooling Analysis Framework (Production-Grade)
# Dataset: Tiny-ImageNet (64x64) | Backbone: Modified ResNet-18 (8x8 feature map)
# Features: Alpha tracking, Entropy calculation, FLOPs/Params/Latency measurement
# Robustness: Per-method checkpointing & auto-resume capability
# ==============================================================================

import os, math, time, random
import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from thop import profile # For FLOPs / Params

# ---- Global config ----
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT     = "data/tiny-imagenet-200" 
OUTPUT_DIR    = "outputs"
EPOCHS        = 120 # 논문용 표준 에포크 상향 (수렴 안정성 극대화 및 리뷰어 지적 방지)
BATCH_SIZE    = 128
BASE_LR       = 1e-3
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1 
NUM_WORKERS   = 2
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
set_seed(42)

# ---- Pooling modules ----
class GeMPooling(nn.Module):
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps
    def forward(self, x):
        p = torch.clamp(self.p, min=1e-3, max=10.0)
        x = torch.clamp(x, min=self.eps)
        x = x.pow(p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        return x.pow(1.0/p).squeeze(-1).squeeze(-1)

class AvgPooling(nn.Module):
    def forward(self, x): return F.adaptive_avg_pool2d(x, 1).flatten(1)

class MaxPooling(nn.Module):
    def forward(self, x): return F.adaptive_max_pool2d(x, 1).flatten(1)

class AttentionPooling(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
    def forward(self, x):
        N, C, H, W = x.shape
        s = self.score(x).view(N, 1, H*W)
        att = F.softmax(s, dim=-1)
        x_flat = x.view(N, C, H*W)
        return torch.bmm(x_flat, att.transpose(1,2)).squeeze(-1)

# ---- Heads ----
class SinglePoolHead(nn.Module):
    def __init__(self, in_channels:int, num_classes:int, kind:str):
        super().__init__()
        self.kind = kind
        if kind == "avg": self.pool = AvgPooling()
        elif kind == "max": self.pool = MaxPooling()
        elif kind == "gem": self.pool = GeMPooling(p_init=3.0)
        elif kind == "att": self.pool = AttentionPooling(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, fmap):
        return self.fc(self.pool(fmap))

class DynaPoolHead(nn.Module):
    def __init__(self, in_channels:int, num_classes:int, hidden=256, dropout=0.1, tau=1.0):
        super().__init__()
        self.tau = tau 
        self.avg, self.max, self.gem = AvgPooling(), MaxPooling(), GeMPooling(p_init=3.0)
        self.att = AttentionPooling(in_channels)
        self.gap = AvgPooling()
        
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4)
        )
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, fmap):
        f_avg, f_max = self.avg(fmap), self.max(fmap)
        f_gem, f_att = self.gem(fmap), self.att(fmap)
        
        logits = self.gate(self.gap(fmap))
        alpha = F.softmax(logits / self.tau, dim=-1)  
        
        f = (alpha[:,0:1]*f_avg + alpha[:,1:2]*f_max + 
             alpha[:,2:3]*f_gem + alpha[:,3:4]*f_att)
        return self.fc(f), alpha

# ---- Backbone ----
class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu) 
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.out_ch = 512
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x 

# ---- Data Loader ----
def get_tiny_imagenet_loaders(root, batch, num_workers=2):
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val', 'images') 
    
    mean, std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
    
    train_tf = T.Compose([
        T.RandomCrop(64, padding=4), 
        T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean, std)
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    
    train_ds = ImageFolder(train_dir, transform=train_tf)
    test_ds = ImageFolder(val_dir, transform=test_tf)
    
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ld, test_ld, train_ds.classes

# ---- Efficiency Measurement ----
def measure_efficiency(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    if device.type == 'cuda':
        for _ in range(50): model(dummy_input)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(200): model(dummy_input)
        torch.cuda.synchronize()
        latency = (time.time() - start) / 200.0 * 1000 
    else:
        latency = 0.0 
        
    return params / 1e6, flops / 1e9, latency

# ---- Train/Eval Functions ----
def accuracy(logits, target): return (logits.argmax(1) == target).float().mean().item()

def cosine_lr(optimizer, base_lr, epoch, total_epochs, warmup=5):
    if epoch < warmup: lr = base_lr*(epoch+1)/warmup
    else:
        progress = (epoch-warmup)/max(1,(total_epochs-warmup))
        lr = 0.5*base_lr*(1+math.cos(math.pi*progress))
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def train_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs, base_lr, label_smoothing):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    run_loss=0; run_acc=0; n=0
    lr = cosine_lr(optimizer, base_lr, epoch, total_epochs)
    
    pbar = tqdm(loader, desc=f"Train {epoch+1}/{total_epochs} (lr={lr:.5f})", ncols=100, leave=True, mininterval=0.5)
    for x,y in pbar:
        x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = ce(logits, y)
        if device.type=='cuda':
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        
        bs = x.size(0); run_loss += loss.item()*bs; run_acc += accuracy(logits,y)*bs; n += bs
        pbar.set_postfix(loss=f"{run_loss/n:.4f}", acc=f"{run_acc/n:.4f}")

@torch.no_grad()
def evaluate_and_analyze(model, loader, device, is_dyna=False):
    model.eval()
    total_acc=0; n=0
    all_alphas, all_labels = [], []
    
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        if is_dyna:
            logits, alpha = out
            all_alphas.append(alpha.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        else:
            logits = out
            
        bs = x.size(0); total_acc += accuracy(logits,y)*bs; n += bs
        
    res = {'acc': total_acc/n}
    if is_dyna:
        alphas = np.vstack(all_alphas)       
        labels = np.concatenate(all_labels)  
        entropy = -np.sum(alphas * np.log(alphas + 1e-9), axis=1)
        res['mean_entropy'] = np.mean(entropy)
        res['alphas'] = alphas
        res['labels'] = labels
        
    return res

# ---- Runner (체크포인트 영구 보존 및 이어받기 메커니즘 탑재) ----
def run_method(method:str, classes):
    backbone = ResNet18Backbone().to(DEVICE)
    if method == "dyna":
        head = DynaPoolHead(backbone.out_ch, 200).to(DEVICE)
    else:
        head = SinglePoolHead(backbone.out_ch, 200, kind=method).to(DEVICE)
        
    class Model(nn.Module):
        def __init__(self, bb, hd): super().__init__(); self.bb=bb; self.hd=hd
        def forward(self, x): return self.hd(self.bb(x))
        
    model = Model(backbone, head).to(DEVICE)
    params, flops, latency = measure_efficiency(model, DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=='cuda'))

    # 경로 정의
    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_{method}.pth")
    best_path = os.path.join(OUTPUT_DIR, f"best_{method}.pth")
    
    start_epoch = 0
    best_acc = 0.0
    best_analysis = {}
    accumulated_time = 0.0

    # 기존 진행 데이터 완전 복구 검사
    if os.path.exists(ckpt_path):
        print(f"[*] '{method}'의 기존 훈련 세션을 감지했습니다. 복구를 시작합니다...")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        best_analysis = checkpoint['best_analysis']
        accumulated_time = checkpoint.get('accumulated_time', 0.0)
        print(f"[*] 복구 완료: 에포크 {start_epoch}부터 시작합니다. (기존 최고 검증 정확도: {best_acc*100:.2f}%)")

    # 만약 이미 끝까지 다 돈 스케줄이라면 연산 없이 결과 즉시 반환 (완전 스킵 메커니즘)
    if start_epoch >= EPOCHS:
        print(f"[+] '{method}'는 이미 {EPOCHS} 에포크 완주 기록이 있으므로 훈련을 완전히 건너뜁니다.")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=DEVICE)
            best_acc = best_ckpt['best_acc']
            best_analysis = best_ckpt['best_analysis']
            accumulated_time = best_ckpt.get('accumulated_time', accumulated_time)
        return best_acc, accumulated_time, params, flops, latency

    # 에포크 루프 구동
    for ep in range(start_epoch, EPOCHS):
        start_t = time.time()
        train_epoch(model, train_loader, opt, scaler, DEVICE, ep, EPOCHS, BASE_LR, LABEL_SMOOTH)
        eval_res = evaluate_and_analyze(model, test_loader, DEVICE, is_dyna=(method=="dyna"))
        
        epoch_mins = (time.time() - start_t) / 60.0
        accumulated_time += epoch_mins
        
        is_best = eval_res['acc'] > best_acc
        if is_best:
            best_acc = eval_res['acc']
            best_analysis = eval_res
            
        # 에포크 종료 시마다 체크포인트 스냅샷 덤프
        checkpoint_state = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc,
            'best_analysis': best_analysis,
            'accumulated_time': accumulated_time
        }
        torch.save(checkpoint_state, ckpt_path)
        if is_best:
            torch.save(checkpoint_state, best_path)

    # 분석 결과 저장 (DynaPool 한정)
    if method == "dyna" and 'alphas' in best_analysis:
        df = pd.DataFrame(best_analysis['alphas'], columns=['Avg', 'Max', 'GeM', 'Attn'])
        df['class_idx'] = best_analysis['labels']
        df['class_name'] = df['class_idx'].map(lambda idx: classes[idx])
        
        class_mean_alphas = df.groupby('class_name')[['Avg', 'Max', 'GeM', 'Attn']].mean()
        csv_path = os.path.join(OUTPUT_DIR, "class_pooling_preference.csv")
        class_mean_alphas.to_csv(csv_path)
        print(f"\n[Analysis] DynaPool Gating Entropy: {best_analysis['mean_entropy']:.4f}")
        print(f"[Analysis] Saved class-wise alpha weights to: {csv_path}")

    return best_acc, accumulated_time, params, flops, latency

# ==========================================
# Main Execution Block
# ==========================================
print("Loading Tiny-ImageNet...")
train_loader, test_loader, class_names = get_tiny_imagenet_loaders(DATA_ROOT, BATCH_SIZE, NUM_WORKERS)

methods = [("Average Pooling","avg"),
           ("Max Pooling","max"),
           ("GeM Pooling","gem"),
           ("Attention Pooling","att"),
           ("DynaPool (Ours)","dyna")]

summary = []
for name, key in methods:
    print(f"\n==== Training Target Method: {name} ====")
    acc, mins, p, f, lat = run_method(key, class_names)
    summary.append((name, acc, p, f, lat))

# ---- Print Final Analysis Table ----
print("\n" + "="*85)
print("Table 1. Enhanced Efficiency & Accuracy Benchmark on Tiny-ImageNet (64x64)")
print("-" * 85)
header = ["Method", "Top-1 Acc(%)", "Params(M)", "FLOPs(G)", "Latency(ms)"]
print(f"{header[0]:<20} {header[1]:>12} {header[2]:>12} {header[3]:>12} {header[4]:>15}")
for row in summary:
    name, acc, p, f, lat = row
    print(f"{name:<20} {acc*100:>12.2f} {p:>12.2f} {f:>12.3f} {lat:>15.2f}")
print("=" * 85)
print("Note: DynaPool's alpha values are exported to the outputs/ folder for visualization.")

# 영구 보존용 결과 백업 시스템 구축 (터미널 버퍼가 지워져도 디스크에 안전하게 남아있도록 함)
df_summary = pd.DataFrame(summary, columns=["Method", "Top-1 Acc(%)", "Params(M)", "FLOPs(G)", "Latency(ms)"])
df_summary["Top-1 Acc(%)"] = df_summary["Top-1 Acc(%)"] * 100
summary_csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
df_summary.to_csv(summary_csv_path, index=False)
print(f"[System] 벤치마크 마스터 데이터가 안전하게 저장되었습니다: {summary_csv_path}")
