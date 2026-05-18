# DynaPool

**Input-adaptive pooling benchmark** on CIFAR-100 with a ResNet-18 backbone.  
DynaPool learns **per-instance mixture weights** over four simple operators: **Average, Max, GeM, Attention** via a lightweight gating MLP.

> **Main finding.** On our controlled benchmark, **GeM** attains the best top-1 accuracy (0.5833); **Max** is close behind (0.5819) with lower training time. **DynaPool** is competitive while offering **instance-wise interpretability** at **modest cost**.

---

## 1. Features
- Unified benchmark for {Avg, Max, GeM, Attention} pooling on Tiny-ImageNet
- DynaPool: two-layer MLP gating → mixture coefficients \( \alpha \) per input
- Optional entropy regularization for non-collapsed \(\alpha\)
- Clean training script & YAML configs for ablations

---

## 2. Environment

```bash
# create environment (example with conda)
conda create -n dynapool python=3.10 -y
conda activate dynapool

# install dependencies
pip install -r requirements.txt
