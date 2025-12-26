# MobileNetV4 : Architecture et Optimisation

## Sommaire
1. [Introduction](#introduction)
2. [Universal Inverted Bottleneck (UIB)](#universal-inverted-bottleneck-uib)
3. [Mobile Multi-Query Attention (Mobile MQA)](#mobile-multi-query-attention-mobile-mqa)
4. [Pourquoi MobileNetV4 est plus rapide](#pourquoi-mobilenetv4-est-plus-rapide)
5. [Variantes du modèle](#variantes-du-modèle)
6. [Quantification INT8](#quantification-int8)

---

## Introduction

MobileNetV4, présenté à ECCV 2024, représente une avancée majeure dans les architectures efficientes pour mobile. Le papier introduit trois innovations principales :

1. **Universal Inverted Bottleneck (UIB)** : Un bloc unifié et flexible
2. **Mobile MQA** : Un mécanisme d'attention optimisé pour mobile (39% plus rapide)
3. **Recette NAS améliorée** : Une stratégie de recherche d'architecture en deux étapes

**Résultat clé** : MobileNetV4-Hybrid-Large atteint **87% de précision sur ImageNet-1K** avec seulement **3.8ms sur Pixel 8 EdgeTPU**.

**Source** : [MobileNetV4 - Universal Models for the Mobile Ecosystem (arXiv)](https://arxiv.org/abs/2404.10518)

---

## Universal Inverted Bottleneck (UIB)

### Principe

Le bloc UIB étend l'Inverted Bottleneck de MobileNetV2 en ajoutant **deux convolutions depthwise optionnelles** :
- Une **avant l'expansion** (comme ConvNext)
- Une **entre l'expansion et la projection** (comme l'IB classique)

### Les 4 Instantiations du UIB

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Universal Inverted Bottleneck                     │
├─────────────────┬───────────────────────────────────────────────────┤
│                 │  [Optional DW #1]  ← Spatial mixing avant expansion│
│   Input         │         ↓                                          │
│     ↓           │  [1×1 Expansion]   ← Augmentation des canaux       │
│   UIB Block     │         ↓                                          │
│     ↓           │  [Optional DW #2]  ← Spatial mixing après expansion│
│   Output        │         ↓                                          │
│                 │  [1×1 Projection]  ← Réduction des canaux          │
└─────────────────┴───────────────────────────────────────────────────┘
```

| Variante | DW #1 | DW #2 | Caractéristiques |
|----------|-------|-------|------------------|
| **Inverted Bottleneck (IB)** | Non | Oui | Mixing spatial sur features expandues, plus grande capacité, coût plus élevé |
| **ConvNext** | Oui | Non | Mixing spatial moins cher avec grands kernels, avant expansion |
| **ExtraDW** | Oui | Oui | Combine les avantages de ConvNext et IB, augmente la profondeur et le champ réceptif à faible coût |
| **FFN** | Non | Non | Deux convolutions 1×1 pointwise, très efficace sur accélérateurs |

### Pourquoi UIB est efficace

1. **Flexibilité** : Le NAS peut choisir la meilleure variante pour chaque couche
2. **Partage de paramètres** : >95% des paramètres sont partagés entre les variantes
3. **Compromis spatial/canal** : Permet d'ajuster finement le mixing spatial vs canal
4. **Champ réceptif adaptatif** : ExtraDW augmente le champ réceptif sans coût majeur

---

## Mobile Multi-Query Attention (Mobile MQA)

### Le problème de l'attention sur mobile

L'attention multi-têtes classique (MHSA) est lente sur mobile car :
- Accès mémoire élevé (O(n²) pour les matrices d'attention)
- Chaque tête a ses propres K et V → beaucoup de paramètres

### Solution : Multi-Query Attention

Mobile MQA partage les clés (K) et valeurs (V) entre toutes les têtes d'attention, ne gardant que des queries (Q) multiples.

**Formule** :
```
attention_j = softmax((X·W^Qj)·(SR(X)·W^K)^T / √dk) · (SR(X)·W^V)
```

Où `SR` = Spatial Reduction via convolution depthwise avec stride 2 (ou identité).

### Optimisations spécifiques

1. **Downsampling asymétrique** : Les K et V sont à résolution réduite, les Q gardent la haute résolution
2. **Réduction spatiale** : Convolution depthwise stride 2 pour réduire les K/V
3. **Moins d'accès mémoire** : Partage de K/V → 39% plus rapide

```
┌────────────────────────────────────────────────────┐
│              Mobile MQA vs MHSA                     │
├────────────────────┬───────────────────────────────┤
│     MHSA           │     Mobile MQA                │
│  Q1 K1 V1          │  Q1 ─┐                        │
│  Q2 K2 V2          │  Q2 ─┼── K (partagé) V (partagé)
│  Q3 K3 V3          │  Q3 ─┘                        │
│  ...               │                               │
│  Beaucoup de       │  Moins de mémoire             │
│  paramètres        │  39% plus rapide              │
└────────────────────┴───────────────────────────────┘
```

---

## Pourquoi MobileNetV4 est plus rapide

### Analyse Roofline

MobileNetV4 utilise une analyse roofline pour optimiser sur tout le spectre hardware :

```
Latence_modèle = Σ max(Temps_MAC_i, Temps_Mémoire_i)
```

Le **Ridge Point** varie de :
- **0 MACs/byte** : CPU-like (compute-bound)
- **500 MACs/byte** : Accélérateurs (memory-bound)

### Stratégie d'allocation des ressources

| Position | Type de couche | Optimisation |
|----------|---------------|--------------|
| **Début du réseau** | Conv2D larges et coûteuses | Investit beaucoup de MACs pour améliorer la capacité. Coûteux uniquement sur hardware low-RP (CPUs) |
| **Fin du réseau** | Couches FC consistantes | Maximise la précision. Coûteux uniquement sur hardware high-RP (accélérateurs) |

**Résultat** : Le modèle n'est jamais lent sur les deux fronts en même temps.

### Comparaison de performance

| Hardware | MNv4 vs MNv3 |
|----------|--------------|
| CPU | ~2× plus rapide à précision égale |
| EdgeTPU | ~2× plus rapide à précision égale |
| GPU mobile | Pareto-optimal |
| Apple Neural Engine | Pareto-optimal |

### NAS en deux étapes

1. **Recherche grossière** : Facteur d'expansion fixe (4), kernel DW 3×3, détermine les tailles de filtres optimales
2. **Recherche fine** : Cherche la config DW du UIB (présence, taille kernel 3×3 ou 5×5)

**Améliorations** :
- Distillation offline sur JFT pour réduire le bruit
- Entraînement étendu à 750 époques → modèles plus profonds et de meilleure qualité

---

## Variantes du modèle

### Spécifications

| Modèle | Params | MACs | ImageNet Top-1 | Pixel 6 CPU |
|--------|--------|------|----------------|-------------|
| **MNv4-Conv-Small** | 3.8M | 0.2G | 73.8% | 2.4ms |
| **MNv4-Conv-Medium** | 9.2M | 1.0G | 79.9% | 11.4ms |
| **MNv4-Hybrid-Medium** | 10.5M | 1.2G | 80.7% | 14.3ms |
| **MNv4-Conv-Large** | 31M | 5.9G | 82.9% | 59.9ms |
| **MNv4-Hybrid-Large** | 35.9M | 7.2G | 83.4% | 87.6ms |

### Conv vs Hybrid

- **Conv** : Utilise uniquement des convolutions (UIB)
- **Hybrid** : Ajoute Mobile MQA dans les dernières couches

Les modèles Hybrid sont meilleurs en précision mais légèrement plus lents sur CPU. Sur accélérateurs dédiés (EdgeTPU), ils sont très rapides.

### Avec distillation avancée

| Modèle | Précision | Notes |
|--------|-----------|-------|
| MNv4-Conv-Large | 85.9% | 2000 époques, distillation JFT |
| MNv4-Hybrid-Large | 87.0% | Pixel 8 EdgeTPU 3.8ms |

---

## Quantification INT8

### Compatibilité AMD CPU (PyTorch)

Bonne nouvelle : PyTorch supporte très bien la quantification INT8 sur CPUs AMD x86_64.

#### Backends disponibles

| Backend | Description | Speedup vs FP32 |
|---------|-------------|-----------------|
| **FBGEMM** (legacy) | Backend original PyTorch x86 | ~1.43× |
| **x86** (recommandé) | Nouveau backend PyTorch 2.0+, hybride FBGEMM/oneDNN | **~2.97×** |

Le backend **x86** est le défaut depuis PyTorch 2.0 et sélectionne automatiquement entre FBGEMM et oneDNN.

**Prérequis** : CPU x86_64 avec AVX2 ou supérieur (tous les AMD Ryzen modernes).

---

### Post-Training Quantization (PTQ)

La PTQ quantifie un modèle déjà entraîné sans réentraînement.

#### Workflow FX Graph Mode (recommandé)

```python
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx

def quantize_ptq(model, calibration_loader):
    """Post-Training Quantization pour MobileNetV4"""

    # 1. Configuration pour x86 (AMD/Intel)
    # Le backend x86 est automatique sur PyTorch 2.0+
    qconfig_mapping = get_default_qconfig_mapping('x86')

    model = model.eval()

    # 2. Préparer avec un exemple d'entrée
    example_input = torch.randn(1, 3, 224, 224)
    # IMPORTANT: channels_last pour performance optimale sur x86
    example_input = example_input.to(memory_format=torch.channels_last)
    model = model.to(memory_format=torch.channels_last)

    prepared_model = prepare_fx(
        model,
        qconfig_mapping,
        example_inputs=example_input
    )

    # 3. Calibration sur données représentatives
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(memory_format=torch.channels_last)
            prepared_model(images)

    # 4. Conversion en INT8
    quantized_model = convert_fx(prepared_model)

    return quantized_model

# Utilisation
model_fp32 = load_mobilenetv4()
calibration_loader = get_calibration_data(num_samples=1000)
model_int8 = quantize_ptq(model_fp32, calibration_loader)
```

#### Points clés pour PTQ

1. **Données de calibration** : 500-2000 échantillons représentatifs suffisent
2. **Format mémoire** : Toujours utiliser `channels_last` pour x86
3. **Batch size** : 1 fonctionne bien pour la calibration

---

### Quantization-Aware Training (QAT)

Le QAT simule la quantification pendant l'entraînement, produisant des modèles plus robustes.

#### Quand utiliser QAT vs PTQ ?

| Critère | PTQ | QAT |
|---------|-----|-----|
| **Précision** | Peut perdre 1-3% | Généralement <0.5% de perte |
| **Effort** | Simple, pas de réentraînement | Nécessite réentraînement |
| **Cas d'usage** | Modèles larges, précision non critique | Modèles compacts (MobileNet), précision critique |

**Pour MobileNetV4** : Le QAT est souvent nécessaire pour atteindre la précision baseline, surtout pour les petites variantes.

#### Workflow QAT

```python
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx

def quantize_qat(model, train_loader, num_epochs=10, lr=1e-4):
    """Quantization-Aware Training pour MobileNetV4"""

    # 1. Configuration QAT pour x86
    qat_qconfig_mapping = get_default_qat_qconfig_mapping('x86')

    # 2. Préparer pour QAT (mode train)
    model.train()
    example_input = torch.randn(1, 3, 224, 224)
    example_input = example_input.to(memory_format=torch.channels_last)
    model = model.to(memory_format=torch.channels_last)

    prepared_model = prepare_qat_fx(
        model,
        qat_qconfig_mapping,
        example_inputs=example_input
    )

    # 3. Entraînement avec fake quantization
    optimizer = torch.optim.Adam(prepared_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(memory_format=torch.channels_last)

            optimizer.zero_grad()
            outputs = prepared_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # 4. Conversion finale en INT8
    prepared_model.eval()
    quantized_model = convert_fx(prepared_model)

    return quantized_model

# Utilisation
model_fp32 = load_mobilenetv4_pretrained()
train_loader = get_training_data()
model_int8 = quantize_qat(model_fp32, train_loader, num_epochs=10)
```

#### Bonnes pratiques QAT

1. **Learning rate** : Utiliser un LR plus faible que l'entraînement initial (1e-4 à 1e-5)
2. **Époques** : 5-15 époques suffisent généralement
3. **Initialisation** : Partir d'un modèle pré-entraîné FP32
4. **Freeze BN** : Certains gèlent les statistiques BN après quelques époques

---

### Défis spécifiques aux convolutions Depthwise

Les architectures MobileNet utilisent massivement les convolutions depthwise, qui posent des défis particuliers pour la quantification.

#### Le problème

- **Peu de poids par canal** : Chaque canal de sortie dépend d'un petit nombre de poids
- **Variabilité élevée** : Les distributions de poids varient énormément entre canaux
- **Batch Norm folding** : Amplifie les déséquilibres entre canaux
- **Zero-variance** : Certains filtres ont des valeurs quasi-nulles, créant des outliers

**Exemple** : Le 6ème filtre peut avoir une plage dynamique 50× plus grande que les autres filtres.

#### Solutions

##### 1. Quantification per-channel (obligatoire)

```python
from torch.ao.quantization import QConfig, PerChannelMinMaxObserver

# Observer per-channel pour les poids
per_channel_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric
)

# La config par défaut x86 utilise déjà per-channel pour les poids
qconfig = get_default_qconfig_mapping('x86')
```

**Amélioration** : Per-channel améliore la précision de **1.98%** sur MobileNetV1 et **70.6%** sur MobileNetV2 (vs per-tensor).

##### 2. Toujours utiliser QAT pour MobileNet

> "Even with per-channel quantization, networks like MobileNet do not reach baseline accuracy with INT8 PTQ and require QAT."

##### 3. Règles d'entraînement

- **Batch Norm après chaque conv** : Surtout après les convolutions depthwise
- **Weight decay sur DW** : Ne pas exclure les DW du weight decay (même si ça aide en FP32, c'est désastreux pour la quantification)
- **Éviter les activations extrêmes** : Préférer ReLU6 ou hardswish plutôt que des activations non bornées

---

### Nouveau : PyTorch 2 Export (PT2E) avec torchao

PyTorch migre vers `torchao` pour la quantification. Voici l'approche moderne :

```python
import torch
from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight

# Option 1: Poids INT8 seulement (plus simple)
quantize_(model, int8_weight_only())

# Option 2: Activations dynamiques + poids INT8
quantize_(model, int8_dynamic_activation_int8_weight())

# Pour PT2E avec export
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

exported_model = torch.export.export(model, (example_input,))
quantizer = X86InductorQuantizer()
prepared = prepare_pt2e(exported_model, quantizer)
# ... calibration ...
quantized = convert_pt2e(prepared)
```

---

### Benchmarking de l'inférence INT8

```python
import torch
import time

def benchmark_inference(model, input_tensor, num_iterations=1000, warmup=100):
    """Benchmark l'inférence du modèle"""

    model.eval()
    input_tensor = input_tensor.to(memory_format=torch.channels_last)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    latency_ms = (end - start) / num_iterations * 1000
    throughput = num_iterations / (end - start)

    return {
        'latency_ms': latency_ms,
        'throughput_fps': throughput
    }

# Comparaison FP32 vs INT8
input_tensor = torch.randn(1, 3, 224, 224)

fp32_results = benchmark_inference(model_fp32, input_tensor)
int8_results = benchmark_inference(model_int8, input_tensor)

speedup = fp32_results['latency_ms'] / int8_results['latency_ms']
print(f"FP32: {fp32_results['latency_ms']:.2f}ms")
print(f"INT8: {int8_results['latency_ms']:.2f}ms")
print(f"Speedup: {speedup:.2f}x")
```

---

### Résumé quantification

| Aspect | Recommandation |
|--------|----------------|
| **Backend** | x86 (défaut PyTorch 2.0+) |
| **Méthode** | QAT recommandé pour MobileNetV4 |
| **Format mémoire** | `channels_last` obligatoire |
| **Quantification poids** | Per-channel (automatique avec x86) |
| **Données calibration** | 500-2000 échantillons |
| **Speedup attendu** | 2-3× sur AMD x86_64 |

---

## Références

### Papier original
- [MobileNetV4 - Universal Models for the Mobile Ecosystem (arXiv)](https://arxiv.org/abs/2404.10518)
- [ECCV 2024 Publication](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php)

### Quantification PyTorch
- [INT8 Quantization for x86 CPU in PyTorch](https://pytorch.org/blog/int8-quantization/)
- [PyTorch Quantization Documentation](https://docs.pytorch.org/docs/stable/quantization.html)
- [Accelerate PyTorch INT8 Inference with X86 Backend](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-int8-inf-with-new-x86-backend.html)

### Quantification MobileNet spécifique
- [A Quantization-Friendly Separable Convolution for MobileNets](https://arxiv.org/pdf/1803.08607)
- [INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE (NVIDIA)](https://arxiv.org/pdf/2004.09602)

### Tutoriels QAT
- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [QAT Guide with PyTorch (Weights & Biases)](https://wandb.ai/byyoung3/Generative-AI/reports/Quantization-Aware-Training-QAT-A-step-by-step-guide-with-PyTorch--VmlldzoxMTk2NTY2Mw)
- [Kaggle: QAT MobileNetV2 PyTorch](https://www.kaggle.com/code/sushovansaha9/quantize-aware-training-qat-mobilenetv2-pytorch)
