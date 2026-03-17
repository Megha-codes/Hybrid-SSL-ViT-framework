# Hybrid Self-Supervised Vision Transformer (Hybrid-SSL-ViT)

The proposed method in this project combines contrastive learning (SimCLR), bootstrap-based learning (BYOL), self-distillation (DINO), and masked feature reconstruction (MAE) using a Vision Transformer (ViT) backbone.

The objective is to learn robust and generalizable visual representations without reliance on labelled data, enabling strong performance in downstream tasks such as classification, medical imaging, and low-data regimes.


## 2. Methodology

### 2.1 Multi-View Augmentation
Given an input image \( x \), two stochastic augmentations are generated:

$\[x_1 = t_1(x), \quad x_2 = t_2(x)\]$

where $\( t_1, t_2 \sim \mathcal{T} \)$ are sampled from a distribution of augmentations.


### 2.2 Encoder and Projection
A Vision Transformer encoder \( f_\theta \) extracts representations:

$\[h_i = f_\theta(x_i)\]$

These representations are passed through a projection head \( g_\theta \):

$\[z_i = g_\theta(h_i)\]$

Normalised embeddings:

$\[\tilde{z}_i = \frac{z_i}{\|z_i\|}\]$

### 2.3 SimCLR Loss (Contrastive Objective)

The InfoNCE loss is defined as:

\[
\mathcal{L}_{SimCLR} = -\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{j=1}^{2N} \exp(\text{sim}(z_i, z_j)/\tau)}
\]

where:
- \( \text{sim}(a,b) = \frac{a^\top b}{\|a\|\|b\|} \)
- \( \tau \) is a temperature parameter


### 2.4 BYOL Loss (Bootstrap Objective)

A predictor network \( q_\theta \) is applied:

\[
p_i = q_\theta(z_i)
\]

A momentum encoder \( f_{\theta'} \) produces targets:

\[
z_i' = g_{\theta'}(f_{\theta'}(x_i))
\]

Loss:

\[
\mathcal{L}_{BYOL} = 2 - 2 \cdot \frac{p_i \cdot z_j'}{\|p_i\|\|z_j'\|}
\]


### 2.5 DINO Loss (Self-Distillation)

Student and teacher outputs:

\[
p_s = \text{softmax}(z_s / T_s), \quad p_t = \text{softmax}(z_t / T_t)
\]

Loss:

\[
\mathcal{L}_{DINO} = - \sum p_t \log p_s
\]


### 2.6 Masked Autoencoder (MAE)

Random masking is applied to latent features:

\[
\hat{h} = M(h)
\]

A decoder reconstructs masked features:

\[
\tilde{h} = d_\theta(\hat{h})
\]

Reconstruction loss:

\[
\mathcal{L}_{MAE} = \| \tilde{h} - h \|_2^2
\]


### 2.7 Total Objective

The final loss is a weighted combination:

\[
\mathcal{L}_{total} =
\lambda_1 \mathcal{L}_{SimCLR} +
\lambda_2 \mathcal{L}_{BYOL} +
\lambda_3 \mathcal{L}_{DINO} +
\lambda_4 \mathcal{L}_{MAE}
\]

## 4. Implementation Details

- Backbone: Vision Transformer (ViT-Base, patch size 16)
- Optimizer: AdamW
- Learning Rate: 3e-4
- Batch Size: 128
- Temperature (SimCLR): 0.2
- EMA Decay (Teacher): 0.996
- Mask Ratio (MAE): 0.75
- Mixed Precision Training enabled

---

## 5. Linear Evaluation Protocol

To evaluate representation quality, a linear classifier is trained on frozen encoder features:

\[
y = W h + b
\]

where \( h \) is the frozen representation.

---

## 6. Use Cases

- Medical imaging (e.g., OCT scans)
- Low-label or unlabeled datasets
- Pretraining for downstream vision tasks
- Feature extraction pipelines

---

## 7. Future Work

- Integration with multimodal learning (vision-language models)
- Scaling to larger datasets (ImageNet, LAION)
- Incorporation of masked token prediction (MAE-ViT variants)
- Deployment in real-world medical AI systems
