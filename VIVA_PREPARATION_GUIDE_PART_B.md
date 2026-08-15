# CSE 445 Computer Vision — Assignment Part B: Master Viva Preparation Guide
**Author:** Md. Sadik Shahriar (Student ID: 2023-2-60-103)  
**Instructor:** Dr. Mohammad Rifat Ahmmad Rashid  
**Course:** CSE 445 — Computer Vision (Summer 2026), East West University  
**Project:** Self-Supervised Pretraining for Label-Efficient Object Detection & Video Tracking on AgroPest-12  

---

# SECTION 1: MASTER CHEAT SHEET (INSTANT RECALL)

### 1.1. Core Benchmark Numbers (Memorize These Exact Values)

| Model Initialisation | Family / Objective | Val $\text{mAP}_{50}$ | Val $\text{mAP}_{50-95}$ | Test P | Test R | Test F1 | Test $\text{mAP}_{50}$ | Test $\text{mAP}_{50-95}$ | Rel. Gain over Rand | COCO Recovery | Part-A Recovery |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Init (20%)** | Baseline (`yolo12m.yaml`) | 0.2340 | 0.1070 | 0.3420 | 0.3030 | 0.3213 | 0.2530 | **0.1110** | Baseline | 34.8% | 23.3% |
| **BYOL (20%)** | Self-Distillation (EMA Target) | 0.2170 | 0.1000 | 0.3160 | 0.3080 | 0.3119 | 0.2380 | **0.1050** | **−5.4%** | 32.9% | 22.0% |
| **I-JEPA (20%)** | Masked Latent Prediction | 0.3550 | 0.1920 | 0.4230 | 0.3910 | 0.4064 | 0.3960 | **0.2080** | **+87.4%** | 65.2% | 43.6% |
| **SimCLR (20%)** | Contrastive (NT-Xent) | 0.3680 | 0.2000 | 0.4350 | 0.4110 | 0.4227 | 0.4070 | **0.2140** | **+92.8%** | 67.1% | 44.8% |
| **DINOv3 (20%)** 🥇 | Foundation Self-Distillation | **0.4410** | **0.2310** | **0.4920** | **0.4690** | **0.4802** | **0.4620** | **0.2550** | **+129.7%** | **79.9%** | **53.4%** |
| **COCO Baseline (20%)**| Supervised (1.5M generic labels)| 0.5920 | 0.3180 | 0.6290 | 0.5910 | 0.6094 | 0.5890 | **0.3190** | +187.4% | 100.0% | 66.8% |
| *Part A Supervised (100%)*| *Fully Supervised (11,475 labels)*| *0.8080* | *0.4700* | *0.8660* | *0.7560* | *0.8010* | *0.8102* | ***0.4773*** | *+330.0%* | *149.6%* | *100.0%* |

---

### 1.2. Key Dataset & Training Constants

- **Total Clean Dataset**: **13,107 images** across 12 pest classes (36 Roboflow label-corrupted pairs removed during Part A EDA).
- **Partitioning Breakdown (`AgroPest-12-PartB-Safe`)**:
  - **SSL Pool**: **9,180 images (70.04%)** — Unlabelled only (pretraining).
  - **Fine-Tuning Set ($\rho=0.20$)**: **2,295 images (17.51%)** — 20% labelled training budget.
  - **Validation Set**: **1,089 images (8.31%)** — Model selection and early stopping.
  - **Test Holdout**: **543 images (4.14%)** — Final unbiased evaluation.
- **Fixed Seed**: **`445`** for all sampling, DDP rank shuffling, and subset generation.
- **Hardware Platform**: Kaggle Dual Tesla T4 GPUs (2x 16GB VRAM), PyTorch 2.4+, CUDA, Mixed Precision (AMP).
- **Target Architecture**: **YOLOv12m** (`20.1M` parameters, `70.4 GFLOPs`, `292` total layers, `169` fused layers).
- **Weight Surgery Transfer**: **402 backbone weight tensors (100% parameter coverage)** mapped into modules `0..8`. Modules `9..20` (PANet Neck) and Module `21` (Detection Head) freshly initialized.
- **Image Resolution Policy**:
  - **Pretraining**: $224 \times 224$ px (permits large batch size $B=128$/GPU $\rightarrow 256$ views $\rightarrow 254$ local negatives on T4).
  - **Fine-Tuning & Test Evaluation**: $640 \times 640$ px.
  - **Tracking**: $960 \times 960$ px for 1080p clips, $640 \times 640$ px for 720p clips.

---

# SECTION 2: TASK 1 VIVA QUESTIONS & COMPLETE ANSWERS

---

## 2.1. SimCLR (Notebook 1a / 1b)

#### Q1: Why does SimCLR require negative pairs while BYOL does not? What failure mode do negatives prevent?
> **Answer**:
> SimCLR uses a symmetric architecture with identical encoders and a contrastive loss (NT-Xent). Without negative pairs, the trivial optimal solution for minimizing distance between positive views is **representation collapse** (where the encoder outputs a constant vector $z = c$ for all inputs, making loss zero).
> In SimCLR, negative pairs provide a repulsive gradient pushing representations of distinct images apart uniformly across the unit hypersphere, preventing collapse. BYOL avoids collapse without negatives by introducing architectural asymmetry (online predictor + stop-gradient on EMA target network).

#### Q2: Derive the role of the temperature parameter $\tau$ in the NT-Xent loss. What happens at very small $\tau$?
> **Answer**:
> The NT-Xent loss for positive pair $(i, j)$ in a batch of $2N$ views is:
> $$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$
> - **Role of $\tau$**: The temperature $\tau$ acts as a **hardness-aware scaling parameter**. The gradient with respect to negative similarities scales inversely with $\tau$:
>   $$\frac{\partial \mathcal{L}_{i,j}}{\partial \text{sim}(z_i, z_k)} \propto \frac{1}{\tau} \frac{\exp(\text{sim}(z_i, z_k)/\tau)}{\sum_m \exp(\text{sim}(z_i, z_m)/\tau)}$$
> - **At very small $\tau$ ($\tau \to 0$)**: The softmax acts like an $\arg\max$, over-penalizing only the single hardest negative (most similar). This makes gradients extremely noisy, unstable, and causes training to diverge.
> - **At large $\tau$ ($\tau \to \infty$)**: All negatives are weighted equally regardless of similarity, losing hardness awareness and leading to suboptimal representation clustering. We set $\tau = 0.20$.

#### Q3: Why is the projection head discarded after pretraining instead of transferred to the detector?
> **Answer**:
> The non-linear projection MLP $g(\cdot)$ (Linear $\to$ BatchNorm $\to$ ReLU $\to$ Linear) maps the encoder representation $h$ to projection space $z$. The contrastive objective enforces **invariance to data augmentations** (color jitter, cropping, rotation) in $z$.
> Discarding $g(\cdot)$ preserves richer semantic and spatial information in $h$. The projection head strips away downstream-relevant features (like exact bounding box coordinates and fine textures) to satisfy contrastive invariance. Empirical studies (Chen et al., 2020) prove that transferring $h$ achieves $10\text{–}15\%$ higher downstream mAP than transferring $z$.

#### Q4: Your batch size on a T4 is far smaller than in the original paper ($B=4096$). How does this affect the contrastive objective, and what did you do about it?
> **Answer**:
> In the SimCLR paper, $B=4096$ provided $8190$ negative pairs per anchor, allowing the model to sample diverse negative distributions. On dual-T4 GPUs (16GB VRAM), we used **$224\times224$ resolution** and **mixed precision (AMP)**, achieving **$B=128$ per GPU** ($256$ total views $\to 254$ local in-batch negatives per anchor).
> To compensate for fewer negatives, we tuned temperature $\tau = 0.20$ to maintain gradient sharpness on hard negatives and used strong color jittering and Gaussian blurring to make positive pairs challenging.

#### Q5: Which augmentations matter most for SimCLR on agricultural imagery, and which could be harmful?
> **Answer**:
> - **Most Beneficial**: **Random Resized Crop (scale 0.2–1.0)** and **Color Jitter (brightness/contrast/saturation 0.8, hue 0.2)**. Color jitter is critical because without it, the network exploits simple background green/brown color histograms as a shortcut to solve positive matching without learning pest morphology.
> - **Harmful Augmentations**: **Extreme Rotation / Vertical Flips** on gravity-dependent pests, or **Over-aggressive CutMix/Erasing** that completely obliterates tiny insect bodies ($<16\times16$ px).

#### Q6: How did your t-SNE / retrieval diagnostics change between an early and a final checkpoint?
> **Answer**:
> - **Epoch 1**: t-SNE embeddings formed an unorganized, spherical cloud where target pests overlapped completely. Nearest neighbors retrieved random background leaf patches.
> - **Epoch 30**: t-SNE formed distinct semantic clusters corresponding to pest taxonomy (e.g., winged insects like Moths/Wasps separated cleanly from mollusks like Snails/Slugs). Top-5 nearest-neighbor retrieval returned semantically identical pest species under varying lighting.

---

## 2.2. BYOL (Notebook 2a / 2b)

#### Q1: Explain the roles of the online network, target network, and predictor. Why is the predictor essential?
> **Answer**:
> - **Online Network**: Composed of encoder $f_\theta$, projector $g_\theta$, and predictor $q_\theta$. Trained via gradient descent.
> - **Target Network**: Composed of encoder $f_\xi$ and projector $g_\xi$. Weights $\xi$ are updated via an Exponential Moving Average (EMA) of online weights $\theta$, not gradients ($\text{stop-gradient}$).
> - **Why the Predictor is Essential**: The predictor $q_\theta$ breaks symmetry between online and target representations. If $q_\theta$ is removed, the optimal update for online weights is simply to match target weights directly, which collapses the representation space into a constant vector ($z = c$). The predictor forces the online network to learn a predictive representation of the target.

#### Q2: What is the EMA update rule for the target network, and how does the momentum coefficient affect training stability?
> **Answer**:
> The EMA update rule at training step $k$ is:
> $$\xi_k \leftarrow \tau \xi_{k-1} + (1 - \tau) \theta_k$$
> where $\tau \in [0, 1]$ is the momentum coefficient (we used $\tau = 0.996$).
> - **High $\tau$ ($\approx 0.996\text{–}0.999$)**: The target network updates very slowly, providing a stable, stationary prediction target that prevents representation collapse.
> - **Low $\tau$ ($\to 0$)**: The target network tracks the online network too quickly, behaving like an ordinary autoencoder without negatives, leading to rapid representation collapse.

#### Q3: Why does BYOL not collapse to a constant representation despite having no negative pairs?
> **Answer**:
> BYOL avoids collapse through **three interconnected asymmetric mechanisms**:
> 1. **Architectural Asymmetry**: The online network has an extra predictor MLP $q_\theta$ that the target network lacks.
> 2. **Optimization Asymmetry**: Gradients only backpropagate through the online network; the target network uses a $\text{stop-gradient}$ operation.
> 3. **Time-Scale Dynamics**: The target network is an EMA average over past online iterations, acting as an implicit ensemble that introduces variance into target representations.

#### Q4: Compare BYOL’s compute and memory footprint to SimCLR’s on your Kaggle runs.
> **Answer**:
> - **SimCLR**: Lower memory footprint. Stores 1 encoder + 1 projection head. Compute involves computing full $2N \times 2N$ cosine similarity matrix for NT-Xent loss.
> - **BYOL**: Higher memory footprint (~25% more VRAM). Requires maintaining two complete model graphs in memory (Online + Target network) plus the additional predictor MLP $q_\theta$, and performing EMA weight updates at every single batch step.

#### Q5: Which loss is minimised in BYOL, and between which quantities exactly?
> **Answer**:
> BYOL minimizes the **Mean Squared Error between $\ell_2$-normalized predictions and normalized target projections**:
> $$\mathcal{L}_{\text{BYOL}} = 2 - 2 \frac{\langle q_\theta(z_\theta), z'_\xi \rangle}{\|q_\theta(z_\theta)\|_2 \cdot \|z'_\xi\|_2} = \left\| \frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|_2} - \frac{z'_\xi}{\|z'_\xi\|_2} \right\|_2^2$$
> Evaluated symmetrically by swapping the two augmented views ($v, v'$) and adding both losses.

#### Q6: Did BYOL or SimCLR produce a better backbone for detection at 20% labels — and can you explain why mechanistically?
> **Answer**:
> **SimCLR substantially outperformed BYOL** (**0.2140 vs. 0.1050 $\text{mAP}_{50-95}$**; **0.4070 vs. 0.2380 $\text{mAP}_{50}$**).
> **Mechanistic Cause**: BYOL is notoriously hyperparameter- and batch-size sensitive. The original BYOL paper used $B=4096$ with cross-GPU Synchronized Batch Normalization. On our dual-T4 setup with smaller mini-batches, the EMA target network lacked sufficient negative contrast, causing dimensional representation collapse where feature dimensions became highly correlated. SimCLR’s explicit in-batch negatives provided the necessary repulsive force to maintain high-rank, diverse representations.

---

## 2.3. I-JEPA (Notebook 3a / 3b)

#### Q1: What does it mean that I-JEPA predicts in *representation space* rather than pixel space? Why is this preferable to a masked autoencoder (MAE) for semantics?
> **Answer**:
> - **MAE (Pixel Space)**: Reconstructs raw missing RGB pixels ($\text{MSE}(I_{\text{pred}}, I_{\text{orig}})$). This wastes capacity predicting high-frequency pixel noise (leaf veins, specular lighting, motion blur) that is irrelevant for semantic categorization.
> - **I-JEPA (Representation Space)**: Predicts the *latent embedding* of the masked target block ($s_y$) from the context block ($s_x$). This forces the encoder to capture abstract semantic features and part-whole structural relationships (e.g., insect legs, antennae, head morphology) without wasting parameter budget on pixel-level texture reconstruction.

#### Q2: How are context and target blocks sampled, and why must target blocks be masked out of the context?
> **Answer**:
> - **Sampling Strategy**: A single large **Context Block** is sampled with high aspect ratio and scale (e.g., $0.85\text{–}1.0$), while 4 small **Target Blocks** are sampled with scale ($0.15\text{–}0.20$).
> - **Masking Requirement**: Target block patch regions are strictly **removed (masked out)** from the context block. If target patches overlapped with the context, the context encoder would have direct access to target tokens, allowing the predictor to solve the task via trivial identity copy rather than semantic inference.

#### Q3: Why does I-JEPA avoid hand-crafted augmentations, and what replaces the invariance signal they normally provide?
> **Answer**:
> - I-JEPA applies only minimal spatial resizing without color distortion, Gaussian blur, or solarization.
> - The invariance signal is replaced by **Masked Spatial Context Prediction**: predicting non-visible target blocks from visible context blocks forces the network to understand spatial coherence, object geometry, and structural context across the visual scene.

#### Q4: Which integration route (ViT adapter vs. distillation into CNN) did you choose, and what trade-off drove the choice?
> **Answer**:
> We chose the **Distillation Route into the YOLOv12m CNN backbone**.
> - **Trade-off**: A direct ViT replacement requires building a complex multi-scale feature pyramid adapter (mapping 1D patch tokens to multi-scale P3/P4/P5 pyramid levels) and increases inference latency. Distilling the learned latent representations into the native YOLOv12m CNN preserves the exact Ultralytics graph routing (`f`, `i`, `save` attributes) and retains native 23+ FPS real-time speed on Tesla T4.

#### Q5: How did you map ViT patch tokens to multi-scale P3/P4/P5 features?
> **Answer**:
> In the distillation pipeline, ViT patch tokens $\mathbf{Z} \in \mathbb{R}^{N \times D}$ ($N = 14 \times 14 = 196$, $D = 768$) are reshaped into a 2D spatial feature map $\mathbf{F} \in \mathbb{R}^{768 \times 14 \times 14}$. We align these features with the YOLOv12m Stage-8 backbone feature map using a $1\times1$ convolutional projection head matched to 512 channels.

#### Q6: What evidence in your diagnostics suggests I-JEPA learned spatially localised features useful for detection?
> **Answer**:
> On the `Beetles` class, I-JEPA achieved **0.1056 $\text{mAP}_{50-95}$** and **0.2484 $\text{mAP}_{50}$**, outperforming both SimCLR (**0.0638**) and DINOv3 (**0.0930**). Beetles have distinct, localized shell textures (elytra) and compact boundaries; I-JEPA's masked part-prediction mechanism explicitly encouraged localized feature extraction.

---

## 2.4. DINOv3 (Notebook 4a / 4b)

#### Q1: What problem in dense (patch-level) features does DINOv3’s Gram anchoring address, and why does it matter for detection?
> **Answer**:
> Standard self-distillation (like DINOv1/v2) can suffer from **patch-token feature collapse or drift**, where background tokens dominate and local dense representations lose discriminative power.
> **Gram Anchoring** computes the Gram matrix of pairwise patch-token correlations ($\mathbf{G} = \mathbf{F}\mathbf{F}^T$) and enforces correlation consistency across views. This preserves dense relational structure, ensuring that patch embeddings retain sharp boundary definitions essential for bounding-box regression.

#### Q2: Explain the student–teacher self-distillation setup and the multi-crop strategy.
> **Answer**:
> - **Student ($S_\theta$)**: Trainable YOLOv12m CNN backbone.
> - **Teacher ($T_\xi$)**: Frozen foundation model (DINOv3 ViT-B/16).
> - **Multi-Crop Strategy**: Two global views ($224\times224$, covering $>50\%$ of image) are passed to both student and teacher, while several small local crops ($96\times96$, covering $<50\%$) are passed only to the student. This forces the student to learn local-to-global representation consistency.

#### Q3: You continued pretraining from released weights: what distinguishes domain-adaptive pretraining from training from scratch, and how does it change fairness?
> **Answer**:
> - **Domain-Adaptive Pretraining**: Initializes from generic foundation weights (trained on millions of web images) and adapts the feature distribution to the target domain (9,180 unlabelled agricultural pest images).
> - **Fairness Accounting**: Pretraining from scratch (SimCLR/BYOL) evaluates pure self-supervision on a closed dataset, whereas DINOv3 evaluates foundation distillation. In the report, we explicitly state that DINOv3 leverages foundation transfer, making it the most practical industrial route, while SimCLR is the fair winner among from-scratch CNN methods.

#### Q4: How do frozen DINOv3 features compare with fine-tuned ones in your downstream runs?
> **Answer**:
> Fine-tuning all backbone layers yielded **0.2550 $\text{mAP}_{50-95}$** vs. ~**0.1720** for a frozen backbone. Fine-tuning allows convolutional kernels to adapt specifically to pest aspect ratios and multi-scale agricultural foliage textures.

#### Q5: Why are ViT-based foundation backbones often more label-efficient than CNNs pretrained from scratch?
> **Answer**:
> ViTs lack the hard inductive biases (translation equivariance, locality) of CNNs, allowing them to learn arbitrary long-range global dependencies when trained on massive web datasets. Distilling these rich global attention representations into a CNN detector gives it an optimal starting initialization that requires far fewer labelled bounding boxes to converge.

#### Q6: If you could only ship one SSL backbone to an edge device in your deployment setting, which would you pick and why?
> **Answer**:
> **YOLOv12m with DINOv3 Distilled Backbone**.
> - **Accuracy**: Highest test mAP (**0.2550 $\text{mAP}_{50-95}$**, **0.4620 $\text{mAP}_{50}$**).
> - **Latency & Memory**: Because DINOv3 was distilled into the standard `YOLOv12m` CNN backbone, it has **zero ViT inference overhead**—it runs at **23.0 FPS single-image** and **104.2 FPS batch mode** on Tesla T4 with identical 20.1M parameter count.

---

# SECTION 3: TASK 2 TRACKING VIVA QUESTIONS & COMPLETE ANSWERS

---

#### Q1: Explain the tracking-by-detection paradigm. Why does tracker quality depend so strongly on detector mAP?
> **Answer**:
> In tracking-by-detection, an independent object detector runs on each video frame to output bounding boxes and class probabilities. The tracker (e.g., ByteTrack) then associates these detections across time using motion estimation (Kalman Filter) and spatial overlap (IoU distance).
> **Why Tracker Depends on Detector**:
> - If the detector **misses an object (false negative)**, the tracker must coast on Kalman velocity predictions; consecutive misses cause the track to be terminated.
> - If the detector outputs a **false positive**, a spurious trajectory is created.
> - If detector **localization is imprecise**, the IoU distance between frames drops below the association threshold, causing track fragmentation and ID switches.

#### Q2: How does ByteTrack use *low-confidence* detections, and why does this reduce ID switches?
> **Answer**:
> Traditional trackers discard all detections below a high threshold (e.g., $\text{Conf} < 0.5$). However, occluded or motion-blurred pests often exhibit low confidence scores ($0.10 \le \text{Conf} < 0.25$).
> **ByteTrack Two-Stage Matching**:
> 1. **First Association**: High-confidence detections ($\text{Conf} \ge 0.25$) are matched to existing active tracks using Kalman filter IoU.
> 2. **Second Association**: Unmatched active tracks are matched against **low-confidence detections ($0.10 \le \text{Conf} < 0.25$)**.
> This allows the tracker to maintain track continuity across leaves and temporary occlusions without swapping IDs or terminating trajectories.

#### Q3: Define MOTA and IDF1. Which failure types does each punish, and why can a tracker have high MOTA but low IDF1?
> **Answer**:
> - **MOTA (Multiple Object Tracking Accuracy)**: Focuses on detection and association errors:
>   $$\text{MOTA} = 1 - \frac{\sum_t (\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$
>   Punishes False Negatives (misses), False Positives (ghost boxes), and ID Switches.
> - **IDF1 (Identification F1-Score)**: Focuses on identity preservation across time:
>   $$\text{IDF1} = \frac{2 \text{IDTP}}{2 \text{IDTP} + \text{IDFP} + \text{IDFN}}$$
>   Measures the percentage of correctly identified trajectory time steps.
> - **Why High MOTA but Low IDF1?**: If a tracker detects all objects in every frame (zero FN/FP) but swaps their IDs every 5 frames, MOTA remains very high (IDSW is only a small penalty per swap), but IDF1 plummets because no single trajectory retains its true identity over time.

#### Q4: What association cues does BoT-SORT add over ByteTrack, and when do they matter?
> **Answer**:
> BoT-SORT adds two key features over ByteTrack:
> 1. **Camera Motion Compensation (GMC)**: Uses affine motion estimation via image feature matching to compensate for camera panning, tilting, or drone movement.
> 2. **Re-ID Visual Appearance Cues**: Fuses deep Re-ID cosine distance with Kalman IoU distance.
> **When BoT-SORT Matters**: On fast-moving handheld cameras or drone footage where camera motion breaks linear Kalman velocity assumptions.

#### Q5: What domain gap did you observe between your training images and video, and how did it show up in tracks?
> **Answer**:
> - **Domain Gap**: Training images are static, close-up macro shots with controlled lighting. Video sequences contain continuous camera panning, dynamic leaf motion, shadows, and motion blur.
> - **Manifestation in Tracks**: Fast insect motion caused temporary confidence drops; ByteTrack's secondary association stage successfully bridged these gaps.

#### Q6: If your video was AI-generated or synthetic: what are the risks and how did you mitigate them?
> **Answer**:
> - **Risks**: Synthetic videos may have unrealistic physics (pests morphing or clipping through surfaces) or unnatural motion trajectories.
> - **Mitigation**: We cross-evaluated on real YouTube field recordings and synthetic clips, confirming that Kalman filtering parameters generalized robustly across both.

---

# SECTION 4: DATASET, PARTITIONING & WEIGHT SURGERY

---

#### Q1: Why is our 80/20 partitioning protocol mathematically safe from data leakage?
> **Answer**:
> 1. **Triplet-Aware Grouping**: The training set contains consecutive triplets (1 source + 2 augmented variants). Splitting at the triplet level ensured all 3 variants were placed into either the SSL pool or the fine-tuning set, never split across partitions.
> 2. **Zero Test/Val Contamination**: Validation (1,089 images) and Test (543 images) splits were 100% excluded from the SSL pool ($\text{Intersection} = \emptyset$).
> 3. **Unlabelled Pretraining**: Labels were strictly removed from the 9,180 SSL pool images during pretraining. Restoring labels for a 20% fine-tuning subset is the canonical label-efficiency protocol established in SimCLR and BYOL literature.

#### Q2: Explain the exact weight surgery procedure used to load SSL checkpoints into YOLOv12m.
> **Answer**:
> 1. The downstream notebook instantiates a fresh model from `yolo12m.yaml` (all 20.1M parameters randomly initialized).
> 2. We load the exported `backbone_only.pt` checkpoint (containing 402 tensors).
> 3. We iterate through model modules `0..8` (Backbone) and copy parameter weights directly using strict tensor shape matching.
> 4. Modules `9..20` (PANet Neck) and Module `21` (Detection Head) are left freshly initialized.
> 5. A layer-by-layer mapping report verifies that **402/402 backbone tensors (100%)** are loaded without silent shape mismatch.

---

# SECTION 5: "IF THE PROFESSOR ASKS..." RAPID DEFENSE

| Question Trigger | 30-Second Winning Answer |
|:---|:---|
| *"Why did you pick YOLOv12m for Part B?"* | *"In Part A, YOLOv12m achieved the highest test $\text{mAP}_{50-95}$ ($0.4773$) and highest precision ($0.866$), beating YOLOv10m ($0.4730$), YOLOv26m ($0.4750$), and RF-DETR ($0.3620$). It won on 8 of 12 classes and generalized best."* |
| *"Why did DINOv3 beat COCO on Weevils?"* | *"COCO has 80 generic classes with zero pest categories. DINOv3 pretraining distilled representations directly on 9,180 agricultural pest images, learning domain-specific snout and shell features."* |
| *"Why did you pretrain at 224px but detect at 640px?"* | *"Pretraining at 224px allowed a batch size of 128/GPU on dual T4s (254 negatives for SimCLR). YOLO's fully convolutional backbone is resolution-agnostic, allowing seamless fine-tuning at 640px for precise small-pest localization."* |
| *"Which pest classes were the hardest and why?"* | *"Caterpillars ($0.0321$), Earthworms ($0.0771$), and Grasshoppers ($0.0707$). They have deformable non-rigid morphology and extreme chromatic camouflage matching crop leaves and soil."* |
| *"What is the practical annotation saving?"* | *"DINOv3 at 20% labels recovers 79.9% of COCO-pretrained performance, saving 80% of manual annotation costs (9,180 images left unlabelled)."* |
