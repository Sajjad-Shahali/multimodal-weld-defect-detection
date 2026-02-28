# 🔥 Multimodal Weld Defect Detection  
## Therness Hackathon — **DATA forge**  
**100% Monitoring, 0% Guesswork**

---

## 🚀 Overview

This repository contains a **reproducible multimodal weld defect detection pipeline** built for the **Therness Hackathon**.

It supports:

- 🔊 Audio (MFCC + spectral features)
- 🎥 Video (MobileNetV3 visual embedding + temporal pooling)
- 📊 Optional sensor CSV branch
- 🤖 Multitask learning (7-class + auxiliary binary head)
- 📈 Temperature scaling calibration
- 🧪 Leakage auditing
- 🏁 End-to-end submission generation & post-processing optimization

---

# 🏆 Final Leaderboard Result (Team: **DATA forge**)

**Final Score:** **0.9567**  
**Binary F1:** **0.9677**  
**Type Macro F1:** **0.9401**  

> Group name on leaderboard: **DATA forge**

---

# ✅ Validation Performance (Local)

From our internal validation evaluation (Step 13):

```text
Binary F1  : 0.9835
Macro F1   : 0.9830
FinalScore : 0.9833
ECE        : 0.0308
Temperature: 0.5527
```

> Note: leaderboard score is based on the hidden test set, so it can differ from validation.

---

# 🧠 Architecture — WeldFusionNet

```text
Audio (18×25) → 1D CNN → 64d
Video (5 frames) → MobileNetV3 → 128d
                  ↓
           Concatenate (192d)
                  ↓
        FC → BN → ReLU → Dropout
                  ↓
      ┌───────────────┴──────────────┐
   7-Class Head                Binary Head
```

### Loss Function

Configured training objective:

```text
Total Loss = 0.7 * FocalLoss + 0.3 * BCE
focal_gamma = 1.0
```

---

# 📁 Repository Structure

```text
multimodal-weld-defect-detection/
│
├── dataset/
│   ├── defect-weld/
│   └── good-weld/
│
├── pipeline/
│   ├── step1_validate.py
│   ├── step2_sensor.py
│   ├── step3_audio.py
│   ├── step4_video.py
│   ├── step5_align.py
│   ├── step6_dataset.py
│   ├── step7_tabular_baseline.py
│   ├── step8_dataset_torch.py
│   ├── step9_model.py
│   ├── step10_losses.py
│   ├── step11_train.py
│   ├── step12_calibrate.py
│   ├── step13_evaluate.py
│   ├── step14_inference.py
│   ├── step15_sanity_leakage.py
│   └── run_all.py
│
├── generate_final_submission.py
├── optimize_postprocessing.py
├── fix_submission.py
├── config.yaml
├── requirements.txt
├── run_dashboard.ps1
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-weld-defect-detection.git
cd multimodal-weld-defect-detection

python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

# 🏗 Run the Pipeline

## Phase 1 — Preprocessing

```bash
python -m pipeline.run_all --steps 1 2 3 4 5 6
```

## Phase 2 — Training

```bash
python -m pipeline.run_all --steps 7 8 9 10 11
```

## Phase 3 — Calibration & Evaluation

```bash
python -m pipeline.run_all --steps 12 13
```

---

# 🏁 Submission Generation (Final)

### 1) Generate submission directly (robust end-to-end)

```bash
python generate_final_submission.py
```

### 2) Optimize post-processing (grid search on validation)

```bash
python optimize_postprocessing.py
```

### 3) Fix grader contradictions (CSV consistency)

```bash
python fix_submission.py --input submission.csv --output submission_fixed.csv
```

---

# 📊 Label Map

| Code | Label |
|------|------------------------|
| 00 | good_weld |
| 01 | excessive_penetration |
| 02 | burn_through |
| 06 | overlap |
| 07 | lack_of_fusion |
| 08 | excessive_convexity |
| 11 | crater_cracks |

---

# 🛡 Calibration Strategy

We use **temperature scaling** to improve confidence quality and reduce ECE.

---

# 🧪 Engineering Quality

- Deterministic split
- Config-driven pipeline (`config.yaml`)
- End-to-end reproducibility
- Optional leakage audit (Step 15)
- Crash-safe inference on each test sample

---

# 👥 Team

**DATA forge**

---

# 📜 License

MIT License
