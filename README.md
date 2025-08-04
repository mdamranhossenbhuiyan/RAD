## RAD: Relation-Aware Distillation for Text-to-Image Person Re-Identification

This repository implements the RAD approach on top of the [RDE framework](https://github.com/QinYang79/RDE). It includes the full pipeline to train and evaluate the model across multiple datasets.

---

### 🔧 Setup & Environment

Please refer to the [RDE repository](https://github.com/QinYang79/RDE) for detailed instructions on environment setup, required packages, and dataset preparation. Make sure the dataset directory is defined properly before training.

---

### 🏃‍♂️ How to Run

#### 1. **Stage 1: Train the Teacher Model**
To begin training the teacher model, run:

```bash
bash run_stage1.sh

#### 2. **Stage 2: Train the Student Model using RAD**
Once the teacher model is trained, define its checkpoint path in run_stage2.sh and then run:

```bash
bash run_stage2.sh
