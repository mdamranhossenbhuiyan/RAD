## RAD: Relation-Aware Distillation for Text-to-Image Person Re-Identification

This repository implements the RAD approach on top of the [RDE framework](https://github.com/QinYang79/RDE). It includes the full pipeline to train and evaluate the model across multiple datasets.

---

### 🔧 Setup & Environment

Please refer to the [RDE repository](https://github.com/QinYang79/RDE) for detailed instructions on environment setup, required packages, and dataset preparation. Make sure the dataset directory is properly configured before training.

---

### 🏃‍♂️ How to Run

#### 1. **Stage 1: Train the Teacher Model**

To train the teacher model, run the following command:

```bash
bash run_stage1.sh
```

#### 2. **Stage 2: Train the Student Model using RAD**

After the teacher model is trained, specify the checkpoint path in `run_stage2.sh`, then execute:

```bash
bash run_stage2.sh
```
