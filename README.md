## From Reflection to Relation: Evaluating Agentic Captioning and Relation-Aware Distillation for Text-to-Image Person Re-Identification

This repository provides the full implementation of our framework, which 
consists of two main components: (1) an agentic caption refinement pipeline 
for generating higher-quality textual supervision, and (2) Relation-Aware 
Distillation (RAD) built on top of the 
[RDE framework](https://github.com/QinYang79/RDE) for efficient cross-modal 
retrieval. The repository includes the complete pipeline to generate refined 
captions, train and distill models, and evaluate across multiple benchmarks.

---

### 🔧 Setup & Environment

Please refer to the [RDE repository](https://github.com/QinYang79/RDE) for 
detailed instructions on environment setup, required packages, and dataset 
preparation. Make sure the dataset directory is properly configured before 
running any stage.

Required models for agentic captioning:
- [LLaVA1.6-7B](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)
- [Qwen-VL-Chat-7B](https://huggingface.co/Qwen/Qwen-VL-Chat)

---

### 🏃‍♂️ How to Run

#### Step 1: Agentic Caption Generation

The agentic captioning pipeline runs all three stages automatically in 
sequence — candidate generation, critic-based reflection, and caption 
refinement — given a data directory. Simply configure the dataset path 
in the script and run:

```bash
bash agent_llava.sh --data_dir /path/to/dataset
```

This single script sequentially executes:
- **Stage 1**: Generate $K$ diverse candidate captions per image using 
  LLaVA1.6-7B as the generator VLM
- **Stage 2**: Apply critic-based self-reflection to identify factual 
  errors, missing attributes, and vague wording
- **Stage 3**: Refine the selected candidate using critic feedback to 
  produce a final, visually grounded caption

The refined captions will be saved to the `captions/` directory and used 
automatically in subsequent training stages.

---

#### Step 2: Train the Teacher Model

Using the refined captions, train a strong teacher model (ViT-L/14) on 
SYNTH-PEDES and fine-tune on each target benchmark:

```bash
bash run_stage1.sh
```

---

#### Step 3: Train the Student Model using RAD

After the teacher model is trained, specify the teacher checkpoint path in 
`run_stage2.sh`, then distill a compact student model (ViT-B/16) using 
Relation-Aware Distillation:

```bash
bash run_stage2.sh
```

RAD aligns pairwise Euclidean distances and cosine similarities between 
teacher and student embeddings in both image and text spaces, preserving 
the relational geometry that governs retrieval ranking.

---

### 📊 Evaluation

To evaluate on CUHK-PEDES, ICFG-PEDES, and RSTPReid:

```bash
bash run_eval.sh
```

Results are reported in terms of Rank-1 (R1), mean Average Precision (mAP), 
and mean Inverse Negative Penalty (mINP).

---

### 🔗 Code Availability

Code is anonymously available via the supplemental material submission 
on OpenReview for review purposes.
