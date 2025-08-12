## Summary:
- Self-supervised ViT model pretrained on unlabeled frames.
- Fine-tuned on minimal labeled data; excellent performance in steering prediction.

## Technologies Used:
- PyTorch, CARLA, Vision Transformers (timm), EMA teacher-student, cosine & variance loss, NYU Greene HPC (A100/V100 GPUs).

### Details:
A deep dive into label-efficient learning for autonomous driving using I-JEPA. Pretrained on ~86K unlabeled CARLA frames, the model was fine-tuned using only 5–11% labeled data. I addressed the rare-turn vs. straight-driving imbalance via balanced sampling and achieved a striking validation MSE of 0.0018—outperforming fully supervised baselines. All experiments ran on NYU’s HPC infrastructure with efficient job scheduling and model checkpointing.

### Contributors:
- Tejdeep Chippa
- Venkat Kumar Laxmikanth Nemala

### Link to demos:
https://drive.google.com/file/d/1KQ2Q6DdwY2mlRaXluGG_HoDV0WUJZdzW/view
https://drive.google.com/file/d/1szZe35-DaQnRKZE5AUF8ADE9-T6qyRuL/view

