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
