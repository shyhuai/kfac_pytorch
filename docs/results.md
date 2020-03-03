| Optimizer | Model    | GPUs | LR    | LR-Decay | Epochs | Val-Acc (Train) | t/epoch  | time    |
|-----------|----------|------|-------|----------|--------|-----------------|----------|---------|
| SGD       | ResNet32 | 1    | 0.1   | 100, 150 | 200    | 92.89% (99.83%) | 00:27.50 | 1:45:22 |
| SGD       | ResNet32 | 4    | 0.4   | 100, 150 | 200    |                 | 00:08.50 |         |
| KFAC (10) | ResNet32 | 1    | 0.1   | 100, 150 | 200    |                 | 01:25.00 |         |
| KFAC (10) | ResNet32 | 4    | 0.4   | 100, 150 | 200    |                 |          |         |