# Deep Multi-Modal Fusion and Kalman Online Adaptation for Electric Load Forecasting

This repository provides the source code for the paper:

**"Deep Multi-Modal Fusion and Kalman Online Adaptation for Electric Load Forecasting Against Extreme Rainfall Events"**  
*(submitted to IEEE Transactions on Smart Grid)*

For questions or issues, please open a GitHub issue or contact:  
📧 zhaopf@std.uestc.edu.cn

---

## Requirements

All experiments were conducted using the following environment:

- Python == 3.8  
- torch == 2.4.1  
- torchvision == 0.19.1  
- numpy == 1.19.5  
- pandas == 1.2.5  
- fvcore == 0.1.5.post20221221  
- einops == 0.8.0  
- Pillow == 10.0.0

---

## Dataset

Place the dataset under the directory `data/`. Each dataset should include a tabular file and corresponding image folders.

Example structure:

```bash
data/
 ├── Gym.npz
 ├── processed_image/
```

---

## How to Run

Run the model with default settings:

```bash
python main.py --datasets Gympie
```

Run with full configuration:

```bash
python main.py \
    --datasets Gympie \
    --his_len 6 \
    --pre_len 1 \
    --batch_size 32 \
    --epochs 200 \
    --lr 1e-3
```

Optional arguments:

- --his_len : history length (default: 6)  
- --pre_len : prediction horizon (default: 1)  
- --batch_size : batch size (default: 32)  
- --epochs : number of training epochs (default: 500)  
- --lr : learning rate (default: 1e-3)  

Run multiple datasets:

```bash
python main.py --datasets Gympie Coolum Noosaville
```

---

## Output

After training, results are saved to:

- results/model_saved/  
- results/logs/  

---

## Evaluation

The following metrics are reported:

- MAPE  
- RMSE  
- MAE  

---

## Notes

Disable weather noise augmentation if needed:

```bash
python main.py --no_weather_noise
```

GPU is recommended for faster training.

---

## Reproducibility

Random seed is fixed:

```python
torch.manual_seed(42)
np.random.seed(42)
```

Time-based data split is adopted for evaluation.
