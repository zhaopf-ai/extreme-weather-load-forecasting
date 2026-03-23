This repository provides the source code for the paper: "Deep Multi-Modal Fusion and Kalman Online Adaptation for Electric Load Forecasting Against Extreme Rainfall Events" (submitted to IEEE Transactions on Smart Grid).

For questions or issues, please open a GitHub issue or contact: zhaopf@std.uestc.edu.cn

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
- CUDA == 11.3 (optional, GPU recommended)

### Core libraries

- torch == 1.12.1  
- torchvision == 0.13.1  
- numpy == 1.21.6  
- pandas == 1.3.5  
- fvcore == 0.1.5.post20221221  
- einops == 0.6.0  
- pillow == 9.5.0  

### Install dependencies

```bash
pip install torch==1.12.1 torchvision==0.13.1
pip install numpy==1.21.6 pandas==1.3.5
pip install fvcore einops pillow

---

## How to Run

### Basic command

```bash
python main.py --datasets Gympie


