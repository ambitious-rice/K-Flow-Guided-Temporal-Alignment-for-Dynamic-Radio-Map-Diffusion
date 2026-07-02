# 📡 PhyRMDM: Physics-Informed Representation Alignment for Sparse Radio-Map Reconstruction 🚀

> **"When Einstein Meets Deep Learning"** — We make radio signals elegantly dance to the laws of physics in virtual cities. 💃

## 📢 NEWS

🎉 **[NEW!]** Our paper has been **accepted** by **ACM MM BNI 2025** as an **Oral Presentation**! 🏆  
🔓 **All code is now available** — Ready for researchers and practitioners to explore and build upon our work! 💻  
🎯 **Pre-trained model weights are now available** — Download from Baidu Netdisk: [Link](https://pan.baidu.com/s/14p0aofKzp0jhreg-9NMKdQ) (Code: dnd4) 📦

## 🌟 Project Highlights

![RMDM Model Structure](RMDM.jpeg)

- 🧠 **Physics-Informed AI**: Equipping neural networks with electromagnetic wisdom, enabling AI to think using Helmholtz equations.
- 🎭 **Dual U-Net Architecture**: Two neural nets—one handling physical laws, the other refining details—working seamlessly to reconstruct radio maps.
- 📉 **Record-Breaking Accuracy**: Achieved an unprecedented 0.0031 NMSE error in static scenarios, 2× better than state-of-the-art methods!
- 🌪️ **Dynamic Scene Mastery**: Robust reconstruction in dynamic, interference-rich environments (vehicles, moving obstacles) with an impressive 0.0047 NMSE.
- 🕵️ **Sparse Data Champion**: Capable of accurately reconstructing complete radio maps even from a mere 1% sampling—like Sherlock deducing from minimal clues.

## 🎯 Problems Solved

- 🧩 **Signal Reconstruction Puzzle**: Restoring complete electromagnetic fields from fragmented measurements.
- 🌆 **Urban Maze Complexity**: Seamlessly handling complex obstructions from buildings, moving vehicles, and urban environments.
- ⚡ **Real-Time Performance**: Achieving inference speeds up to 10× faster than traditional methods—ideal for real-time 5G/6G applications.

## 🧠 Core Technical Innovations

### 🎵 **Dual U-Net Symphony**

1. **Physics-Conductor U-Net**: Embeds physical laws (Helmholtz equations) through Physics-Informed Neural Networks (PINNs).
2. **Detail-Sculptor U-Net**: Uses advanced diffusion models for ultra-fine precision in radio map reconstruction.

### 🔥 **Three Innovative Modules**

- 🎯 **Anchor Conditional Mechanism**: Precisely locking onto critical physical landmarks (like GPS for radio signals).
- 🌐 **RF-Space Attention**: Models "frequency symphonies" enabling focused learning of electromagnetic signal characteristics.
- ⚖️ **Multi-Objective Loss**: Harmonizing physics-based constraints and data-driven fitting to achieve optimal results.

## 📂 Benchmark Dataset

Leveraged the authoritative **RadioMapSeer dataset**:

- 700+ real-world urban scenarios (London, Berlin, Tel Aviv, etc.)
- 80 base stations per map with high-resolution 256×256 grids
- Incorporates static and dynamic challenges (buildings, vehicles)

## 🚀 Quick Start

### 📋 Prerequisites

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **Dataset**: RadioMapSeer dataset

### 🛠️ Installation

1. **Clone the repository**:
```bash
git clone git@github.com:Hxxxz0/RMDM.git
cd PhyRMDM
```

2. **Create and activate conda environment**:
```bash
conda create -n RMDM python=3.8
conda activate RMDM
```

3. **Install dependencies**:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirement.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 📁 Dataset Setup

Download and organize the RadioMapSeer dataset:
```
RadioMapSeer/
├── gain/
│   ├── DPM/
│   ├── IRT2/
│   └── cars*/
├── png/
│   ├── buildings_complete/
│   ├── antennas/
│   └── cars/
└── dataset.csv
```

### 🎯 Training

**Single GPU Training (SRM)**:
```bash
conda activate RMDM
python train.py \
  --data_name Radio \
  --data_dir /path/to/RadioMapSeer \
  --batch_size 16 \
  --mixed_precision no \
  --use_checkpoint True \
  --num_channels 96 \
  --attention_resolutions 16 \
  --log_interval 50 \
  --max_steps 100000 \
  --save_interval 10000 \
  --save_dir ./checkpoints_phy
```

**Multi-GPU Training (SRM)**:
```bash
accelerate launch --num_processes=2 --multi_gpu --mixed_precision=no \
  train.py \
  --data_name Radio \
  --data_dir /path/to/RadioMapSeer \
  --batch_size 32 \
  --mixed_precision no \
  --use_checkpoint True \
  --num_channels 96 \
  --attention_resolutions 16 \
  --log_interval 50 \
  --max_steps 100000 \
  --save_interval 10000 \
  --save_dir ./checkpoints_phy
```

**Resume Training**:
```bash
python train.py \
  --resume_from ./checkpoints_phy/model_phy_step5000.pth \
  --data_name Radio \
  --data_dir /path/to/RadioMapSeer \
  --batch_size 16 \
  --mixed_precision no \
  --save_dir ./checkpoints_phy
```

**DynamicRadioMap Training (ShadowDenoiseV1)**:
```bash
python train.py \
  --data_name DynamicRadio \
  --data_dir /data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1 \
  --image_size 128 \
  --batch_size 32 \
  --workers 4 \
  --num_channels 96 \
  --attention_resolutions 16 \
  --diffusion_steps 1000 \
  --noise_schedule linear \
  --lr 1e-4 \
  --max_steps 100000 \
  --save_interval 5000 \
  --save_dir ./checkpoints_dynamic_rmdm
```

### 🔮 Inference & Evaluation

**Quick Inference Test (SRM)**:
```bash
python sample_test.py \
  --scheduler_type ddpm \
  --data_dir /path/to/RadioMapSeer \
  --checkpoint_path ./checkpoints_phy/model_phy_step100000.pth \
  --output_dir ./inference_results \
  --ddpm_steps 1000 \
  --batch_size 4 \
  --num_samples 100
```

**Full Test Set Evaluation (SRM)**:
```bash
python sample_test.py \
  --scheduler_type ddpm \
  --data_dir /path/to/RadioMapSeer \
  --checkpoint_path ./checkpoints_phy/model_phy_step10000.pth \
  --output_dir ./full_evaluation \
  --ddpm_steps 1000 \
  --batch_size 4 \
  --num_samples -1
```

**Inference with Image Saving** 🖼️:
```bash
python sample_test.py \
  --scheduler_type ddpm \
  --data_dir /path/to/RadioMapSeer \
  --checkpoint_path ./checkpoints_phy/model_phy_step10000.pth \
  --output_dir ./results_with_images \
  --ddpm_steps 1000 \
  --batch_size 4 \
  --num_samples 50 \
  --save_images
```

This will generate:
- 📊 **Generated radio maps**: `generated/` folder
- 🎯 **Ground truth maps**: `ground_truth/` folder  
- 🏗️ **Input conditions**: `conditions/` folder (buildings + transmitters)
- 🔍 **Comparison plots**: `comparison/` folder (generated vs. ground truth vs. difference)

**DynamicRadioMap Evaluation**:
```bash
python sample_test.py \
  --scheduler_type ddim \
  --data_name DynamicRadio \
  --data_dir /data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1 \
  --checkpoint_path ./checkpoints_dynamic_rmdm/model_phy_step100000.pth \
  --output_dir ./eval_dynamic_rmdm \
  --image_size 128 \
  --batch_size 8 \
  --ddim_steps 50 \
  --num_samples 1000 \
  --save_images
```

## 📜 Academic Citation

```bibtex
@misc{jia2025rmdmradiomapdiffusion,
      title={RMDM: Radio Map Diffusion Model with Physics Informed}, 
      author={Haozhe Jia and Wenshuo Chen and Zhihui Huang and Hongru Xiao and Nanqian Jia and Keming Wu and Songning Lai and Yutao Yue},
      year={2025},
      eprint={2501.19160},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.19160}, 
}
```

## 🙌 Acknowledgments

Special thanks to:
- 🏫 Joint Laboratory of Hong Kong University of Science and Technology (Guangzhou) & Shandong University
- 🌉 Guangzhou Education Bureau's Key Research Project
- 🤖 DIILab for generous computational support



---

**License**: This project is distributed under the **Academic Free License v3.0**. Please cite accordingly for academic use. For commercial applications, contact the authors directly.
