# AI-Enabled Choreography: Dance Beyond Music

This project explores the use of AI for choreography representation and generation, connecting dance movements with natural language descriptions through contrastive learning.

## Project Overview

Dance is an art form that combines movement, space, and time, but its representation in digital form presents unique challenges. This project aims to create a multimodal representation that connects dance movement with natural language, enabling bidirectional translation between these modalities.

Key features:
- 3D visualization of dance motion capture data
- Semi-supervised labeling of dance sequences
- Contrastive learning model to embed dance and text in a shared space
- Bidirectional generation between modalities

## Repository Structure

- `notebooks/`: Jupyter notebooks for exploration, visualization, and training
  - `1_data_exploration.ipynb`: Initial data loading and analysis
  - `2_visualization.ipynb`: 3D animations of dance sequences
  - `3_sequence_labeling.ipynb`: Creating and propagating text labels
  - `4_model_training.ipynb`: Training the contrastive learning model
  - `5_generation_examples.ipynb`: Demonstrations of text-to-dance and dance-to-text generation
- `src/`: Source code modules
  - `data/`: Data loading and preprocessing
  - `visualization/`: Dance sequence animation
  - `labeling/`: Feature extraction and label propagation
  - `model/`: Contrastive learning model implementation
  - `generation/`: Text-to-dance and dance-to-text conversion
- `results/`: Generated outputs (animations, plots, trained models)
- `data/`: Motion capture data and processed versions

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Enabled-Choreography-GSoC2025.git
cd AI-Enabled-Choreography-GSoC2025

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt