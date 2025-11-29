# Physics Informed Machine Learning Models

## 🛠️ Installation

### Prerequisites
- Python 3.12.11
- CUDA / TPU (optional, for GPU / TPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/piml-models.git
cd piml-models

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12.11
source .venv/bin/activate

# Install dependencies via uv
uv sync --all-groups

# Install pre-commit
pre-commit install
```

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

## 🚀 Usage

### Training

To train a model, run:

```bash
uv run train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

**Note:** Using `uv run` is recommended. Alternatively, you can use `python3 train.py ...`
