#!/bin/bash

# Autonomous Self-Improving AI System - Setup Script

set -e  # Exit on error

echo "========================================================================"
echo "🧠 AUTONOMOUS SELF-IMPROVING AI SYSTEM - SETUP"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 11 ]; then
    echo -e "${RED}❌ Python 3.11+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo -e "${GREEN}✅ Virtual environment created${NC}"

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo -e "${GREEN}✅ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a while)..."
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Install PyTorch with CUDA support if available
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo -e "${GREEN}✅ PyTorch with CUDA installed${NC}"
else
    echo -e "${YELLOW}⚠️ No NVIDIA GPU detected. Using CPU-only PyTorch${NC}"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/{knowledge,performance,models,sandbox}
mkdir -p data/knowledge/{vector_db,graph_db}
mkdir -p logs
mkdir -p backups
mkdir -p skills/{learned_skills,evolved_algorithms}
echo -e "${GREEN}✅ Directories created${NC}"

# Copy environment template
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️ Please edit .env with your API keys before running${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Initialize databases (if Docker is available)
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo ""
    echo "Docker detected. Would you like to start database services? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Starting database services..."
        docker-compose up -d postgres redis chromadb neo4j
        echo -e "${GREEN}✅ Database services started${NC}"
        echo "Waiting for databases to initialize..."
        sleep 10
    fi
else
    echo -e "${YELLOW}⚠️ Docker not found. You'll need to setup databases manually${NC}"
fi

# Download initial models (optional)
echo ""
echo "Would you like to download initial AI models? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Downloading models..."
    python scripts/download_models.py
    echo -e "${GREEN}✅ Models downloaded${NC}"
fi

# Run initialization script
echo ""
echo "Initializing system..."
python scripts/initialize_system.py
echo -e "${GREEN}✅ System initialized${NC}"

# Run tests
echo ""
echo "Would you like to run tests? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Running tests..."
    pytest tests/ -v
    echo -e "${GREEN}✅ Tests completed${NC}"
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}✅ SETUP COMPLETE${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Review config/system_config.yaml for settings"
echo "3. Start the system:"
echo "   - Development: python src/main.py"
echo "   - Production: docker-compose up -d"
echo ""
echo "Access points:"
echo "- API: http://localhost:8000"
echo "- Dashboard: http://localhost:8080"
echo "- Grafana: http://localhost:3000"
echo "- Jupyter: http://localhost:8888"
echo ""
echo "========================================================================"
