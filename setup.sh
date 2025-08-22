#!/bin/bash

# PointStream Quick Setup Script
# This script helps you set up the PointStream environment and run your first reconstruction

set -e

echo "🎬 PointStream Quick Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
POINTSTREAM_DIR="$SCRIPT_DIR"

cd "$POINTSTREAM_DIR"

print_step "Setting up PointStream environments..."

# Setup mode selection
echo ""
echo "Choose setup mode:"
echo "1) Full setup (server + client environments)"
echo "2) Server only"
echo "3) Client only"
echo "4) Single environment (development mode)"
read -p "Enter choice (1-4): " setup_mode

case $setup_mode in
    1)
        print_step "Setting up server environment..."
        if [ -f "server/environment.yml" ]; then
            conda env create -f server/environment.yml --force
            print_status "Server environment created: pointstream"
        else
            print_error "Server environment.yml not found!"
            exit 1
        fi

        print_step "Setting up client environment..."
        if [ -f "client/environment.yml" ]; then
            conda env create -f client/environment.yml --force
            print_status "Client environment created: pointstream-client"
        else
            print_error "Client environment.yml not found!"
            exit 1
        fi

        print_step "Installing additional client dependencies..."
        conda run -n pointstream-client pip install -r client/requirements.txt
        ;;
    2)
        print_step "Setting up server environment only..."
        if [ -f "server/environment.yml" ]; then
            conda env create -f server/environment.yml --force
            print_status "Server environment created: pointstream"
        else
            print_error "Server environment.yml not found!"
            exit 1
        fi
        ;;
    3)
        print_step "Setting up client environment only..."
        if [ -f "client/environment.yml" ]; then
            conda env create -f client/environment.yml --force
            conda run -n pointstream-client pip install -r client/requirements.txt
            print_status "Client environment created: pointstream-client"
        else
            print_error "Client environment.yml not found!"
            exit 1
        fi
        ;;
    4)
        print_step "Setting up single development environment..."
        if [ -f "server/environment.yml" ]; then
            conda env create -f server/environment.yml --force
            conda run -n pointstream pip install -r client/requirements.txt
            print_status "Single environment created: pointstream"
        else
            print_error "Server environment.yml not found!"
            exit 1
        fi
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    mkdir -p models
    print_status "Created models directory"
fi

# Check for GPU support
print_step "Checking GPU support..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected - hardware acceleration available"
    else
        print_warning "NVIDIA drivers may not be properly installed"
    fi
else
    print_warning "No NVIDIA GPU detected - will use CPU (slower training)"
fi

# Test video suggestion
echo ""
print_step "Quick test run suggestion:"
echo ""
echo "To test PointStream with a sample video:"
echo ""
echo "1. Download a test video:"
echo "   wget https://sample-videos.com/zip/10/mp4/SampleVideo_720x480_1mb.mp4"
echo ""
echo "2. Run the full pipeline:"
if [ "$setup_mode" == "4" ]; then
    echo "   conda activate pointstream"
    echo "   python main.py SampleVideo_720x480_1mb.mp4 --full-pipeline"
else
    echo "   # For server processing:"
    echo "   conda activate pointstream"
    echo "   python main.py SampleVideo_720x480_1mb.mp4 --server-only"
    echo ""
    echo "   # For client processing:"
    echo "   conda activate pointstream-client"
    echo "   python main.py --client-only --metadata-dir ./metadata"
fi

echo ""
echo "3. For a quick server-only test:"
if [ "$setup_mode" == "4" ]; then
    echo "   conda activate pointstream"
else
    echo "   conda activate pointstream"
fi
echo "   cd server"
echo "   python server.py ../SampleVideo_720x480_1mb.mp4"

echo ""
print_step "Configuration customization:"
echo ""
echo "Edit configuration files:"
echo "- Server config: server/config.ini"
echo "- Client config: client/config.ini"
echo ""
echo "Key settings to adjust:"
echo "- Batch sizes (if you get memory errors)"
echo "- Model training epochs"
echo "- Video quality settings"
echo "- Output directories"

echo ""
print_status "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Activate the appropriate conda environment"
echo "2. Review and customize configuration files"
echo "3. Run your first video reconstruction"
echo ""
echo "For help:"
echo "  python main.py --help"
echo "  python server/server.py --help"
echo "  python client/client.py --help"
echo ""
echo "For detailed documentation, see README.md"

# Create a quick launcher script
print_step "Creating launcher script..."
cat > run_pointstream.sh << 'EOF'
#!/bin/bash

# PointStream Launcher Script
# Quick launcher for common operations

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "PointStream Launcher"
    echo "==================="
    echo ""
    echo "Usage:"
    echo "  $0 full <video_file>          # Run full pipeline"
    echo "  $0 server <video_file>        # Server processing only"
    echo "  $0 client <metadata_dir>      # Client reconstruction only"
    echo "  $0 assess <original> <recon>  # Quality assessment only"
    echo "  $0 train <metadata_dir>       # Train models only"
    echo ""
    echo "Examples:"
    echo "  $0 full video.mp4"
    echo "  $0 server video.mp4"
    echo "  $0 client ./metadata"
    echo "  $0 assess video.mp4 ./reconstructed"
    exit 1
fi

case $1 in
    full)
        if [ -z "$2" ]; then
            echo "Error: Video file required"
            exit 1
        fi
        echo "Running full pipeline on $2..."
        conda run -n pointstream python main.py "$2" --full-pipeline
        ;;
    server)
        if [ -z "$2" ]; then
            echo "Error: Video file required"
            exit 1
        fi
        echo "Running server processing on $2..."
        conda run -n pointstream python main.py "$2" --server-only
        ;;
    client)
        if [ -z "$2" ]; then
            echo "Error: Metadata directory required"
            exit 1
        fi
        echo "Running client reconstruction from $2..."
        conda run -n pointstream-client python main.py --client-only --metadata-dir "$2"
        ;;
    assess)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Error: Original and reconstructed video paths required"
            exit 1
        fi
        echo "Assessing quality: $2 vs $3..."
        conda run -n pointstream python main.py "$2" --assess-quality "$3"
        ;;
    train)
        if [ -z "$2" ]; then
            echo "Error: Metadata directory required"
            exit 1
        fi
        echo "Training models from $2..."
        conda run -n pointstream-client python client/scripts/model_trainer.py "$2"
        ;;
    *)
        echo "Error: Unknown command '$1'"
        echo "Use '$0' without arguments to see usage"
        exit 1
        ;;
esac
EOF

chmod +x run_pointstream.sh
print_status "Created launcher script: run_pointstream.sh"

echo ""
echo "You can now use the launcher for common operations:"
echo "  ./run_pointstream.sh full video.mp4"
echo "  ./run_pointstream.sh server video.mp4"
echo "  ./run_pointstream.sh client ./metadata"
