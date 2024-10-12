#!/usr/bin/env fish
#SBATCH --job-name=diffmet.ntuple
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=./logs/run_%j.out
#SBATCH --error=./logs/run_%j.err

echo "START: $(date)"
echo "USER: $(whoami)"
echo "HOST: $(hostname)"
echo "CWD: $(pwd)"

echo "dataset=$dataset"
echo "counter=$counter"

micromamba shell hook --shell fish | source
micromamba activate diffmet-ntuple-py311

./ntupleise.py batch -d $dataset -c $counter

echo "END: $(date)"
