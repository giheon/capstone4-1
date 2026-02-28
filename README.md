# VaDE (PyTorch)

This directory now contains a modern PyTorch implementation of VaDE (Variational Deep Embedding).

The implementation keeps the core behavior of the original code:
- VaDE objective (negative ELBO structure)
- jointly train neural network parameters and GMM parameters
- separate learning rates for NN and GMM parts
- dataset-specific hyperparameters (`mnist`, `reuters10k`, `har`)

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- matplotlib (optional, for MNIST visualization)
- Pillow (for saving generated MNIST grids)

## Files

- `VaDE.py`: PyTorch training entrypoint
- `vade_core.py`: shared model/data/loss/training utilities
- `VaDE_test_mnist.py`: evaluate MNIST checkpoint + class-conditioned digit generation
- `VaDE_test_reuters_all.py`: evaluate Reuters-all checkpoint

## Train

```bash
python VaDE.py mnist
python VaDE.py reuters10k
python VaDE.py har
```

Optional examples:

```bash
python VaDE.py mnist --epochs 200 --lr-nn 0.001 --lr-gmm 0.0005
python VaDE.py reuters_all --data-root dataset --save-path checkpoints/vade_reuters_all.pt
```

## Test (MNIST)

```bash
python VaDE_test_mnist.py --checkpoint checkpoints/vade_mnist.pt --show
```

This prints clustering accuracy and saves a generated digit grid (`digits.jpg` by default).

## Test (Reuters-all)

```bash
python VaDE_test_reuters_all.py --checkpoint checkpoints/vade_reuters_all.pt
```

If Reuters raw files are not present, download them with:

```bash
cd dataset/reuters
./get_data.sh
```
