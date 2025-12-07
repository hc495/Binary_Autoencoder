## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--feature_path` | str | required | Path to the feature set. Should be a `.pkl` file containing either a list-like of `torch.Tensor(dim)` or a list of list-like `torch.Tensor(dim)`. |
| `--layered_feature` | flag | False | Whether the feature set is layered. If set, `feature_path` should point to a list of list-like `torch.Tensor(dim)`. Otherwise, it should point to a list-like `torch.Tensor(dim)`. |
| `--layer` | int | -1 | Layer index for the feature set. Used only if `--layered_feature` is set. |
| `--inner_expand_rate` | int | 4 | Inner expand rate for the autoencoder. |
| `--binarization_type` | str | 'sign_s' | Binarization type for the autoencoder: `'sign'`, `'sign_s'`, or `'squarewave'`. |
| `--cycle_for_squarewave` | int | 2 | Cycle parameter for squarewave binarization. |
| `--output_activation` | str | 'none' | Output activation function for the autoencoder: `'none'`, `'sigmoid'`, `'tanh'`, or `'relu'`. |
| `--bias` | flag | False | Whether to use bias in the autoencoder. |
| `--cuda` | flag | False | Use CUDA for training. |
| `--dataset_split` | float | 0.8 | Split ratio for training and validation datasets. |
| `--lr` | float | 1e-3 | Learning rate for the optimizer. |
| `--weight_decay` | float | 0 | Weight decay for the optimizer. |
| `--batch_size` | int | 512 | Batch size for training. |
| `--num_epochs` | int | 3500 | Number of epochs for training. |
| `--entropy_weight` | float | 0e-8 | Weight for the entropy loss. |
| `--covarience_weight` | float | 0e-8 | Weight for the covariance loss. |
| `--entropy_threshold` | float | 0 | Threshold for the entropy loss. |
| `--entropy_start_epoch` | int | 500 | Epoch to start applying the entropy loss. |
| `--save_dir` | str | 'logs' | Directory to save the results (logs, model, plots). |
| `--dont_save_model` | flag | False | Do not save the model after training. |
| `--dont_save_log` | flag | False | Do not save the log after training. |

## Standard Usage

```bash
python feature_reconstruction.py \
    --feature_path "feature.pkl" \
    --cuda \
    --layered_feature \
    --layer $layer \
    --lr 5e-4 \
    --inner_expand_rate 4 \
    --num_epochs 2000 \
    --bias \
    --covarience_weight 1e-7 \
    --entropy_weight 1e-7 \
    --binarization_type "sign_s" \
    --save_dir "logs/" 

```