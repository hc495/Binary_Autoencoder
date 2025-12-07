import torch
import argparse
import os
import pickle
from module import SAE_baseline
from train import trainer
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

parser = argparse.ArgumentParser(description="Feature Reconstruction on Binary Autoencoder")
parser.add_argument("--feature_path", type=str, required=True, help="Path to the feature set")
parser.add_argument("--MLP_input_path", type=str, default=None, help="Path to the MLP input feature set")
parser.add_argument("--layered_feature", action='store_true', help="Whether the feature set is layered")
parser.add_argument("--layer", type=int, default=-1, help="Layer index for the feature set")
parser.add_argument("--inner_expand_rate", type=int, default=4, help="Inner expand rate for the autoencoder")
parser.add_argument("--inner_activation", type=str, default='none', help="Output activation function for the autoencoder: 'none', 'sigmoid', 'tanh', 'relu', 'jumprelu' and 'topk'")
parser.add_argument("--topk", type=int, default=10, help="Top k features to consider for the inner activation function if 'topk' is selected")
parser.add_argument("--theta", type=float, default=0.5, help="Threshold for the 'jumprelu' activation function")
parser.add_argument("--output_activation", type=str, default='none', help="Output activation function for the autoencoder: 'none', 'sigmoid', 'tanh', or 'relu'")
parser.add_argument("--bias", action='store_true', help="Whether to use bias in the autoencoder")
parser.add_argument("--cuda", action='store_true', help="Use CUDA for training")
parser.add_argument("--dataset_split", type=float, default=0.8, help="Split ratio for training and validation datasets")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=3500, help="Number of epochs for training")
parser.add_argument("--l1_norm_weight", type=float, default=0.0, help="Weight for the L1 norm regularization")
parser.add_argument("--l1_start_epoch", type=int, default=0, help="Epoch to start applying L1 norm regularization")
parser.add_argument("--save_dir", type=str, default="logs", help="Directory to save the results")
parser.add_argument("--dont_save_model", action='store_true', help="Don't save the model after training")
parser.add_argument("--dont_save_log", action='store_true', help="Don't save the log after training")
args = parser.parse_args()

# If using 'topk' activation, set l1_norm_weight to 0
if args.inner_activation == 'topk':
    args.l1_norm_weight = 0.0
    print("Warning: 'topk' activation is selected, setting l1_norm_weight to 0.0")

# Load the feature set

if os.path.exists(args.feature_path):
    with open(args.feature_path, "rb") as f:
        hidden_state = pickle.load(f)
else:
    raise FileNotFoundError(f"Feature path {args.feature_path} does not exist. Please check the path.")

if args.MLP_input_path is not None and os.path.exists(args.MLP_input_path):
    with open(args.MLP_input_path, "rb") as f:
        mlp_input_state = pickle.load(f)
    print(f"Using MLP input features from {args.MLP_input_path}")
else:
    mlp_input_state = None
    print("No MLP input feature path provided or file does not exist, using hidden states instead.")

dimensions = 0
if args.layered_feature:
    dimensions = hidden_state[args.layer][0].shape[0]
    dataset = hidden_state[args.layer]
else:
    dimensions = hidden_state[0].shape[0]
    dataset = hidden_state


# Create the autoencoder

auto_encoder = SAE_baseline.SAE(
    input_dim = dimensions,
    hidden_dim = args.inner_expand_rate * dimensions,
    output_dim = dimensions,
    bias = args.bias,
    inner_activation = args.inner_activation,
    output_activation = args.output_activation,
    topk = args.topk,
    theta = args.theta,
)
if args.cuda:
    auto_encoder = auto_encoder.cuda()
else:
    print("Warning: CUDA is not enabled. Running on CPU.")
    print("Warning: CUDA is not enabled. Running on CPU.")
    print("Warning: CUDA is not enabled. Running on CPU.")
    print("Warning: CUDA is not enabled. Running on CPU.")

wandb.init(
    project="BinaryAutoencoder",
    name=f"SAEFeatureReconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=args,
    save_code=True
)

# Train

if mlp_input_state is None:
    log = trainer.SAE_trainer(
        auto_encoder,
        train_dataset = dataset[0 : int(args.dataset_split*len(dataset))],
        val_dataset = dataset[int(args.dataset_split*len(dataset)) : ],
        optimizer = torch.optim.Adam(
            auto_encoder.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        ),
        batch_size = args.batch_size,
        num_epochs = args.num_epochs,
        L1_weight = args.l1_norm_weight,
        L1_start_epoch = args.l1_start_epoch,
        cuda = args.cuda,
    )
else:
    log = trainer.SAE_trainer(
        auto_encoder,
        train_dataset = mlp_input_state[0 : int(args.dataset_split*len(dataset))],
        val_dataset = mlp_input_state[int(args.dataset_split*len(dataset)) : ],
        train_target_dataset = dataset[0 : int(args.dataset_split*len(dataset))],
        val_target_dataset = dataset[int(args.dataset_split*len(dataset)) : ],
        optimizer = torch.optim.Adam(
            auto_encoder.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        ),
        batch_size = args.batch_size,
        num_epochs = args.num_epochs,
        L1_weight = args.l1_norm_weight,
        L1_start_epoch = args.l1_start_epoch,
        cuda = args.cuda,
    )

# Create the save directory
os.makedirs(args.save_dir, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

experiment_params = {
    "feature_path": args.feature_path,
    "layered_feature": args.layered_feature,
    "layer": args.layer,
    "inner_expand_rate": args.inner_expand_rate,
    "inner_activation": args.inner_activation,
    "output_activation": args.output_activation,
    "bias": args.bias,
    "cuda": args.cuda,
    "dataset_split": args.dataset_split,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "batch_size": args.batch_size,
    "num_epochs": args.num_epochs,
    "l1_norm_weight": args.l1_norm_weight,
    "l1_start_epoch": args.l1_start_epoch,
    "save_dir": args.save_dir,
    "dont_save_model": args.dont_save_model,
    "dont_save_log": args.dont_save_log,
    "current_time": current_time,
}

report_path = os.path.join(args.save_dir, f"report_{current_time}")
os.makedirs(report_path, exist_ok=True)

params_txt_path = os.path.join(report_path, "experiment_params.txt")
with open(params_txt_path, "w") as f:
    for key, value in experiment_params.items():
        f.write(f"{key}: {value}\n")
print(f"Experiment parameters saved to {params_txt_path}")

wandb.finish()

# 可视化log中的六个内容
log_keys = list(log.keys())
for key in log_keys:
    plt.figure()
    plt.plot(log[key])
    plt.title(key)
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.grid(True)
    if "loss" in key:
        plt.ylim(-0.05, 0.7)
    fig_path = os.path.join(report_path, f"{key}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"{key} plot saved to {fig_path}")

if not args.dont_save_model:
    model_save_path = os.path.join(report_path, f"model_{current_time}.pt")
    torch.save(auto_encoder.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
if not args.dont_save_log:
    log_dic = {
        "log": log,
        "params": experiment_params
    }
    log_save_path = os.path.join(report_path, f"log_{current_time}.pkl")
    with open(log_save_path, "wb") as f:
        pickle.dump(log_dic, f)
    print(f"Log saved to {log_save_path}")