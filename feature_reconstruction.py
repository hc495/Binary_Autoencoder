import torch
import argparse
import os
import pickle
from module import binarized_autoencoder
from train import trainer
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

parser = argparse.ArgumentParser(description="Feature Reconstruction on Binary Autoencoder")
parser.add_argument("--feature_path", type=str, required=True, help="Path to the feature set. Should be a .pkl file of: listlike[torch.Tensor(dim)] or list[listlike[torch.Tensor(dim)]]")
parser.add_argument("--layered_feature", action='store_true', help="Whether the feature set is layered. If set, feature_path should point to a list[listlike[torch.Tensor(dim)]]. If not, it should point to a listlike[torch.Tensor(dim)]")
parser.add_argument("--layer", type=int, default=-1, help="Layer index for the feature set, used only if layered_feature is set.")
parser.add_argument("--inner_expand_rate", type=int, default=4, help="Inner expand rate for the autoencoder")
parser.add_argument("--binarization_type", type=str, default='sign_s', help="Binarization type for the autoencoder: 'sign', 'sign_s' or 'squarewave'")
parser.add_argument("--cycle_for_squarewave", type=int, default=2, help="Cycle for squarewave binarization")
parser.add_argument("--output_activation", type=str, default='none', help="Output activation function for the autoencoder: 'none', 'sigmoid', 'tanh', or 'relu'")
parser.add_argument("--bias", action='store_true', help="Whether to use bias in the autoencoder")
parser.add_argument("--cuda", action='store_true', help="Use CUDA for training")
parser.add_argument("--dataset_split", type=float, default=0.8, help="Split ratio for training and validation datasets")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=3500, help="Number of epochs for training")
parser.add_argument("--entropy_weight", type=float, default=0e-8, help="Weight for the entropy loss")
parser.add_argument("--covarience_weight", type=float, default=0e-8, help="Weight for the covariance loss")
parser.add_argument("--entropy_threshold", type=float, default=0, help="Threshold for the entropy loss")
parser.add_argument("--entropy_start_epoch", type=int, default=500, help="Epoch to start applying the entropy loss")
parser.add_argument("--save_dir", type=str, default="logs", help="Directory to save the results (logs, model, plots)")
parser.add_argument("--dont_save_model", action='store_true', help="Don't save the model after training")
parser.add_argument("--dont_save_log", action='store_true', help="Don't save the log after training")
args = parser.parse_args()

# Load the feature set

if os.path.exists(args.feature_path):
    with open(args.feature_path, "rb") as f:
        hidden_state = pickle.load(f)
else:
    raise FileNotFoundError(f"Feature path {args.feature_path} does not exist. Please check the path.")

dimensions = 0
if args.layered_feature:
    dimensions = hidden_state[args.layer][0].shape[0]
    dataset = hidden_state[args.layer]
else:
    dimensions = hidden_state[0].shape[0]
    dataset = hidden_state


# Create the autoencoder

auto_encoder = binarized_autoencoder.BinarizedAutoencoder(
    input_dim = dimensions,
    hidden_dim = args.inner_expand_rate * dimensions,
    output_dim = dimensions,
    bias = args.bias,
    binarization_type = args.binarization_type,
    output_activation = args.output_activation,
    cycle_for_squarewave = args.cycle_for_squarewave,
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
    name=f"FeatureReconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=args,
    save_code=True
)

# Train

log = trainer.auto_encoder_trainer(
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
    entropy_weight = args.entropy_weight,
    covarience_weight = args.covarience_weight,
    entropy_threshold = args.entropy_threshold,
    entropy_start_epoch = args.entropy_start_epoch,
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
    "binarization_type": args.binarization_type,
    "cycle_for_squarewave": args.cycle_for_squarewave,
    "output_activation": args.output_activation,
    "cuda": args.cuda,
    "dataset_split": args.dataset_split,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "batch_size": args.batch_size,
    "num_epochs": args.num_epochs,
    "entropy_weight": args.entropy_weight,
    "covarience_weight": args.covarience_weight,
    "entropy_threshold": args.entropy_threshold,
    "entropy_start_epoch": args.entropy_start_epoch
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