from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import os
import pickle
from module import binarized_autoencoder, SAE_baseline
from train import trainer
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
from util import injected_inference
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser(description="Autoencoder Injection Test")
parser.add_argument("--autoencoder_path", type=str, required=True, help="Path to the pre-trained autoencoder model.")
parser.add_argument("--config_path", type=str, help="Path to the configuration file for the Binary Autoencoder / SAE.")
parser.add_argument("--type", type=str, choices=["BAE", "SAE"], default="BAE", help="Type of the autoencoder model.")
parser.add_argument("--layer", type=int, required=True, help="Layer index where the autoencoder will be injected.")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained Llama model.")
parser.add_argument("--dataset_name_or_path", type=str, required=True, help="Path to the dataset for evaluation.")
parser.add_argument("--split", type=str, default="train", help="Dataset split to use for evaluation.")
parser.add_argument("--samples", type=int, default=1024, help="Number of samples to evaluate.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results.")
parser.add_argument("--openai_token", type=str, default=None, help="OpenAI API key for using the backend model")
args = parser.parse_args()

if args.openai_token is None:
    try:
        args.openai_token = os.environ['OAI_TOKEN']
    except KeyError:
        print("OpenAI token must be provided either through --openai_token or OAI_TOKEN environment variable. Use empty string in default.")
        args.openai_token = ""

wandb.init(
    project="BinaryAutoencoder_interpret",
    name=f"FeatureInterpret_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=args,
    save_code=True
)

with open(args.config_path, "rb") as f:
    parameters = pickle.load(f)["params"]

dataset = load_dataset(args.dataset_name_or_path, split=args.split, streaming=True)
sentence = []
for i, example in enumerate(dataset):
    if 'text' in example:
        sentence.append(example['text'])
    elif 'sentence' in example:
        sentence.append(example['sentence'])
    else:
        raise ValueError("Dataset does not contain 'text' or 'sentence' field.")
    if i >= args.samples - 1:
        break

LM = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
dimensions = LM.config.hidden_size

if args.type == "BAE":
    BAE = binarized_autoencoder.BinarizedAutoencoder(
        input_dim = dimensions,
        hidden_dim = parameters["inner_expand_rate"] * dimensions,
        output_dim = dimensions,
        bias = True,
        binarization_type = parameters["binarization_type"],
        output_activation = parameters["output_activation"],
        cycle_for_squarewave = parameters["cycle_for_squarewave"],
    ).cuda()
    with open(args.autoencoder_path, "rb") as f:
        BAE.load_state_dict(torch.load(f))
elif args.type == "SAE" or args.type == "TRC":
    BAE = SAE_baseline.SAE(
        input_dim = dimensions,
        hidden_dim = parameters["inner_expand_rate"] * dimensions,
        output_dim = dimensions,
        bias = True,
        inner_activation = parameters["inner_activation"],
        output_activation = parameters["output_activation"],
    ).cuda()
    with open(args.autoencoder_path, "rb") as f:
        BAE.load_state_dict(torch.load(f))

model = injected_inference.Llama3_injected(
    llama3_model = LM,
    auto_encoder = BAE,
    injected_layer_num = args.layer,
)

injected_run_res = []
for i, text in tqdm(enumerate(sentence)):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    model.injected_run()
    with torch.no_grad():
        injected_outputs = model.forward(**inputs)
    injected_logits = injected_outputs.logits[0][-1:]
    for j in range(len(injected_logits)):
        injected_run_res.append(torch.softmax(injected_logits[j], dim=0).detach().cpu())

model.clean_run()
clean_run_res = []
for i, text in tqdm(enumerate(sentence)):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        clean_outputs = model.forward(**inputs)
    clean_logits = clean_outputs.logits[0][-1:]
    for j in range(len(clean_logits)):
        clean_run_res.append(torch.softmax(clean_logits[j], dim=0).detach().cpu())

injection_clean_JSD = []
for i in range(len(injected_run_res)):
    m = 0.5 * (injected_run_res[i] + clean_run_res[i])
    kl_injected = torch.sum(injected_run_res[i] * (torch.log(injected_run_res[i] + 1e-10) - torch.log(m + 1e-10)))
    kl_clean = torch.sum(clean_run_res[i] * (torch.log(clean_run_res[i] + 1e-10) - torch.log(m + 1e-10)))
    jsd = 0.5 * (kl_injected + kl_clean)
    injection_clean_JSD.append(jsd.item())

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "injection_clean_JSD.pkl"), "wb") as f:
    pickle.dump({'injection_clean_JSD': injection_clean_JSD, 'param': args}, f)

params_txt_path = os.path.join(args.output_dir, "experiment_params.txt")
with open(params_txt_path, "w") as f:
    for key, value in args.__dict__.items():
        f.write(f"{key}: {value}\n")
    f.write("Mean JSD: {:.6f}\n".format(sum(injection_clean_JSD) / len(injection_clean_JSD)))
    print("Mean JSD: {:.6f}".format(sum(injection_clean_JSD) / len(injection_clean_JSD)))
print(f"Experiment parameters saved to {params_txt_path}")