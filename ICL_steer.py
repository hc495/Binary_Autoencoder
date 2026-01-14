import argparse
import StaICC
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from module import binarized_autoencoder, SAE_baseline
import pickle
from util.injected_inference_for_steering.llama3 import Llama3_injected
from util.injected_inference_for_steering.my_model_kernel import test_model
from tqdm import tqdm as tqdm
import os
from datetime import datetime

def locate_feature_index(
    good_activations, # torch.Tensor(num_samples, d_hidden)
    bad_activations,  # torch.Tensor(num_samples, d_hidden)
    top_k = 20,
    method = 'binary' # 'binary' or 'magnitude'
):
    good_activation_frequency = None
    bad_activation_frequency = None
    if method == 'binary':
        good_activation_frequency = good_activations.sum(dim=0) / good_activations.size(0)
        bad_activation_frequency = bad_activations.sum(dim=0) / bad_activations.size(0)
    elif method == 'frequency':
        good_activation_frequency = (good_activations > 0).sum(dim=0) / good_activations.size(0)
        bad_activation_frequency = (bad_activations > 0).sum(dim=0) / bad_activations.size(0)
    elif method == 'magnitude':
        good_activation_frequency = good_activations.mean(dim=0)
        bad_activation_frequency = bad_activations.mean(dim=0)
    else:
        raise ValueError("Invalid method. Choose 'binary' or 'magnitude'.")
    
    print("Good activation frequency:", max(good_activation_frequency))
    print("Bad activation frequency:", max(bad_activation_frequency))
    activation_difference = good_activation_frequency - bad_activation_frequency
    print("Max activation difference:", max(activation_difference))
    _, top_indices = torch.topk(torch.abs(activation_difference), top_k)
    print("Top indices to modify:", top_indices)

    tv_index_vector = []
    if method == 'binary':
        for index in range(good_activations.size(1)):
            if index in top_indices:
                if activation_difference[index] > 0:
                    tv_index_vector.append(1.0)
                elif activation_difference[index] < 0:
                    tv_index_vector.append(-1.0)
                else:
                    tv_index_vector.append(0.0)
            else:
                tv_index_vector.append(0.0)
    elif method == 'magnitude':
        for index in range(good_activations.size(1)):
            if index in top_indices:
                tv_index_vector.append(activation_difference[index])
            else:
                tv_index_vector.append(0.0)

    return torch.tensor(tv_index_vector).float()

def normalize_tv(
    tv_vector,
    original_good_activations,
    original_bad_activations
):
    original_tv = torch.mean(torch.stack(original_good_activations), dim=0).to("cuda") - torch.mean(torch.stack(original_bad_activations), dim=0).to("cuda")
    scale = torch.norm(original_tv) / torch.norm(tv_vector)
    return tv_vector * scale

parser = argparse.ArgumentParser(description="TV injection")
parser.add_argument("--model_name", type=str, required=True, help="Path to the pretrained model")
parser.add_argument("--ICL_dataset_index", type=int, default=0, help="ICL dataset index, defined in StaICC.")
parser.add_argument("--top_k", type=int, default=20, help="Top K features to modify")
parser.add_argument("--BAE_path", type=str, help="Path to the Binary Autoencoder model / SAE model / TransCoder model")
parser.add_argument("--type", type=str, default="BAE", help="Type of the model to use for feature extraction, SAE, BAE, none. None for centroid method.")
parser.add_argument("--config_path", type=str, help="Path to the configuration file for the Binary Autoencoder / SAE / TransCoder")
parser.add_argument("--layer", type=int, default=-1, help="Layer to extract and inject features from")
parser.add_argument("--injection_strength", type=float, default=0.2, help="Strength of the feature injection")
parser.add_argument("--estimation_sample_size", type=int, default=512, help="Number of samples to estimate the good/bad activation statistics")
parser.add_argument("--save_path", type=str, default="logs/ICL_steer", help="Path to save the logs")
parser.add_argument("--normalize", action="store_true", help="Whether to normalize the TV vector")

args = parser.parse_args()

with open(args.config_path, "rb") as f:
    parameters = pickle.load(f)["params"]

LM = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
dimensions = LM.config.hidden_size

injected_model = Llama3_injected(
    llama3_model = LM,
    auto_encoder = None,
    injected_layer_num = -1,
    hook = True,
    output_hidden_states = True,
    output_attentions = False,
    only_last_token_hidden_states = True
).cuda()

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
    with open(args.BAE_path, "rb") as f:
        BAE.load_state_dict(torch.load(f))
    method = "binary"
elif args.type == "SAE":
    BAE = SAE_baseline.SAE(
        input_dim = dimensions,
        hidden_dim = parameters["inner_expand_rate"] * dimensions,
        output_dim = dimensions,
        bias = True,
        inner_activation = parameters["inner_activation"],
        output_activation = parameters["output_activation"],
    ).cuda()
    with open(args.BAE_path, "rb") as f:
        BAE.load_state_dict(torch.load(f))
    method = "magnitude"
elif args.type == "none":
    BAE = None

benchmark = StaICC.Normal(k=0)
experimentor = benchmark[args.ICL_dataset_index]
experimentor.prompt_former.replace_space_to_label()

calibration_set = experimentor.calibration_set()

good_prompts = []
bad_prompts = []
for sample_index in range(args.estimation_sample_size):
    query_dataline = calibration_set.get_input_text(sample_index)
    demons_datalines = []
    for i in range(4):
        demons_datalines.append(calibration_set[sample_index + i])
    prompt = experimentor.prompt_former.write_prompt_from_dataline(demons_datalines, query_dataline)
    good_prompts.append(prompt)
    prompt = experimentor.prompt_former.write_prompt_from_dataline([], query_dataline)
    bad_prompts.append(prompt)

good_hs = []
bad_hs = []
good_activations = []
bad_activations = []
print("Extracting activations for good and bad prompts...")
with torch.no_grad():
    for prompt in tqdm(good_prompts):
        inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to("cuda")
        outputs = injected_model(inputs)
        hidden_states = torch.Tensor(outputs.hidden_states["hidden_states"][args.layer][0, -1, :])
        good_hs.append(hidden_states)
        if args.type != "none":
            if args.type != "BAE" and parameters["inner_activation"] == 'topk':
                activation = BAE.encode(hidden_states.unsqueeze(0).to("cuda"))
                good_activations.append(activation[0])
            else:
                activation = BAE.encode(hidden_states.to("cuda"))
                good_activations.append(activation)
    
    for prompt in tqdm(bad_prompts):
        inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to("cuda")
        outputs = injected_model(inputs)
        hidden_states = torch.Tensor(outputs.hidden_states["hidden_states"][args.layer][0, -1, :])
        bad_hs.append(hidden_states)
        if args.type != "none":
            if args.type != "BAE" and parameters["inner_activation"] == 'topk':
                activation = BAE.encode(hidden_states.unsqueeze(0).to("cuda"))
                bad_activations.append(activation[0])
            else:
                activation = BAE.encode(hidden_states.to("cuda"))
                bad_activations.append(activation)

print("Calculating feature index to modify...")
if args.type != "none":
    good_activations_tensor = torch.stack(good_activations)
    bad_activations_tensor = torch.stack(bad_activations)
    tv_index_vector = locate_feature_index(
        good_activations = good_activations_tensor,
        bad_activations = bad_activations_tensor,
        top_k = args.top_k,
        method = method
    ).to("cuda")
    tv = BAE.decoder.weight.data @ tv_index_vector
    print(tv.shape)
else:
    tv = torch.mean(torch.stack(good_hs), dim=0).to("cuda") - torch.mean(torch.stack(bad_hs), dim=0).to("cuda")

if args.normalize: 
    tv = normalize_tv(
        tv_vector = tv,
        original_good_activations = good_hs,
        original_bad_activations = bad_hs
    )

print("Performing ICL with feature steering...")
res = test_model(
    model = injected_model,
    tokenizer = tokenizer,
    experimentor = experimentor,
    task_vector = tv * args.injection_strength,
    task_vector_layer = args.layer,
)

print(res['res'])

path = args.save_path
os.makedirs(path, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
filename = f"ICL_steer_{args.model_name.split('/')[-1]}_dataset{args.ICL_dataset_index}_layer{args.layer}_topk{args.top_k}_strength{args.injection_strength}_{timestamp}.txt"

with open(os.path.join(path, filename), "w") as f:
    for keys, value in parameters.items():
        f.write(f"{keys}: {value}\n")
    f.write(f"{res['res']}\n")

pkl_save_path = os.path.join(path, filename.replace(".txt", ".pkl"))
with open(pkl_save_path, "wb") as f:
    pickle.dump({
        "parameters": parameters,
        "results": res['res'],
    }, f)