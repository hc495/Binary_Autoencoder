from datasets import load_dataset
import argparse
from util import interp_score
import os
from module import binarized_autoencoder, SAE_baseline
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import torch
from tqdm import tqdm
from datetime import datetime
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="bookcorpus", help="Name of the dataset to load")
parser.add_argument("--split", type=str, default="train", help="Split of the dataset to load")
parser.add_argument("--sample_size", type=int, default=8192, help="Number of samples to print from the dataset")
parser.add_argument("--LM_name", type=str, default="meta-llama/Llama-3.2-1B", help="Name of the language model to use")
parser.add_argument("--layer", type=int, default=11, help="Layer of the language model to use for feature extraction")
parser.add_argument("--backend_model_name", type=str, default="gpt-4.1-mini", help="Name of the backend model for interpretation")
parser.add_argument("--BAE_path", type=str, help="Path to the Binary Autoencoder model / SAE model / TransCoder model")
parser.add_argument("--type", type=str, default="BAE", help="Type of the model to use for feature extraction, SAE, BAE, TRC, random")
parser.add_argument("--config_path", type=str, help="Path to the configuration file for the Binary Autoencoder / SAE / TransCoder")
parser.add_argument("--top_k", type=int, default=10, help="Top k features to consider for interpretation")
parser.add_argument("--openai_token", type=str, default=None, help="OpenAI API key for using the backend model")
parser.add_argument("--save_path", type=str, default="logs/interp_score_results/", help="Path to save the interpretation results")
parser.add_argument("--no_interpret", action='store_true', help="If set, will not run the interpretation process")
parser.add_argument("--SAE_rescale", action='store_true', help="If set, will use the SAE baseline with rescaling")

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

dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)
sentence = []
for i, example in enumerate(dataset):
    if 'text' in example:
        sentence.append(example['text'])
    elif 'sentence' in example:
        sentence.append(example['sentence'])
    else:
        raise ValueError("Dataset does not contain 'text' or 'sentence' field.")
    if i >= args.sample_size - 1:
        break

with open(args.config_path, "rb") as f:
    parameters = pickle.load(f)["params"]

LM = AutoModelForCausalLM.from_pretrained(args.LM_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(args.LM_name)
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
    with open(args.BAE_path, "rb") as f:
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
    with open(args.BAE_path, "rb") as f:
        BAE.load_state_dict(torch.load(f))
elif args.type == "random":
    BAE = None

if args.type == "BAE":
    feature_source = interp_score.feature_extractor_for_BAE(
        LM=LM,
        tokenizer=tokenizer,
        sentences=sentence,
        trained_BAE=BAE,
        layer_num=args.layer
    )
elif args.type == "SAE":
    feature_source = interp_score.feature_extractor_for_SAE(
        LM=LM,
        tokenizer=tokenizer,
        sentences=sentence,
        trained_SAE=BAE,
        layer_num=args.layer,
        rescale=args.SAE_rescale
    )
elif args.type == "random":
    feature_source = interp_score.feature_extractor_for_random(
        dimensions = parameters["inner_expand_rate"] * dimensions,
        tokenizer=tokenizer,
        sentences=sentence,
    )
elif args.type == "TRC":
    feature_source = interp_score.feature_extractor_for_Transcoder(
        LM=LM,
        tokenizer=tokenizer,
        sentences=sentence,
        trained_SAE=BAE,
        layer_num=args.layer,
        rescale=args.SAE_rescale
    )
feature_source.encode_sentences()
interpreter = interp_score.interp_score(
    feature_num=parameters["inner_expand_rate"] * dimensions,
    backend_model_name=args.backend_model_name,
    top_k=args.top_k,
    interpretation_length=5,
    openai_api_key=args.openai_token,
)

for i in tqdm(range(len(feature_source))):
    feature = feature_source.get_encoded_features(i)
    interpreter.add_original_feature(
        feature['token'],
        feature['location'],
        feature['sentence'],
        feature['encoded_feature']
    )
if args.no_interpret:
    print("Skipping interpretation process as --no_interpret is set.")
else:
    interpreter.run()

save_path = args.save_path
save_name = f"interp_score_{args.dataset_name}_{args.split}_{args.LM_name.split('/')[-1]}_layer_{args.layer}_topk_{args.top_k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, save_name), "wb") as f:
    pickle.dump((interpreter.feature_interpretation, interpreter.original_feature_set), f)