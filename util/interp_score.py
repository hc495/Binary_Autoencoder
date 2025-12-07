import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import wandb
from scipy.stats import chi2
from util.hooked_Llama_for_transcoder import make_hooked_llama3

def find_tokenized_position_in_another_tokenizer(
    str,
    tokenizer,
    location,
    target_tokenizer
):
    tokenized = tokenizer(str, return_tensors="pt")
    tokenized_str = tokenized["input_ids"][0]
    if tokenized_str[0] == tokenizer.bos_token_id:
        if location == 0:
            return 0
        tokenized_str = tokenized_str[1:]
        location -= 1
    tokenized_str_cutted = tokenized_str[:location + 1]
    str_cutted = tokenizer.decode(tokenized_str_cutted)
    # Find the longest common prefix between str_cutted and str
    max_prefix_len = 0
    for i in range(min(len(str_cutted), len(str)) + 1):
        if str_cutted[:i] == str[:i]:
            max_prefix_len = i
        else:
            break
    longest_prefix = str[:max_prefix_len]
    another_tokenized = target_tokenizer(longest_prefix, return_tensors="pt")
    another_tokenized_str = another_tokenized["input_ids"][0]
    return len(another_tokenized_str) - 1


class interp_score():
    def __init__(
        self,
        feature_num,
        backend_model_name = "o4-mini",
        top_k = 10,
        interpretation_length = 10,
        openai_api_key=None,
        max_test_samples = 8
    ):
        self.feature_num = feature_num
        self.backend_model_name = backend_model_name
        self.top_k = top_k
        self.interpretation_length = interpretation_length
        self.max_test_samples = max_test_samples

        self.original_feature_set = {}
        self.feature_interpretation = {}
        for i in range(self.feature_num):
            self.original_feature_set[i] = []
            self.feature_interpretation[i] = {
                "semantic": "",
                "interpretable": False,
                "test_score": None
            }

        self.client = OpenAI(api_key = openai_api_key)
        
        self.instructions = '''
            Instruction: \n
            I will provide a set of tokens along with their positions (this position may vary depending on the tokenizer) and the surrounding context. Please describe what these tokens have in common using concise expressions such as "date expressions", "words ending in 'ing'", or "adjectives". \n
            Please choose the most specific term while ensuring commonality, and avoid using overly general terms like "words", "English tokens", "high-frequency English lexemes", or "phrases". \n
            Non-semantic or non-linguistic terms such as "BPE Subword Token" are strictly prohibited. \n
            Any additional information, explaination, or context are strongly prohibited. Only return one phrase. \n
            \n
            Example 1:\n
            Token: "running" at position 3 in sentence: "She is running in the park." \n
            Token: "eating" at position 3 in sentence: "He is eating an apple." \n
            Token: "sleeping" at position 4 in sentence: "The baby is sleeping on the sofa." \n
            Token: "jumping" at position 2 in sentence: "They are jumping over the fence." \n
            Token: "talking" at position 3 in sentence: "We are talking about the project." \n
            The commonality is: -ing verbs of human behavior\n
            \n
            Example 2:\n
            Token: "yesterday" at position 4 in sentence: "I went there yesterday."\n
            Token: "last week" at position 5 in sentence: "She arrived last week."\n
            Token: "in 1998" at position 6 in sentence: "They moved here in 1998."\n
            Token: "last year" at position 5 in sentence: "We met last year."\n
            The commonality is: past time expressions\n
            \n
            Example 3:\n
            Token: "happy" at position 4 in sentence: "She looks very happy today."\n
            Token: "angry" at position 5 in sentence: "They were extremely angry about it."\n
            Token: "sad" at position 4 in sentence: "He felt really sad after the call."\n
            The commonality is: emotional adjectives\n
            \n
            Example 4:\n
            Token: "dog" at position 2 in sentence: "The dog barked loudly."\n
            Token: "cat" at position 2 in sentence: "The cat chased the mouse."\n
            Token: "bird" at position 2 in sentence: "The bird sang beautifully."\n
            Token: "fish" at position 2 in sentence: "The fish swam gracefully in the tank."\n
            The commonality is: animal nouns\n
            \n
            Example 5:\n
            Token: "quickly" at position 3 in sentence: "He ran quickly toward the exit."\n
            Token: "silently" at position 3 in sentence: "She moved silently through the room."\n
            Token: "carefully" at position 6 in sentence: "They handled the fragile vase carefully."\n
            Token: "eagerly" at position 4 in sentence: "The children waited eagerly for the show to start."\n
            The commonality is: manner adverbs\n
            \n
            Example 6:\n
            Token: "quick" at position 2 in sentence: "She is quick to respond."\n
            Token: "slow" at position 2 in sentence: "He is slow to understand."\n
            Token: "fast" at position 4 in sentence: "The car is fast on the highway."\n
            The commonality is: adjectives describing speed\n
            \n
            Example 7:\n
            Token: "first" at position 4 in sentence: "This is the first time I have seen this."\n
            Token: "second" at position 4 in sentence: "This is the second time I have visited this place."\n
            Token: "third" at position 4 in sentence: "This is the third time I have heard this story."\n
            Token: "fourth" at position 4 in sentence: "This is the fourth time I have tried this recipe."\n
            The commonality is: ordinal numbers describing sequence\n
            \n
            Example 8:\n
            Token: "date" at position 2 in sentence: "The date of the meeting is next Monday."\n
            Token: "time" at position 2 in sentence: "The time of the event is 3 PM."\n
            Token: "schedule" at position 2 in sentence: "The schedule for the conference is available online."\n
            The commonality is: time-related nouns\n
            \n
            Example 9:\n
            Token: "looking" at position 3 in sentence: "She is looking forward to the event."\n
            Token: "scanned" at position 2 in sentence: "He scanned the document quickly."\n
            Token: "searching" at position 4 in sentence: "They are searching for the lost keys."\n
            The commonality is: verbs related to visual perception\n
            \n
            Example 10:\n
            Token: "to" at position 5 in sentence: "I want to go to the store."\n
            Token: "for" at position 4 in sentence: "This gift is for you."\n
            Token: "with" at position 4 in sentence: "She is going with her friends."\n
            The commonality is: prepositions indicating direction or purpose\n
            \n
            Example 11:\n
            Token: "sad" at position 4 in sentence: "He felt really sad after the call."\n
            Token: "angry" at position 5 in sentence: "They were extremely angry about it."\n
            Token: "negative" at position 4 in sentence: "She had a negative reaction to the news."\n
            The commonality is: negative emotion adjectives\n
            \n
            Now, please analyze the following tokens and their contexts:\n
        '''

        self.instructions_test = '''
            Background:\n
            I will provide a token, its position in the sentence, the surrounding context, and a candidate description of the token's role or type given the context. \n
            Your task is to judge whether the given description accurately characterizes the token in its context. \n
            Please respond with either:\n
            - "Yes" (if the description is accurate), or \n
            - "No" (if it is inaccurate)\n
            Any additional information, explaination, or context are strongly prohibited. Only return "Yes" and "No". \n
            \n
            Example 1:\n
            Token: "running" at position 3 in sentence: "She is running in the park."\n
            Candidate description: "present participle"\n
            Answer: Yes\n
            \n
            Example 2:\n
            Token: "dog" at position 2 in sentence: "The dog barked loudly."\n
            Candidate description: "adjective"\n
            Answer: No\n
            \n
            Example 3:\n
            Token: "quickly" at position 4 in sentence: "He ran quickly toward the exit."\n
            Candidate description: "manner adverb"\n
            Answer: Yes\n
            \n
            Example 4:\n
            Token: "first" at position 4 in sentence: "This is the first time I have seen this."\n
            Candidate description: "ordinal number"\n
            Answer: Yes\n
            \n
            Example 5:\n
            Token: "to" at position 5 in sentence: "I want to go to the store."\n
            Candidate description: "emotional verb"\n
            Answer: No\n
            \n
            Example 6:\n
            Token: "looking" at position 3 in sentence: "She is looking forward to the event."\n
            Candidate description: "verb related to oral communication"\n
            Answer: No\n
            \n
            Example 7:\n
            Token: "for" at position 4 in sentence: "This gift is for you."\n
            Candidate description: "preposition indicating direction or purpose"\n
            Answer: Yes\n
            \n
            Example 8:\n
            Token: "yesterday" at position 4 in sentence: "I went there yesterday."\n
            Candidate description: "future time expression"\n
            Answer: No\n
            \n
            Example 9:\n
            Token: "happy" at position 4 in sentence: "She looks very happy today."\n
            Candidate description: "adjective describing a state of being"\n
            Answer: Yes\n
            \n
            Example 10:\n
            Token: "1998" at position 6 in sentence: "They moved here in 1998."\n
            Candidate description: "past time expression"\n
            Answer: Yes\n
            \n
            Example 11:\n
            Token: "cat" at position 2 in sentence: "The cat chased the mouse."\n
            Candidate description: "noun describing an animal"\n
            Answer: Yes\n
            \n
            Example 12:\n
            Token: "sad" at position 4 in sentence: "He felt really sad after the call."\n
            Candidate description: "positive emotion adjective"\n
            Answer: No\n
            \n
            Example 13:\n
            Token: "scanned" at position 2 in sentence: "He scanned the document quickly."\n
            Candidate description: "verb related to movement"\n
            Answer: No\n
            \n
            Example 14:\n
            Token: "searching" at position 4 in sentence: "They are searching for the lost keys."\n
            Candidate description: "verb related to visual perception"\n
            Answer: Yes\n
            \n
            Example 15:\n
            Token: "time" at position 2 in sentence: "The time of the event is 3 PM."\n
            Candidate description: "noun indicating a specific moment"\n
            Answer: Yes\n
            \n
            Example 16:\n
            Token: "schedule" at position 2 in sentence: "The schedule for the conference is available online."\n
            Candidate description: "noun indicating an animal"\n
            Answer: No\n
            \n
            Example 17:\n
            Token: "last week" at position 5 in sentence: "She arrived last week."\n
            Candidate description: "future time expression"\n
            Answer: No\n
            \n
            Example 18:\n
            Token: "first" at position 4 in sentence: "This is the first time I have seen this."\n
            Candidate description: "ordinal number"\n
            Answer: Yes\n
            \n
            Example 19:\n
            Token: "fighting" at position 3 in sentence: "They are fighting for their rights."\n
            Candidate description: "verb related to conflict"\n
            Answer: Yes\n
            \n
            Example 20:\n
            Token: "to" at position 5 in sentence: "I want to go to the store."\n
            Candidate description: "number"\n
            Answer: No\n
            \n
            Example 21:\n
            Token: "cat" at position 2 in sentence: "The cat chased the mouse."\n
            Candidate description: "noun describing tools or objects"\n
            Answer: No\n
            \n
            Example 22:\n
            Token: "fish" at position 2 in sentence: "The fish swam gracefully in the tank."\n
            Candidate description: "noun describing an animal"\n
            Answer: Yes\n
            \n
            Now, please analyze the following tokens, their positions, contexts, and candidate descriptions:\n
        '''
        
    def add_original_feature(
        self,
        token_str,
        token_location,
        sentence_str,
        activation_scores
    ):
        if len(activation_scores) != self.feature_num:
            raise ValueError("Activation scores length does not match feature number.")
        top_indices = sorted(range(len(activation_scores)), key=lambda i: activation_scores[i], reverse=True)[:self.top_k]
        for index in top_indices:
            self.original_feature_set[index].append({"token": token_str, "location": token_location, "sentence": sentence_str, "score": activation_scores[index]})

    def interpret_single_feature(
        self,
        feature_index
    ):
        if len(self.original_feature_set[feature_index]) < self.interpretation_length + 1:
            print(f"Not enough data to interpret feature {feature_index}.")
            return ("", False)
        samples = self.original_feature_set[feature_index][:self.interpretation_length]
        prompt = self.instructions
        for sample in samples:
            prompt += f"Token: {sample['token']} at position {sample['location']} in sentence: \"{sample['sentence']}\". \n"
        prompt += "The commonality is:"
        # Here you would call the backend model to get the interpretation
        # Example using OpenAI API (assuming backend_model_name is an OpenAI model)

        response = self.client.chat.completions.create(
            model=self.backend_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for interpreting token commonalities."},
                {"role": "user", "content": prompt}
            ],
        )
        token_cost = response.usage
        print(token_cost)
        interpretation = response.choices[0].message.content.strip()
        print(f"Interpretation for feature {feature_index}: {interpretation} from {len(self.original_feature_set[feature_index])} samples.")
        return (interpretation, True)
    
    def test_single_feature( 
        self,
        feature_index
    ):
        if feature_index not in self.feature_interpretation:
            print(f"Feature index {feature_index} not found.")
            return -1
        if not self.feature_interpretation[feature_index]["interpretable"]:
            print(f"Feature {feature_index} is not interpretable.")
            return 0
        interpretation = self.feature_interpretation[feature_index]["semantic"]
        test_samples = self.original_feature_set[feature_index][self.interpretation_length:]
        print(f"Testing feature {feature_index} with interpretation: {interpretation}")
        print(f"Number of test samples: {len(test_samples)}")
        print("example test samples:")
        test_samples = test_samples[:self.max_test_samples]  # Limit to max_test_samples
        for sample in test_samples: 
            print(f"\t\tToken: {sample['token']}, Location: {sample['location']}\"")
        correct_count = 0
        for sample in test_samples:
            prompt = self.instructions_test
            prompt += f"Token: {sample['token']} at position {sample['location']} in sentence: \"{sample['sentence']}\". \n"
            prompt += f"Candidate description: \"{interpretation}\"\nAnswer:"
            response = self.client.chat.completions.create(
                model=self.backend_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for testing token interpretations."},
                    {"role": "user", "content": prompt}
                ],
            )
            answer = response.choices[0].message.content.strip()
            if answer.lower() == "yes":
                correct_count += 1
            token_cost = response.usage
            print(token_cost)
        score = correct_count / len(test_samples) if test_samples else 0.0
        print(f"Test score for feature {feature_index}: {score} ({correct_count}/{len(test_samples)})")
        return score

    def run(self):
        for feature_index in range(self.feature_num):
            interpretation, interpretable = self.interpret_single_feature(feature_index)
            self.feature_interpretation[feature_index]["semantic"] = interpretation
            self.feature_interpretation[feature_index]["interpretable"] = interpretable
        
        for feature_index in range(self.feature_num):
            interpretable = self.feature_interpretation[feature_index]["interpretable"]
            if interpretable:
                test_score = self.test_single_feature(feature_index)
                self.feature_interpretation[feature_index]["test_score"] = test_score
            else:
                self.feature_interpretation[feature_index]["test_score"] = None


class interp_score_embclu():
    def __init__(
        self,
        feature_num,
        backend_model,
        backend_model_type,
        backend_tokenizer,
        front_tokenizer,
        top_k = 10,
        interpretation_length = 10,
        max_test_samples = 8
    ):
        self.feature_num = feature_num
        self.backend_model = backend_model
        self.backend_tokenizer = backend_tokenizer
        self.front_tokenizer = front_tokenizer
        self.backend_model_type = backend_model_type
        self.backend_model.eval()
        self.backend_model.cuda()
        self.top_k = top_k
        self.interpretation_length = interpretation_length
        self.max_test_samples = max_test_samples

        self.original_feature_set = {}
        self.feature_interpretation = {}
        for i in range(self.feature_num):
            self.original_feature_set[i] = []
            self.feature_interpretation[i] = {
                "semantic": "",
                "interpretable": False,
                "test_score": None
            }

    def add_original_feature(
        self,
        token_str,
        token_location,
        sentence_str,
        activation_scores
    ):
        if len(activation_scores) != self.feature_num:
            raise ValueError("Activation scores length does not match feature number.")
        top_indices = sorted(range(len(activation_scores)), key=lambda i: activation_scores[i], reverse=True)[:self.top_k]
        for index in top_indices:
            self.original_feature_set[index].append({"token": token_str, "location": token_location, "sentence": sentence_str, "score": activation_scores[index]})

    def interpret_single_feature(
        self,
        feature_index
    ):
        return
    
    def test_single_feature(
        self,
        feature_index
    ):
        with torch.no_grad():   
            if feature_index not in self.feature_interpretation:
                print(f"Feature index {feature_index} not found.")
                return -1
            total_samples = self.original_feature_set[feature_index]
            if len(total_samples) < self.interpretation_length + 1:
                self.feature_interpretation[feature_index]["interpretable"] = False
                return 0.0
            samples_for_distribution = total_samples[:self.interpretation_length]
            samples_for_test = total_samples[self.interpretation_length:]
            encodings = []
            for sample in samples_for_distribution:
                tokenized_new = self.backend_tokenizer(sample["sentence"], return_tensors="pt")['input_ids'].cuda()
                token_loca = find_tokenized_position_in_another_tokenizer(
                    sample["sentence"],
                    self.front_tokenizer,
                    sample["location"],
                    self.backend_tokenizer
                )
                encoded = self.backend_model(tokenized_new, output_hidden_states=True)['last_hidden_state'][0, token_loca, :].detach().cpu().numpy()
                encodings.append(encoded)
            encodings = np.array(encodings)
            mean_encoding = np.mean(encodings, axis=0)
            covariance_matrix = np.cov(encodings, rowvar=False)
            # If covariance_matrix is singular, use pseudo-inverse
            try:
                cov_rev = np.linalg.inv(covariance_matrix)
            except np.linalg.LinAlgError:
                cov_rev = np.linalg.pinv(covariance_matrix)
            
            p_value = 0
            for sample in samples_for_test:
                tokenized_new = self.backend_tokenizer(sample["sentence"], return_tensors="pt")['input_ids'].cuda()
                token_loca = find_tokenized_position_in_another_tokenizer(
                    sample["sentence"],
                    self.front_tokenizer,
                    sample["location"],
                    self.backend_tokenizer
                )
                encoded = self.backend_model(tokenized_new, output_hidden_states=True)['last_hidden_state'][0, token_loca, :].detach().cpu().numpy()
                diff = encoded - mean_encoding
                mahalanobis_distance = np.dot(np.dot(diff, cov_rev), diff.T)
                sample["mahalanobis_distance"] = mahalanobis_distance
                d = len(mean_encoding)
                chi2_score = chi2.sf(mahalanobis_distance, d)
                p_value += 1 - chi2_score
            p_value /= len(samples_for_test) if samples_for_test else 1
            self.feature_interpretation[feature_index]["interpretable"] = True
            self.feature_interpretation[feature_index]["test_score"] = p_value
            print(f"Test score for feature {feature_index}: {p_value}")
            return p_value
    
    def run(self):   
        for feature_index in range(self.feature_num):
            self.test_single_feature(feature_index)

class feature_extractor_for_BAE():
    def __init__(
        self, 
        LM, 
        tokenizer, 
        sentences,
        trained_BAE,
        layer_num,
    ):
        self.LM = LM
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.trained_BAE = trained_BAE
        self.layer_num = layer_num

        self.encoded_features = []
        self.averaged_features = None
    
    def __len__(self):
        return len(self.encoded_features)

    def encode_sentences(self):
        with torch.no_grad():
            for sentence in tqdm(self.sentences):
                tokenized = self.tokenizer(sentence, return_tensors="pt")
                input_ids = tokenized["input_ids"].cuda()
                inference = self.LM(input_ids, output_hidden_states=True)
                hidden_states = inference.hidden_states
                for token_index in range(hidden_states[self.layer_num].shape[1]):
                    feature_vector = hidden_states[self.layer_num][0, token_index, :].detach()
                    feature_encoded = self.trained_BAE.encode(feature_vector).detach().cpu().numpy()
                    self.encoded_features.append({
                        "token": self.tokenizer.decode(input_ids[0, token_index]),
                        "location": token_index,
                        "sentence": sentence,
                        "encoded_feature": feature_encoded
                    })
                    if self.averaged_features is None:
                        self.averaged_features = feature_encoded
                    else:
                        self.averaged_features += feature_encoded
            self.averaged_features /= len(self.encoded_features)
    
    def get_encoded_features(self, index):
        if index < 0 or index >= len(self.encoded_features):
            raise IndexError("Index out of range.")
        feature_new = self.encoded_features[index]["encoded_feature"].copy()
        feature_new = np.log2(np.abs(feature_new - self.averaged_features) + 1e-6)
        return {
            "token": self.encoded_features[index]["token"],
            "location": self.encoded_features[index]["location"],
            "sentence": self.encoded_features[index]["sentence"],
            "encoded_feature": feature_new
        }
    
class feature_extractor_for_SAE():
    def __init__(
        self, 
        LM, 
        tokenizer, 
        sentences,
        trained_SAE,
        layer_num = 11,
        rescale = False
    ):
        self.LM = LM
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.trained_BAE = trained_SAE
        self.layer_num = layer_num
        self.rescale = rescale

        self.averaged_features = None
        self.std_features = None

        self.encoded_features = []
    
    def __len__(self):
        return len(self.encoded_features)

    def encode_sentences(self):
        with torch.no_grad():
            for sentence in tqdm(self.sentences):
                tokenized = self.tokenizer(sentence, return_tensors="pt")
                input_ids = tokenized["input_ids"].cuda()
                inference = self.LM(input_ids, output_hidden_states=True)
                hidden_states = inference.hidden_states
                for token_index in range(hidden_states[self.layer_num].shape[1]):
                    feature_vector = hidden_states[self.layer_num][0, token_index, :].detach()
                    if feature_vector.dim() == 1:
                        feature_vector = feature_vector.unsqueeze(0)
                    feature_encoded = self.trained_BAE.encode(feature_vector).detach().cpu().numpy()
                    if feature_encoded.shape[0] == 1:
                        feature_encoded = feature_encoded[0]
                    self.encoded_features.append({
                        "token": self.tokenizer.decode(input_ids[0, token_index]),
                        "location": token_index,
                        "sentence": sentence,
                        "encoded_feature": feature_encoded
                    })
            if self.rescale:
                feature_np = np.array([feature["encoded_feature"] for feature in self.encoded_features])
                self.averaged_features = np.mean(feature_np, axis=0)
                self.std_features = np.std(feature_np, axis=0)
    
    def get_encoded_features(self, index):
        if index < 0 or index >= len(self.encoded_features):
            raise IndexError("Index out of range.")
        feature_new = self.encoded_features[index]["encoded_feature"].copy()
        if self.rescale:
            feature_new = (feature_new - self.averaged_features) / (self.std_features + 1e-6)
        return {
            "token": self.encoded_features[index]["token"],
            "location": self.encoded_features[index]["location"],
            "sentence": self.encoded_features[index]["sentence"],
            "encoded_feature": feature_new
        }

class feature_extractor_for_Transcoder():
    def __init__(
        self, 
        LM, 
        tokenizer, 
        sentences,
        trained_SAE,
        layer_num = 11,
        rescale = False
    ):
        self.LM = make_hooked_llama3(LM)
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.trained_BAE = trained_SAE
        self.layer_num = layer_num
        self.rescale = rescale

        self.averaged_features = None
        self.std_features = None

        self.encoded_features = []
    
    def __len__(self):
        return len(self.encoded_features)

    def encode_sentences(self):
        with torch.no_grad():
            for sentence in tqdm(self.sentences):
                tokenized = self.tokenizer(sentence, return_tensors="pt")
                input_ids = tokenized["input_ids"].cuda()
                inference = self.LM(input_ids, output_hidden_states=True)
                hidden_states = inference.hidden_states
                for token_index in range(hidden_states[1][self.layer_num - 1].shape[1]):
                    feature_vector = hidden_states[1][self.layer_num - 1][0, token_index, :].detach()
                    if feature_vector.dim() == 1:
                        feature_vector = feature_vector.unsqueeze(0)
                    feature_encoded = self.trained_BAE.encode(feature_vector).detach().cpu().numpy()
                    if feature_encoded.shape[0] == 1:
                        feature_encoded = feature_encoded[0]
                    self.encoded_features.append({
                        "token": self.tokenizer.decode(input_ids[0, token_index]),
                        "location": token_index,
                        "sentence": sentence,
                        "encoded_feature": feature_encoded
                    })
            if self.rescale:
                feature_np = np.array([feature["encoded_feature"] for feature in self.encoded_features])
                self.averaged_features = np.mean(feature_np, axis=0)
                self.std_features = np.std(feature_np, axis=0)
    
    def get_encoded_features(self, index):
        if index < 0 or index >= len(self.encoded_features):
            raise IndexError("Index out of range.")
        feature_new = self.encoded_features[index]["encoded_feature"].copy()
        if self.rescale:
            feature_new = (feature_new - self.averaged_features) / (self.std_features + 1e-6)
        return {
            "token": self.encoded_features[index]["token"],
            "location": self.encoded_features[index]["location"],
            "sentence": self.encoded_features[index]["sentence"],
            "encoded_feature": feature_new
        }

class feature_extractor_for_random():
    def __init__(
        self, 
        dimensions,
        tokenizer, 
        sentences,
    ):
        self.dimensions = dimensions
        self.tokenizer = tokenizer
        self.sentences = sentences

        self.encoded_features = []
    
    def __len__(self):
        return len(self.encoded_features)

    def encode_sentences(self):
        with torch.no_grad():
            for sentence in tqdm(self.sentences):
                tokenized = self.tokenizer(sentence, return_tensors="pt")
                input_ids = tokenized["input_ids"]
                for token_index in range(input_ids.shape[1]):
                    feature_encoded = torch.randn(self.dimensions).numpy()
                    if feature_encoded.shape[0] == 1:
                        feature_encoded = feature_encoded[0]
                    self.encoded_features.append({
                        "token": self.tokenizer.decode(input_ids[0, token_index]),
                        "location": token_index,
                        "sentence": sentence,
                        "encoded_feature": feature_encoded
                    })
    
    def get_encoded_features(self, index):
        if index < 0 or index >= len(self.encoded_features):
            raise IndexError("Index out of range.")
        feature_new = self.encoded_features[index]["encoded_feature"].copy()
        return {
            "token": self.encoded_features[index]["token"],
            "location": self.encoded_features[index]["location"],
            "sentence": self.encoded_features[index]["sentence"],
            "encoded_feature": feature_new
        }