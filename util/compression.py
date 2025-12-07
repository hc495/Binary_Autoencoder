import numpy as np
import pickle
import torch
from tqdm import tqdm as tqdm

class BAE_feature_compression():
    def __init__(
        self, 
        BAE = None
    ):
        self.model = BAE
        self.first_batch_finished = False
        self.average = np.zeros(BAE.hidden_dim)
        self.average_round = np.zeros(BAE.hidden_dim)
        self.encoded = []
        self.cache_feature = []

    def add_feature(self, x):
        self.cache_feature.append(x.detach().cpu())

    def encode(self):
        if self.model is None:
            raise ValueError("BAE model is not initialized. Use load_from_file or initialize the BAE in the __init__.")
        if self.first_batch_finished:
            for x in tqdm(self.cache_feature):
                x = x.to(next(self.model.parameters()).device)
                z = self.model.encode(x).detach().cpu().numpy()
                delta = z - self.average
                saved = []
                for index in range(len(delta)):
                    if abs(delta[index]) > 0.5:
                        saved.append(index)
                self.encoded.append(saved)
        else:
            temp_encoding = []
            for x in tqdm(self.cache_feature):
                x = x.to(next(self.model.parameters()).device)
                z = self.model.encode(x).detach().cpu().numpy()
                temp_encoding.append(z)
            self.average = np.mean(temp_encoding, axis=0)
            self.average_round = np.round(self.average)
            for z in tqdm(temp_encoding):
                delta = z - self.average
                saved = []
                for index in range(len(delta)):
                    if abs(delta[index]) > 0.5:
                        saved.append(index)
                self.encoded.append(saved)
            self.first_batch_finished = True
        self.cache_feature = []

    def decode(self, index):
        encoding = self.average_round.copy()
        for i in self.encoded[index]:
            encoding[i] = 1 - encoding[i]
        return self.model.decode(torch.tensor(encoding, dtype=torch.float32).unsqueeze(0).to(next(self.model.parameters()).device)).detach().cpu().numpy()

    def save_to_file(self, path):
        with open(path + ".pkl", "wb") as f:
            pickle.dump({
                "average": self.average,
                "average_round": self.average_round,
                "encoded": self.encoded
            }, f)
        torch.save(self.model.state_dict(), path + "_bae.pt")

    def load_from_file(self, path):
        self.first_batch_finished = True
        with open(path + ".pkl", "rb") as f:
            data = pickle.load(f)
            self.average = data["average"]
            self.average_round = data["average_round"]
            self.encoded = data["encoded"]
        self.model.load_state_dict(torch.load(path + "_bae.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        self.model.eval()