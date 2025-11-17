import torch
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

class FedProtoServer:
    def __init__(self, num_classes=10):
        self.global_prototypes = {}
        self.num_classes = num_classes

    def aggregate_prototypes(self, client_protos_list):
        merged = defaultdict(list)
        for proto in client_protos_list:
            for cls, vec in proto.items():
                merged[cls].append(vec)
        self.global_prototypes = {cls: torch.stack(vecs).mean(dim=0) for cls, vecs in merged.items()}

    def broadcast(self):
        return self.global_prototypes

    def clear(self):
        pass

    def visualize_prototypes(self, round_num, output_dir="vis"):
        if not self.global_prototypes:
            return
        os.makedirs(output_dir, exist_ok=True)
        labels, vecs = zip(*sorted(self.global_prototypes.items()))
        mat = torch.stack(vecs)
        coords = PCA(n_components=2).fit_transform(mat.numpy())
        plt.figure()
        for i, coord in enumerate(coords):
            plt.scatter(coord[0], coord[1], label=f"Class {labels[i]}")
        plt.title(f"Prototype PCA (Round {round_num})")
        plt.legend()
        plt.savefig(f"{output_dir}/proto_pca_round{round_num}.png")
        plt.close()
