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
        agg_protos_label = dict()
        for idx in client_protos_list:
            local_protos = client_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for label, proto_list in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = torch.stack(proto_list).mean(dim=0)
                self.global_prototypes[label] = proto
            else:
                self.global_prototypes[label] = proto_list[0]

    def broadcast(self):
        return self.global_prototypes

    def clear(self):
        pass

    def visualize_prototypes(self, round_num, output_dir="vis"):
        if not self.global_prototypes:
            return
        os.makedirs(output_dir, exist_ok=True)
        labels, vecs = zip(*sorted(self.global_prototypes.items()))
        mat = torch.stack(list(vecs))

        if mat.is_cuda:
            mat = mat.cpu()

        coords = PCA(n_components=2).fit_transform(mat.detach().numpy())
        plt.figure()
        for i, coord in enumerate(coords):
            plt.scatter(coord[0], coord[1], label=f"Class {labels[i]}")
        plt.title(f"Prototype PCA (Round {round_num})")
        plt.legend()
        plt.savefig(f"{output_dir}/proto_pca_round{round_num}.png")
        plt.close()
