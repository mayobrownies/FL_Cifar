import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config


def assign_clients_to_gpus(num_clients, gpu_ids):

    client_to_gpu = {}
    for client_id in range(num_clients):
        gpu_id = gpu_ids[client_id % len(gpu_ids)]
        client_to_gpu[client_id] = gpu_id
    return client_to_gpu


def train_client_on_gpu(client, gpu_id, epochs, server_prototypes, round_num):

    # Set the current GPU device
    torch.cuda.set_device(gpu_id)

    # Move client model to the assigned GPU (models are already on GPU from init)
    # Just ensure we're on the correct one
    client.model = client.model.to(f'cuda:{gpu_id}')

    # Train the client
    client.train(epochs=epochs, server_prototypes=server_prototypes, round_num=round_num)

    return client


def parallel_train_clients(clients, epochs, server_prototypes, round_num):

    if not config.USE_MULTI_GPU or len(config.GPU_IDS) <= 1:
        # Fallback to sequential training
        for client in clients:
            client.train(epochs=epochs, server_prototypes=server_prototypes, round_num=round_num)
        return clients

    # Assign clients to GPUs
    client_to_gpu = assign_clients_to_gpus(len(clients), config.GPU_IDS)

    print(f"\nMulti-GPU Training:")
    for client_id, gpu_id in client_to_gpu.items():
        print(f"  Client {client_id} → GPU {gpu_id}")
    print()

    # Train clients in parallel using ThreadPoolExecutor
    # Note: Using threads instead of processes because PyTorch CUDA contexts
    # work better with threads, and we're just dispatching GPU work
    trained_clients = [None] * len(clients)

    with ThreadPoolExecutor(max_workers=len(config.GPU_IDS)) as executor:
        # Submit training jobs
        future_to_client_id = {}
        for client_id, client in enumerate(clients):
            gpu_id = client_to_gpu[client_id]
            future = executor.submit(
                train_client_on_gpu,
                client,
                gpu_id,
                epochs,
                server_prototypes,
                round_num
            )
            future_to_client_id[future] = client_id

        # Collect results as they complete
        for future in as_completed(future_to_client_id):
            client_id = future_to_client_id[future]
            try:
                trained_client = future.result()
                trained_clients[client_id] = trained_client
            except Exception as exc:
                print(f'Client {client_id} generated an exception: {exc}')
                raise exc

    # Move all models back to primary GPU for evaluation
    primary_gpu = f'cuda:{config.GPU_IDS[0]}'
    for client in trained_clients:
        client.model = client.model.to(primary_gpu)

    return trained_clients


def fedmd_parallel_train_clients(clients, logits_matching_epochs, private_training_epochs,
                                  alignment_data, consensus_logits, y_alignment, round_num):

    if not config.USE_MULTI_GPU or len(config.GPU_IDS) <= 1:
        # Fallback to sequential training
        for client in clients:
            client.train(
                logits_matching_epochs=logits_matching_epochs,
                private_training_epochs=private_training_epochs,
                alignment_data=alignment_data,
                consensus_logits=consensus_logits,
                y_alignment=y_alignment,
                round_num=round_num
            )
        return clients

    # Assign clients to GPUs
    client_to_gpu = assign_clients_to_gpus(len(clients), config.GPU_IDS)

    print(f"\nMulti-GPU Training (FedMD):")
    for client_id, gpu_id in client_to_gpu.items():
        print(f"  Client {client_id} → GPU {gpu_id}")
    print()

    def train_fedmd_client_on_gpu(client, gpu_id):
        """Helper function to train FedMD client on specific GPU."""
        torch.cuda.set_device(gpu_id)
        client.model = client.model.to(f'cuda:{gpu_id}')

        client.train(
            logits_matching_epochs=logits_matching_epochs,
            private_training_epochs=private_training_epochs,
            alignment_data=alignment_data,
            consensus_logits=consensus_logits,
            y_alignment=y_alignment,
            round_num=round_num
        )
        return client

    trained_clients = [None] * len(clients)

    with ThreadPoolExecutor(max_workers=len(config.GPU_IDS)) as executor:
        future_to_client_id = {}
        for client_id, client in enumerate(clients):
            gpu_id = client_to_gpu[client_id]
            future = executor.submit(train_fedmd_client_on_gpu, client, gpu_id)
            future_to_client_id[future] = client_id

        for future in as_completed(future_to_client_id):
            client_id = future_to_client_id[future]
            try:
                trained_client = future.result()
                trained_clients[client_id] = trained_client
            except Exception as exc:
                print(f'Client {client_id} generated an exception: {exc}')
                raise exc

    # Move all models back to primary GPU for evaluation
    primary_gpu = f'cuda:{config.GPU_IDS[0]}'
    for client in trained_clients:
        client.model = client.model.to(primary_gpu)

    return trained_clients
