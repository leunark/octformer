import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.nn as nn
import torch.optim as optim

def setup(rank, world_size):
    print(f"Rank {rank}: Starting setup.")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    print(f"Rank {rank}: Initializing process group.")
    try:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=torch.distributed.timedelta(seconds=60))
        print(f"Rank {rank}: Process group initialized successfully.")
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group: {e}")
        raise
    
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: CUDA device set to {torch.cuda.current_device()}.")

def cleanup(rank):
    print(f"Rank {rank}: Cleaning up process group.")
    dist.destroy_process_group()
    print(f"Rank {rank}: Process group destroyed.")

def train(rank, world_size):
    print(f"Rank {rank}: Training started.")
    setup(rank, world_size)

    try:
        # Dummy dataset
        print(f"Rank {rank}: Initializing dataset and dataloader.")
        dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
        print(f"Rank {rank}: Dataset and dataloader initialized.")

        # Model
        print(f"Rank {rank}: Initializing model.")
        model = nn.Linear(10, 2).to(rank)
        print(f"Rank {rank}: Model moved to GPU {rank}.")
        print(f"Rank {rank}: Wrapping model with DDP.")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        print(f"Rank {rank}: Model wrapped with DDP successfully.")

        # Loss and optimizer
        print(f"Rank {rank}: Initializing optimizer and loss function.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        print(f"Rank {rank}: Optimizer and loss function initialized.")

        # Training loop
        print(f"Rank {rank}: Starting training loop.")
        for epoch in range(3):
            print(f"Rank {rank}: Epoch {epoch} started.")
            sampler.set_epoch(epoch)  # Ensure proper shuffling
            for batch_idx, (data, target) in enumerate(dataloader):
                print(f"Rank {rank}: Processing batch {batch_idx}.")
                data, target = data.to(rank), target.to(rank)
                print(f"Rank {rank}: Data and target moved to GPU {rank}.")

                print(f"Rank {rank}: Forward pass started.")
                optimizer.zero_grad()
                output = model(data)
                print(f"Rank {rank}: Forward pass completed.")

                print(f"Rank {rank}: Calculating loss.")
                loss = criterion(output, target)
                print(f"Rank {rank}: Loss calculated: {loss.item()}.")

                print(f"Rank {rank}: Backward pass started.")
                loss.backward()
                print(f"Rank {rank}: Backward pass completed.")

                print(f"Rank {rank}: Optimizer step started.")
                optimizer.step()
                print(f"Rank {rank}: Optimizer step completed.")
            print(f"Rank {rank}: Epoch {epoch} completed.")
        print(f"Rank {rank}: Training loop completed.")
    except Exception as e:
        print(f"Rank {rank}: Error during training: {e}")
    finally:
        cleanup(rank)

if __name__ == "__main__":
    world_size = 4  # Adjust to the number of GPUs you have
    print(f"Starting training with {world_size} GPUs.")
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    print(f"Training completed.")