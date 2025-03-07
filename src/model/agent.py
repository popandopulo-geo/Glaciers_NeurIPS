import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def ddp_setup(world_size, rank):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_destroy():
    dist.destroy_process_group()

class TrainAgent:
    def __init__(self, device, model, optimizer, criterion, scheduler, logger=None):
        """
        Initializes the training agent.
        
        Args:
            device: The local device (e.g. cuda:0).
            model: The sequence-to-sequence transformer model.
            optimizer: Optimizer.
            criterion: Loss function.
            scheduler: Learning rate scheduler.
            logger: A logger instance supporting dictionary-like keys and a method log_image(tag, image_np).
        """
        self.local_rank = device  # e.g. "cuda:0"
        self.global_rank = int(os.environ.get("SLURM_PROCID", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        if self.world_size > 1:
            process_group = dist.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group).to(self.local_rank)
            self.model = DDP(self.model, find_unused_parameters=True, device_ids=[self.local_rank])
        else:
            self.model = model.to(self.local_rank)
        
        # Initialize environment (only on global_rank 0)
        if self.global_rank == 0:
            self._init_environment(logger)
    
    def train(self, n_epochs, train_loader, valid_loader) -> None:
        """
        Runs training and validation for a specified number of epochs.
        Logs the loss metric and sample images from validation.
        Snapshots are saved based on validation loss.
        """
        for epoch in range(1, n_epochs + 1):
            self.current_epoch = epoch
            
            # ------------------- Training Phase -------------------
            self.stage = 'train'
            self.model.train()
            self._init_records()
            self._run_epoch(train_loader)
            self._reduce_records()
            self._process_records()
            self._log_scalars()
            
            # ------------------- Validation Phase -------------------
            self.stage = 'valid'
            self.model.eval()
            with torch.no_grad():
                self._init_records()
                self._run_epoch(valid_loader)
                self._reduce_records()
                self._process_records()
                self._log_scalars()
                self._log_images(valid_loader)
            
            # Save snapshots on rank 0.
            if self.global_rank == 0:
                self._save_snapshot("LATEST.pth")
                cur_loss = self.records['loss'].item()
                if cur_loss < self.best_loss:
                    self.best_loss = cur_loss
                    self._save_snapshot("BEST.pth")
                    self.logger['metrics/best_loss'] = self.best_loss
                    self.logger['metrics/best_epoch'] = self.current_epoch
            
            print(f"Epoch {epoch}/{n_epochs} | Train Loss: {self.train_loss:.4f} | Val Loss: {self.valid_loss:.4f}")
    
    def _run_epoch(self, loader) -> None:
        """
        Runs one epoch over the provided DataLoader.
        For each batch, stacks the input and target sequences, computes the loss,
        and updates the model (if in training stage).
        """
        for i, batch in enumerate(loader):
            # Expecting batch to be a dictionary with keys:
            # "input": list of input image tensors,
            # "target": list of target image tensors,
            # "input_temporal": list of day differences for input images,
            # "target_temporal": list of day differences for target images.
            inputs = torch.stack(batch["input"], dim=1).to(self.local_rank)    # shape: (B, n1, 1, 128, 128)
            targets = torch.stack(batch["target"], dim=1).to(self.local_rank)   # shape: (B, n2, 1, 128, 128)
            input_temporal = torch.tensor(batch["input_temporal"], dtype=torch.float32).to(self.local_rank)   # (B, n1)
            target_temporal = torch.tensor(batch["target_temporal"], dtype=torch.float32).to(self.local_rank) # (B, n2)
            
            output = self.model(inputs, input_temporal, target_temporal)        # shape: (B, n2, 1, 128, 128)
            loss = self.criterion(output, targets)
            
            if self.stage == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            batch_size = targets.shape[0]
            self._update_records({"loss": loss.item()}, batch_size)
        
        if self.stage == 'train' and self.scheduler is not None:
            self.scheduler.step()
    
    def _init_records(self) -> None:
        """
        Initializes records for accumulating loss.
        """
        self.records = {"loss": torch.tensor(0.0, device=self.local_rank)}
        self.n_samples = torch.tensor(0, device=self.local_rank)
    
    def _update_records(self, record, n_samples) -> None:
        """
        Updates loss records.
        
        Args:
            record: Dictionary with loss.
            n_samples: Batch size.
        """
        self.records["loss"] += record["loss"] * n_samples
        self.n_samples += n_samples
    
    def _reduce_records(self) -> None:
        """
        Reduces the records across processes (for DDP).
        """
        if self.world_size > 1:
            dist.reduce(self.records["loss"], dst=0)
            dist.reduce(self.n_samples, dst=0)
    
    def _process_records(self):
        """
        Processes records on rank 0 to compute average loss.
        Also stores train_loss and valid_loss for printing.
        """
        if self.global_rank == 0:
            avg_loss = self.records["loss"] / self.n_samples
            if self.stage == "train":
                self.train_loss = avg_loss.item()
            elif self.stage == "valid":
                self.valid_loss = avg_loss.item()
            self.records["loss"] = avg_loss
    
    def _log_scalars(self) -> None:
        """
        Logs scalar metrics (loss and learning rate) to the logger.
        """
        if self.global_rank == 0:
            self.logger[f'metrics/{self.stage}/loss'].append(self.records["loss"].item())
            if self.stage == 'valid':
                self.logger['metrics/lr'].append(self.optimizer.param_groups[0]['lr'])
    
    def _log_images(self, loader) -> None:
        """
        Logs sample images from the first batch of the validation loader.
        Logs one input image, its corresponding prediction, and target.
        """
        # Get one sample batch from loader
        sample = next(iter(loader))
        inputs = torch.stack(sample["input"], dim=1).to(self.local_rank)
        targets = torch.stack(sample["target"], dim=1).to(self.local_rank)
        input_temporal = torch.tensor(sample["input_temporal"], dtype=torch.float32).to(self.local_rank)
        target_temporal = torch.tensor(sample["target_temporal"], dtype=torch.float32).to(self.local_rank)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs, input_temporal, target_temporal)
        
        # Select the first sample in the batch and the first image in the sequence.
        input_img = inputs[0, 0, 0]   # (128, 128)
        pred_img = outputs[0, 0, 0]     # (128, 128)
        target_img = targets[0, 0, 0]   # (128, 128)
        
        # Convert tensors to numpy arrays.
        input_np = input_img.cpu().numpy()
        pred_np = pred_img.cpu().numpy()
        target_np = target_img.cpu().numpy()
        
        if hasattr(self.logger, "log_image"):
            self.logger.log_image(f"Epoch_{self.current_epoch}_input", input_np)
            self.logger.log_image(f"Epoch_{self.current_epoch}_pred", pred_np)
            self.logger.log_image(f"Epoch_{self.current_epoch}_target", target_np)
    
    def _save_snapshot(self, snapshot_name):
        """
        Saves a checkpoint snapshot including model, optimizer, scheduler states, and current epoch.
        """
        snapshot_path = os.path.join(self.snapshots_root, snapshot_name)
        snapshot = {
            "PARAMS": self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict(),
            "CURRENT_EPOCH": self.current_epoch,
            "SCHEDULER": self.scheduler.state_dict() if self.scheduler is not None else None,
            "BEST_LOSS": self.best_loss
        }
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {self.current_epoch} | Snapshot saved at {snapshot_path}")
    
    def _init_environment(self, logger):
        """
        Initializes logging and snapshot directories.
        """
        self.logger = logger
        # For example, the logger may provide a unique id:
        self.snapshots_root = os.path.join("exp", self.logger["sys/id"].fetch())
        os.makedirs("exp", exist_ok=True)
        os.makedirs(self.snapshots_root, exist_ok=True)
        self.current_epoch = 1
        self.best_loss = float("inf")
