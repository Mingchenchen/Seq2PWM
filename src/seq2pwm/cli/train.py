import os
import argparse
from functools import partial
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
import logging
import random
import numpy as np
import pandas as pd

from seq2pwm.dataset import Seq2PWMDataset, collate_fn
from seq2pwm.embedding_cache import ensure_esm_cache
from seq2pwm.model import PWMGenerator
from seq2pwm.weights import load_weights_strict
from seq2pwm.config import resolve_placeholders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def js_divergence_pytorch(p, q):
    eps = 1e-10
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    
    term1 = torch.where(p != 0, p * torch.log(p / m), torch.tensor(0.0, device=p.device))
    term2 = torch.where(q != 0, q * torch.log(q / m), torch.tensor(0.0, device=q.device))
    
    kl_p_m = torch.sum(term1, dim=-1)
    kl_q_m = torch.sum(term2, dim=-1)
    
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd

class DBDContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, is_dbd):
        device = z.device
        batch_size = z.shape[0]

        # 1. Compute the similarity matrix.
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        exp_sim = torch.exp(sim_matrix)

        # 2. Construct the denominator, excluding self-comparisons.
        mask_self = torch.eye(batch_size, device=device)
        neg_mask = 1.0 - mask_self
        # The denominator contains every other sample, including non-DBD samples.
        denominator = (exp_sim * neg_mask).sum(dim=1, keepdim=True)

        # 3. Compute log probabilities.
        log_prob = sim_matrix - torch.log(denominator + 1e-8)

        # 4. Mark a pair as positive only when both samples are DBDs.
        pos_mask = is_dbd.unsqueeze(1) * is_dbd.unsqueeze(0)
        pos_mask = pos_mask * (1.0 - mask_self) # Exclude self-comparisons.
        
        # A valid anchor has at least one other DBD sample as a positive.
        valid_anchors = (pos_mask.sum(dim=1) > 0)
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Average each valid anchor's log probability across its positives.
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        
        # Compute loss only for valid DBD anchors, pulling DBDs together and separating non-DBDs.
        loss = -mean_log_prob_pos[valid_anchors].mean()
        return loss

class PWMLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.weights = dict(loss_weights)

    def cal_jsd_metrics(self, pred_pwm, true_pwm, valid_mask, batch_size):
        bg_dist = torch.tensor([0.25, 0.25, 0.25, 0.25], device=pred_pwm.device)
        metrics = {'jsd_loss': [], 'bg_reg_loss': []}

        for i in range(batch_size):
            pred_slice = pred_pwm[i]
            true_slice = true_pwm[i]
            mask_b = valid_mask[i].bool()
            
            true_valid = true_slice[mask_b]
            pred_len = pred_slice.shape[0]
            true_len = true_valid.shape[0]

            pred_rc = pred_slice.flip(0)[:, [3, 2, 1, 0]]

            # Global JSD is the best forward/reverse-complement alignment.
            # Background regularization keeps the historical center-near
            # alignment rule, but it is no longer exposed as a separate loss.
            idx = torch.where(mask_b)[0]
            start_true = idx[0].item()
            end_true = idx[-1].item() + 1
            ext_left = max(0, start_true - 3)
            ext_right = min(pred_len - true_len + 1, end_true + 3)

            min_jsd = None
            min_center_jsd = float('inf')
            best_bg_pos = 0
            best_is_rc = False

            for pos in range(pred_len - true_len + 1):
                win_pred = pred_slice[pos:pos+true_len]
                jsd_per_pos_pos = js_divergence_pytorch(true_valid, win_pred)
                avg_pos = jsd_per_pos_pos.mean()

                if min_jsd is None or avg_pos < min_jsd:
                    min_jsd = avg_pos

                if ext_left <= pos < ext_right and avg_pos < min_center_jsd:
                    min_center_jsd = avg_pos
                    best_bg_pos = pos
                    best_is_rc = False

                win_pred_rc = pred_rc[pos:pos+true_len]
                jsd_per_pos_rc = js_divergence_pytorch(true_valid, win_pred_rc)
                avg_rc = jsd_per_pos_rc.mean()

                if avg_rc < min_jsd:
                    min_jsd = avg_rc

                if ext_left <= pos < ext_right and avg_rc < min_center_jsd:
                    min_center_jsd = avg_rc
                    best_bg_pos = pos
                    best_is_rc = True

            # Encourage positions outside the selected motif window to be
            # close to a uniform A/C/G/T background distribution.
            current_pred = pred_rc if best_is_rc else pred_slice
            bg_mask = torch.ones(pred_len, device=pred_pwm.device)
            bg_mask[best_bg_pos : best_bg_pos + true_len] = 0.0
            jsd_bg = js_divergence_pytorch(current_pred, bg_dist.expand_as(current_pred))

            if bg_mask.sum() > 0:
                bg_loss = (jsd_bg * bg_mask).sum() / bg_mask.sum()
            else:
                bg_loss = torch.tensor(0.0, device=pred_pwm.device)

            metrics['bg_reg_loss'].append(bg_loss)
            metrics['jsd_loss'].append(min_jsd)

        return {k: torch.stack(v).mean() for k, v in metrics.items() if len(v) > 0}

    def forward(self, pred_pwm, true_pwm, pwm_lengths):

        pwm_lengths = torch.tensor(pwm_lengths, dtype=torch.int64, device=true_pwm.device)
        batch_size, max_len = true_pwm.shape[:2]

        center_start = (max_len - pwm_lengths) // 2
        length_mask = torch.zeros((batch_size, max_len), device=true_pwm.device, dtype=torch.bool)
        for i in range(batch_size):
            start = center_start[i]
            length_mask[i, start:start + pwm_lengths[i]] = True

        valid_mask = length_mask.float()

        metrics = self.cal_jsd_metrics(pred_pwm, true_pwm, valid_mask, batch_size)

        total_loss = torch.tensor(0.0, device=true_pwm.device)
        for loss_name, weight in self.weights.items():
            total_loss += weight * metrics[loss_name]
        metrics['total_loss'] = total_loss
        return metrics

class Trainer:
    def __init__(self, model, optimizer, criterion, contrastive_criterion, device, rank, loss_weights, aux_loss_weights):
        self.model = model
        self.model_without_ddp = model.module if hasattr(model, 'module') else model
        self.optimizer = optimizer
        self.criterion = criterion
        self.contrastive_criterion = contrastive_criterion
        self.device = device
        self.rank = rank
        self.best_val_loss = float('inf')
        self.scaler = torch.cuda.amp.GradScaler()
        self.results_dir = None
        self.loss_weights = dict(loss_weights)
        self.aux_loss_weights = aux_loss_weights
        self.base_loss_weights = dict(loss_weights)

    def update_loss_weights(self, epoch):
        """Use bg_reg=0.1 for epochs 0-5, then the configured weight."""
        new_weights = dict(self.base_loss_weights)
        if epoch <= 5:
            new_weights['bg_reg_loss'] = 0.1

        self.criterion.weights = new_weights
        self.loss_weights = new_weights

        if self.rank == 0:
            print(f"--- Epoch {epoch}: Active Loss Weights: {new_weights} ---")


    def parse_single_pwm(self, pwm_str: str) -> torch.Tensor:
        rows = pwm_str.strip().split(';')
        pwm = [[float(x) for x in r.split(',')] for r in rows]
        return torch.tensor(pwm, dtype=torch.float32)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        self.model.train()
        
        protein_embeddings = batch['protein_embeddings'].to(self.device)

        if self.rank == 0 and torch.abs(protein_embeddings).sum() < 1e-6:
            print("!!! Warning: Input Embedding is all 0, please check the ESM cache loading !!!")

        asym_ids = batch['asym_ids'].to(self.device)
        protein_masks = batch['protein_masks'].to(self.device)
        true_pwm = batch['pwms'].to(self.device)
        pwm_lengths = batch['pwm_lengths']
        is_dbd = batch['is_dbd'].to(self.device)  # 1=TF, 0=Non-TF

        with torch.cuda.amp.autocast():
            pred_pwm, _, aux_outputs = self.model(
                protein_embeddings,
                asym_ids,
                protein_masks,
            )
            metrics = {}
            total_loss = torch.tensor(0.0, device=self.device)

            # PWM losses apply only to TF/DBD samples.
            dbd_mask = (is_dbd == 1.0)
            if dbd_mask.sum() > 0:
                pred_pwm_dbd = pred_pwm[dbd_mask]
                true_pwm_dbd = true_pwm[dbd_mask]
                pwm_lengths_dbd = [pwm_lengths[i] for i in range(len(pwm_lengths)) if dbd_mask[i]]
                pwm_metrics = self.criterion(
                    pred_pwm_dbd,
                    true_pwm_dbd,
                    pwm_lengths_dbd,
                )

                for loss_name, val in pwm_metrics.items():
                    if loss_name != 'total_loss':
                        metrics[f'pwm_{loss_name}'] = val

                for loss_name, weight in self.loss_weights.items():
                    total_loss += weight * pwm_metrics[loss_name]

            # Auxiliary objectives use the complete batch.
            if self.aux_loss_weights.get('dbd_class_loss', 0.0) > 0.0:
                dbd_class_loss = F.binary_cross_entropy_with_logits(
                    aux_outputs['dbd_logits'].squeeze(), is_dbd
                )
                metrics['dbd_class_loss'] = dbd_class_loss
                total_loss += self.aux_loss_weights['dbd_class_loss'] * dbd_class_loss

            if self.aux_loss_weights.get('contrastive_loss', 0.0) > 0.0:
                contra_loss = self.contrastive_criterion(
                    z=aux_outputs['projection'],
                    is_dbd=is_dbd
                )
                metrics['contrastive_loss'] = contra_loss
                total_loss += self.aux_loss_weights['contrastive_loss'] * contra_loss
            
            metrics['total_loss'] = total_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {k: v.item() for k, v in metrics.items()}

    def save_valid_test_results(self, all_results, epoch, prefix):
        if self.rank != 0:
            return
        folder = os.path.join(self.results_dir, prefix)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{prefix}_epoch_{epoch:03d}.csv")
        pd.DataFrame(all_results).to_csv(path, index=False)

    @torch.no_grad()
    def validate(self, val_loader, epoch_idx, writer=None, prefix='valid'):
        self.model.eval()
        pwm_total_metrics = {}
        aux_total_metrics = {}
        num_dbd_samples = 0
        num_batches = 0
        all_results = []

        for batch in tqdm(val_loader, desc=f'{prefix.capitalize()} Epoch {epoch_idx}', leave=False, disable=self.rank != 0):
            protein_embeddings = batch['protein_embeddings'].to(self.device)
            
            if self.rank == 0 and torch.abs(protein_embeddings).sum() < 1e-6:
                print("!!! Warning: Input Embedding is all 0, please check the ESM cache loading !!!")

            asym_ids = batch['asym_ids'].to(self.device)
            protein_masks = batch['protein_masks'].to(self.device)
            is_dbd = batch['is_dbd'].to(self.device)

            with torch.cuda.amp.autocast():
                pred_pwm, _, aux_outputs = self.model(
                    protein_embeddings,
                    asym_ids,
                    protein_masks,
                )

                num_batches += 1
                if self.aux_loss_weights.get('dbd_class_loss', 0.0) > 0.0:
                    l_class = F.binary_cross_entropy_with_logits(aux_outputs['dbd_logits'].squeeze(), is_dbd)
                    aux_total_metrics['dbd_class_loss'] = aux_total_metrics.get('dbd_class_loss', 0.0) + l_class.item()
                
                if self.aux_loss_weights.get('contrastive_loss', 0.0) > 0.0:
                    l_contra = self.contrastive_criterion(aux_outputs['projection'], is_dbd)
                    aux_total_metrics['contrastive_loss'] = aux_total_metrics.get('contrastive_loss', 0.0) + l_contra.item()
                    
            B = protein_embeddings.size(0)
            for b in range(B):
                if is_dbd[b] == 1.0: # Compute PWM metrics only for DBD samples.
                    pwm_strs = batch['pwm_lists'][b]
                    pwm_lens = batch['pwm_len_lists'][b]
                    best_jsd = float('inf')
                    best_pwm_losses = None
                    
                    for pwm_str, pwm_len in zip(pwm_strs, pwm_lens):
                        true_pwm = self.parse_single_pwm(pwm_str).unsqueeze(0).to(self.device)
                        losses = self.criterion(
                            pred_pwm[b:b+1],
                            true_pwm,
                            [int(pwm_len)],
                        )
                        if losses['jsd_loss'].item() < best_jsd:
                            best_jsd = losses['jsd_loss'].item()
                            best_pwm_losses = {k: v.item() for k, v in losses.items()}

                    if best_pwm_losses:
                        num_dbd_samples += 1
                        for k, v in best_pwm_losses.items():
                            pwm_total_metrics[k] = pwm_total_metrics.get(k, 0.0) + v
                        
                        all_results.append({
                            "epoch": epoch_idx,
                            "name": batch['indices'][b],
                            "jsd": best_jsd
                        })

        # Aggregate and report results.
        final_val_metrics = {}
        # Average PWM metrics over the actual number of DBD samples.
        for k, v in pwm_total_metrics.items():
            final_val_metrics[f'pwm_{k}'] = v / max(1, num_dbd_samples)
        # Average auxiliary losses over the number of batches.
        for k, v in aux_total_metrics.items():
            final_val_metrics[k] = v / max(1, num_batches)

        if self.rank == 0:
            # Write metrics to TensorBoard.
            for k, v in final_val_metrics.items():
                if writer: writer.add_scalar(f'{prefix}/{k}', v, epoch_idx)
            
            # Save CSV results.
            self.save_valid_test_results(all_results, epoch_idx, prefix)
            
            # Print the log summary.
            log_str = ", ".join([f"{k}: {v:.4f}" for k, v in final_val_metrics.items()])
            print(f"{prefix.capitalize()} Epoch {epoch_idx} -> {log_str} (DBD Samples: {num_dbd_samples})")
        
        return final_val_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config file')
    parser.add_argument('--fold', type=int, choices=range(10), help='Ten-fold split index (0-9)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = resolve_placeholders(config, fold=args.fold)

    data_cfg = config['data']
    train_cfg = config['training']
    loss_weights = config['loss_weights']
    aux_loss_weights = config.get('aux_loss_weights', {})
    contra_cfg = config.get('contrastive', {})

    # First-run ESM cache generation can take longer than one hour while the
    # other ranks wait at a barrier.
    timeout = datetime.timedelta(hours=24)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timeout)
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    set_seed(train_cfg['seed'])

    if rank == 0:
        os.makedirs(train_cfg['save_dir'], exist_ok=True)
        writer = SummaryWriter(f"{train_cfg['save_dir']}/train_pwm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logging.basicConfig(level=logging.INFO)
        checkpoints_dir = os.path.join(train_cfg['save_dir'], "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
    else:
        writer, checkpoints_dir = None, None

    results_dir = os.path.join(train_cfg['save_dir'], "results")
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)

    cache_path = data_cfg['esm_cache_path']
    cache_inputs = [
        data_cfg['train_tf_data'],
        data_cfg['valid_data'],
        data_cfg['test_data'],
    ]
    if rank == 0:
        ensure_esm_cache(
            cache_inputs,
            cache_path,
            model_cache=data_cfg.get('esm_model_cache', '.cache/esm'),
            device=device,
        )
    dist.barrier()

    train_dataset = Seq2PWMDataset(
        data_cfg['train_tf_data'], cache_path=cache_path, mode='train'
    )
    valid_dataset = Seq2PWMDataset(
        data_cfg['valid_data'], cache_path=cache_path, mode='val'
    )
    test_dataset = Seq2PWMDataset(
        data_cfg['test_data'], cache_path=cache_path, mode='test'
    )

    eval_collate = partial(collate_fn, mode='val')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=eval_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=eval_collate,
    )

    model = PWMGenerator().to(device)

    initial_weights = train_cfg.get('weights') or train_cfg.get('checkpoint')
    if initial_weights:
        print(f"Loading initial model weights from {initial_weights}")
        load_weights_strict(model, initial_weights, map_location=device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg['learning_rate']),
        weight_decay=float(train_cfg['weight_decay']),
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg['learning_rate']),
        epochs=train_cfg['epochs'],
        steps_per_epoch=1,
        pct_start=0.4,
        div_factor=25.0,
        final_div_factor=1e2
    )

    criterion = PWMLoss(loss_weights=loss_weights).to(device)
    # Initialize the contrastive loss with configuration parameters.
    contrastive_criterion = DBDContrastiveLoss(
        temperature=contra_cfg.get('temperature', 0.1)
    ).to(device)

    # Initialize the trainer with the complete configuration.
    trainer = Trainer(
        model, optimizer, criterion, contrastive_criterion, 
        device, rank, loss_weights, aux_loss_weights
    )
    trainer.results_dir = results_dir

    for epoch in range(train_cfg['epochs']):
        trainer.update_loss_weights(epoch)
        train_sampler.set_epoch(epoch)
        train_metrics = {}
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', disable=rank != 0):
            batch_metrics = trainer.train_step(batch)

            for k, v in batch_metrics.items():
                train_metrics[k] = train_metrics.get(k, 0.0) + v
            num_batches += 1
            
        scheduler.step()
        
        for k in train_metrics:
            train_metrics[k] /= num_batches
            if rank == 0:
                writer.add_scalar(f'train/{k}', train_metrics[k], epoch)
                
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/lr', current_lr, epoch)
            logging.info(f"Epoch {epoch} Train Avg -> " + ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))

        val_metrics = trainer.validate(val_loader, epoch, writer=writer, prefix='valid')
        trainer.validate(test_loader, epoch, writer=writer, prefix='test')

        if rank == 0:
            target_metric = val_metrics.get('pwm_jsd_loss', float('inf'))
                
            if target_metric < trainer.best_val_loss:
                trainer.best_val_loss = target_metric
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, os.path.join(checkpoints_dir, 'best_model_epoch.pt'))
            
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, os.path.join(checkpoints_dir, f'normal_model_epoch_{epoch}.pt'))

        dist.barrier()

    if rank == 0:
        writer.close()

if __name__ == '__main__':
    main()

