import tables
import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import time
import torch.backends.cudnn as cudnn
import random
from functools import partial
import sys
sys.path.append("..")
import torch.nn as nn

from models.standard import __dict__ as standard_dict
from models.meta import __dict__ as meta_dict
from dataset.utils import Split
from methods import __dict__ as all_methods
from losses import __dict__ as all_losses
from utils import load_cfg_from_cfg_file, merge_cfg_from_list, AverageMeter, \
                   save_checkpoint, get_model_dir, make_episode_visualization

import warnings
warnings.filterwarnings("ignore")
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config', type=str, default=True, help='Base config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg





def main(args):
    import numpy as np
    if args.seeds:
        args.seed = args.seeds[0]
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

    # ============ Device ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_model_dir(args, args.seed)

    # ============ Training method ================
    method = all_methods[args.method](args=args)
    print(f"=> Using {args.method} method")

    # ============ Data loaders =========
    from dataset.loader import DatasetFolder
    from torch.utils.data import DataLoader



    from dataset import Tasks_Generator, CategoriesSampler, get_dataset, get_dataloader


    

    train_set = get_dataset(split='train',scenario=args.scenario)
    sampler = CategoriesSampler(label=train_set.labels, n_batch=args.batch_size, n_cls=args.num_ways, s_shot=args.num_support, q_shot=args.num_support,balanced='balanced', alpha=2)

    loader = DataLoader(train_set, sampler=sampler,
                        num_workers=0, pin_memory=False)

    task_generator = Tasks_Generator(n_ways=args.num_ways, shot=args.num_support, loader=loader,
                                 train_mean=0, log_file=1)    
    

    
    
    val_set = get_dataset(split='val',scenario=args.scenario)
    sampler_val = CategoriesSampler(label=val_set.labels, n_batch=args.batch_size, n_cls=args.num_ways, s_shot=args.num_support, q_shot=args.num_support, balanced='balanced', alpha=2)

    loader_val = DataLoader(val_set, sampler=sampler_val,
                        num_workers=0, pin_memory=False)

    task_generator_val = Tasks_Generator(n_ways=args.num_ways, shot=args.num_support, loader=loader_val,
                                 train_mean=0, log_file=1)      
    
    
    num_classes=args.num_ways


    if not args.episodic_training:
        loss_fn = all_losses[args.loss](args=args, num_classes=num_classes, reduction='none')



    # ============ Model and optim ================
    if 'MAML' in args.method:
        print(f"Meta {args.arch} loaded")
        model = meta_dict[args.arch](num_classes=num_classes)
    else:
        print(f"Standard {args.arch} loaded")
        model = standard_dict[args.arch](num_classes=num_classes)
    #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.train_iter, eta_min=1e-9)

    # ============ Prepare metrics ================
    metrics: Dict[str, torch.tensor] = {"train_loss": torch.zeros(int(args.train_iter / args.train_freq)).type(torch.float32),
                                        "train_acc": torch.zeros(int(args.train_iter / args.train_freq)).type(torch.float32),
                                        "val_acc": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        "val_loss": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        "test_acc": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        "test_loss": torch.zeros(int(args.train_iter / args.val_freq)).type(torch.float32),
                                        }
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    best_val_acc = 0.

    # ============ Training loop ============
    model.train()
    tqdm_bar = tqdm(range(args.train_iter), total=args.train_iter, ascii=True)
    i = 0
    for i in tqdm(range(args.train_iter)):

        if i >= args.train_iter:
            break

        # ============ Make a training iteration ============
        t0 = time.time()
        if args.episodic_training:
            
            tasks = task_generator.generate_tasks()

            support_labels, target = tasks['y_s'][...,0], tasks['y_q'][...,0]
            support, query = tasks['x_s'], tasks['x_q']
            
            
            
            
            
            
            
            support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
            query, target = query.to(device), target.to(device, non_blocking=True)

            loss, preds_q = method(x_s=support,
                                   x_q=query,
                                   y_s=support_labels,
                                   y_q=target,
                                   model=model)  # [batch, q_shot]
        else:
            (input_, target) = data
            input_, target = input_.to(device), target.to(device, non_blocking=True).long()
            loss = loss_fn(input_, target, model)

            model.eval()
            with torch.no_grad():
                preds_q = model(input_).softmax(-1).argmax(-1)
            model.train()

        # Perform optim
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Log metrics
        train_loss.update(loss.mean().detach(), i == 0)
        train_acc.update((preds_q == target).float().mean(), i == 0)
        batch_time.update(time.time() - t0, i == 0)

        if i % args.train_freq == 0:
            tqdm_bar.set_description(
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                        'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                                batch_time=batch_time,
                                loss=train_loss,
                                acc=train_acc))
            for k in metrics:
                if 'train' in k:
                    metrics[k][int(i / args.train_freq)] = eval(k).avg

        # ============ Evaluation ============
        if i % args.val_freq == 0:
            evaluate(i, task_generator_val, model, method, model_dir, metrics, best_val_acc, device)



        i += 1
        print(i)

def test(i, loader, model, method, model_dir, metrics, best_val_acc, device):
    print('Starting testing ...')
    model.eval()
    method.eval()
    tqdm_test_bar = tqdm(loader, total=args.val_iter, ascii=True)
    test_acc = 0.
    test_loss = 0.
    for j, data in enumerate(tqdm_test_bar):
        support, query, support_labels, query_labels = data
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        loss, preds_q = method(x_s=support,
                                    x_q=query,
                                    y_s=support_labels,
                                    y_q=query_labels,
                                    model=model)
        if args.visu and j == 0:
            task_id = 0
            root = os.path.join(model_dir, 'visu', 'test')
            os.makedirs(root, exist_ok=True)
            save_path = os.path.join(root, f'{i}.png')
            make_episode_visualization(
                       args,
                       support[task_id].cpu().numpy(),
                       query[task_id].cpu().numpy(),
                       support_labels[task_id].cpu().numpy(),
                       query_labels[task_id].cpu().numpy(),
                       preds_q[task_id].cpu().numpy(),
                       save_path)
        test_acc += (preds_q == query_labels).float().mean()
        tqdm_test_bar.set_description(
            f'Test Prec@1 {test_acc/(j+1):.3f})')
        if loss is not None:
            test_loss += loss.detach().mean()
        if j >= args.val_iter:
            break
    test_acc /= args.val_iter
    test_loss /= args.val_iter

    print(f'Iteration: [{i}/{args.train_iter}] \t Test Prec@1 {test_acc:.3f} ')

    for k in metrics:
        if 'test' in k:
            metrics[k][int(i / args.val_freq)] = eval(k)

    model.train()
    method.train()


def evaluate(i, loader, model, method, model_dir, metrics, best_val_acc, device):
    print('Starting validation ...')
    model.eval()
    method.eval()

    tqdm_eval_bar = tqdm(range(args.val_iter), total=args.val_iter, ascii=True)
    val_acc = 0.
    val_loss = 0.
    for j, data in enumerate(tqdm_eval_bar):
        tasks = loader.generate_tasks()

        support_labels, query_labels = tasks['y_s'][...,0], tasks['y_q'][...,0]
        support, query = tasks['x_s'], tasks['x_q']
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
        loss, preds_q = method(x_s=support,
                                    x_q=query,
                                    y_s=support_labels,
                                    y_q=query_labels,
                                    model=model)

        if args.visu and j == 0:
            task_id = 0
            root = os.path.join(model_dir, 'visu', 'valid')
            os.makedirs(root, exist_ok=True)
            save_path = os.path.join(root, f'{i}.png')
            make_episode_visualization(
                       args,
                       support[task_id].cpu().numpy(),
                       query[task_id].cpu().numpy(),
                       support_labels[task_id].cpu().numpy(),
                       query_labels[task_id].cpu().numpy(),
                       preds_q[task_id].cpu().numpy(),
                       save_path)
        val_acc += (preds_q == query_labels).float().mean()
        tqdm_eval_bar.set_description(
            f'Val Prec@1 {val_acc/(j+1):.3f})')
        if loss is not None:
            val_loss += loss.detach().mean()
        if j >= args.val_iter:
            break
    val_acc /= args.val_iter
    val_loss /= args.val_iter

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(state={'iter': i,
                               'arch': args.arch,
                               'state_dict': model.state_dict(),
                               'best_acc': best_val_acc},
                        folder=model_dir)
    print(f'Iteration: [{i}/{args.train_iter}] \t Val Prec@1 {val_acc:.3f} ({best_val_acc:.3f})\t')

    for k in metrics:
        if 'val' in k:
            metrics[k][int(i / args.val_freq)] = eval(k)

    for k, e in metrics.items():
        metrics_path = os.path.join(model_dir, f"{k}.npy")
        np.save(metrics_path, e.detach().cpu().numpy())

    model.train()
    method.train()
    return best_val_acc


if __name__ == '__main__':
    args = parse_args()
    main(args)
