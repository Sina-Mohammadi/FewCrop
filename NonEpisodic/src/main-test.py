
import os
import sys

    
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from visdom_logger import VisdomLogger
from src.utils import warp_tqdm, save_checkpoint, load_cfg_from_cfg_file, merge_cfg_from_list, Logger, get_log_file
from src.trainer import Trainer
from src.eval import Evaluator
from src.optim import get_optimizer, get_scheduler
from src.models import Conv4
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config', type=str, required=True, help='Method config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.base_config is not None
    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    callback = None if args.visdom_port is None else VisdomLogger(port=args.visdom_port)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    torch.cuda.set_device(0)
    print(args.log_path)
    log_file = get_log_file(log_path=args.log_path,method=args.method)
    logger = Logger(__name__, log_file)

    # create model
    net=Conv4.Conv4(num_classes=args.num_classes)
    model = torch.nn.DataParallel(net).cuda()


    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = get_optimizer(args=args, model=model)
    
    

    start_epoch = 0
    best_prec1 = -1

    cudnn.benchmark = True


    trainer = Trainer(device=device, args=args)
    scheduler = get_scheduler(optimizer=optimizer,
                              num_batches=len(trainer.train_loader),
                             epochs=args.epochs,
                              args=args)
    tqdm_loop = warp_tqdm(list(range(start_epoch, args.epochs)),
                          disable_tqdm=False)

    # Final evaluation on test set
    net=Conv4.Conv4(num_classes=args.num_classes)
    model = torch.nn.DataParallel(net).cuda()

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    evaluator = Evaluator(device=device, args=args, log_file=log_file)

    results = evaluator.run_full_evaluation(model=model)
    return results

if __name__ == "__main__":
    main()
