from src.dataset.loader import DatasetFolder
from torch.utils.data import DataLoader

def get_dataloader(sets, args, sampler=None, shuffle=True, pin_memory=False):
    if sampler is not None:
        loader = DataLoader(sets, batch_sampler=sampler,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        loader = DataLoader(sets, batch_size=args.batch_size_loader, shuffle=shuffle,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    return loader

def get_dataset(split,scenario):

    sets = DatasetFolder(split,scenario)
    return sets
