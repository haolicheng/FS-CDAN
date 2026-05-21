from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class CrimeTensorDataset(Dataset):
    def __init__(
        self,
        xc,
        xp,
        xt,
        y,
        aux: Optional[torch.Tensor] = None,
    ):
        self.xc = torch.as_tensor(xc, dtype=torch.float32)
        self.xp = torch.as_tensor(xp, dtype=torch.float32)
        self.xt = torch.as_tensor(xt, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.aux = None if aux is None else torch.as_tensor(aux, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        aux = None if self.aux is None else self.aux[idx]
        return self.xc[idx], self.xp[idx], self.xt[idx], self.y[idx], aux


def collate_crime_batch(batch):
    xc, xp, xt, y, aux = zip(*batch)
    aux_batch = None if aux[0] is None else torch.stack(aux)
    return (
        torch.stack(xc),
        torch.stack(xp),
        torch.stack(xt),
        torch.stack(y),
        aux_batch,
    )


def make_loader(dataset: CrimeTensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_crime_batch,
    )


def split_few_shot(dataset: CrimeTensorDataset, ratio: float, seed: int = 42) -> CrimeTensorDataset:
    n = len(dataset)
    k = max(1, int(n * ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:k]
    aux = None if dataset.aux is None else dataset.aux[indices]
    return CrimeTensorDataset(
        dataset.xc[indices],
        dataset.xp[indices],
        dataset.xt[indices],
        dataset.y[indices],
        aux,
    )
