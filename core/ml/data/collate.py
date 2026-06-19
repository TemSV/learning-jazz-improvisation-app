import torch
from torch.nn.utils.rnn import pad_sequence

def transformer_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    discrete_list = [item['discrete'] for item in batch]
    continuous_list = [item['continuous'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    lengths = torch.tensor([len(seq) for seq in labels_list], dtype=torch.long)
    max_len = lengths.max().item()
    batch_size = len(batch)

    padded_discrete = pad_sequence(discrete_list, batch_first=True, padding_value=0)
    padded_continuous = pad_sequence(continuous_list, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)

    return {
        "discrete": padded_discrete,
        "continuous": padded_continuous,
        "labels": padded_labels,
        "padding_mask": mask,
        "lengths": lengths
    }
