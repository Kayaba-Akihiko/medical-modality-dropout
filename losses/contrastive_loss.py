import torch.nn as nn
import torch
from typing import Literal, Optional, Union

class ContrastiveLoss(nn.Module):
    def __init__(
            self,
            criterion: Literal['bce', 'ce'] = 'ce',
    ):
        super().__init__()
        self.criterion = criterion

        log_scale = torch.log(torch.tensor(10, dtype=torch.float32))
        self.log_scale = nn.Parameter(log_scale)
        if self.criterion == 'bce':
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
            self.bias = nn.Parameter(torch.tensor(-10, dtype=torch.float32))
        elif self.criterion == 'ce':
            self.loss = nn.CrossEntropyLoss(reduction='none')
            self.register_buffer('bias', torch.tensor(0, dtype=torch.float32))
        else:
            raise NotImplementedError

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            label=None,
            memory_key: Optional[torch.Tensor] = None,
            memory_label: Optional[torch.Tensor] = None,
    )-> torch.Tensor:
        # query: (B, C), key: (B, D), labels: (B, )
        B, C = query.shape
        device = query.device
        dtype = query.dtype
        factory_kwargs = {'device': device, 'dtype': dtype}

        with torch.no_grad():
            if label is not None:
                label = label.view(-1, 1).contiguous()

            pair_label = torch.arange(B, device=device, dtype=torch.long)
            if label is None:
                positive_mask = torch.eye(B, **factory_kwargs)
            else:
                positive_mask = torch.eq(
                    label, label.T).to(dtype)

        # logits = torch.matmul(query, key.T)
        logits = torch.einsum('nc,kc->nk', query, key)
        if memory_key is not None:
            memory_size = len(memory_key)
            memory_key = memory_key.detach()
            memory_logits = torch.einsum('nc,kc->nk', query, memory_key)
            logits = torch.cat([logits, memory_logits], dim=1)
            del memory_logits

            with torch.no_grad():
                if memory_label is not None:
                    memory_positive_mask = torch.eq(
                        label, memory_label[None, ...].contiguous(),
                    ).to(dtype)
                else:
                    memory_positive_mask = torch.zeros(
                        B, memory_size, **factory_kwargs)
                positive_mask = torch.cat(
                    [positive_mask, memory_positive_mask], dim=1)
                del memory_positive_mask

        logits = torch.mul(
            logits, torch.exp(self.log_scale.to(device))) + self.bias.to(device)

        if self.criterion == 'ce':
            loss = self.loss(logits, pair_label)
            mask_pos_pairs = positive_mask.sum(1)
            loss = torch.div(loss, mask_pos_pairs)
        else:
            loss = self.loss(logits, positive_mask)
        return loss.mean()



class MemoryBank(nn.Module):
    def __init__(
            self, memory_size: int, dim: int, store_labels: bool = False
    ) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        self.store_labels = store_labels
        self.register_buffer(
            'feats', torch.randn(memory_size, dim))
        self.register_buffer(
            'ptr', torch.tensor(0, dtype=torch.long))
        if self.store_labels:
            self.register_buffer(
                'labels',
                -torch.arange(memory_size, dtype=torch.long) - 1,
            )

    @torch.no_grad()
    def forward(
            self,
            x: Optional[torch.Tensor] = None,
            label: Optional[torch.Tensor] = None,
    ):

        memory_feats = self.feats.detach()
        memory_labels = None

        if x is None:
            if self.store_labels:
                return memory_feats, self.labels.detach()
            return memory_feats

        B, C = x.shape

        ptr_start = self.ptr
        ptr_end = (ptr_start + B).clip_(max=self.memory_size)
        self.feats[ptr_start: ptr_end] = x[:ptr_end - ptr_start].detach()

        if self.store_labels:
            assert label is not None
            memory_labels = self.labels.detach()
            self.labels[ptr_start: ptr_end] = label[:ptr_end - ptr_start]

        self.ptr = ptr_end % self.memory_size

        if memory_labels is None:
            return memory_feats
        return memory_feats, memory_labels



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 512
    memory_size = 2048
    contrast_dim = 128

    bank = MemoryBank(
        memory_size, contrast_dim, store_labels=True).to(device)

    query = torch.randn(batch_size, contrast_dim, device=device)
    key = torch.randn(batch_size, contrast_dim, device=device)
    label = torch.randint(
        0, 10, size=(batch_size,), device=device)

    query = nn.functional.normalize(query, dim=1)
    key = nn.functional.normalize(key, dim=1)

    calc_loss = ContrastiveLoss(criterion='ce').to(device)
    loss = calc_loss(query, key)
    print('qk_ce', loss)

    memory_feats, memory_labels = bank(query, label)
    loss = calc_loss(query, key, memory_key=memory_feats)
    print('qk_ce_memory', loss)

    loss = calc_loss(query, key, label)
    print('qk_ce_label', loss)

    memory_feats, memory_labels = bank(query, label)
    loss = calc_loss(query, key, label, memory_feats, memory_labels)
    print('qk_ce_label_memory', loss)

    calc_loss = ContrastiveLoss(criterion='bce').to(device)
    loss = calc_loss(query, key)
    print('qk_bce', loss)

    memory_feats, memory_labels = bank(query, label)
    loss = calc_loss(query, key, memory_key=memory_feats)
    print('qk_bce_memory', loss)

    loss = calc_loss(query, key, label)
    print('qk_bce_label', loss)

    memory_feats, memory_labels = bank(query, label)
    loss = calc_loss(query, key, label, memory_feats, memory_labels)
    print('qk_bce_label_memory', loss)