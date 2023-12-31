from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtyping import TensorType

from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.pipeline import BaseRolloutStore


class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id, mixin=False):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.mixin = mixin
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                # Left padding of already left-padded queries
                pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1),
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.values for elem in elems], padding_value=0.0, batch_first=True
                ),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                torch.Tensor([elem.score_train for elem in elems]),
                pad_sequence(
                    [elem.input_ids_mixin.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1) if self.mixin else None,
                pad_sequence(
                    [elem.attention_mask_mixin.flip(0) for elem in elems],
                    padding_value=0,
                    batch_first=True,
                ).flip(1) if self.mixin else None,
                pad_sequence(
                    [elem.token_type_ids_mixin.flip(0) for elem in elems],
                    padding_value=0,
                    batch_first=True,
                ).flip(1) if self.mixin else None,
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)
