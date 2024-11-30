# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from squeeze_call.seqdist import sparse
from squeeze_call.seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from squeeze_call.seqdist.core import SequenceDist, Max, Log, semiring


class CTC(torch.nn.Module):
    """CTC module"""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        use_focal_loss: bool = False,
        blank_id: int = 0,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
            blank_id: blank label.
        """
        super().__init__()
        eprojs = encoder_output_size
        self.odim = odim
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.use_focal_loss = use_focal_loss
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction=reduction_type,
                                         zero_infinity=True)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        no_log_ys_hat = F.softmax(ys_hat, dim=2)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.square(1 - weight)
            loss = torch.multiply(loss, weight)
            loss = loss.mean()
        else:
            # Batch-size average
            loss = loss / ys_hat.size(1)
        ys_hat = ys_hat.transpose(0, 1)
        return loss, ys_hat, no_log_ys_hat.transpose(0, 1)

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


class CRF(torch.nn.Module):

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        use_focal_loss: bool = False,
        blank_id: int = 0,
    ):
        super().__init__()
        self.blank_score = 2.0
        self.odim = 4096
        self.n_base = 4
        self.scale = 5
        self.expand_blanks = True
        self.dropout_rate = dropout_rate
        self.linear = torch.nn.Linear(encoder_output_size, self.odim)
        self.reduce = reduce
        self.use_focal_loss = use_focal_loss
        self.compute_loss = CTC_CRF(5, ["N", "A", "C", "G", "T"])

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        hs_pad = hs_pad.transpose(0, 1)
        scores = self.linear(F.dropout(hs_pad, p=self.dropout_rate))
        scores = torch.nn.Tanh()(scores)
        if self.scale is not None:
            scores = scores * self.scale
        T, N, C = scores.shape
        scores = torch.nn.functional.pad(
            scores.view(T, N, C // self.n_base, self.n_base),
            (1, 0, 0, 0, 0, 0, 0, 0),
            value=self.blank_score
        ).view(T, N, -1)

        if self.use_focal_loss:
            loss = self.compute_loss.ctc_loss(scores.to(torch.float32), ys_pad, ys_lens, reduction=None)
            weight = torch.exp(-loss)
            weight = torch.square(1 - weight)
            loss = torch.multiply(loss, weight)
            loss = loss.mean()
        else:
            loss = self.compute_loss.ctc_loss(scores.to(torch.float32), ys_pad, ys_lens)
        return loss, scores.transpose(0, 1), scores.transpose(0, 1)

    def decode(self, scores):
        scores = self.compute_loss.posteriors(scores.to(torch.float32)) + 1e-8
        tracebacks = self.compute_loss.viterbi(scores.log()).to(torch.int16).T
        return tracebacks

    def ctc_lo(self, hs_pad: torch.Tensor):
        hs_pad = hs_pad.transpose(0, 1)
        scores = self.linear(hs_pad)
        scores = torch.nn.Tanh()(scores)
        if self.scale is not None:
            scores = scores * self.scale
        T, N, C = scores.shape
        return scores

    def to_dict(self, include_weights=False):
        res = {
            'insize': self.linear.in_features,
            'n_base': self.n_base,
            'state_len': self.state_len,
            'bias': self.linear.bias is not None,
            'scale': self.scale,
            'blank_score': self.blank_score,
            'expand_blanks': self.expand_blanks,
        }
        if self.activation is not None:
            res['activation'] = self.activation.name
        if self.permute is not None:
            res['permute'] = self.permute
        if include_weights:
            res['params'] = {
                'W': self.linear.weight, 'b': self.linear.bias
                if self.linear.bias is not None else []
            }
        return res


class CTC_CRF(SequenceDist):
    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat(
            [
                torch.arange(self.n_base ** (self.state_len))[:, None],
                torch.arange(self.n_base ** (self.state_len))
                .repeat_interleave(self.n_base)
                .reshape(self.n_base, -1)
                .T,
            ],
            dim=1,
        ).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base ** (self.state_len)

    def logZ(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return scores - self.logZ(scores)[:, None] / len(scores)

    def forward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.bwd_scores_cupy(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (
            scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None]
        )
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat(
            [
                log_trans_probs[:, :, :, [0]],
                log_trans_probs[:, :, :, 1:]
                .transpose(3, 2)
                .reshape(T, N, -1, self.n_base),
            ],
            dim=-1,
        )
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        # convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = (
            T,
            N,
            *(self.n_base for _ in range(self.state_len)),
            self.n_base + 1,
        )
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(
            scores[..., 0]
            .permute(0, 1, *range(self.state_len + 1, 1, -1))
            .reshape(T, N, -1, 1),
            [0, 2],
        )
        emissions = torch.flip(
            scores[..., 1:]
            .permute(
                0,
                1,
                *range(self.state_len, 1, -1),
                self.state_len + 2,
                self.state_len + 1
            )
            .reshape(T, N, -1, self.n_base),
            [0, 2, 3],
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths

    def path_to_str(self, path):
        alphabet = np.frombuffer("".join(self.alphabet).encode(), dtype="u1")
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i : n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)  # indices 用于在self.idx.flatten()里找kmer的下标
        move_indices = stay_indices[:, 1:] + targets[:, : n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(
        self,
        scores,
        targets,
        target_lengths,
        loss_clip=None,
        reduction=None,
        normalise_scores=True,
    ):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = -(logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == "mean":
            return loss.mean()
        elif reduction in ("none", None):
            return loss
        else:
            raise ValueError("Unknown reduction type {}".format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(
            stay_scores, move_scores, target_lengths + 1 - self.state_len
        )
