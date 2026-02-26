import functools
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from pytorch_lightning import LightningModule
from .base_models import CategoricalModel, OrdinalModel, ConvEmbedding, NoEmbedding

from deepgene.loss import CrossEntropyLoss, KLDivLoss, EMD_squared_loss, FocalLoss


class ReformerLabeler(transformers.ReformerModel, CategoricalModel):
    def __init__(self, embedding: Union[NoEmbedding, nn.Embedding, ConvEmbedding], config, predictor, n_class=32, loss: str = "CrossEntropyLoss",
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__(config=config)
        print(type(embedding))
        print(type(config))
        print(type(predictor))
        self.embedding = embedding
        self.predictor = predictor
        
        # self.loss = functools.partial(KLDivLoss, 32)
        # self.loss = functools.partial(EMD_squared_loss, 32)
        #self.loss = CrossEntropyLoss
        self.n_class = n_class
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # resolve loss from config
        if isinstance(loss, str):
            if loss == "CrossEntropyLoss":
                self.loss1 = CrossEntropyLoss
            elif loss == "KLDivLoss":
                self.loss1 = functools.partial(KLDivLoss, self.n_class)
            elif loss == "EMD_squared_loss":
                self.loss1 = functools.partial(EMD_squared_loss, self.n_class)
            else:
                self.loss1 = CrossEntropyLoss
        else:
            self.loss1 = loss
    
    def forward(self, X):
        # Сохраняем исходную длину последовательности
        original_seq_len = X.shape[1]
        
        # Получаем axial_pos_shape из конфигурации для вычисления требуемой длины
        axial_pos_shape = getattr(self.config, 'axial_pos_shape', [128, 235])
        required_length = axial_pos_shape[0] * axial_pos_shape[1]  # prod(axial_pos_shape)
        
        # Вычисляем необходимое количество padding до требуемой длины
        if original_seq_len < required_length:
            pad_length = required_length - original_seq_len
            # Добавляем padding справа (форма X: (batch_size, seq_len))
            X = F.pad(X, (0, pad_length), mode='constant', value=0.0)
        elif original_seq_len > required_length:
            # Если длина больше требуемой, обрезаем (не должно происходить, но на всякий случай)
            X = X[:, :required_length]
        
        output = self.embedding(X)
        # Создаем attention_mask для указания реальной длины последовательности
        # Форма X: (batch_size, sequence_length, ...)
        batch_size, seq_length = X.shape[0], X.shape[1]
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=output.device)
        
        output = super().forward(
            inputs_embeds=output,
            attention_mask=attention_mask
        )
        output = self.predictor(output[0])
        
        # Обрезаем выходные данные до исходной длины последовательности
        if original_seq_len < required_length:
            output = output[:, :original_seq_len, :]
        
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    @property
    def name(self):
        return "RM"


class ReformerPreTrainerLM(transformers.ReformerForMaskedLM):
    def __init__(self, embedding: Union[NoEmbedding, nn.Embedding, ConvEmbedding], config):
        super(ReformerPreTrainerLM, self).__init__(config=config)
        self.masking = Masking(value=2, prob=0.15)
        self.embedding = embedding
        
        self.loss = FocalLoss(gamma=2, alpha=0.002)
    
    def forward(self, X):
        # Сохраняем исходную длину последовательности
        original_seq_len = X.shape[1]
        
        # Получаем axial_pos_shape из конфигурации для вычисления требуемой длины
        axial_pos_shape = getattr(self.config, 'axial_pos_shape', [128, 235])
        required_length = axial_pos_shape[0] * axial_pos_shape[1]  # prod(axial_pos_shape)
        
        # Вычисляем необходимое количество padding до требуемой длины
        if original_seq_len < required_length:
            pad_length = required_length - original_seq_len
            # Добавляем padding справа (форма X: (batch_size, seq_len))
            X = F.pad(X, (0, pad_length), mode='constant', value=0.0)
        elif original_seq_len > required_length:
            # Если длина больше требуемой, обрезаем (не должно происходить, но на всякий случай)
            X = X[:, :required_length]
        
        output = self.masking(X)
        output = self.embedding(output)
        # Создаем attention_mask для указания реальной длины последовательности
        batch_size, seq_length = X.shape[0], X.shape[1]
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=output.device)
        
        output = super().forward(
            inputs_embeds=output,
            attention_mask=attention_mask
        )
        output = output[0]
        
        # Обрезаем выходные данные до исходной длины последовательности
        if original_seq_len < required_length:
            output = output[:, :original_seq_len, :]
        
        return output
    
    def training_step(self, batch, batch_ix):
        X_batch, _ = batch
        labels = torch.clone(X_batch)
        logits = self.forward(X_batch)
        loss = self.loss(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss}
    
    @property
    def name(self):
        return "RM-pretrained"


class Masking(LightningModule):
    def __init__(self, value, prob=0.15):
        super(Masking, self).__init__()
        self.prob = prob
        self.value = value
    
    def forward(self, X):
        rand = torch.rand(X.shape)
        # where the random array is less than 0.15, we set true
        mask_arr = rand < self.prob
        
        selection = []
        for i in range(X.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(X.shape[0]):
            X[i, selection[i]] = self.value
        
        return X


class ReformerLabelerOrdinal(OrdinalModel):
    def __init__(self, reformer_model, ordinal_head):
        super().__init__()
        self.reformer_model = reformer_model
        self.ordinal_head = ordinal_head
    
    def forward(self, X):
        output = self.reformer_model(X)
        assert len(output) == 1
        output = self.ordinal_head(output)
        return output
    
    @property
    def name(self):
        return "RM-ordinal"
