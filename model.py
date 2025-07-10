import torch.nn as nn
from transformers import AutoModel


class KoELECTRABinaryClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(KoELECTRABinaryClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, 1)  # 이진 분류

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0]  # 첫 번째 토큰 [CLS]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits.squeeze(-1)  # (B,) → BCEWithLogitsLoss 사용 가능
    