import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, encoder):
        super(TextEncoder, self).__init__()
        self.encoder = encoder
    
    def forward(self, text):
        outputs = self.encoder.model(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask'],
            output_hidden_states=False,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        return embeddings


class ConvoRouter(nn.Module):
    def __init__(self, encoder, tokenizer, hidden_dim=1024, num_models=2, max_seq_length=1408, device='cuda'):
        super(ConvoRouter, self).__init__()
        self.backbone = TextEncoder(encoder)
        p = ['encoder.model.layers.23', 'encoder.model.layers.22', 'encoder.model.layers.21']
        for name, param in self.backbone.named_parameters():
            if p[0] not in name and p[1] not in name and p[2] not in name:
                param.requires_grad = False
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 1, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_models)
        )
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device

    def forward(self, x, ref):
        outputs = self.backbone(x).float()
        ref_out = self.backbone(ref).float()

        combined = torch.cat([outputs, ref_out], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        context = gate * outputs + (1 - gate) * ref_out

        logits = self.classifier(context)
        return logits
    
    def build_input(self, state):
        dial_id = []
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > self.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        inp = s[:1] + dial_id
        
        seq_length = len(inp)
        attention_mask = [1] * seq_length

        if seq_length < self.max_seq_length:
            pad_len = self.max_seq_length - seq_length
            inp = inp + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        else:
            inp = inp[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        
        return {
            "input_ids": torch.tensor([inp]).long().to(self.device),
            "attention_mask": torch.tensor([attention_mask]).to(self.device),
        }

    def select_action(self, state, next_state=None):
        state = self.build_input(state)
        next_state = self.build_input(next_state)
        outputs = self.backbone(state).float()
        ref_out = self.backbone(next_state).float()

        combined = torch.cat([outputs, ref_out], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        context = gate * outputs + (1 - gate) * ref_out

        logits = self.classifier(context)

        probs = nn.functional.softmax(logits, dim=1)
        action = probs.argmax()
        return action
