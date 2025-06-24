import torch
import torch.nn as nn
import torch.optim as optim
import copy
from models import DNN 

class Client:
    def __init__(self, cid, model, train_loader, device,
                 lr=0.001, epochs=10, beta=1.0, is_zeroday_client=False):
        self.cid = cid
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.lr = lr
        self.epochs = epochs
        self.beta = beta
        self.is_zeroday_client = is_zeroday_client
        self.full_weights = None  # zeroday 포함 전체 weight 저장용

    def train(self):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        if self.is_zeroday_client and self.full_weights is None:
            self.full_weights = copy.deepcopy(self.model.state_dict())

    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0

    def get_weights(self, server_round=None, zeroday_round=None):
        state_dict = copy.deepcopy(self.model.state_dict())

        if self.is_zeroday_client:
            self.full_weights = copy.deepcopy(state_dict)

        if self.is_zeroday_client and server_round is not None and zeroday_round is not None:
            if server_round < zeroday_round:
                print(f"[Client {self.cid}] Slicing weights for pre-zeroday round {server_round}")
                print(f"  BEFORE: weight={state_dict['net.4.weight'].shape}, bias={state_dict['net.4.bias'].shape}")
                state_dict['net.4.weight'] = state_dict['net.4.weight'][:-1]
                state_dict['net.4.bias'] = state_dict['net.4.bias'][:-1]
                print(f"  AFTER : weight={state_dict['net.4.weight'].shape}, bias={state_dict['net.4.bias'].shape}")

        return state_dict

    def set_weights(self, state_dict):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_class_weight(self, class_idx):
        return self.model.net[-1].weight[class_idx].detach().cpu().numpy()

    def get_full_weights(self):
        return self.full_weights if self.full_weights is not None else self.model.state_dict()

    def expand_model_to_include_new_class(self, global_weights):
        """
        서버로부터 확장된 global model weight을 받아 6-class 모델로 확장
        """
        current_output_dim = self.model.net[-1].out_features
        target_output_dim = global_weights["net.4.bias"].shape[0]

        if current_output_dim >= target_output_dim:
            print(f"[Client {self.cid}] Model already includes zeroday class. Skipping expansion.")
            return

        print(f"[Client {self.cid}] Expanding model to include zeroday class.")

        old_state_dict = self.model.state_dict()
        hidden_dim = self.model.net[0].out_features
        input_dim = self.model.net[0].in_features

        new_model = DNN(input_dim, hidden_dim, target_output_dim).to(self.device)
        new_model.load_state_dict(copy.deepcopy(global_weights))
        self.model = new_model
