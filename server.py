import torch
import copy
from models import DNN 

class Server:
    def __init__(self, model, clients, device, alpha=1.0, zeroday_class_index=None, zeroday_round=3):
        self.global_model = model
        self.clients = clients
        self.device = device
        self.alpha = alpha
        self.zeroday_class_index = zeroday_class_index
        self.zeroday_round = zeroday_round
        self.zeroday_weights = None
        self.current_round = 0

        self.input_dim = model.net[0].in_features
        self.hidden_dim = model.net[0].out_features
        self.output_dim = model.net[-1].out_features  

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def set_zeroday_weights(self, full_weights):
        self.zeroday_weights = {
            k: v.clone().detach() for k, v in full_weights.items()
            if 'net.4' in k
        }

    def expand_model_to_include_new_class(self):
        print("[Server] Expanding global model to include zeroday class...")

        state_dict = self.global_model.state_dict()
        new_model = DNN(self.input_dim, self.hidden_dim, self.output_dim + 1).to(self.device)

        with torch.no_grad():
            for name, param in new_model.named_parameters():
                if 'net.4' not in name:
                    param.copy_(state_dict[name])

            new_model.net[4].weight[:self.output_dim] = state_dict['net.4.weight']
            new_model.net[4].bias[:self.output_dim] = state_dict['net.4.bias']
            # 나머지 1개 클래스는 랜덤 초기화 그대로

        self.global_model = new_model
        self.output_dim += 1

    def aggregate(self, client_weights):
        new_state_dict = {}

        for key in self.global_model.state_dict().keys():
            if 'net.4.weight' in key or 'net.4.bias' in key:
                stacked = torch.stack([cw[key] for cw in client_weights], dim=0)

                if self.current_round < self.zeroday_round:
                    print(f"[Server] Pre-zeroday aggregation for {key}")
                    avg = torch.mean(stacked, dim=0)
                    new_state_dict[key] = avg

                elif self.current_round == self.zeroday_round and self.output_dim == 5:
                    print(f"[Server] Zeroday-round aggregation for {key}")
                    avg = torch.mean(stacked, dim=0)
                    zeroday_vector = self.zeroday_weights[key][-1:].clone()
                    new_tensor = torch.cat([avg, zeroday_vector], dim=0)
                    new_state_dict[key] = new_tensor

                else:
                    print(f"[Server] Post-zeroday aggregation for {key}")
                    # 평균은 0 ~ (output_dim-2) 까지만 계산
                    avg_all = torch.mean(stacked[:, :-1], dim=0)  # 일반 클래스 평균
                    # 마지막 클래스는 client 0의 값 사용
                    client0_value = client_weights[0][key][-1:].clone()  # 마지막 row
                    # 두 부분 합치기
                    new_tensor = torch.cat([avg_all, client0_value], dim=0)
                    new_state_dict[key] = new_tensor

            else:
                stacked = torch.stack([cw[key] for cw in client_weights], dim=0)
                new_state_dict[key] = torch.mean(stacked, dim=0)

        print("[Server] Final shape check before load_state_dict:")
        for k, v in new_state_dict.items():
            print(f"  - {k}: aggregated shape = {v.shape}, expected = {self.global_model.state_dict()[k].shape}")

        self.global_model.load_state_dict(new_state_dict)


    def evaluate_global(self, test_loader):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.global_model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total
