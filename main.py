import argparse
import torch
from models import DNN
from client import Client
from server import Server
from data_loader import load_data
from utils import (
    set_seed, evaluate_zeroday_accuracy,
    get_result_filepath, save_results_with_metadata,
    analyze_zeroday_predictions,
    print_confusion_matrix  
)
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='/home/nid/data/UNR-IDD/UNR-IDD_train.feather')
    parser.add_argument('--test_path', type=str, default='/home/nid/data/UNR-IDD/UNR-IDD_test.feather')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--zeroday_client', type=int, default=0)
    parser.add_argument('--exclude', nargs='+', default=["blackhole"])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--zeroday_round', type=int, default=10)

    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders_all, train_loaders_excluded, test_loader, input_dim, output_dim, label_map = load_data(
        args.train_path, args.test_path, args.num_clients, args.batch_size, exclude_labels=args.exclude
    )

    zeroday_class_index = output_dim

    print("=== Label Mapping (LabelCode → Label) ===")
    for code, label in sorted(label_map.items()):
        print(f"{code}: {label}")
    print("==========================================")

    clients = []
    for cid in range(args.num_clients):
        if cid == args.zeroday_client:
            train_loader = train_loaders_all[cid]
            model_output_dim = output_dim
        else:
            train_loader = train_loaders_excluded[cid]
            model_output_dim = output_dim - 1

        model = DNN(input_dim=input_dim, hidden_dim=64, output_dim=model_output_dim)
        client = Client(cid, model, train_loader, device,
                        lr=args.lr, epochs=args.epochs, beta=1.0,
                        is_zeroday_client=(cid == args.zeroday_client))
        clients.append(client)

    non_zeroday_client = next(c for c in clients if not c.is_zeroday_client)
    server = Server(model=non_zeroday_client.model, clients=clients, device=device,
                    alpha=args.alpha, zeroday_class_index=zeroday_class_index,
                    zeroday_round=args.zeroday_round)

    results = []

    for rnd in range(args.rounds):
        print(f"\n--- Round {rnd+1} ---")
        server.current_round = rnd + 1

        for client in clients:
            print(f"[Main] Training Client {client.cid}")
            client.train()

        if rnd == 0:
            print("[Main] Setting zeroday weights from Client 0 (full model)...")
            server.set_zeroday_weights(clients[args.zeroday_client].full_weights)

        if rnd + 1 == args.zeroday_round:
            print(f"[Round {rnd+1}] Expanding all models to include the new class...")
            server.expand_model_to_include_new_class()
            for client in clients:
                client.expand_model_to_include_new_class(server.get_global_weights())

        client_weights = [
            client.get_weights(server_round=server.current_round, zeroday_round=args.zeroday_round)
            for client in clients
        ]

        for i, w in enumerate(client_weights):
            print(f"[Main] Client {i} weight net.4.weight shape: {w['net.4.weight'].shape}")

        print("[Main] Calling server.aggregate()...")
        server.aggregate(client_weights)

        global_acc = server.evaluate_global(test_loader)
        print(f"Round {rnd+1} | Global Acc: {global_acc:.4f}")
        results.append({"round": rnd+1, "global_acc": global_acc})

    # 마지막 라운드 후, zero-day 공격이 어떻게 분류되었는지 출력
    analyze_zeroday_predictions(
        model=server.global_model,
        dataloader=test_loader,
        device=device,
        zeroday_class_index=zeroday_class_index - 1,  # zero-day class는 마지막 index
        label_map=label_map
    )

    print_confusion_matrix(
    model=server.global_model,
    dataloader=test_loader,
    device=device,
    label_map=label_map
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"/home/nid/yeon/FL_class_unr/results/FL_results_{timestamp}.csv"
    save_results_with_metadata(results, args, filename)

if __name__ == '__main__':
    main()
