import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn

from constants import DATA_SETUPS, MODEL_CKPT_DIR
from datasets import get_test_dataset, get_train_dataset, construct_train_dataset
from utils_models import get_model, initialize_checkpoint

def get_subset_train_loader(setup_key, num_dfr_train, batch_size, num_workers, normalize=True):
    train_ds = construct_train_dataset(setup_key=setup_key,
                                       normalize=normalize,
                                       transforms_key='train_transform')
    setup = DATA_SETUPS[setup_key].copy()
    dataset_name = setup['dataset_name']

    # if dataset_name exists in dfr_indices, load the indices
    # otherwise, create the indices and save them
    dfr_indices_dir = os.path.join('dfr_indices', dataset_name)
    if not os.path.exists(dfr_indices_dir):
        os.makedirs(dfr_indices_dir)
    dfr_indices_path = os.path.join(dfr_indices_dir, f"train_{num_dfr_train}samples.npy")
    if os.path.exists(dfr_indices_path):
        train_idx = np.load(dfr_indices_path)
        print(f"Loaded {num_dfr_train} indices from {dfr_indices_path}.")
    else:
        train_idx = np.random.choice(len(train_ds), num_dfr_train, replace=False)
        np.save(dfr_indices_path, train_idx)
        print(f"No indices file. Saved {num_dfr_train} indices to {dfr_indices_path}.")

    train_ds = torch.utils.data.Subset(train_ds, train_idx)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    return train_loader

def evaluate_acc_and_loss(model, loader):
    losses = []
    correct = 0
    total = 0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = criterion(logits, y).item()
            losses.append(loss)
            preds = torch.argmax(logits, axis=1)
            correct_batch = (preds == y)
            correct += correct_batch.sum().item()
            total += correct_batch.shape[0]
    accuracy = correct / total
    avg_loss = np.mean(losses)
    return accuracy, avg_loss

def main():
    parser = argparse.ArgumentParser(description="Tune and evaluate DFR on a checkpoint.")
    parser.add_argument("poison_identifier", type=str, help="Name of the poison data.")
    parser.add_argument("--model_name", type=str, default='resnet18', help="Name of the architecture.")
    parser.add_argument("--epoch", type=int, required=False, default=-1, help="Epoch of the checkpoint.")
    parser.add_argument("--clean_data_name", type=str, default='cifar10', help="Name of the clean data.")
    parser.add_argument("--num_workers", type=int, default=4, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--num_dfr_train", type=int, default=5000, required=False, help="Number of clean training data to use")
    parser.add_argument("--random_init", action="store_true", help="Whether to use random init model.")
    args = parser.parse_args()
    print(args)

    # randomly initialize model or load checkpoint
    ckpt_path = os.path.join(MODEL_CKPT_DIR, args.poison_identifier, args.model_name, f'epoch={args.epoch}.pt')
    if args.random_init:
        model = get_model(args.model_name, setup_key=args.clean_data_name)
        print("Using random init model")
        result_path = os.path.join(os.path.dirname(ckpt_path), f'num_dfr={args.num_dfr_train}', 'random-init')
    else:
        
        model = initialize_checkpoint(args.model_name, 
                                    setup_key=args.clean_data_name,
                                    ckpt_path=ckpt_path)
        print("Loaded model checkpoint", ckpt_path)
        result_path = os.path.join(os.path.dirname(ckpt_path), f'num_dfr={args.num_dfr_train}', 'ckpt-init')

    result_file = os.path.join(result_path, f"{args.epoch}.json")
    # if result file already exists, return
    if os.path.exists(result_file):
        print("Result file already exists. Skipping.")
        return

    # load data 
    train_loader = get_subset_train_loader(setup_key=args.clean_data_name, # Important! Must use clean data!
                                            num_dfr_train=args.num_dfr_train,
                                            batch_size=8*args.batch_size,
                                            num_workers=args.num_workers)

    full_train_loader = get_train_dataset(setup_key=args.clean_data_name, 
                                        batch_size=8*args.batch_size, 
                                        num_workers=args.num_workers, 
                                        normalize=True)
                
    test_loader  = get_test_dataset(setup_key=args.poison_identifier, 
                                    batch_size=8*args.batch_size, 
                                    num_workers=args.num_workers, 
                                    normalize=True)
    
    # create result payload
    result_payload = {}

    # evaluate test acc and loss before DFR
    ckpt_test_acc, ckpt_test_loss = evaluate_acc_and_loss(model, test_loader)
    print("[Before] Test Accuracy:", ckpt_test_acc)
    print("[Before] Test Loss:", ckpt_test_loss)
    result_payload["test_acc"] = ckpt_test_acc
    result_payload["test_loss"] = ckpt_test_loss
    # evaluate train acc and loss before DFR
    ckpt_train_acc, ckpt_train_loss = evaluate_acc_and_loss(model, full_train_loader)
    print("[Before] Train Accuracy:", ckpt_train_acc)
    print("[Before] Train Loss:", ckpt_train_loss)
    result_payload["train_acc"] = ckpt_train_acc
    result_payload["train_loss"] = ckpt_train_loss

    # finetune last layer using LBFGS
    model = lbfgs_finetune(model, train_loader)
    dfr_acc, dfr_loss = evaluate_acc_and_loss(model, test_loader)
    print("[DFR] Test Accuracy:", dfr_acc)
    print("[DFR] Test Loss:", dfr_loss)
    result_payload["dfr_test_acc"] = dfr_acc
    result_payload["dfr_test_loss"] = dfr_loss
    # evaluate train acc and loss after DFR
    dfr_train_acc, dfr_train_loss = evaluate_acc_and_loss(model, full_train_loader)
    print("[DFR] Train Accuracy:", dfr_train_acc)
    print("[DFR] Train Loss:", dfr_train_loss)
    result_payload["dfr_train_acc"] = dfr_train_acc
    result_payload["dfr_train_loss"] = dfr_train_loss

    # save result payload
    os.makedirs(result_path, exist_ok=True)
    with open(result_file, 'w') as f:
        # dump all_results as readable json
        json.dump(result_payload, f, indent=4)
    
def lbfgs_finetune(model, train_loader):
    model.train()
    # no gradient is required for all params except those in last linear layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = True

    # train last linear layer using LBFGS
    print("Training on dataset of size", len(train_loader.dataset))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(model.linear.parameters(), lr=0.5)

    def closure():
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        return loss

    for epoch in range(15):
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.step(closure)

    return model

if __name__ == '__main__':
    main()