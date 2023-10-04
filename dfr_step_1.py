import os
import argparse
import torch
import torch.nn as nn

from datasets import get_train_dataset, get_test_dataset
from utils_models import get_model
from constants import MODEL_CKPT_DIR

def train_model(model, train_loader, test_loader, optim, lr_scheduler, num_epochs=60, ckpt_dir=None, save_every=-1):
    """
    desc: trains model on data from loader, projecting the data to be orthogonal to the
          learned feature vectors of the linear model
    returns: loss list and accuracy list
    """
    loss_fn = nn.CrossEntropyLoss()

    for e in range(num_epochs):
        model.train()
        e_losses = []
        e_accs = []
        for b in train_loader:
            im, label = [x.cuda() for x in b]
            bs = im.size(0)

            optim.zero_grad()
            o = model(im)

            loss = loss_fn(o, label)
            loss.backward()
            optim.step()
            e_losses.append(loss.item())
            
            # Labels can be of of shape (bs, 1) or (bs,)
            if label.ndim == 2:
                e_accs.append((o.argmax(dim=1) == label.argmax(dim=1)).sum().item() / bs)
            else:
                e_accs.append((o.argmax(dim=1) == label).sum().item() / bs)
            
        epoch_loss = sum(e_losses) / len(e_losses)
        print({'loss': epoch_loss, 'epoch': e})

        epoch_acc = sum(e_accs) / len(e_accs)
        print({'train_acc': epoch_acc, 'epoch': e})

        # evaluate on test set
        test_acc = eval_on_loader(model, test_loader)
        print({'test_acc': test_acc, 'epoch': e})

        lr_scheduler.step()

        should_save = (save_every > 0 and (e+1) % save_every == 0) or (save_every == -1 and e == num_epochs-1)
        if ckpt_dir and should_save:
            # filename should include epoch number
            ckpt_path = os.path.join(ckpt_dir, f'epoch={e}.pt')
            torch.save(model.state_dict(), ckpt_path)

def eval_on_loader(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b in loader:
            im, label = [x.cuda() for x in b]
            o = model(im)
            pred = o.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += im.size(0)
    acc = (correct / total)
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('poison_identifier', type=str, default='cifar10',
                        help='Poison identifier (from DATA_SETUPS in constants.py). Determines what poison training data to use.')
    parser.add_argument('--model', type=str, default='resnet18', help='Name of the architecture.')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--scheduler', type=str, default='cosine', help='cosine, step, or multistep')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_normalize', action='store_true', help='disable normalizing data')
    parser.add_argument('--save_every', type=int, default=1, help='save checkpoint every n epochs. -1 to save only last epoch')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='directory to save checkpoints')
    args = parser.parse_args()

    # Get dataset
    normalize = not args.no_normalize
    train_loader = get_train_dataset(args.poison_identifier, args.batch_size, args.num_workers, normalize=normalize)
    test_loader  = get_test_dataset(args.poison_identifier, 8*args.batch_size, args.num_workers, normalize=normalize)

    # Create model, defense, optimizer, and lr scheduler
    model = get_model(args.model, args.poison_identifier)
    ckpt_dir = None
    if args.checkpoint_dir is None:
        # default path is (checkpoint_dir)/(setup_key)/(defense_name)/(model)
        ckpt_dir = os.path.join(MODEL_CKPT_DIR, args.poison_identifier, args.model)
    else: 
        ckpt_dir = os.path.join(args.checkpoint_dir, args.poison_identifier)
    os.makedirs(ckpt_dir, exist_ok=True)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    elif args.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)
    elif args.scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[75, 90, 100], gamma=0.1)
    else:
        raise ValueError(f'Invalid lr scheduler {args.scheduler}')

    train_model(model, train_loader, test_loader, optim, lr_scheduler, num_epochs=args.epochs, ckpt_dir=ckpt_dir, save_every=args.save_every)

if __name__ == '__main__':
    main()