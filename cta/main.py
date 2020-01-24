import argparse
import torch
# import lib
import numpy as np
import os
import datetime

from loss import *
from model import *
from optimizer import *
from trainer import *
from torch.utils import data
from dataset import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=50, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--dropout_hidden', default=.2, type=float)
parser.add_argument('--kernel_type', default='exp-1', type=str)
parser.add_argument('--context', default=None, type=str)

# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--final_act', default='tanh', type=str)
parser.add_argument('--lr', default=.05, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--eps', default=1e-6, type=float)

parser.add_argument("-seed", type=int, default=7,
					 help="Seed for random initialization")
parser.add_argument("-sigma", type=float, default=None,
					 help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument("--embedding_dim", type=int, default=-1,
					 help="using embedding")
# parse the loss type
parser.add_argument('--loss_type', default='TOP1', type=str)
# parser.add_argument('--loss_type', default='BPR', type=str)
parser.add_argument('--topk', default=5, type=int)
parser.add_argument('--warm_start', default=5, type=int)
# etc
parser.add_argument('--bptt', default=1, type=int)
parser.add_argument('--test_observed', default=1, type=int) ### sequence with length of at least 1
parser.add_argument('--window_size', default=30, type=int)
parser.add_argument('--position_embedding', default=0, type=int)
parser.add_argument('--shared_embedding', default=1, type=int)


parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--time_sort', default=False, type=bool)
parser.add_argument('--model_name', default='GRU4REC', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--train_data', default='train', type=str)
parser.add_argument('--valid_data', default='valid', type=str)
parser.add_argument('--test_data', default='test', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument("--save_model", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_name', default=None, type=str)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def make_checkpoint_dir():
    print("PARAMETER" + "-"*10)
    now = datetime.datetime.now()
    S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    save_dir = os.path.join(args.checkpoint_dir, S)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    args.checkpoint_dir = save_dir

    with open(os.path.join(args.checkpoint_dir, 'parameter.txt'), 'w') as f:
        for attr, value in sorted(args.__dict__.items()):
            print("{}={}".format(attr.upper(), value))
            f.write("{}={}\n".format(attr.upper(), value))

    print("---------" + "-"*10)

def count_parameters(model):
    parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameter_num", parameter_num) 

def main():
    final_act = args.final_act
    BPTT = args.bptt
    time_sort = args.time_sort
    window_size = args.window_size

    train_data = args.data_folder+args.train_data
    valid_data = args.data_folder+args.valid_data
    test_data = args.data_folder+args.test_data

    print("Loading train data from {}".format(train_data))
    print("Loading valid data from {}".format(valid_data))
    print("Loading test data from {}\n".format(test_data))

    data_name = args.data_name

    observed_threshold = args.test_observed
    
    train_data = Dataset(train_data, data_name, observed_threshold, window_size, '_item.pickle', '_time.pickle')
    valid_data = Dataset(valid_data, data_name, observed_threshold, window_size, '_item.pickle', '_time.pickle')
    test_data = Dataset(test_data, data_name, observed_threshold, window_size, '_item.pickle', '_time.pickle')

    if not args.is_eval:
        make_checkpoint_dir()

    input_size = len(train_data.items)+1
    output_size = input_size
    print("input_size", input_size)

    train_data_loader = dataset.DataLoader(train_data, args.batch_size)
    valid_data_loader = dataset.DataLoader(valid_data, args.batch_size)

    if not args.is_eval:
        model = SATT(input_size, args.hidden_size, output_size,
                                                num_layers=args.num_layers,
                                                num_heads=args.num_heads,
                                                use_cuda=args.cuda,
                                                batch_size=args.batch_size,
                                                dropout_input=args.dropout_input,
                                                dropout_hidden=args.dropout_hidden,
                                                embedding_dim=args.embedding_dim,
                                                position_embedding=args.position_embedding,
                                                shared_embedding=args.shared_embedding,
                                                window_size = window_size,
                                                kernel_type = args.kernel_type,
                                                contextualize_opt = args.context
                                                )
        
        count_parameters(model)


        optimizer = Optimizer(model.parameters(), optimizer_type=args.optimizer_type,
                                                  lr=args.lr,
                                                  weight_decay=args.weight_decay,
                                                  momentum=args.momentum,
                                                  eps=args.eps)

        loss_function = LossFunction(loss_type=args.loss_type, use_cuda=args.cuda)

        trainer = Trainer(model, train_data=train_data_loader, eval_data=valid_data_loader,
                                  optim=optimizer,
                                  use_cuda=args.cuda,
                                  loss_func=loss_function,
                                  topk = args.topk,
                                  args=args)

        trainer.train(0, args.n_epochs - 1, args.batch_size, saving=args.save_model)
        

        
    else:
        if args.load_model is not None:
            print("Loading pre trained model from {}".format(args.load_model))
            checkpoint = torch.load(args.load_model)
            model = checkpoint["model"]
            model.gru.flatten_parameters()
            optim = checkpoint["optim"]
            loss_function = LossFunction(loss_type=args.loss_type, use_cuda=args.cuda)
            evaluation = Evaluation(model, loss_function, use_cuda=args.cuda)
            loss, recall, mrr = evaluation.eval(valid_data)
            print("Final result: recall = {:.2f}, mrr = {:.2f}".format(recall, mrr))
        else:
            print("Pre trained model is None!")


if __name__ == '__main__':
    main()
