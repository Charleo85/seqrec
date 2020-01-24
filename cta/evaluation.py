import numpy as np
import torch
import dataset
from metric import *

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20, warm_start=5):
        self.model = model
        self.loss_func = loss_func

        self.warm_start = warm_start
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def eval(self, eval_data, batch_size, debug=False):
        self.model.eval()

        losses = []
        recalls = []
        mrrs = []
        weights = []

        dataloader = eval_data

        eval_iter = 0

        with torch.no_grad():
            total_test_num = []
            for input_x_batch, target_y_batch, input_t_batch, idx_batch, feature_batch in dataloader:
                input_x_batch = input_x_batch.to(self.device)
                target_y_batch = target_y_batch.to(self.device)
                input_t_batch = input_t_batch.to(self.device)
                feature_batch = feature_batch.to(self.device)
                
                warm_start_mask = (idx_batch>=self.warm_start).to(self.device)
                
                logit_batch = self.model(input_x_batch, input_t_batch, feature_batch, target=target_y_batch)

                logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
                loss_batch = self.loss_func(logit_sampled_batch, target_y_batch)

                losses.append(loss_batch.item())

                recall_batch, mrr_batch = evaluate(logit_batch, target_y_batch, warm_start_mask, k=self.topk)

                weights.append( int( warm_start_mask.int().sum() ) )
                recalls.append(recall_batch)
                mrrs.append(mrr_batch)

                total_test_num.append(target_y_batch.view(-1).size(0))

        mean_losses = np.mean(losses)
        mean_recall = np.average(recalls, weights=weights)
        mean_mrr = np.average(mrrs, weights=weights)
#         print(recalls, mrrs, weights)
        return mean_losses, mean_recall, mean_mrr

