import argparse
from sched import scheduler
import sched
from time import time
from tqdm import tqdm
import numpy as np
import torch as th
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from data_loader import Data
from models import CompGCN_ConvE, CompGCN_InteractE, ResE
from GAT_models import Attention_Layer
from utils import in_out_norm
from SAttLE import SAttLE
# from SEGAT import Final_Model
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler
from LGCN import Final_Model
import math
from math import cos, pi
# python test_main.py --score_func conve --opn ccorr --gpu 1 --data FB15k-237
# python test_main.py -d FB15k-237 --gpu 0 --batch-size 2048 --evaluate-every 25 --n-epochs 1500 --lr 0.001 --lr-decay 0.995 --lr-step-decay 2 --n-layers 1 --d-embed 200 --num-head 64 --d-k 32 --d-v 50 --d-model 200 --d-inner 1024 --start-test-at 1000 --save-epochs 1100 --dr-enc 0.4 --dr-pff 0.2 --dr-sdp 0.1 --dr-mha 0.3 --decoder twomult
# predict the tail for (head, rel, -1) or head for (-1, rel, tail)
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段，学习率线性增加
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]

def predict(model, graph, device, data_iter, split="valid", mode="tail"):
    model.eval()
    with th.no_grad():
        results = {}
        train_iter = iter(data_iter["{}_{}".format(split, mode)])

        for step, batch in enumerate(train_iter):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            pred = model(graph, sub, rel)
            # pred = model(triple)
            b_range = th.arange(pred.size()[0], device=device)
            target_pred = pred[b_range, obj]
            pred = th.where(label.bool(), -th.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred

            # compute metrics
            ranks = (
                1
                + th.argsort(
                    th.argsort(pred, dim=1, descending=True),
                    dim=1,
                    descending=False,
                )[b_range, obj]
            )
            ranks = ranks.float()
            results["count"] = th.numel(ranks) + results.get("count", 0.0)
            results["mr"] = th.sum(ranks).item() + results.get("mr", 0.0)
            results["mrr"] = th.sum(1.0 / ranks).item() + results.get(
                "mrr", 0.0
            )
            for k in [1, 3, 10]:
                results["hits@{}".format(k)] = th.numel(
                    ranks[ranks <= (k)]
                ) + results.get("hits@{}".format(k), 0.0)

    return results


# evaluation function, evaluate the head and tail prediction and then combine the results
def evaluate(model, graph, device, data_iter, split="valid"):
    # predict for head and tail
    left_results = predict(model, graph, device, data_iter, split, mode="tail")
    right_results = predict(model, graph, device, data_iter, split, mode="head")
    results = {}
    count = float(left_results["count"])

    # combine the head and tail prediction results
    # Metrics: MRR, MR, and Hit@k
    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round(
        (left_results["mr"] + right_results["mr"]) / (2 * count), 5
    )
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )
    for k in [1, 3, 10]:
        results["left_hits@{}".format(k)] = round(
            left_results["hits@{}".format(k)] / count, 5
        )
        results["right_hits@{}".format(k)] = round(
            right_results["hits@{}".format(k)] / count, 5
        )
        results["hits@{}".format(k)] = round(
            (
                left_results["hits@{}".format(k)]
                + right_results["hits@{}".format(k)]
            )
            / (2 * count),
            5,
        )
    # results["left_mr"] = round(left_results["mr"] / count, 5)
    # results["left_mrr"] = round(left_results["mrr"] / count, 5)

    # results["mr"] = round(
    #     left_results["mr"]/ (count), 5
    # )
    # results["mrr"] = round(
    #     left_results["mrr"]  / (count), 5
    # )
    
    # for k in [1, 3, 10]:
    #     results["left_hits@{}".format(k)] = round(
    #         left_results["hits@{}".format(k)] / count, 5
    #     )
    #     results["hits@{}".format(k)] = round(
    #         (
    #             left_results["hits@{}".format(k)]
    #         )
    #         / (count),
    #         5,
    #     )
    return results


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # construct graph, split in/out edges and prepare train/validation/test data_loader
    data = Data(
        args.dataset, args.label_smoothing, args.num_workers, args.batch_size
    )
    data_iter = data.data_iter  # train/validation/test data_loader
    graph = data.g.to(device)
    num_rel = th.max(graph.edata["etype"]).item() + 1

    # Compute in/out edge norms and store in edata
    graph = in_out_norm(graph)

    # Step 2: Create model =================================================================== #
    # compgcn_model = SAttLE(
    #     num_bases=args.num_bases,
    #     num_rel=num_rel,
    #     num_ent=graph.num_nodes(),
    #     in_dim=args.init_dim,
    #     layer_size=args.layer_size,
    #     comp_fn=args.opn,
    #     batchnorm=True,
    #     dropout=args.dropout,
    #     layer_dropout=args.layer_dropout,
    #     num_filt=args.num_filt,
    #     hid_drop=args.hid_drop,
    #     feat_drop=args.feat_drop,
    #     ker_sz=args.ker_sz,
    #     k_w=args.k_w,
    #     k_h=args.k_h,
    # )
    # model = SAttLE(
    #     graph.num_nodes(),
    #     num_rel,
    #     args.n_layers,
    #     args.d_embed,
    #     args.d_k,
    #     args.d_v,
    #     args.d_model,
    #     args.d_inner,
    #     args.num_heads,
    #     **{'dr_enc': args.dr_enc,
    #        'dr_pff': args.dr_pff,
    #        'dr_sdp': args.dr_sdp,
    #        'dr_mha': args.dr_mha,
    #        'decoder': args.decoder}
    # )
    model = Final_Model(
        graph.num_nodes(),
        num_rel,
        args.n_layers,
        args.d_embed,
        args.d_k,
        args.d_v,
        args.d_model,
        args.d_inner,
        args.num_heads,
        **{'dr_enc': args.dr_enc,
           'dr_pff': args.dr_pff,
           'dr_sdp': args.dr_sdp,
           'dr_mha': args.dr_mha,
           'decoder': args.decoder}       
    )
 
    model = model.to(device)
    # model.eval()
    # model.load_state_dict(th.load("comp_link" + "_" + args.dataset, weights_only=True))
    # valid_results = evaluate(
    #     model, graph, device, data_iter, split="valid"
    # )
    # print(
    #     "Valid MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
    #         valid_results["mrr"],
    #         valid_results["mr"],
    #         valid_results["hits@10"],
    #         valid_results["hits@3"],
    #         valid_results["hits@1"],
    #     )
    # )

    # Step 3: Create training components ===================================================== #
    # loss_fn = th.nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr
    )
    scheduler = StepLR(optimizer, step_size=args.lr_step_decay, gamma=args.lr_decay)
    # scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps=100, total_steps=50000, eta_min=0.0002)
    # Step 4: training epoches =============================================================== #
    best_mrr = 0.0
    kill_cnt = 0
    
    for epoch in range(args.n_epochs):
        # Training and validation using a full graph
        model.train()
        train_loss = []
        t0 = time()
        # 初始化进度条（每个epoch一个进度条）
        
        progress_bar = tqdm(data_iter["train"], desc=f'Epoch {epoch + 1}/{args.n_epochs}', leave=True)
        # 破案了，GNN编码的部分只是对图进行操作，没有分batch，后面解码的部分才做了分batch处理，没绷住
        for step, batch in enumerate(progress_bar):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            # triple  = th.as_tensor(triple)
            # print(triple.shape)
            logits = model(graph, sub, rel)
            # print("Logits shape:", logits.shape)
            
            # compute loss
            tr_loss = model.cal_loss(logits, labels=label)
            train_loss.append(tr_loss.item())

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        if epoch<=args.decay_until:
            scheduler.step()
        
        train_loss = np.sum(train_loss)

        t1 = time()
        val_results = evaluate(
            model, graph, device, data_iter, split="test"
        )
        t2 = time()

        # validate
        if val_results["mrr"] > best_mrr:
            best_mrr = val_results["mrr"]
            th.save(
                model.state_dict(), "comp_link" + "_" + args.dataset
            )
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt > args.kill_cnt:
                print("early stop.")
                break
        print(
            "In epoch {}, Train Loss: {:.4f}, Test MRR: {:.5}, Train time: {}, Test time: {}".format(
                epoch, train_loss, val_results["mrr"], t1 - t0, t2 - t1
            )
        )

    # test use the best model
    model.eval()
    model.load_state_dict(th.load("comp_link" + "_" + args.dataset, weights_only=True))
    test_results = evaluate(
        model, graph, device, data_iter, split="test"
    )
    print(
        "Test MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
            test_results["mrr"],
            test_results["mr"],
            test_results["hits@10"],
            test_results["hits@3"],
            test_results["hits@1"],
        )
    )
    valid_results = evaluate(
        model, graph, device, data_iter, split="valid"
    )
    print(
        "Valid MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
            valid_results["mrr"],
            valid_results["mr"],
            valid_results["hits@10"],
            valid_results["hits@3"],
            valid_results["hits@1"],
        )
    )


if __name__ == "__main__":
    # Step 1: Parse arguments
    parser = argparse.ArgumentParser(description='SAttLE')
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.999,
                        help="learning rate decay rate")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of encoder layers")
    parser.add_argument("--n-epochs", type=int, default=1500,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use: fb15k-237 or wn18rr")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="number of triples to sample at each iteration")
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")
    parser.add_argument("--start-test-at", type=int, default=7000,
                        help="firs epoch to evaluate on test data for each epoch")
    parser.add_argument("--lr-step-decay", type=int, default=5,
                        help="decay lr every x steps")
    parser.add_argument("--save-epochs", type=int, default=1000,
                        help="save per epoch")
    parser.add_argument("--num-heads", type=int, default=64,
                        help="number of attention heads")
    parser.add_argument('--d-k', default=32, type=int,
                        help='Dimension of key')
    parser.add_argument('--d-v', default=50, type=int,
                        help='Dimension of value')
    parser.add_argument('--d-model', default=200, type=int,
                        help='Dimension of model')
    parser.add_argument('--d-embed', default=200, type=int,
                        help='Dimension of embedding')
    parser.add_argument('--d-inner', default=512, type=int,
                        help='Dimension of inner (FFN)')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        help='label smoothing')
    parser.add_argument('--dr-enc', default=0.2, type=float,
                        help='encoder dropout')
    parser.add_argument('--dr-pff', default=0.3, type=float,
                        help='position feedforward dropout')
    parser.add_argument('--dr-sdp', default=0.2, type=float,
                        help='scaled dot product dropout')
    parser.add_argument('--dr-mha', default=0.3, type=float,
                        help='multi-head attention dropout')
    parser.add_argument('--decay-until', default=1050, type=int,
                        help='decay learning rate until')
    parser.add_argument('--decoder', default='twomult', type=str,
                        help='decoder')
    # 2411
    parser.add_argument("--seed", dest="seed", default = 41504, type=int, 
                        help="Seed for randomization")
    parser.add_argument("--num-workers", type=int, default=10, 
                        help="Number of processes to construct batches")
    parser.add_argument("--kill_cnt", default=100, type=int, 
                        help='Number of epochs to wait before early stopping')
    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    print(args)

    # args.layer_size = eval(args.layer_size)
    # args.layer_dropout = eval(args.layer_dropout)

    main(args)
