import time
import argparse
import csv
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from apmeter import APMeter
import os
from Evaluation import print_second_metric
from torch.nn import BCEWithLogitsLoss
# from calflops import calculate_flops
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-gpu', type=str, default='5')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.0001')
parser.add_argument('-epoch', type=str, default=50)
parser.add_argument('-model', type=str, default='TDNet')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-batch_size', type=int, default=5)
parser.add_argument('-num_clips', type=str, default=256)
parser.add_argument('-skip', type=int, default=0)
parser.add_argument('-num_layer', type=str, default='False')
parser.add_argument('-unisize', type=str, default='True')
parser.add_argument('-num_classes', type=int, default=157)
parser.add_argument('-annotation_file', type=str, default='data/charades.json')
parser.add_argument('-fine_weight', type=float, default=0.1)
parser.add_argument('-coarse_weight', type=float, default=0.9)
parser.add_argument('-save_logit_path', type=str, default='./save_logit_path')
parser.add_argument('-step_size', type=int, default=7)
parser.add_argument('-gamma', type=float, default=0.1)

args = parser.parse_args()

# set random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)


# batch_size = int(args.batch_size)
batch_size = args.batch_size
new_loss =  AsymmetricLoss()
# new_loss = FocalLoss2d()

if args.dataset == 'charades':
    from charades_dataloader import Charades as Dataset

    if str(args.unisize) == "True":
        print("uni-size padd all T to",args.num_clips)
        from charades_dataloader import collate_fn_unisize
        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        from charades_dataloader  import mt_collate_fn as collate_fn


def load_data(train_split, val_split, root):
    # Load Data
    print('load data', root)

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, args.num_classes, args.num_clips, args.skip)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root, batch_size, args.num_classes, args.num_clips, args.skip)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    return dataloaders, datasets


def run(models, criterion, num_epochs=50):
    since = time.time()
    Best_val_map = 0.
    for epoch in range(num_epochs):
        since1 = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            _, _ = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch)
            # sched.step(val_loss)
            sched.step(val_loss)
            # Time
            print("epoch", epoch, "Total_Time",time.time()-since, "Epoch_time",time.time()-since1)

            if Best_val_map < val_map:
                Best_val_map = val_map
                save_path = "./save_logit_path/best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": Best_val_map,
                    "args": vars(args)
                }, save_path)

                print(f"[Checkpoint] Best model saved at epoch {epoch}, mAP={Best_val_map:.2f}")
            pickle.dump(prob_val,
                        open('./save_logit_path/' + str(epoch) + 'val_map' + str(val_map) + '.pkl',
                             'wb'), pickle.HIGHEST_PROTOCOL)
            print("logit_saved at:",
                  "./save_logit_path/" + str(epoch) + 'val_map' + str(val_map) + ".pkl")
            print_second_metric("./save_logit_path/" + str(epoch) + 'val_map' + str(val_map) + ".pkl",
                                args.annotation_file, args.num_classes)


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def run_network(model, data, gpu):
    inputs, mask, labels, other, hm = data
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))
    inputs = inputs.squeeze(3).squeeze(3)

    fine_probs, coarse_probs, phase_prob = model(inputs) 
    finall_f = torch.stack([args.fine_weight * fine_probs, args.coarse_weight * coarse_probs])
    finall_f = torch.sum(finall_f, dim=0)
    probs_f = F.sigmoid(finall_f) * mask.unsqueeze(2)

    new_loss = AsymmetricLoss()
    loss_coarse = new_loss(coarse_probs, labels)/ torch.sum(mask)
    loss_fine = new_loss(fine_probs, labels)/ torch.sum(mask)

  
    def build_4phase_gt(labels, mask):
      
        import torch.nn.functional as F
        B, T, C = labels.shape  # B:批次, T:时序, C:动作类别数
        device = labels.device

        global_start_gt = torch.zeros(B, T, device=device)
        global_end_gt = torch.zeros(B, T, device=device)

    
        for c in range(C):
            
            cls_action_mask = labels[..., c].float() * mask 
            cls_diff = cls_action_mask[:, 1:] - cls_action_mask[:, :-1] 
            cls_start = (cls_diff > 0).float()  
            cls_end = (cls_diff < 0).float() 
            # 3. 补0对齐维度，初始首帧/末帧为0
            cls_start = F.pad(cls_start, (1, 0), value=0.0)  
            cls_end = F.pad(cls_end, (0, 1), value=0.0) 

            cls_first_action = (cls_action_mask[:, 0] == 1.0) & (mask[:, 0] == 1.0)
            cls_start[cls_first_action, 0] = 1.0
            cls_last_action = (cls_action_mask[:, -1] == 1.0) & (mask[:, -1] == 1.0)
            cls_end[cls_last_action, -1] = 1.0

            cls_start = cls_start * mask
            cls_end = cls_end * mask
            global_start_gt += cls_start
            global_end_gt += cls_end

        global_start_gt = (global_start_gt > 0).float()  # (B,T)
        global_end_gt = (global_end_gt > 0).float()  # (B,T) 

   
        global_action_mask = (labels.sum(dim=-1) > 0).float() * mask  # (B,T)
        bg_gt = 1 - global_action_mask  # (B,T) 


        interior_gt = global_action_mask - global_start_gt - global_end_gt
        interior_gt = torch.clamp(interior_gt, 0, 1)  # (B,T)

        phase_gt = torch.stack([bg_gt, global_start_gt, interior_gt, global_end_gt], dim=-1)  # (B,T,4)
        return phase_gt

    phase_gt = build_4phase_gt(labels, mask)  # (B,T,4)


    p_bg = phase_prob[..., 0]  
    p_start = phase_prob[..., 1] 
    p_end = phase_prob[..., 3] 

    gt_bg = phase_gt[..., 0]
    gt_start = phase_gt[..., 1]
    gt_end = phase_gt[..., 3]


    loss_bg = F.binary_cross_entropy(p_bg * mask, gt_bg, reduction='sum') / (mask.sum() + 1e-6)

    loss_start = F.binary_cross_entropy(p_start * mask, gt_start, reduction='sum') / (mask.sum() + 1e-6)
    loss_end = F.binary_cross_entropy(p_end * mask, gt_end, reduction='sum') / (mask.sum() + 1e-6)

    loss_phase = loss_bg + loss_start + loss_end 
  
    loss = loss_coarse + args.fine_weight * loss_fine + 0.05 * loss_phase  
    corr = torch.sum(mask)
    tot = torch.sum(mask)
    return finall_f, loss, probs_f, corr / tot

def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data

        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    print('epoch', epoch, 'train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    sampled_apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu)
        if sum(data[1].numpy()[0]) > 25:
            p1, l1 = sampled_25(probs.data.cpu().numpy()[0], data[2].numpy()[0], data[1].numpy()[0])
            sampled_apm.add(p1, l1)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs_1 = mask_probs(probs.data.cpu().numpy()[0], data[1].numpy()[0]).squeeze()

        full_probs[other[0][0]] = probs_1.T

    epoch_loss = tot_loss / num_iter
    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    # sample_val_map = torch.sum(100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    print('epoch', epoch, 'Full-val-map:', val_map)
    # print('epoch', epoch, 'sampled-val-map:', sample_val_map)
    # print(100 * sampled_apm.value())
    apm.reset()
    # sampled_apm.reset()
    return full_probs, epoch_loss, val_map

# # for rgb
if __name__ == '__main__':
    train_split = 'data/charades.json'
    test_split = train_split
    dataloaders, datasets = load_data(train_split, test_split, args.rgb_root)
    print(len(dataloaders['train']))
    print(len(dataloaders['val']))

    if not os.path.exists(args.save_logit_path):
        os.makedirs(args.save_logit_path)
    if args.train:

        if args.model == "TDNet":
            print("TDNet")
            from models.TDNet_Model import TDNet
            num_clips = args.num_clips
            num_classes = args.num_classes
            inter_channels = [512, 512, 512, 512]
            num_block = 3
            # H
            head = 8
            # theta
            mlp_ratio = 8
            # D_0
            in_feat_dim = 768
            # D_v
            final_embedding_dim = 512

            rgb_model = TDNet(inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes, num_clips)
            print("loaded", args.load_model)

        rgb_model.cuda()
        from thop import profile, clever_format
        import torch


        dummy_input = torch.randn(1, 768, 256).cuda()

        flops, params = profile(rgb_model, inputs=(dummy_input,))
        flops, params = clever_format([flops, params], "%.3f")

        print(f"Model: {args.model}")
        print(f"Shape: {dummy_input.shape}")
        print(f"Params: {params}")
        print(f"FLOPs: {flops}")


        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        optimizer = optim.AdamW(rgb_model.parameters(), lr=lr)
        # lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.step_size), gamma=args.gamma)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)
        if not os.path.exists('./save_logit_path'):
            os.makedirs('./save_logit_path')
        run([(rgb_model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))
