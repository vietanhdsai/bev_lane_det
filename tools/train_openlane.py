import sys
sys.path.append('/home/vietanh/Documents/LaneLine Detection/bev_lane_det')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import time


class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, train=True):
        res = self.model(inputs)
        pred, emb, offset_y, z = res[0]
        pred_2d, emb_2d = res[1]
        if train:
            ## 3d
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
            loss_emb = self.poopoo(emb, gt_instance)
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
            loss_z = self.mse_loss(gt_seg * z, gt_z)
            loss_total = 3 * loss_seg + 0.5 * loss_emb
            loss_total = loss_total.unsqueeze(0)
            loss_offset = 60 * loss_offset.unsqueeze(0)
            loss_z = 30 * loss_z.unsqueeze(0)
            ## 2d
            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
            loss_total_2d = loss_total_2d.unsqueeze(0)
            return pred, loss_total, loss_total_2d, loss_offset, loss_z
        else:
            return pred


def train_epoch(model, dataset, optimizer, configs, epoch, loss_back_bev_hist, loss_offset_hist, loss_z_hist, f1_bev_seg_hist):
    # Last iter as mean loss of whole epoch
    model.train()
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance) in enumerate(
            dataset):
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda()
        gt_seg_data = gt_seg_data.cuda()
        gt_emb_data = gt_emb_data.cuda()
        offset_y_data = offset_y_data.cuda()
        z_data = z_data.cuda()
        image_gt_segment = image_gt_segment.cuda()
        image_gt_instance = image_gt_instance.cuda()
        prediction, loss_total_bev, loss_total_2d, loss_offset, loss_z = model(input_data,
                                                                                gt_seg_data,
                                                                                gt_emb_data,
                                                                                offset_y_data, z_data,
                                                                                image_gt_segment,
                                                                                image_gt_instance)
        loss_back_bev = loss_total_bev.mean()
        loss_back_2d = loss_total_2d.mean()
        loss_offset = loss_offset.mean()
        loss_z = loss_z.mean()
        loss_back_total = loss_back_bev + 0.5 * loss_back_2d + loss_offset + loss_z
        ''' caclute loss '''

        optimizer.zero_grad()
        loss_back_total.backward()
        optimizer.step()
        # if idx % 50 == 0:
        #     print(idx, loss_back_bev.item(), '*' * 10)
        if idx % 3000 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            loss_iter = {"BEV Loss": loss_back_bev.item(), 'offset loss': loss_offset.item(), 'z loss': loss_z.item(),
                            "F1_BEV_seg": f1_bev_seg}
            # losses_show = loss_iter
            loss_back_bev_hist.append(loss_back_bev.item())
            loss_offset_hist.append(loss_offset.item())
            loss_z_hist.append(loss_z.item())
            f1_bev_seg_hist.append(f1_bev_seg)
            print(idx, loss_iter)


def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is'+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)

    ''' models and optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)
    if Dataset is None:
        Dataset = configs.training_dataset
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean) 
    #         return
    loss_back_bev_hist = []
    loss_offset_hist = []
    loss_z_hist = []
    f1_bev_seg_hist = []


    for epoch in range(20, 50):#configs.epochs):
        start_time = time.time()
        print('*' * 100, epoch)
        train_epoch(model, train_loader, optimizer, configs, epoch, loss_back_bev_hist, loss_offset_hist, loss_z_hist, f1_bev_seg_hist)
        scheduler.step()
        if (epoch+1) % 10 == 0:
            save_model_dp(model, optimizer, configs.model_save_path, 'ep%03d.pth' % epoch)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch} - Time: {epoch_time:.2f} seconds")
    save_model_dp(model, None, configs.model_save_path, 'latest.pth')

    # 1. loss_back_bev
    plt.figure()
    plt.plot(loss_back_bev_hist, label="loss_back_bev")
    plt.xlabel("x3000 batches")
    plt.ylabel("Loss")
    plt.title("Loss Back BEV")
    plt.legend()
    plt.savefig("loss_back_bev.png")
    plt.close()

    # 2. loss_offset
    plt.figure()
    plt.plot(loss_offset_hist, label="loss_offset", color="orange")
    plt.xlabel("x3000 batches")
    plt.ylabel("Loss")
    plt.title("Loss Offset")
    plt.legend()
    plt.savefig("loss_offset.png")
    plt.close()

    # 3. loss_z
    plt.figure()
    plt.plot(loss_z_hist, label="loss_z", color="red")
    plt.xlabel("x3000 batches")
    plt.ylabel("Loss")
    plt.title("Loss Z")
    plt.legend()
    plt.savefig("loss_z.png")
    plt.close()

    # 4. f1_bev_seg
    plt.figure()
    plt.plot(f1_bev_seg_hist, label="f1_bev_seg", color="green")
    plt.xlabel("x3000 batches")
    plt.ylabel("F1-score")
    plt.title("F1 Score BEV Seg")
    plt.legend()
    plt.savefig("f1_bev_seg.png")
    plt.close()


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('./openlane_config.py', gpu_id=[0], checkpoint_path='/home/vietanh/Documents/LaneLine Detection/openlane/openlane/ep019.pth')
