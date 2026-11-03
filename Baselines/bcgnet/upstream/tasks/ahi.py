import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import pandas as pd
import rich
import torch
import torch.nn as nn
import torchaudio
import typing as t
import numpy as np
from wuji_dl.display import df_to_rich, df_to_wandb
from wuji_dl.factory import create_instance
from wuji_dl.metrics import ClassificationMetrics
from wuji_dl.optim import SingleOptimizerFactory
from data.ahi import AHIBatch, AHINetInput
import pdb
import neurokit2 as nk
from metrics.event_eval import cal_cls_confusion_matrix, cal_batch_cls_segs, cal_cls_AP_matrix, process_cls_for_AP_matrix
import wandb
from wuji import sed_eval

class AHITask(L.LightningModule):
    def __init__(
        self,
        model: t.Dict[str, t.Any],
        criterion: t.Dict[str, t.Any],
        optimizer: t.Dict[str, t.Any],
        lr_scheduler: t.Optional[t.Dict[str, t.Any]] = None,
        bad_case_study: bool = False,
        predict: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = create_instance(nn.Module, model)
        self.criterion = create_instance(nn.Module, criterion)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.predict = predict

    def forward(self, net_input: AHINetInput) -> t.Dict[str, t.Any]:
        #return self.model.forward(net_input["source"])
        #net_input.pop('length')
        #pdb.set_trace()
        return self.model.forward(**net_input)
        #return self.model.forward(net_input["bcg_raw"])

    def configure_optimizers(self) -> t.Any:
        optimizer = create_instance(
            torch.optim.Optimizer, self.optimizer, self.parameters()
        )
        if self.lr_scheduler is None:
            return optimizer
        lr_scheduler = self.lr_scheduler.copy()
        lr_scheduler["scheduler"] = create_instance(
            object, lr_scheduler["scheduler"], optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_fit_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)

    def training_step(
        self, batch: AHIBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        #pdb.set_trace()

        
        net_output = self.forward(batch["net_input"])
        batch['cls_labels'] = batch['ah_event']
        loss = self.criterion(net_output, batch)
        #self.log("train/loss", loss["loss"], prog_bar=True)
        self.log_dict(
            {"train/" + k: v for k, v in loss.items()},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss["loss"]
    
    def on_validation_start(self) -> None:
        self.overall_Metrics_segment_detection = np.zeros((100,3))
        self.overall_Metrics_segment_classification = np.zeros((2,2))

    
    @torch.no_grad()
    def validation_step(
        self, batch: AHIBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        

        
        id_index = [str(idx) for idx in batch["id"]]
        #pdb.set_trace()
        net_output = self.forward(batch["net_input"]) # ['ahi_event']
        batch['cls_labels'] = batch['ah_event']
        loss = self.criterion(net_output, batch)
        
        self.log_dict(
            {"val/" + k: v for k, v in loss.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )
        for i in range(len(id_index)):
            #pdb.set_trace()
            #confusion_matrix = cal_cls_confusion_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output[i].detach().to(torch.float).cpu().numpy())
            confusion_matrix = cal_cls_confusion_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output['ahi_event'][i].detach().to(torch.float).cpu().numpy())
            #AP_matrix = process_cls_for_AP_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output[i].detach().to(torch.float).cpu().numpy())
            AP_matrix = process_cls_for_AP_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output['ahi_event'][i].detach().to(torch.float).cpu().numpy())
            self.overall_Metrics_segment_classification += confusion_matrix
            self.overall_Metrics_segment_detection += AP_matrix

            
    def on_validation_epoch_end(self):
        overall_AP = cal_cls_AP_matrix(self.overall_Metrics_segment_detection)
        TP, FP, FN = self.overall_Metrics_segment_detection[50]
        overall_precision, overall_recall = TP/(TP+FP), TP/(TP+FN)
        overall_f1 = 2*overall_precision*overall_recall/(overall_precision+overall_recall)
        overall_cls_precision, overall_cls_recall = self.overall_Metrics_segment_classification[0,0]/(self.overall_Metrics_segment_classification[0,0]+self.overall_Metrics_segment_classification[1,0]), self.overall_Metrics_segment_classification[0,0]/(self.overall_Metrics_segment_classification[0,0]+self.overall_Metrics_segment_classification[0,1])
        overall_cls_f1 = 2*overall_cls_precision*overall_cls_recall/(overall_cls_precision+overall_cls_recall)
        self.log_dict(
            {"val/precision": overall_precision, "val/recall": overall_recall, "val/f1": overall_f1, "val/AP": overall_AP},
        )
        table = wandb.Table(columns=["id", "precision", "recall", "f1","cls_pre","cls_recall","correct_label_num"])

        table = wandb.Table(columns=["Criterion", "Value"])
        # 添加数据到表格
        table.add_data("precision", overall_precision)
        table.add_data("recall", overall_recall)
        table.add_data("f1", overall_f1)
        table.add_data("AP", overall_AP)
        table.add_data("cls_precision", overall_cls_precision)
        table.add_data("cls_recall", overall_cls_recall)
        table.add_data("cls_f1", overall_cls_f1)
        
        # 记录表格到 W&B
        self.logger.experiment.log({"val/Metrics": table})
    

    @torch.no_grad()
    def test_step(
        self, batch: AHIBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        
      
        
        id_index = [str(idx) for idx in batch["id"]]
        net_output = self.forward(batch["net_input"])
        #pdb.set_trace()
        batch['cls_labels'] = batch['ah_event']

        for i in range(len(id_index)):
            #pdb.set_trace()
            #confusion_matrix = cal_cls_confusion_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output[i].detach().to(torch.float).cpu().numpy())
            confusion_matrix = cal_cls_confusion_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output['ahi_event'][i].detach().to(torch.float).cpu().numpy())
            #AP_matrix = process_cls_for_AP_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output[i].detach().to(torch.float).cpu().numpy())
            AP_matrix = process_cls_for_AP_matrix(batch["cls_labels"][i].detach().cpu().numpy(), net_output['ahi_event'][i].detach().to(torch.float).cpu().numpy())
            self.overall_Metrics_segment_classification += confusion_matrix
            self.overall_Metrics_segment_detection += AP_matrix
            if id_index[i] not in self.id_Metrics_segment_detection:
                self.id_Metrics_segment_classification[id_index[i]] = confusion_matrix
                self.id_Metrics_segment_detection[id_index[i]] = AP_matrix
            else:
                self.id_Metrics_segment_classification[id_index[i]] += confusion_matrix
                self.id_Metrics_segment_detection[id_index[i]] += AP_matrix
        
        
    def on_test_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)
        self.overall_Metrics_segment_detection = np.zeros((100,3))
        self.overall_Metrics_segment_classification = np.zeros((2,2))
        self.id_Metrics_segment_detection = {}
        self.id_Metrics_segment_classification = {}

    @torch.no_grad()
    def on_test_epoch_end(self):
        overall_AP = cal_cls_AP_matrix(self.overall_Metrics_segment_detection)
        TP, FP, FN = self.overall_Metrics_segment_detection[50]
        overall_precision, overall_recall = TP/(TP+FP), TP/(TP+FN)
        overall_f1 = 2*overall_precision*overall_recall/(overall_precision+overall_recall)
        overall_cls_precision, overall_cls_recall = self.overall_Metrics_segment_classification[0,0]/(self.overall_Metrics_segment_classification[0,0]+self.overall_Metrics_segment_classification[1,0]), self.overall_Metrics_segment_classification[0,0]/(self.overall_Metrics_segment_classification[0,0]+self.overall_Metrics_segment_classification[0,1])
        overall_cls_f1 = 2*overall_cls_precision*overall_cls_recall/(overall_cls_precision+overall_cls_recall)
        self.log_dict(
            {"test/precision": overall_precision, "test/recall": overall_recall, "test/f1": overall_f1, "test/AP": overall_AP},
        )
        table = wandb.Table(columns=["id", "AP", "precision", "recall", "f1","cls_pre","cls_recall","correct_label_num"])
        for id in self.id_Metrics_segment_classification:
            AP = cal_cls_AP_matrix(self.id_Metrics_segment_detection[id])
            TP, FP, FN = self.id_Metrics_segment_detection[id][50]
            precision, recall = TP/(TP+FP), TP/(TP+FN)
            f1 = 2*precision*recall/(precision+recall)
            Nref = TP+FN
            cls_pre, cls_recall = self.id_Metrics_segment_classification[id][0,0]/(self.id_Metrics_segment_classification[id][0,0]+self.id_Metrics_segment_classification[id][1,0]), self.id_Metrics_segment_classification[id][0,0]/(self.id_Metrics_segment_classification[id][0,0]+self.id_Metrics_segment_classification[id][0,1])
            table.add_data(id, AP, precision, recall, f1, cls_pre, cls_recall, Nref)
        
        self.logger.experiment.log({"test/id_Metrics": table})
        table = wandb.Table(columns=["Criterion", "Value"])
        # 添加数据到表格
        table.add_data("precision", overall_precision)
        table.add_data("recall", overall_recall)
        table.add_data("f1", overall_f1)
        table.add_data("AP", overall_AP)
        table.add_data("cls_precision", overall_cls_precision)
        table.add_data("cls_recall", overall_cls_recall)
        table.add_data("cls_f1", overall_cls_f1)
        
        # 记录表格到 W&B
        self.logger.experiment.log({"test/Metrics": table})
