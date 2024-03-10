import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchattacks
import pdb
from pathlib import Path
import argparse
from pytorch_lightning import LightningModule, Trainer, seed_everything
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR,MultiStepLR
from torchmetrics import Accuracy
from utils_data import get_dataset,NORMALIZERS,Stage
import sys
sys.path.insert(0, './libnas/lib')
from config_utils import dict2config
from models import CellStructure,get_cell_based_tiny_net

    
def create_model(num_classes,archidx,data):
    if archidx == '-1':
        # model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        # model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # model.maxpool = nn.Identity()
        from resnet import ResNet18 
        model_kwargs = dict(num_classes=num_classes,pool_adapt=0)
        model = ResNet18(**model_kwargs)
            
    else:
        meta_path = Path('./libnas/output/NAS-BENCH-201-4/meta-node-4.pth') 
        meta_info = torch.load(meta_path)
        all_archs = meta_info['archs']
        arch = all_archs[archidx]
        arch_config = {'channel': 16, 'num_cells': 5}
        model = get_cell_based_tiny_net(dict2config({'name': 'infer.tiny',
                                                'C': arch_config['channel'], 'N': arch_config['num_cells'],
                                                'genotype': CellStructure.str2structure(arch), 
                                                'num_classes':num_classes }, None))
    normalizer = torchvision.transforms.Normalize(*NORMALIZERS[data])
    model = nn.Sequential(normalizer, model).cuda()
    return model

class NASModel(LightningModule):
    def __init__(self, lrschedule,lr,num_classes,len_train_loader,archidx,data,epochs,nas101 = False):
        super().__init__()

        self.save_hyperparameters()
        if not nas101:
            self.model = create_model(num_classes,archidx,data)
        self.num_classes = num_classes
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).cuda()
        self.len_train_loader = len_train_loader
        
        self.attackname_test = [None,'pgd','fgsm']
        self.eps_test = [3/255,8/255]
        self.obfuscated_test = False
        self.corrupt = None
        
    def forward(self, x):
        return self.model(x)

    def loss(self, logits, y, reduction="mean"):
        loss = F.cross_entropy(logits, y, reduction=reduction)
        return loss
    
    def _attack_train(self, batch,attackname):
        if attackname != 'pgd':
            raise NotImplementedError
        img, label = batch
        self.eval()
        with torch.enable_grad():
            attack = torchattacks.PGD(self.model, eps=8/255, alpha=2/255, steps=7, random_start=True)
            adv_img = attack(img, label)
            self.train()
            return adv_img, label

    def _attack_test(self, batch,attack,eps_test=None):
        img, label = batch
        self.eval()
        with torch.enable_grad():
            if attack == 'pgd':
                step = 20
                attack = torchattacks.PGD(self.model, eps=eps_test, alpha=2.5*eps_test/step, steps=step, random_start=True)
            elif attack == 'fgsm':
                attack = torchattacks.FGSM(self.model, eps = eps_test)
            elif attack == 'autoattack':
                attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=eps_test, version='standard', n_classes=self.num_classes)
            # elif attack == 'autoattack_apgdce':
            #     # attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=eps_test, version='apgd-ce', n_classes=self.num_classes) #changed standard
            #     step = 10 # TODO 20
            #     attack = torchattacks.APGD(self.model, norm='Linf', eps=eps_test, steps=step, loss='ce')
            elif attack == 'square':
                attack = torchattacks.Square(self.model, eps=eps_test)
            elif attack == 'apgd':
                step = 20
                attack = torchattacks.APGD(self.model, norm='Linf', eps=eps_test, steps=step, loss='ce')
            # elif attack == 'bpda':
            #     step = 20
            #     from advertorch.attacks import LinfPGDAttack
            #     attack = LinfPGDAttack(
            #         self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps_test,nb_iter=step, eps_iter=2.5*eps_test/step, rand_init=False, clip_min=0.0, clip_max=1.0,targeted=False)
            # elif attack == 'apgdt':
            #     attack = torchattacks.APGDT(self.model, eps=eps_test, norm='Linf',n_classes=self.num_classes, n_restarts=1)
            # elif attack == 'fab':
            #     step = 20
            #     attack = torchattacks.FAB(self.model, norm='Linf', steps=step, eps=eps_test, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, n_classes=10)
            else:
                raise NotImplementedError
            adv_img = attack(img, label)
            return adv_img, label
    
    def _get_loss_and_acc_from_batch(self,batch):
        x, y = batch
        logits = self(x)
        y_hat =  torch.argmax(logits, dim=1)
        loss = self.loss(logits, y, reduction="mean").mean()
        acc = self.accuracy(y_hat, y)
        return acc, loss
    
    def _compute_loss(self, batch, stage,attackname,eps_test=None): 
        if stage == Stage.train:
            batch = self._attack_train(batch,attackname) #train with pgd attack
            acc, loss = self._get_loss_and_acc_from_batch(batch)
            self.log(f'{stage.name}_acc',acc,on_step=False,on_epoch=True, prog_bar=True)
        else:
            if attackname == None:
                # clean acc
                acc, loss = self._get_loss_and_acc_from_batch(batch)
                if self.corrupt != None:
                    self.log(f'{stage.name, self.corrupt}_acc',acc,on_step=False,on_epoch=True, prog_bar=False)
                else:
                    self.log(f'{stage.name}_acc',acc,on_step=False,on_epoch=True, prog_bar=True)
            else:
                #robust acc
                batch = self._attack_test(batch,attackname,eps_test) 
                acc, loss = self._get_loss_and_acc_from_batch(batch)   
                self.log(f'{stage.name}_{attackname}_{eps_test*255}_acc',acc,on_step=False,on_epoch=True, prog_bar=True)        
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch,stage=Stage.train,attackname='pgd')
        return loss

    def myeval(self,batch,stage):
        # self.evaluate(batch, "val")
        
        if self.obfuscated_test:
            self._compute_loss(batch,stage=stage,attackname=None)
            self._compute_loss(batch,stage=stage,attackname='fgsm', eps_test = 3/255)
            self._compute_loss(batch,stage=stage,attackname='square',eps_test = 3/255)    
            for eps_test in [3/255,8/255,16/255, 32/255, 64/255, 128/255, 255/255]:
                self._compute_loss(batch,stage=stage,attackname='pgd', eps_test = eps_test)  

        else:
            for attackname in self.attackname_test: #,'apgd','autoattack' square
                for eps_test in self.eps_test:
                    self._compute_loss(batch, 
                                    stage=stage,
                                    attackname=attackname,
                                    eps_test = eps_test)
                 
    def validation_step(self, batch, batch_idx):
        self.myeval(batch,Stage.val)
    
    def test_step(self, batch, batch_idx):
        self.myeval(batch,Stage.test)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        
        if self.hparams.lrschedule == 'cycle':
            steps_per_epoch = self.len_train_loader
            scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=steps_per_epoch,
                ),
                "interval": "step",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        
        elif self.hparams.lrschedule == 'step':
            milestones = [int(i*self.hparams.epochs) for i in [0.5, 0.75]]
            scheduler_dict = {
                'scheduler':MultiStepLR(
                    optimizer, 
                    milestones,
                    gamma=0.1),
                'interval':'epoch',
                'frequency': 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
            
        elif self.hparams.lrschedule == 'constant':
            return {"optimizer": optimizer}      
        else:
            raise NotImplementedError

def main(args):
    print(args.data)
    #init set up'
    output_dir = './output/seed%s_%s_epoch%s_%s%s/%s/'%(args.seed,args.data,args.epochs,args.lrschedule,args.cyclelr,args.archidx)
    os.makedirs(output_dir, exist_ok = True)
    seed_everything(args.seed)
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project='lightningfinal',name= output_dir)
    csv_logger = CSVLogger(save_dir=output_dir,name = 'log')
                       
    train_data,valid_data, num_classes = get_dataset(args.data,'./dataset/')
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=256, shuffle=True, pin_memory=True, num_workers=4) 
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
  
    #model
    model = NASModel(lrschedule = args.lrschedule,
                     lr=args.cyclelr if args.lrschedule == 'cycle' else 0.1,
                     num_classes=num_classes,
                     len_train_loader = len(train_queue),
                     archidx = args.archidx,
                     data = args.data,
                     epochs = args.epochs,
                     )

    #trainer
    trainer = Trainer(
        default_root_dir = output_dir,
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpu,
        check_val_every_n_epoch = args.epochs, # todo 1, #10, #args.epochs, : at the end
        logger=[wandb_logger,csv_logger] if args.wandb else csv_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)
                #    ,ModelCheckpoint(save_last=True)
                   ],
        num_sanity_val_steps = 0
    )
    trainer.fit(model, train_queue,valid_queue)

    # trainer.test(model, dataloaders=valid_queue)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser("adversarial training")
  parser.add_argument('--data', type=str, default='cifar10', help='name of dataset')
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--lrschedule', type=str,default='cycle',help='lr schedule')
  parser.add_argument('--cyclelr', type=float,default=0.05,help='initial lr for onecycle schedule')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--archidx', type=int, default=2581,help='id of architectures')
  parser.add_argument('--wandb', type=int, default=0,help='using wandb')
  parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
  args = parser.parse_args()
  
  main(args)