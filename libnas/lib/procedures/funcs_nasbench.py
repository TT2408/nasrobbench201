#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
import os, time, copy, torch, pathlib

# import datasets
from config_utils import load_config
from procedures   import prepare_seed, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net


__all__ = ['evaluate_for_seed', 'pure_evaluate', 
          #  'get_nas_bench_loaders'
           ]


def pure_evaluate(xloader, network, criterion=torch.nn.CrossEntropyLoss()):
  data_time, batch_time, batch = AverageMeter(), AverageMeter(), None
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  latencies, device = [], torch.cuda.current_device()
  network.eval()
  with torch.no_grad():
    end = time.time()
    for i, (inputs, targets) in enumerate(xloader):
      targets = targets.cuda(device=device, non_blocking=True)
      inputs  = inputs.cuda(device=device, non_blocking=True)
      data_time.update(time.time() - end)
      # forward
      features, logits = network(inputs)
      loss             = criterion(logits, targets)
      batch_time.update(time.time() - end)
      if batch is None or batch == inputs.size(0):
        batch = inputs.size(0)
        latencies.append( batch_time.val - data_time.val )
      # record loss and accuracy
      prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      losses.update(loss.item(),  inputs.size(0))
      top1.update  (prec1.item(), inputs.size(0))
      top5.update  (prec5.item(), inputs.size(0))
      end = time.time()
  if len(latencies) > 2: latencies = latencies[1:]
  return losses.avg, top1.avg, top5.avg, latencies



def procedure(xloader, network, criterion, scheduler, optimizer, mode: str):
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  if mode == 'train'  : network.train()
  elif mode == 'valid': network.eval()
  else: raise ValueError("The mode is not right : {:}".format(mode))
  device = torch.cuda.current_device()
  data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()
  for i, (inputs, targets) in enumerate(xloader):
    if mode == 'train': scheduler.update(None, 1.0 * i / len(xloader))

    targets = targets.cuda(device=device, non_blocking=True)
    if mode == 'train': optimizer.zero_grad()
    # forward
    features, logits = network(inputs)
    loss             = criterion(logits, targets)
    # backward
    if mode == 'train':
      loss.backward()
      optimizer.step()
    # record loss and accuracy
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))
    # count time
    batch_time.update(time.time() - end)
    end = time.time()
  return losses.avg, top1.avg, top5.avg, batch_time.sum


def evaluate_for_seed(arch_config, opt_config, train_loader, valid_loaders, seed: int, logger):

  prepare_seed(seed) # random seed
  net = get_cell_based_tiny_net(arch_config)
  #net = TinyNetwork(arch_config['channel'], arch_config['num_cells'], arch, config.class_num)
  flop, param  = get_model_infos(net, opt_config.xshape)
  logger.log('Network : {:}'.format(net.get_message()), False)
  logger.log('{:} Seed-------------------------- {:} --------------------------'.format(time_string(), seed))
  logger.log('FLOP = {:} MB, Param = {:} MB'.format(flop, param))
  # train and valid
  optimizer, scheduler, criterion = get_optim_scheduler(net.parameters(), opt_config)
  default_device = torch.cuda.current_device()
  network = torch.nn.DataParallel(net, device_ids=[default_device]).cuda(device=default_device)
  criterion = criterion.cuda(device=default_device)
  # start training
  start_time, epoch_time, total_epoch = time.time(), AverageMeter(), opt_config.epochs + opt_config.warmup
  train_losses, train_acc1es, train_acc5es, valid_losses, valid_acc1es, valid_acc5es = {}, {}, {}, {}, {}, {}
  train_times , valid_times, lrs = {}, {}, {}
  for epoch in range(total_epoch):
    scheduler.update(epoch, 0.0)
    lr = min(scheduler.get_lr())
    train_loss, train_acc1, train_acc5, train_tm = procedure(train_loader, network, criterion, scheduler, optimizer, 'train')
    train_losses[epoch] = train_loss
    train_acc1es[epoch] = train_acc1 
    train_acc5es[epoch] = train_acc5
    train_times [epoch] = train_tm
    lrs[epoch] = lr
    with torch.no_grad():
      for key, xloder in valid_loaders.items():
        valid_loss, valid_acc1, valid_acc5, valid_tm = procedure(xloder  , network, criterion,      None,      None, 'valid')
        valid_losses['{:}@{:}'.format(key,epoch)] = valid_loss
        valid_acc1es['{:}@{:}'.format(key,epoch)] = valid_acc1 
        valid_acc5es['{:}@{:}'.format(key,epoch)] = valid_acc5
        valid_times ['{:}@{:}'.format(key,epoch)] = valid_tm

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.avg * (total_epoch-epoch-1), True) )
    logger.log('{:} {:} epoch={:03d}/{:03d} :: Train [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%] Valid [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%], lr={:}'.format(time_string(), need_time, epoch, total_epoch, train_loss, train_acc1, train_acc5, valid_loss, valid_acc1, valid_acc5, lr))
  info_seed = {'flop' : flop,
               'param': param,
               'arch_config' : arch_config._asdict(),
               'opt_config'  : opt_config._asdict(),
               'total_epoch' : total_epoch ,
               'train_losses': train_losses,
               'train_acc1es': train_acc1es,
               'train_acc5es': train_acc5es,
               'train_times' : train_times,
               'valid_losses': valid_losses,
               'valid_acc1es': valid_acc1es,
               'valid_acc5es': valid_acc5es,
               'valid_times' : valid_times,
               'learning_rates': lrs,
               'net_state_dict': net.state_dict(),
               'net_string'  : '{:}'.format(net),
               'finish-train': True
              }
  return info_seed