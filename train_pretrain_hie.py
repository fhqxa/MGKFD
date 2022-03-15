import argparse
import os
import time

import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Models.dataloader.samplers import CategoriesSampler

from Models.models.Network import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *

DATA_DIR='/datas'


parser = argparse.ArgumentParser()
# about dataset and network
parser.add_argument('-dataset', type=str, default='cifar_fs', choices=['tieredimagenet','fc100','cifar_fs'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR)
# about pre-training
parser.add_argument('-max_epoch', type=int, default=120)
parser.add_argument('-lr', type=float, default=0.1)
parser.add_argument('-step_size', type=int, default=30)
parser.add_argument('-gamma', type=float, default=0.2)
parser.add_argument('-bs', type=int, default=128)
parser.add_argument("-c","--size_percent",type = float, default=1/2)
# about validation
parser.add_argument('-set', type=str, default='val', choices=['val', 'test'], help='the set for validation')
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=15)
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-metric', type=str, default='cosine')
parser.add_argument('-num_episode', type=int, default=100)
parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
parser.add_argument('-random_val_task', action='store_true', help='random samples tasks for validation in each epoch')

# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=10, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')

# about deepemd setting
parser.add_argument('-norm', type=str, default='center', choices=[ 'center'])
parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn'])
parser.add_argument('-feature_pyramid', type=str, default=None)
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
# about training
parser.add_argument('-gpu', default='1')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')

args = parser.parse_args()
pprint(vars(args))

num_gpu = set_gpu(args)
set_seed(args.seed)

dataset_name = args.dataset
args.save_path = 'pre_train/%s/%d-%.4f-%d-%.2f/' % \
                 (dataset_name, args.bs, args.lr, args.step_size, args.gamma)
args.save_path = osp.join('/hie_loss_cg_fgl/', args.save_path)
if args.extra_dir is not None:
    args.save_path=osp.join(args.save_path,args.extra_dir)
ensure_path(args.save_path)


Dataset=set_up_datasets(args)
trainset = Dataset('train', args)
train_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

valset = Dataset(args.set, args)
val_sampler = CategoriesSampler(valset.label, args.num_episode, args.way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
if not args.random_val_task:
    print('fix val set for all epochs')
    val_loader = [x for x in val_loader]
print('save all checkpoint models:', (args.save_all is True))

model = DeepEMD(args, mode='pre_train')
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()

# label of query images.
label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)  # shape[75]:012340123401234...
label = label.type(torch.LongTensor)
label = label.cuda()

optimizer = torch.optim.SGD([{'params': model.module.encoder.parameters(), 'lr': args.lr},
                             {'params': model.module.fc.parameters(), 'lr': args.lr}
                             ], momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


def save_model(name):
    torch.save(dict(params=model.module.encoder.state_dict()), osp.join(args.save_path, name + '.pth'))

def get_coarse_label(label):
    
    c_label = []
    for i in range(args.way):
        if label[i].item() in c_label:
            continue
        else:
            c_label.append(label[i].item())
    c_label = dict(zip(c_label, range(len(c_label))))
    coarse_label = torch.arange(label.size(0))
    for i in range(label.size(0)):
        coarse_label[i] = c_label[label[i].item()]
    return coarse_label.cuda()
    
def get_sample_parent(sample_coarse_labels):
    sample_parents = {}  # 整合到父节点的数组下
    for i in range(sample_coarse_labels.size(0)):
        if sample_coarse_labels[i].item() not in sample_parents.keys():
            sample_parents[sample_coarse_labels[i].item()] = [i]
        else:
            sample_parents[sample_coarse_labels[i].item()].append(i)
    return sample_parents


trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0

global_count = 0
writer = SummaryWriter(osp.join(args.save_path, 'tf'))

result_list = [args.save_path]
for epoch in range(1, args.max_epoch + 1):
    print(args.save_path)
    start_time = time.time()
    model = model.train()
    model.module.mode = 'pre_train'
    tl = Averager()
    ta = Averager()
    #standard classification for pretrain
    tqdm_gen = tqdm.tqdm(train_loader)
    for i, batch in enumerate(tqdm_gen, 1):
        global_count = global_count + 1
        data, train_label,_ = [_.cuda() for _ in batch]
        logits = model(data)
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        writer.add_scalar('data/loss', float(loss), global_count)
        writer.add_scalar('data/acc', float(acc), global_count)
        total_loss = loss
        writer.add_scalar('data/total_loss', float(total_loss), global_count)
        tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
  

    tl = tl.item()
    ta = ta.item()

    model = model.eval()
    model.module.mode = 'meta'
    vl = Averager()
    va = Averager()
    va_f = Averager()
    va_c = Averager()
    #use deepemd fcn for validation
    with torch.no_grad():
        tqdm_gen = tqdm.tqdm(val_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, _, coarse_label = [_ for _ in batch]
            coarse_label = get_coarse_label(coarse_label)

            k = args.way * args.shot
            model.module.mode = 'encoder'
            data = model(data.cuda())
            data_shot, data_query = data[:k], data[k:]   # 5*640*13*1
            coarse_shot,coarse_query = coarse_label[:k], coarse_label[k:]
            if args.shot > 1:
                data_shot = model.module.get_sfc(data_shot)
                coarse_shot = coarse_shot[:args.shot]
            sample_parents = get_sample_parent(coarse_shot)
            for i, (key,value) in enumerate(sample_parents.items()):
                sample_coarse_ext = data_shot[value,:].mean(dim=0).unsqueeze(0)  # 获取每个父类特征的平均值
                if i == 0:
                    sample_coarse_exts = sample_coarse_ext
                else:
                    sample_coarse_exts = torch.cat((sample_coarse_exts,sample_coarse_ext),0) 

            model.module.mode = 'meta'
            model.module.grain = 'coarse'
            coarse_logits = model((sample_coarse_exts.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            loss_c = F.cross_entropy(coarse_logits, coarse_query)


            model.module.grain = 'fine'
            logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            

            all_logits = torch.zeros(data_query.size(0), args.way).cuda()
            for i,(key,value) in enumerate(sample_parents.items()):
                for j in value:
                    all_logits[:,j] = 0.4*logits[:,j] + 0.6*coarse_logits[:,i]
            pred = torch.argmax(all_logits, dim=1)
            loss_f = 0
            
            for j in range(data_query.size(0)):
                loss0 = F.cross_entropy(all_logits[j,:].view(1,-1), label[j].reshape((1,)))
                if coarse_shot[pred[j].item()] == coarse_query[j]:
                    if pred[j] == label[j]:
                        w = 0
                    else:
                        w = 1/2
                else:
                    w = 1
                loss_f += (1+w)*loss0
            loss_f = loss_f/data_query.size(0)

            loss = loss_f 
            acc_f = count_acc(logits, label)
            acc_c = count_acc(coarse_logits, coarse_query)
            acc = count_acc(all_logits, label)

            vl.add(loss.item())
            va.add(acc)
            va_f.add(acc_f)
            va_c.add(acc_c)

        vl = vl.item()
        va = va.item()
        va_f = va_f.item()
        va_c = va_c.item()
    writer.add_scalar('data/val_loss', float(vl), epoch)
    writer.add_scalar('data/val_acc', float(va), epoch)
    writer.add_scalar('data/val_f_acc', float(va_f), epoch)
    writer.add_scalar('data/val_c_acc', float(va_c), epoch)
    tqdm_gen.set_description('epo {}, val, loss={:.4f} acc_f={:.4f} acc_c={:.4f} acc={:.4f}'.format(epoch, vl, va_f,va_c,va))

    print (' val_f acc:%.4f val_c acc:%.4f val acc:%.4f '%(va_f,va_c,va))

    if va >= trlog['max_acc']:
        print('A better model is found!!')
        trlog['max_acc'] = va
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc')
        torch.save(optimizer.state_dict(), osp.join(args.save_path, 'optimizer_best.pth'))

    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)

    result_list.append(
        'epoch:%03d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (epoch, tl, ta, vl, va))
    torch.save(trlog, osp.join(args.save_path, 'trlog'))
    if args.save_all:
        save_model('epoch-%d' % epoch)
        torch.save(optimizer.state_dict(), osp.join(args.save_path, 'optimizer_latest.pth'))
    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('This epoch takes %d seconds' % (time.time() - start_time),
          '\nstill need around %.2f hour to finish' % ((time.time() - start_time) * (args.max_epoch - epoch) / 3600))
    lr_scheduler.step()

writer.close()
result_list.append('Val Best Epoch {},\nbest val Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ))
save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)