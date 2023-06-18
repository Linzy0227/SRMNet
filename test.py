import argparse
import os

import torch
import setproctitle

from model.net import Model
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='user of name', type=str)
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--resume', default='path of checkpoint', type=str)
parser.add_argument('--datapath', default='path of datasets', type=str)
parser.add_argument('--savepath', default='BrsTS20VisualFloder', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
             [True, False, False, False],
             [False, True, False, True], [False, True, True, False], [True, False, True, False],
             [False, False, True, True], [True, False, False, True], [True, True, False, False],
             [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
             [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair',
                 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                 'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                 'flairt1cet1t2']

    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train2.txt'
        test_file = 'test2.txt'

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    num_cls = 4

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=args.datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            mask_n = mask_name[i]
            dice_score = test_softmax(
                test_loader,
                model,
                savepath=args.savepath,
                dataname=args.dataname,
                feature_mask=mask,
                mask_name=mask_n)
            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))