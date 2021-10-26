from tqdm import tqdm
from argparse import ArgumentParser
from utils.ShapeNetDataset import ShapeNetDataset
from utils.model_utils import calc_cd, calc_emd
from utils.train_utils import setup_seed, AverageValueMeter
from torch.utils.data import DataLoader

from model.CovNet import CovNet


def load_model():
    try:
        model = CovNet(num_coarse=1024, num_dense=2048, lrate=args.lrate)
        logs = 'logs/epoch=200-batch=32.ckpt'
    except:
        raise Exception('cannot load model and logs')

    model = model.load_from_checkpoint(logs, **dict(model.hparams))
    print('Successfully load:', logs)

    model.cuda()
    model.freeze()
    model.eval()
    return model


def main(args):
    names = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4,
             'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9,
             'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

    model = load_model()
    setup_seed(0)

    loss = [AverageValueMeter() for i in range(3)]
    loss_avg = [AverageValueMeter() for i in range(3)]
    print('Categorie\tChamfer\tEmd\tF1')

    for k, [name, v] in enumerate(names.items()):
        [l.reset() for l in loss]
        dataset = ShapeNetDataset(npoints=args.num_points, class_choice={name: v}, split='test', hole_size=0.35)
        dataloader = DataLoader(dataset, args.batchSize, shuffle=False, num_workers=8, drop_last=False)
        counts = len(dataset)
        with tqdm(total=counts, desc=f"Processing {name} ", leave=False) as pbar:
            for x, y in dataloader:
                pbar.update(y.shape[0])

                coarse, out = model(x.cuda())

                _, chamfer_loss, f1 = calc_cd(out, y.cuda(), calc_f1=True)
                emd_loss = calc_emd(out, y)

                for i in range(x.shape[0]):
                    loss[0].update(chamfer_loss[i] * 10000)
                    loss[1].update(emd_loss[i] * 10000)
                    loss[2].update(f1[i])

            loss_avg[0].update(loss[0].avg)
            loss_avg[1].update(loss[1].avg)
            loss_avg[2].update(loss[2].avg)
        print("\r{:15}\t{:.8f}\t{:.8f}\t{:.8f}".format(name, loss[0].avg, loss[1].avg, loss[2].avg))
    print('\r{:15}\t{:.8f}\t{:.8f}\t{:.8f}'.format("avg", loss_avg[0].avg, loss_avg[1].avg, loss_avg[2].avg))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='gpus')
    parser.add_argument('--lrate', type=float, default=0.001, help='lrate')
    parser.add_argument('--epochs', type=int, default=1, help='train epoch')
    parser.add_argument('--batchSize', type=int, default=32, help='batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='the number of points')
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='bottleneck_size')
    args = parser.parse_args()

    main(args)
