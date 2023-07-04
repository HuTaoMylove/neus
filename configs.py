import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='../../dataset/nerf_synthetic/')
    parser.add_argument("--things", type=str, default='drums')
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=999)

    parser.add_argument("--render", action='store_true', default=False)
    parser.add_argument("--model_idx", type=str, default='latest')
    parser.add_argument("--reload", action='store_true', default=False)
    parser.add_argument("--perturb", action='store_true', default=False)
    parser.add_argument("--white_bkgd", action='store_true', default=False)
    parser.add_argument("--use_view", action='store_false', default=True)
    parser.add_argument("--only_reconstruct", action='store_true', default=False)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--test_skip", type=int, default=1)
    parser.add_argument("--val_skip", type=int, default=100)
    parser.add_argument("--bias", type=float, default=1.5)

    parser.add_argument("--Batch_size", type=int, default=512)
    parser.add_argument("--co_samples", type=int, default=28)
    parser.add_argument("--re_samples", type=int, default=28)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--warm_up", type=float, default=5)
    parser.add_argument("--anneal", type=float, default=40)

    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--reconstruct_interval", type=int, default=5)

    return parser.parse_args()
