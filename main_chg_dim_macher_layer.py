
import argparse

import torch


def main():
    args = get_args()

    model = torch.load(args.inp_model)
    model_params = model['params'] if 'params' in model.keys() else model

    # find 'dim_matcher_layer' layers.
    matcher_layers = [p for p in model_params.keys() if 'dim_matcher_layer' in p]
    # pair weights and biases
    matcher_layers = list(zip(matcher_layers[::2], matcher_layers[1::2]))

    for w, b in matcher_layers:
        linear = torch.nn.Linear(args.modulation_dim, args.nework_dim)
        model_params[w], model_params[b] = linear.weight.data, linear.bias.data

    torch.save(model, args.out_model)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp_model', type=str)
    parser.add_argument('out_model', type=str)
    parser.add_argument('-m', '--modulation_dim', type=int, default=1024)
    parser.add_argument('-n', '--nework_dim', type=int, default=180)
    return parser.parse_args()


if __name__ == '__main__':
    main()
