import os

import torch

def last_checkpoints(path):
    states = torch.load(path, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
    return states

def average_checkpoints(last):
    avg = None
    for path in last:
        states = torch.load(path, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
        states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


def ensemble(args):
    last = [
        os.path.join(args.exp_dir, args.exp_name, f"epoch={n}.ckpt")
        for n in range(
            args.trainer.max_epochs - args.checkpoint.save_top_k,
            args.trainer.max_epochs,
        )
    ]
    model_path = os.path.join(
        args.exp_dir, args.exp_name, f"model_avg_{args.checkpoint.save_top_k}.pth"
    )

    model_path_last = os.path.join(
        args.exp_dir, args.exp_name, f"model_last.pth"
    )

    torch.save(average_checkpoints(last), model_path)
    torch.save(last_checkpoints(last[-1]), model_path_last)
    return model_path
