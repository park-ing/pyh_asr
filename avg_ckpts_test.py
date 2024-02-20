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


def ensemble():
    path = ['/home/nas4/user/yh/ai_hub_korean/outputs/2023-10-26/16-58-18/Tr2DZ/new_midV3_uni_zeroshot/last.ckpt']
    #model_path = os.path.join(
    #    args.exp_dir, args.exp_name, f"model_avg_{args.checkpoint.save_top_k}.pth"
    #)

    model_path_last = os.path.join(
        '/home/nas4/user/yh/ai_hub_korean/outputs/2023-10-26/16-58-18/Tr2DZ/new_midV3_uni_zeroshot/', f"model_last.pth"
    )

    #token_list=['<unk>','안','녕','하','세','요']

    #torch.save(average_checkpoints(last), model_path)
    #state = {'model': average_checkpoints(path),
    #        'tokenize': token_list}
    
    
    torch.save(average_checkpoints(path), model_path_last)
    #torch.save(state, model_path_last)

    return model_path_last

print("start")

temp = ensemble()
print(temp)