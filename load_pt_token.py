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
    path = ['/home/nas4/user/yh/ai_hub_korean/outputs/2023-10-17/12-18-54/Tr2DZ/small_zeroshot_v2/last.ckpt']
    #model_path = os.path.join(
    #    args.exp_dir, args.exp_name, f"model_avg_{args.checkpoint.save_top_k}.pth"
    #)

    model_path_last = os.path.join(
        '//home/nas4/user/yh/ai_hub_korean/outputs/2023-10-17/12-18-54/Tr2DZ/small_zeroshot_v2', f"model_last.pt"
    )

    token_list=['<unk>','안','녕','하','세','요']

    vocab_unit_path = '/home/nas4/user/yh/ai_hub_korean/spm/unigram/unigram5000_units.txt'
    f = open(vocab_unit_path, 'r')
    lines = f.readlines()
    vocabs = []
    for token in lines:
        vocabs.append(token.rstrip())
        print(token.rstrip())
    print("[Total vocab]", len(vocabs))

    #torch.save(average_checkpoints(last), model_path)
    state = {'model': average_checkpoints(path),
            'tokenize': vocabs}
    
    
    #torch.save(average_checkpoints(path), model_path_last)
    torch.save(state, model_path_last)

    #return model_path_last

print("start")

#temp = ensemble()
#print(temp)

state = torch.load('/home/nas4/user/yh/ai_hub_korean/model_last.pt')
#print(state['tokenize'])

vocab = state['tokenize']

vocab_path = os.path.join('/home/nas4/user/yh/ai_hub_korean/','units_test.txt')
vocab_file = open(vocab_path, 'w')
for token in vocab:
    vocab_file.write(token + '\n')

'''
def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)
'''

