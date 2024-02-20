import os

import hydra
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio
import torchvision
from lightning_infer import ModelModule
from datamodule.transforms import AudioTransform
from tqdm import tqdm
from datamodule.data_module import DataModule
from torch.utils.data import BatchSampler, SequentialSampler, DistributedSampler
from pytorch_lightning import seed_everything, Trainer, LightningDataModule
from pytorch_lightning.plugins import DDPPlugin
from datamodule.samplers import DistributedSamplerWrapper
import torch.distributed as dist
import time

# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    #print(max_size)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        #pad_val = -1 if data_type == "target" else 0.0
        if data_type=="input":
            pad_val = 0.0
            c_batch, sample_lengths = pad([s[data_type] for s in batch if s[data_type] is not None], pad_val)
            batch_out[data_type + "s"] = c_batch
            batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
        else:
            batch_out["targets"] = [s[data_type] for s in batch if s[data_type] is not None]
    return batch_out



class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="mediapipe"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        if self.modality in ["audio"]:
            self.audio_transform = AudioTransform(subset="test")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
        )
        self.modelmodule.model = torch.compile(self.modelmodule.model)
        self.modelmodule.to('cuda')
        #self.modelmodule = nn.DataParallel(self.modelmodule)
        self.modelmodule.eval()
        #self.modelmodule = torch.compile(self.modelmodule)
        

    def forward(self, data_filename):
        #data_filename = os.path.abspath(data_filename)
        #assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        data_filename = os.path.join(data_filename)

        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)
            with torch.no_grad():
                transcript = self.modelmodule(audio.to('cuda'))

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


class InferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        #root,
        #label_path,
        input_path,
        subset,
        modality,
        audio_transform,
        rate_ratio=640,
    ):

        self.input_path = input_path

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(input_path)

        self.audio_transform = audio_transform

    def load_list(self, input_path):
        paths_counts_labels = []
        for path_count_label in input_path:
            filename = path_count_label.split('/')[-1]
            #dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            paths_counts_labels.append(
                (
                    path_count_label,
                    filename,
                    #rel_path,
                    #int(input_length),
                    #torch.tensor([int(_) for _ in token_id.split()]),
                )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        #dataset_name, rel_path, input_length, token_id = self.list[idx]
        path, filename = self.list[idx]
        #path = os.path.join(self.root, dataset_name, rel_path)
        data_filename = os.path.join(path)
        audio, sample_rate = self.load_audio(data_filename)
        audio = self.audio_process(audio, sample_rate)
        audio = audio.transpose(1, 0)
        audio = self.audio_transform(audio)
        return {"input": audio, "target": filename}

    def __len__(self):
        return len(self.list)


    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate


    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform




class DataModule(LightningDataModule):
    def __init__(self, cfg=None, input_paths=None):
        super().__init__()
        self.cfg = cfg
        self.input_paths = input_paths
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes

    def _dataloader(self, ds, sampler, collate_fn): #_dataloader(self, ds, sampler, collate_fn):
        return torch.utils.data.DataLoader(
            ds,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        pred_dataset = InferDataset(
            #label_path=os.path.join(ds_args.root, ds_args.label_dir, ds_args.test_file),
            input_path=self.input_paths,
            subset="test",
            modality='audio',
            audio_transform=AudioTransform(subset="val"),
        )
        point_sampler = SequentialSampler(pred_dataset)
        #sampler = BatchSampler(point_sampler, batch_size=10, drop_last=True)
        sampler = BatchSampler(point_sampler, batch_size=1, drop_last=False)
        if self.total_gpus > 1:
            #sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=False)
            sampler = DistributedSampler(sampler, shuffle=False)
        #return self._dataloader(pred_dataset, sampler, collate_pad)
        return torch.utils.data.DataLoader(pred_dataset, batch_size=1, num_workers=16, sampler=sampler, shuffle=False, collate_fn=collate_pad)

@hydra.main(config_path="./conf", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    modelmodule = ModelModule(cfg)

    f = open(cfg.infer_path,'r')
    lines = f.readlines()
    input_paths=[]
    for line in tqdm(lines):
        input_paths.append(line.rstrip())
    print("Done")

    datamodule = DataModule(cfg, input_paths)

    trainer = Trainer(
        **cfg.trainer,
        strategy=DDPPlugin(find_unused_parameters=False) if cfg.gpus > 1 else None
    )

    predict = trainer.predict(model=modelmodule, datamodule=datamodule)
    if cfg.gpus > 1:
        import torch.distributed as dist
        object_list = [None] * cfg.gpus
        obj = predict
        dist.all_gather_object(object_list, obj)

    # only 1 process should save the checkpoint and compute WER
    if trainer.is_global_zero:
        start = time.time()
        print("Done")
        #print(len(object_list))
        #print(object_list)
        all_predict = object_list[0] if cfg.gpus > 1 else predict 
        for i in range (cfg.gpus - 1):
            all_predict.extend(object_list[i+1])
        print(len(all_predict))
        all_predict = list(set(all_predict))
        print(len(all_predict))
        print("time :", time.time() - start)

            

    
    # logs를 외부 txt로 저장?

    
    '''
    pipeline = InferencePipeline(cfg)
    #print(cfg.infer_path)
    f = open(cfg.infer_path,'r')
    lines = f.readlines()
    for line in tqdm(lines):
        transcript = pipeline(line.rstrip())
        print(f"transcript: {transcript}")
    print("Done")
    '''
if __name__ == "__main__":
    main()
