import argparse
import glob
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_aud_txt, split_file

import pdb

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="LRS2LRS3 Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="Directory where the sequence data is stored.",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Face detector used in the experiment.",
)
parser.add_argument(
    "--landmarks-dir",
    type=str,
    default=None,
    help="Directory where the landmarks data is stored.",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Directory of saved mouth patches or embeddings.",
)
parser.add_argument(
    "--subset",
    type=str,
    required=True,
    help="Subset of the dataset used in the experiment.",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of the dataset used in the experiment.",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=24,
    help="Length of the segment in seconds.",
)
parser.add_argument(
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merges the audio and video components to a media file.",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel.",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing).",
)
args = parser.parse_args()

seg_duration = args.seg_duration
dataset = args.dataset
text_transform = TextTransform()

# Load Data
args.data_dir = os.path.normpath(args.data_dir)
#vid_dataloader = AVSRDataLoader(
#    modality="video", detector=args.detector, convert_gray=False
#)
aud_dataloader = AVSRDataLoader(modality="audio")

#seg_vid_len = seg_duration * 25
seg_aud_len = seg_duration * 16000

# Label filename
label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.csv"
    if args.groups <= 1
    else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.{args.groups}.{args.job_index}.csv",
)
os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")
# Step 2, extract mouth patches from segments.
dst_aud_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_audio_seg{seg_duration}s"
)
dst_txt_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_text_seg{seg_duration}s"
)

if args.subset == "test":
    filenames = glob.glob(
        os.path.join(args.data_dir, 'Validation', 'doc',"**", "*.wav"), recursive=True      
    )
    #filenames.extend(
    #        glob.glob(
    #            os.path.join(args.data_dir, 'Validation', 'nur',"**", "*.wav"), recursive=True      
    #        )
    #)        
    #filenames.extend(
    #        glob.glob(
    #            os.path.join(args.data_dir, 'Validation', 'pat',"**", "*.wav"), recursive=True      
    #        )
    #)  
    print(dataset, args.subset, len(filenames))
elif args.subset == "train":
    filenames = glob.glob(
        os.path.join(args.data_dir, 'Training', 'doc',"**", "*.wav"), recursive=True      
    )
    #filenames.extend(
    #        glob.glob(
    #            os.path.join(args.data_dir, 'Training', 'nur',"**", "*.wav"), recursive=True      
    #        )
    #)        
    #filenames.extend(
    #        glob.glob(
    #            os.path.join(args.data_dir, 'Training', 'pat',"**", "*.wav"), recursive=True      
    #        )
    #)  
    print(dataset, args.subset, len(filenames))
    filenames.sort()
else:
    raise NotImplementedError


unit = math.ceil(len(filenames) * 1.0 / args.groups)
filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]
for data_filename in tqdm(filenames):
    try:
        audio_data = aud_dataloader.load_data(data_filename)
    except (UnboundLocalError, TypeError, OverflowError, AssertionError):
        continue

    #pdb.set_trace()
    if os.path.normpath(data_filename).split(os.sep)[-3] in [
        "doc",
        "nur",
        #"test",
        "pat",
    ]:
        #dst_vid_filename = (
        #    f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.mp4"
        #)
        dst_aud_filename = (
            f"{data_filename.replace(args.data_dir, dst_aud_dir)[:-4]}.wav"
        )
        dst_txt_filename = (
            f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}.txt"
        )
        #trim_vid_data, trim_aud_data = video_data, audio_data
        trim_aud_data = audio_data
        text_line = (
            #open(data_filename[:-4] + ".txt", "r").read().splitlines()[0].split(" ")
            open(data_filename.replace('Validation','Validation/medv')[:-4] + ".txt", "r").read()
        )  ###
        #print(text_line_list)
        
        #text_line = " ".join(text_line_list[2:])
        content = text_line.replace("?", "").replace("!", "").replace(".","")
        #print(content)
        #sys.exit()

        if trim_aud_data is None:
            continue
        
        audio_length = trim_aud_data.size(1)
        if audio_length == 0:
            continue
        if audio_length > seg_aud_len:
            continue
        
        #save_aud_txt(
        #    dst_aud_filename,
        #    dst_txt_filename,
        #    trim_aud_data,
        #    content,
        #    audio_sample_rate=16000,
        #)

        basename = os.path.relpath(
            dst_aud_filename, start=os.path.join(args.root_dir, dataset)
        )
        token_id_str = " ".join(
            map(str, [_.item() for _ in text_transform.tokenize(content)])
        )
        #f.write(
        #    "{}\n".format(
        #        f"{dataset},{basename},{trim_aud_data.shape[0]},{token_id_str}"
        #    )
        #)
        f.write(
            "{}\n".format(
                f"{dataset},{data_filename},{trim_aud_data.shape[0]},{token_id_str}"
            )
        )
        continue
    
f.close()
