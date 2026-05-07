import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add RAFT to path so we can import it
sys.path.append(os.path.join(os.path.dirname(__file__), '../models/RAFT/core'))

try:
    from raft import RAFT
    from utils.utils import InputPadder
    from utils import flow_viz
except ImportError:
    print("RAFT not found. Please run scripts/setup_raft.sh first.")
    sys.exit(1)

def load_image(imfile):
    img = np.array(cv2.imread(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="../models/RAFT/models/raft-things.pth")
    parser.add_argument('--path', help="dataset for evaluation", default="../data/DAVIS/JPEGImages/480p/bear/")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if not os.path.exists(args.model):
        print(f"Model {args.model} not found. Did you run setup_raft.sh?")
        return
        
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model = model.module
    model.cuda()
    model.eval()
    
    print("RAFT model loaded successfully! Ready for flow extraction.")
    # TODO: Implement loop over image pairs and forward pass

if __name__ == '__main__':
    # main()
    print("RAFT flow extraction skeleton ready.")
