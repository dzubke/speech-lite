# standard libs
import os
import time
# third-party libs
import torch
from warpctc_pytorch import CTCLoss


if __name__ == "__main__":

    while True:
        print(f"data dir /mnt/disks/data_disk/ exists: {os.path.exists('/mnt/disks/data_disk')}")
        print(f"data dir /mnt/disks/data_disk/data/ exists: {os.path.exists('/mnt/disks/data_disk/data')}")
        print(f"gpu available: {torch.cuda.is_available()}")
        print(f"type of naren CTC Loss: {type(CTCLoss)}")
        with open("/tmp/log.txt", 'w') as fid:
            fid.write(f"data dir /mnt/disks/data_disk/ exists: {os.path.exists('/mnt/disks/data_disk')}\n")
            fid.write(f"data dir /mnt/disks/data_disk/data/ exists: {os.path.exists('/mnt/disks/data_disk/data')}\n")
            fid.write(f"gpu available: {torch.cuda.is_available()}\n")
        time.sleep(1)
