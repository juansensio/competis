import os
import math
import torch
import sys

import random
random.seed(42)

import warnings
import numpy as np
import xarray as xr
import pandas as pd

from typing import List
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

from src.util.scale import scale_zero_to_one
from src.dataloader.dataset_stats import X_MIN, X_MAX, Y_REG_MIN, Y_REG_MAX, Y_REG_NORM_MAX


# quite some annoying UserWarnings thrown by xarray 
# when opening datasets with phony_dims=None
warnings.simplefilter("ignore")


WFC_ROOT_DIR    = "/playpen-ssd/levi/w4c/w4c-25/weather4cast_data"
WFC_C1_TEST_DIR = "/playpen-ssd/levi/w4c/w4c-25/weather4cast_data/challenge_one"
Y_REG_NORMS_FP  = "/playpen-ssd/levi/w4c/w4c-25/___old___/11-4-25-y_reg_norms.pth"


def get_y_reg_bin_counts_step(n_classes: int):

    # emp distribution of train label values
    y_regs:torch.Tensor  = torch.load(Y_REG_NORMS_FP)
    data                 = y_regs.squeeze()

    # [N_classes]
    y_reg_norm_bins = torch.arange(0, data.max(), data.max() / n_classes)
    freq            = torch.zeros(y_reg_norm_bins.shape)

    for i, _bin in enumerate(y_reg_norm_bins):

        start = _bin
        if i <= (len(y_reg_norm_bins) - 2):
            end = y_reg_norm_bins[i+1]
        else:
            end = sys.maxsize - 1

        # num samples/bin
        count   = ((data >= start) & (data < end)).sum()
        freq[i] = count
    
    y_reg_norm_bin_step   = Y_REG_NORM_MAX / n_classes
    y_reg_norm_bin_counts = freq
    
    # to avoid any divide by zero errors when calculating pmfs
    y_reg_norm_bin_counts += 1

    return y_reg_norm_bins, y_reg_norm_bin_step, y_reg_norm_bin_counts


class Sat2RadDataset(Dataset):

    TOTAL_TRAIN_T      = 163820
    TOTAL_VAL_T        = 15100
    TOTAL_SAMPLE_LEN_T = 20

    SAMPLE_FULL_H = 252
    SAMPLE_FULL_W = 252
    SAMPLE_H      = 32
    SAMPLE_W      = 32

    TEST_STEPS_PER_EPOCH = 120

    def __init__(
            self, 
            split="train",
            n_classes=128,
        ):

        super().__init__()

        self.split = split
        if not split in ['train', 'val', 'test']: 
            raise Exception(f"Invalid split: {split}")
        
        if   split == "train":
            self.steps_per_epoch = Sat2RadDataset.TOTAL_TRAIN_T // 20
        elif split == "val":
            self.steps_per_epoch = Sat2RadDataset.TOTAL_VAL_T   // 20
        else:
            self.steps_per_epoch = Sat2RadDataset.TEST_STEPS_PER_EPOCH
        
        # modify to keyword for challenge one (cum. precip) data
        if self.split == "test":
            split = "cum1test"
        
        opera_regex   = f"{WFC_ROOT_DIR}/**/*{split}*rates*h5"
        hrit_regex    = f"{WFC_ROOT_DIR}/**/*{split}*reflbt0*h5"

        all_opera_fps = list(set(glob(opera_regex, recursive=True)))
        all_hrit_fps  = list(set(glob(hrit_regex , recursive=True)))

        self.all_fp_dict = {}
        fp_dict          = {}

        # please don't looooookkk...!!!
        # remove some symlinks
        for op_fp in all_opera_fps:
            
            if not Path(op_fp).is_symlink():

                name = Path(op_fp).name
                name = name.split(".")[0]
                if name not in self.all_fp_dict:
                    self.all_fp_dict[name] = {
                        "opera": op_fp,
                        "hrit" : None,
                    }
                else:
                    self.all_fp_dict[name]['opera'] = op_fp

        for hr_fp in all_hrit_fps:

            if not Path(hr_fp).is_symlink():

                name = Path(hr_fp).name
                name = name.split(".")[0]

                if name not in self.all_fp_dict:
                    self.all_fp_dict[name] = {
                        "opera": None,
                        "hrit" : hr_fp,
                    }
                else:
                    self.all_fp_dict[name]['hrit'] = hr_fp

        hrit_fps  = []
        opera_fps = []

        for k, v in self.all_fp_dict.items():
            if v['opera'] != None and v['hrit'] != None: 
                fp_dict[k] = v
                hrit_fps.append(v['hrit'])
                opera_fps.append(v['opera'])

        # multiband satallite data
        # [B, T1, 11, 252, 252]
        self.hrit_buffer   : List[xr.Dataset] = []
        
        # TODO: we should just use dicts to hold all datasets
        self.hrit_test_dict: dict             = {}

        # high-res radar rain-rates
        # [B, T2, 252, 252, 1]
        self.opera_buffer: List[xr.Dataset] = []

        # we have a lot to think about here
        # current, we lazy load huge xarray datasets
        # - samples, once loaded, are NOT cached atm
        if self.split in ['train', 'val']:

            for fp_hr, fp_op in tqdm(zip(hrit_fps, opera_fps), desc="Loading dataset...", total=len(hrit_fps)):
                
                ds = xr.open_dataset(fp_hr, phony_dims=None, chunks="auto", create_default_indexes=False)
                self.hrit_buffer.append(ds)

                ds = xr.open_dataset(fp_op, phony_dims=None, chunks="auto", create_default_indexes=False)
                self.opera_buffer.append(ds)

        # NOTE: we only have HRIT inputs for the test set
        if self.split == "test":
            
            for k, v in self.all_fp_dict.items():
                if v['hrit'] != None:
                    fp_dict[k] = v
                    hrit_fps.append(v['hrit'])

            for fp_hr in tqdm(hrit_fps, desc="Loading dataset...", total=len(hrit_fps)):
                ds = xr.open_dataset(fp_hr, phony_dims='sort')
                self.hrit_buffer.append(ds)
                self.hrit_test_dict[Path(fp_hr).name] = ds

        if self.split == "test":
            
            csvs_rx       = f"{WFC_C1_TEST_DIR}/*.csv"
            
            # [0008, 0009, 0010]
            test_csvs_fps = sorted(glob(csvs_rx))
            dfs           = [pd.read_csv(fp) for fp in test_csvs_fps]
            
            test_df = None
            
            for fp, df in zip(test_csvs_fps, dfs):

                if test_df is None: 
                    test_df   = df
                    names     = [Path(fp).name] * len(df)
                    df['key'] = names

                else:
                    names     = [Path(fp).name] * len(df)
                    df['key'] = names
                    test_df   = pd.concat([test_df, df], axis=0)

            if self.steps_per_epoch != len(test_df): print(f"Warning: changing steps-per-epoch from {self.steps_per_epoch} -> {len(test_df)}")
            self.steps_per_epoch     = len(test_df)
            self.test_df             = test_df

        # calculate the sum of the temporal dim shapes (T) across all regions
        if self.split in ['train', 'val']:
            
            total_T = 0
            for hrit_ds, opera_ds in zip(self.hrit_buffer, self.opera_buffer):
                T = min(hrit_ds['REFL-BT'].shape[0], opera_ds['rates.crop'].shape[0])
                total_T += T
            
            # print(f"Warning: changing steps-per-epoch from {self.steps_per_epoch} -> {total_T // 20}")
            self.total_T = total_T

        # *** calculate dataset statistics ****
        assert n_classes > 0, f"Number of classes must be >= 1"
        self.num_classes = n_classes
        self.y_reg_norm_bins, self.y_reg_norm_bin_step, self.y_reg_norm_bin_counts = get_y_reg_bin_counts_step(self.num_classes)

    def __len__(self) -> int:
        return self.steps_per_epoch
    
    @staticmethod
    def X_pre_proc(X: torch.Tensor) -> torch.Tensor:
        # [ds.min, ds.max] -> [0, 1]
        X = (X - X_MIN.view(1, -1, 1, 1)) / (X_MAX.view(1, -1, 1, 1) - X_MIN.view(1, -1, 1, 1))
        return X

    @staticmethod
    def X_post_proc(X_norm: torch.Tensor) -> torch.Tensor:
        # [0, 1] -> [ds.min, ds.max]
        S = (X_MAX - X_MIN)
        X = (S * X_norm) - X_MIN
        return X

    @staticmethod
    def y_reg_pre_proc(y_reg: torch.Tensor) -> torch.Tensor:
        # [ds.min, ds.max] -> [0, 1]
        y_reg = (y_reg - Y_REG_MIN) / (Y_REG_MAX - Y_REG_MIN)
        return y_reg

    @staticmethod
    def y_reg_post_proc(y_reg_norm: torch.Tensor) -> torch.Tensor:
        # [0, 1] -> [ds.min, ds.max]
        S     = (Y_REG_MAX       - Y_REG_MIN)
        y_reg = (S * y_reg_norm) - Y_REG_MIN
        return y_reg
     
    def get_y_reg_norm_cat_label(self, y_reg_norm: float) -> torch.Tensor:
        """
        args
        ---
        :y_reg: float (original regression target)

        returns
        ---
        :torch.Tensor: one hot categorcial label
        """

        assert y_reg_norm >= 0.0

        # round y to the nearest multiple of bin step
        i                = int(round(y_reg_norm / self.y_reg_norm_bin_step))
        
        # HACK: occasionally we're getting enormous values (e.g., 2000+)
        # TODO: investigate
        i = min(i, len(self.y_reg_norm_bins) - 1)

        one_hot_label    = torch.zeros(self.y_reg_norm_bins.shape)
        one_hot_label[i] = 1

        # HACK: replace one-hot label with a gaussian centered at the onehot idx
        # sigma      = 0.25
        # gauss      = torch.exp(-0.5 * ((torch.arange(len(one_hot_label)) - i) / sigma)**2)
        # gauss_norm = gauss / gauss.sum()

        return one_hot_label

    def get_item_train_val(self, index: int) -> dict:
        """
        from the organizers...

        > *Cumulative rainfall should be averaged over a 32×32 pixel area of hi-res radar rain rates. 
        It should be aggregated over 4h into the future (16 time slots à 15 mins). 
        As rain rates are per hour and there are 4 slots à 15 mins per hour, 
        that means averaging per hour and summing the 4 hours, 
        i.e., averaging the 16 slots and multiplying by 4 (or summing the 16 slots and diving by 4).*

        # [HRIT, OPERA]
        region_0: ((24644, 11, 252, 252), (24644, 1, 1512, 1512))
        region_1: ((20308, 11, 252, 252), (24644, 1, 252, 252))
        region_2: ((20308, 11, 252, 252), (20308, 1, 252, 252))
        region_3: ((20308, 11, 252, 252), (24644, 1, 252, 252))
        region_4: ((20308, 11, 252, 252), (24644, 1, 252, 252))
        region_5: ((20308, 11, 252, 252), (20308, 1, 1512, 1512))
        region_6: ((24644, 11, 252, 252), (20308, 1, 252, 252))

        input
        --- 
        - :X:
        - 1H HRIT satallite context (can be larger); centered about corresponding area of precipitation
        - (B, H, W, C, T) -> (B, (32 // 6) + (32 // 6) + 1, 11, 4) -> (B, 6+, 6+, 11, 4)

        output 
        ---
        - :y:
        - OPERA "average, hourly cummulative precipitation" for 4H over (32x32) pixels
        - (B, H, W, C, T) -> (B, 32, 32, 1, 16)
        """

        # TODO
        # * contact organizers
        # * T1, T2 are DIFFERENT for trainset; (T2 < T1)
        # * verify that index starts at 0 for both datasets

        # 1. select a random region
        region_idx = random.randint(0, len(self.hrit_buffer) - 1)

        # [T1, 11, 252, 252]        
        hrit_ds  = self.hrit_buffer[region_idx]['REFL-BT']

        # [T2, 1, 252 or 1512, 252 or 1512]
        opera_ds = self.opera_buffer[region_idx]['rates.crop']

        # calculate spatial center points of both datasets
        # it is given to us by the organizers that these points are indentical in a shared space
        W_opera, H_opera     = opera_ds.shape[2], opera_ds.shape[3]
        W_hrit,  H_hrit      = hrit_ds.shape[2] , hrit_ds.shape[3]
        c_x_opera, c_y_opera = math.floor(W_opera / 2), math.floor(H_opera / 2)
        c_x_hrit , c_y_hrit  = math.floor(W_hrit  / 2), math.floor(H_hrit  / 2)

        # choose a random (32x32) patch from the OPERA dataset
        opera_x1 = random.randint(0, W_opera - 32 - 1)
        opera_x2 = opera_x1 + 32
        opera_y1 = random.randint(0, H_opera - 32 - 1)
        opera_y2 = opera_y1 + 32

        assert opera_x2-opera_x1 == 32
        assert opera_y2-opera_y1 == 32

        # calculate the center of this random sample in OPERA space
        opera_sample_mp_x = math.floor((opera_x1 + opera_x2) / 2)
        opera_sample_mp_y = math.floor((opera_y1 + opera_y2) / 2)

        # project the OPERA sample center to HRIT space; both samples should share the same center
        hrit_sample_mp_x  = math.floor(((((opera_x2 - c_x_opera) / 6)  + c_x_hrit) + (((opera_x1 - c_x_opera) / 6) + c_x_hrit)) * (1/2))
        hrit_sample_mp_y  = math.floor(((((opera_y2 - c_y_opera) / 6)  + c_y_hrit) + (((opera_y1 - c_y_opera) / 6) + c_y_hrit)) * (1/2))

        # calculate the HRIT ROI relative to its center
        hrit_x1 = hrit_sample_mp_x - 16
        hrit_y1 = hrit_sample_mp_y - 16

        # HACK: manually check out of bounds
        if hrit_x1 < 0:
            hrit_x1 = 0
        if hrit_y1 < 0:
            hrit_y1 = 0
        if hrit_x1 > W_hrit - 32 - 1:
            hrit_x1 = W_hrit - 32 - 1
        if hrit_y1 > H_hrit - 32 - 1:
            hrit_y1 = H_hrit - 32 - 1

        hrit_x2 = hrit_x1 + 32
        hrit_y2 = hrit_y1 + 32

        assert hrit_x2-hrit_x1 == 32
        assert hrit_y2-hrit_y1 == 32

        # calculate starting temporal index of this sample
        T       = min(hrit_ds.shape[0], opera_ds.shape[0])
        start_T = random.randint(0, T - Sat2RadDataset.TOTAL_SAMPLE_LEN_T - 1)
        
        # input: 1H satellite data
        # [T=4, C=11, H=32, W=32]
        X      = hrit_ds[start_T:start_T+4, :, hrit_x1:hrit_x2, hrit_y1:hrit_y2].to_numpy()
        X      = torch.Tensor(X)

        # soft label: 4H proceeding rainfall
        # [T=16, C=1, H=32, W=32]
        y = opera_ds[start_T+4:start_T+20, :, opera_x1:opera_x2, opera_y1:opera_y2].to_numpy()
        y = torch.Tensor(y)
        
        # clip @0; it can't rain a negative amount; large negative values in our datasets
        y = y.clip(0)

        # fill `nan` values with 0.0
        y = y.nan_to_num()

        # wfc challenge #1 target
        # * average hourly cummulative rainfall
        # * individual feature maps (H, W) are 15 minute accumulated rainfall
        # * we derive regression targets: 
        # * * hourly rainfall    : H = (maps) * 4
        # * * avg hourly rainfall: H/16
        
        # [T, C=1, H, W] -> [T, H, W]
        y_reg = y.squeeze(1)
        y_reg = y_reg.mean(dim=(1, 2)) # "should be averaged over a 32×32 pixel area"
        y_reg = y_reg.sum()            # "summing the 16 slots"
        y_reg = y_reg / 4              #  "diving by 4"
        y_reg = y_reg.unsqueeze(0)     # -> [1]

        # [ds.min, ds.max] -> [0, 1]
        X_norm     = self.X_pre_proc(X)
        y_reg_norm = self.y_reg_pre_proc(y_reg)

        # # HACK
        # if y_reg < 0.1: self.get_item_train_val(index)

        # [1] -> [129]; get categorical label for classification/probabilistic task formulation
        y_one_hot_label = self.get_y_reg_norm_cat_label(y_reg_norm[0].item())

        return {
            "X"             : X,
            "X_norm"        : X_norm,
            "y"             : y,
            "y_reg"         : y_reg,
            "y_reg_norm"    : y_reg_norm,
            "y_reg_norm_oh" : y_one_hot_label
        }
    
    def get_item_test(self, index: int) -> dict:
        
        assert index < self.steps_per_epoch, f"Error: get item index {index} out of bounds"

        test_row   = self.test_df.iloc[index]
        case_id    = test_row["Case-id"]
        year       = test_row["year"]
        slot_start = test_row["slot-start"]
        slot_end   = test_row["slot-end"]
        x_tl       = test_row['x-top-left']
        x_br       = test_row['x-bottom-right']
        y_tl       = test_row['y-top-left']
        y_br       = test_row['y-bottom-right']
        tr_key     = test_row['key'].split(".")[0]

        key = ""
        for k in list(self.hrit_test_dict.keys()):
            if tr_key in k: key = k; break
        
        # self.all_fp_dict
        # [T1, 11, 252, 252]
        hrit_ds  = self.hrit_test_dict[key]['REFL-BT']

        # HACK: it looks like the organizers gave us one out of bounds slot
        if slot_end >= hrit_ds.shape[0]:
            slot_end   = hrit_ds.shape[0]
            slot_start = slot_end - 4

        # input: 1H satellite data
        X = hrit_ds[slot_start:slot_end, ...].to_numpy()
        X = torch.Tensor(X)

        T, C, H, W  = X.shape
        
        # 1 HRIT PX = 6x6 OPERA pixels
        x_tl_scaled = int((x_tl / 6).item())
        x_br_scaled = int((x_br / 6).item())
        y_tl_scaled = int((y_tl / 6).item())
        y_br_scaled = int((y_br / 6).item())

        # scale window sizes to (32, 32); center crop around radar ROI
        curr_x_range = x_br_scaled - x_tl_scaled
        curr_x_diff  = (32 - curr_x_range)
        x_tl_scaled  -= (curr_x_diff // 2)
        x_br_scaled  += (32 - (x_br_scaled - x_tl_scaled))

        curr_y_range = y_br_scaled - y_tl_scaled
        curr_y_diff  = (32 - curr_y_range)
        y_tl_scaled  -= (curr_y_diff // 2)
        y_br_scaled  += (32 - (y_br_scaled - y_tl_scaled))

        # --- some samples spill out of bounds
        if x_br_scaled >= W:
            x_tl_scaled = W - 32 - 1
            x_br_scaled = W - 1

        if y_br_scaled >= H:
            y_tl_scaled = H - 32 - 1
            y_br_scaled = H - 1

        if x_tl_scaled < 0:
            x_tl_scaled = 0
            x_br_scaled = 32

        if y_tl_scaled < 0:
            y_tl_scaled = 0
            y_br_scaled = 32

        try:
            assert x_tl_scaled >= 0 and x_br_scaled < W
            assert y_tl_scaled >= 0 and y_br_scaled < H
        except:
            breakpoint()

        assert x_br_scaled - x_tl_scaled == 32
        assert y_br_scaled - y_tl_scaled == 32
        
        # [ds.min, ds.max] -> [0, 1]
        X_norm     = self.X_pre_proc(X)
        X_norm_32  = X_norm[:, :, x_tl_scaled:x_br_scaled, y_tl_scaled:y_br_scaled]

        return {
            "X"             : X,
            "X_norm"        : X_norm,
            "X_norm_32"     : X_norm_32,
            "Case-id"       : case_id,
            "year"          : year,
            "slot-start"    : slot_start,
            "slot-end"      : slot_end,
            "x-top-left"    : x_tl,
            "x-bottom-right": x_br,
            "y-top-left"    : y_tl,
            "y-bottom-right": y_br,
            "file_name"     : key,
        }

    def __getitem__(self, index: int) -> dict:

        if self.split in ["train", "val"]:
            return self.get_item_train_val(index)
        elif self.split == "test":
            return self.get_item_test(index)


if __name__ == "__main__":
    """
    """

    import torch

    ds = Sat2RadDataset(split="train", n_classes=64)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0,)