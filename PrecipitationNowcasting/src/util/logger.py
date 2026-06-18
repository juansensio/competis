import os
import math
import torch
import datetime
import yaml
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

EXPS_DIR = "/playpen-ssd/levi/w4c/w4c-25/__exps__"
FIGURES_DIR_NAME = "figures"
RESULTS_CSV_NAME = "results.csv"


class ExperimentLogger:
    """
    A flexible logger used to record and organize experimental runs.
    """

    def __init__(
        self,
        train_config_dict: dict,
        model_config_dict: Optional[dict] = None,
        root: str = EXPS_DIR,
        exp_name: Optional[str] = "",
        log_interval: int = 100,
        enable_tensorboard=False,
        enable_wandb=False,
        wandb_proj_name: Optional[str] = None,
    ) -> None:
        """
        :param config_fp:           path to a `.yaml` config file containing all hps
        :param root:                path to top experiment dir
        :param exp_name:            name of the experiment
        :param log_interval:        how often to write log results to .csv file
        :param enable_tensorboard:  flag to enable tensorboard logging
        :param enable_wandb:        flag to enable W&B logging [NOT SUPPORTED]
        :param wandb_project_name:  name of W&B project (e.g. "my-project")
        """

        self.config: dict = train_config_dict
        self.model_config: Optional[dict] = model_config_dict
        self.exp_name: str = exp_name

        self.results = pd.DataFrame()
        self.log_buffer = []
        self.log_interval: int = log_interval
        self.log_counter = 0

        self.root: str = root
        self.exp_dir: Optional[str] = None

        # ---- tensorboard support ----
        self.enable_tensorboard: bool = enable_tensorboard
        self.results_out_path: Optional[str] = None
        self.summary_writer: Optional[SummaryWriter] = None

        # ---- wandb support ----
        self.enable_wandb = enable_wandb
        if self.enable_wandb == True:
            assert (
                wandb_proj_name != None
            ), f"Error: must provide a valid name for wandb_proj_name"
        self.wandb_proj_name = wandb_proj_name
        self.wandb_run = None

        self._setup_exp_dir()

    def _flush(self) -> None:
        if not self.log_buffer:
            return
        # init new results table from buffer
        _logs = pd.DataFrame.from_records(self.log_buffer)

        # append results in memory
        self.results = pd.concat([self.results, _logs], ignore_index=True)
        if not os.path.exists(self.results_out_path):
            # create new file
            _logs.to_csv(self.results_out_path, index=False)
        else:
            # write to csv in append mode
            _logs.to_csv(self.results_out_path, mode="a", header=False, index=False)
        self.log_buffer = []

    def _update_csv(self) -> None:
        self.results.to_csv(self.results_out_path, index=False)

    def _setup_exp_dir(self) -> None:

        # get date and time as a string
        date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir_name   = date_time_str + "_" + self.exp_name
        exp_out_dir   = os.path.join(self.root, subdir_name)
        self.exp_dir  = exp_out_dir

        # make new subdir if needed
        os.makedirs(exp_out_dir, exist_ok=True)

        # save config in subdir
        config_save_fp = os.path.join(exp_out_dir, "config.yaml")
        with open(config_save_fp, "w") as f:
            yaml.dump(self.config, f, indent=4)

        # path to results csv file
        self.results_out_path = os.path.join(exp_out_dir, RESULTS_CSV_NAME)

        # optional: create a tensorboard writer object
        if self.enable_tensorboard:
            tb_log_dir = os.path.join(self.exp_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=tb_log_dir)

        # optional: create a wandb run
        if self.enable_wandb:
            with open(config_save_fp, "r") as f:
                config_dict = yaml.safe_load(f)
            wandb.init(
                project=self.wandb_proj_name,
                name=self.exp_name,
                config=config_dict,
                dir=self.exp_dir,
            )
            self.wandb_run = wandb.run

        model_config_save_fp = os.path.join(exp_out_dir, "model.yaml")

        # save a copy of the model config to the exp dir
        with open(model_config_save_fp, "w") as f:
            yaml.dump(self.model_config, f, indent=4)

        # TODO: this looks hacky; remove
        self.config_fp = config_save_fp

    def add_result_column(self, name: str) -> None:
        self.results[name] = None
        # HACK: just ignore this for now
        # self._update_csv()

    def add_result_columns(self, names: List[str]) -> None:
        for name in names:
            self.add_result_column(name)
        # HACK: just ignore this for now
        # self._update_csv()

    def log(self, **kwargs) -> None:
        """
        Log a dictionary of items to a csv.
        """

        # append results to mem
        self.log_buffer.append(kwargs)
        self.log_counter += 1

        # write to out
        if len(self.log_buffer) >= self.log_interval:
            self._flush()

        # optional: log -> tensorboard
        if self.enable_tensorboard:
            if step is None:
                step = self.log_counter
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    self.summary_writer.add_scalar(k, v, step)

        # optional: log -> wandb
        if self.enable_wandb:
            step = self.log_counter
            wandb_dict = {
                k: v for k, v in kwargs.items() if isinstance(v, (int, float))
            }
            wandb.log(wandb_dict, step=step)

    def save_weights(
        self,
        model: torch.nn.Module,
        name: str = "best",
    ) -> None:
        """
        Save model weights of a `torch.nn.Module` object to the current exp dir.

        :param model: model to save
        """

        out_fp = Path(self.exp_dir) / Path(f"{name}.pth")
        torch.save(model, str(out_fp))


if __name__ == "__main__":
    pass