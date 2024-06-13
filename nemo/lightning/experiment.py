from dataclasses import dataclass
from typing import Optional, List, Union
import sys
import os
import time
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as PTLModelCheckpoint
import lightning_fabric as fl
from nemo.constants import NEMO_ENV_VARNAME_TESTING, NEMO_ENV_VARNAME_VERSION
from nemo.utils import logging
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.exp_manager import check_explicit_log_dir
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.app_state import AppState
from nemo.utils.mcore_logger import add_handlers_to_mcore_logger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint


@dataclass
class Experiment:
    name: str
    dir: Optional[str] = None
    explicit_log_dir: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: bool = True
    log_local_rank_0_only: bool = False
    log_global_rank_0_only: bool = False
    files_to_copy: Optional[List[str]] = None
    update_logger_directory: bool = True
    
    def __post_init__(self):
        if self.log_local_rank_0_only is True and self.log_global_rank_0_only is True:
            raise ValueError(
                f"Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
            )
    
    def setup(self, trainer: Union[pl.Trainer, fl.Fabric], resume_if_exists: bool = False):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = trainer.node_rank * trainer.world_size + local_rank
        logging.rank = global_rank

        if self.explicit_log_dir and isinstance(trainer, pl.Trainer):  # If explicit log_dir was passed, short circuit
            return check_explicit_log_dir(trainer, self.explicit_log_dir, self.dir, self.name, self.version)

        # Default dir to ./nemo_experiments if None was passed
        _dir = self.dir
        if self.dir is None:
            _dir = str(Path.cwd() / 'nemo_experiments')

        if not self.name:
            self.name = "default"
            
        if isinstance(trainer, pl.Trainer) and trainer.logger is not None:
            if self.update_logger_directory:
                logging.warning(
                    f'"update_logger_directory" is True. Overwriting logger "save_dir" to {_dir} and "name" to {self.name}'
                )
                trainer.logger._root_dir = _dir
                trainer.logger._name = self.name
            
        version = self.version or os.environ.get(NEMO_ENV_VARNAME_VERSION, None)
        if is_global_rank_zero():
            if self.use_datetime_version:
                version = time.strftime('%Y-%m-%d_%H-%M-%S')
        if resume_if_exists:
            logging.warning(
                "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
            )
            version = None  
        if version:
            if is_global_rank_zero():
                os.environ[NEMO_ENV_VARNAME_VERSION] = version

        log_dir = Path(_dir) / Path(str(self.name)) / Path("" if version is None else str(version))
        # update app_state with log_dir, exp_dir, etc
        app_state = AppState()
        app_state.log_dir = log_dir
        app_state.exp_dir = _dir
        app_state.name = self.name
        app_state.version = version
        
        os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
        logging.info(f'Experiments will be logged at {log_dir}')
        
        if isinstance(trainer, pl.Trainer): 
            for callback in trainer.callbacks:
                if isinstance(callback, PTLModelCheckpoint):
                    ## TODO: make configurable
                    callback.dirpath = Path(log_dir / "checkpoints")  # app_state.exp_dir
                    if callback.filename is None:
                        callback.filename = f'{name}--{{{callback.monitor}:.4f}}-{{epoch}}'
                    if callback.prefix is None:
                        callback.prefix = name
                    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback.filename + '-last'

        
        # This is set if the env var NEMO_TESTING is set to True.
        nemo_testing = get_envbool(NEMO_ENV_VARNAME_TESTING, False)

        # Handle logging to file
        log_file = log_dir / f'nemo_log_globalrank-{global_rank}_localrank-{local_rank}.txt'
        if self.log_local_rank_0_only is True and not nemo_testing:
            if local_rank == 0:
                logging.add_file_handler(log_file)
        elif self.log_global_rank_0_only is True and not nemo_testing:
            if global_rank == 0:
                logging.add_file_handler(log_file)
        else:
            # Logs on all ranks.
            logging.add_file_handler(log_file)

        add_handlers_to_mcore_logger()

        app_state.files_to_copy = self.files_to_copy
        app_state.cmd_args = sys.argv
        
        return app_state

    def teardown(self):
        pass