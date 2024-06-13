# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import random
import signal
import sys
import threading
import time

import torch
from pytorch_lightning.callbacks import Callback

from nemo.utils import logging

try:
    import fault_tolerance as ft
    HAVE_FT = True
except (ImportError, ModuleNotFoundError):
    HAVE_FT = False

class _TrainingStateMachine:
    """
    This class encapsulates logic for determining when:
    - training is finished successfully (`.is_training_completed` property)
    - FT timeouts can be updated (`.can_update_timeouts` property)

    `on_ ...` methods update the state and should be called from the corresponding PTL callback methods.
    """

    MIN_ITERS_FOR_TIMEOUT_UPDATE = 2

    def __init__(self):
        self.num_tr_iters_total = 0
        self.num_tr_iter_at_last_save = None
        self.seen_checkpointing = False
        self.loaded_checkpoint = False
        self.caught_exception = False
        self.training_ended = False
        self.timeouts_updated = False

    def on_setup(self):
        pass

    def on_fit_end(self):
        self.training_ended = True

    def on_load_checkpoint(self):
        self.loaded_checkpoint = True

    def on_save_checkpoint(self):
        self.num_tr_iter_at_last_save = self.num_tr_iters_total

    def on_train_heartbeat(self):
        self.num_tr_iters_total += 1
        if not self.seen_checkpointing and self.num_tr_iter_at_last_save is not None:
            # detect mid-epoch checkpointing that makes hearbeat interval longer
            iters_pre_save = self.num_tr_iter_at_last_save
            iters_post_save = self.num_tr_iters_total - self.num_tr_iter_at_last_save
            self.seen_checkpointing = iters_pre_save > 0 and iters_post_save > 0

    def on_eval_heartbeat(self):
        pass

    def on_exception(self):
        self.caught_exception = True

    def on_timeouts_updated(self):
        self.timeouts_updated = True

    @property
    def is_training_completed(self) -> bool:
        """
        Returns True if training is finished sucessfuly, due to the number of iters or time limit.
        """
        # if exiting AND just 0 or 1 training iterations were made AND error is not set,
        # assume training has finished successfully and there is nothing else to do.
        # 1 iteration is made when we run a workload for which 'max_time' elapsed,
        # so need to handle that special case.
        # NOTE: this detection mechanism is sligtly wasteful, as it requires final "empty run"
        return self.training_ended and self.num_tr_iters_total <= 1 and not self.caught_exception

    @property
    def can_update_timeouts(self) -> bool:
        """
        Returns True if new timeouts can be computed.
        `.on_timeouts_updated()` resets this property back to False.
        """
        if self.timeouts_updated:
            # timeouts are updated at most once per training run
            return False
        if self.num_tr_iters_total < self.MIN_ITERS_FOR_TIMEOUT_UPDATE:
            # need a few training iters
            return False
        # check if there was checkoint loading and saving
        # this makes heartbeat iterval longer than usual.
        return self.loaded_checkpoint and self.seen_checkpointing


class FaultToleranceCallback(Callback):
    """
    FaultToleranceCallback class is a Torch Lightning callback that handles fault tolerance.
    """

    STATE_DICT_KEY = "fault_tolerance"

    def __init__(self, autoresume=False, calculate_timeouts=False, simulated_fault_params=None):
        self.fault_tol_client = None
        self.autoresume = autoresume
        self.calculate_timeouts = calculate_timeouts
        self.simulated_fault_params = simulated_fault_params
        self.state_machine = _TrainingStateMachine()
        self._verify_env()

    def _verify_env(self):
        if self.autoresume and not os.environ.get('FAULT_TOL_FINISHED_FLAG_FILE', ''):
            raise RuntimeError(
                "'FAULT_TOL_FINISHED_FLAG_FILE' env variable is not set. Was this job launched with FT launcher?"
            )
        if not HAVE_FT:
            raise RuntimeError(
                "fault_tolerance package is required to use FaultToleranceCallback."
            )

    def _setup_fault_tolerance(self):

        self.fault_tol_client = ft.RankMonitorClient()

        # Initialize the FT client
        # Disable in-memory checkpoint manager, as it is not implemented yet.
        self.fault_tol_client.init_workload_monitoring(chkpt_manager=ft.CheckpointManagerType.NONE)

        ft_timeouts = self.fault_tol_client.timeouts
        if ft_timeouts.are_valid:
            logging.info(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")
        else:
            if self.calculate_timeouts:
                logging.info("Fault tolerance client initialized. Timeouts: not calculated yet.")
            else:
                raise RuntimeError(
                    "Fault tolerance doesn't have valid timeouts set and 'calculate_timeouts' is False."
                )
        # Simulated fault for testing/debug purposes
        if self.simulated_fault_params:
            self._setup_simulated_fault()

    def setup(self, trainer, pl_module, stage):
        self.state_machine.on_setup()
        if self.fault_tol_client is None:
            self._setup_fault_tolerance()

    def on_fit_end(self, trainer, pl_module):
        self.state_machine.on_fit_end()
        if trainer.global_rank == 0:
            if self.autoresume and self.state_machine.is_training_completed:
                self._create_finished_flag_file()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.state_machine.on_load_checkpoint()
        loaded_ft_state_dict = checkpoint.get(self.STATE_DICT_KEY, None)
        if loaded_ft_state_dict:
            self.fault_tol_client.load_state_dict(loaded_ft_state_dict)
            ft_timeouts = self.fault_tol_client.timeouts
            logging.info(f"Fault tolerance timeouts loaded from chkpt: {ft_timeouts}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.state_machine.on_save_checkpoint()
        if trainer.global_rank == 0:
            # FT state is the same on all ranks, so we can save it only on rank 0
            checkpoint[self.STATE_DICT_KEY] = self.fault_tol_client.state_dict()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.state_machine.on_train_heartbeat()
        self.fault_tol_client.send_heartbeat()
        if self.calculate_timeouts and self.state_machine.can_update_timeouts:
            self.fault_tol_client.calculate_and_set_timeouts()
            self.state_machine.on_timeouts_updated()
            logging.info(f'Updated FT timeouts. New values: {self.fault_tol_client.timeouts}')
            # verify that can_update_timeouts is cleared
            assert not self.state_machine.can_update_timeouts

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.state_machine.on_eval_heartbeat()
        self.fault_tol_client.send_heartbeat()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.state_machine.on_eval_heartbeat()
        self.fault_tol_client.send_heartbeat()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.state_machine.on_eval_heartbeat()
        self.fault_tol_client.send_heartbeat()

    def on_exception(self, trainer, pl_module, exception):
        self.state_machine.on_exception()

    def _create_finished_flag_file(self):
        try:
            flag_file_path = pathlib.Path(os.environ["FAULT_TOL_FINISHED_FLAG_FILE"])
            flag_file_path.touch()
        except Exception as e:
            logging.error(f"_create_finished_flag_file exception: {e}")

    def _setup_simulated_fault(self):

        # TODO: this if for testing only, should be removed in release version

        rng = random.Random()

        fault_desc = self.simulated_fault_params

        logging.info(f"Initializing simulated fault: {fault_desc}")

        rank = torch.distributed.get_rank()
        rand_rank = rng.randint(0, torch.distributed.get_world_size() - 1)
        rank_to_fail = int(fault_desc.get('rank_to_fail', rand_rank))
        rank_to_fail = torch.tensor([rank_to_fail], device=torch.cuda.current_device())
        torch.distributed.broadcast(rank_to_fail, 0)
        rank_to_fail = int(rank_to_fail.item())

        if rank != rank_to_fail:
            return

        fault_type = fault_desc.fault_type
        if fault_type == 'random':
            fault_type = rng.choice(['rank_killed', 'rank_hung'])

        if fault_type == 'rank_killed':
            target_pid = os.getpid()
        elif fault_type == 'rank_hung':
            target_pid = os.getpid()
        else:
            raise Exception(f"Unknown fault type {fault_type}")

        delay = fault_desc.base_delay + fault_desc.get('rand_delay', 0) * rng.random()

        logging.info(f"Selected fault={fault_type}; target rank={rank_to_fail}; delay={delay}")

        def __fault_thread():
            time.sleep(delay)
            for of in [sys.stdout, sys.stderr]:
                print(f"\n####\nSimulating fault: {fault_type}; rank to fail: {rank_to_fail}\n#####\n", file=of)
            if fault_type == 'rank_hung':
                os.kill(target_pid, signal.SIGSTOP)
            else:
                os.kill(target_pid, signal.SIGKILL)

        fault_sim_thread = threading.Thread(target=__fault_thread)
        fault_sim_thread.daemon = True
        fault_sim_thread.start()
