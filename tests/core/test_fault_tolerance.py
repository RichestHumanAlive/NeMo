# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import pytest

from nemo.utils.callbacks.fault_tolerance import _TrainingStateMachine


class TestFaultTolerance:
    @pytest.mark.unit
    def test_training_ended_ok(self):
        # Training ended if there were no training iterations nor error
        sm = _TrainingStateMachine()
        assert sm.is_training_completed is False
        sm.on_eval_heartbeat()
        sm.on_eval_heartbeat()
        sm.on_fit_end()
        assert sm.is_training_completed is True

    @pytest.mark.unit
    def test_training_ended_false_00(self):
        # Training is not completed if there was an error
        sm = _TrainingStateMachine()
        assert sm.is_training_completed is False
        sm.on_exception()
        sm.on_fit_end()
        assert sm.is_training_completed is False

    @pytest.mark.unit
    def test_training_ended_false_01(self):
        # Training is not completed if there were some training iterations
        # (we detect last "empty run" to determine that the training is completed)
        sm = _TrainingStateMachine()
        assert sm.is_training_completed is False
        sm.on_train_heartbeat()
        sm.on_train_heartbeat()
        sm.on_fit_end()
        assert sm.is_training_completed is False

    @pytest.mark.unit
    def test_training_can_upd_timeouts(self):
        sm = _TrainingStateMachine()
        assert sm.can_update_timeouts is False
        sm.on_load_checkpoint()
        sm.on_save_checkpoint()
        assert sm.can_update_timeouts is False
        sm.on_train_heartbeat()
        sm.on_train_heartbeat()
        sm.on_train_heartbeat()
        # cant save, as mid-epoch checkpoint saving not seen
        assert sm.can_update_timeouts is False
        sm.on_save_checkpoint()
        # now checkpointing was done, but need following heartbeat
        assert sm.can_update_timeouts is False
        sm.on_train_heartbeat()
        assert sm.can_update_timeouts is True
        sm.on_timeouts_updated()
        # on_timeouts_updated() resets the flag
        # should not back to True, only one update per run is allowed
        assert sm.can_update_timeouts is False
        sm.on_train_heartbeat()
        sm.on_save_checkpoint()
        sm.on_train_heartbeat()
        assert sm.can_update_timeouts is False
