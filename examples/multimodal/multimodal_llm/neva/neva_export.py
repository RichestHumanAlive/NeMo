# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo.core.config import hydra_runner
from nemo.export.multimodal_exporter import MultimodalExporter


@hydra_runner(config_path='conf', config_name='neva_export')
def main(cfg):
    exporter = MultimodalExporter(model_dir=cfg.infer.output_dir, load_model=False)
    exporter.export(
        visual_checkpoint_path=cfg.model.visual_model_path,
        llm_checkpoint_path=cfg.model.llm_model_path,
        model_type=cfg.model.type,
        llm_model_type=cfg.model.llm_model_type,
        tensor_parallel_size=cfg.infer.llm.tensor_parallelism,
        max_input_len=cfg.infer.llm.max_input_len,
        max_output_len=cfg.infer.llm.max_output_len,
        max_batch_size=cfg.infer.llm.max_batch_size,
        max_multimodal_len=cfg.infer.llm.max_multimodal_len,
        dtype=cfg.model.precision,
        load_model=False,
    )


if __name__ == '__main__':
    main()
