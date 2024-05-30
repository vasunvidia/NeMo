# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


# To suppress BF16 compile related issue in the CI runs with turing/V100
import time
import torch
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from megatron.core import parallel_state
from megatron.core.transformer.custom_layers.transformer_engine import TEDelayedScaling
from pytorch_lightning import Callback
from transformer_engine.common import recipe
from transformer_engine.pytorch import make_graphed_callables

from apex.transformer.pipeline_parallel.utils import get_num_microbatches
torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)

class CudaGraphCallback(Callback):
    def __init__(self, cfg):
        super().__init__()

    def get_microbatch_schedule(self, num_microbatches, num_model_chunks):
        schedule = []
        pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

        def get_model_chunk_id(microbatch_id, forward):
            """Helper method to get the model chunk ID given the iteration number."""
            microbatch_id_in_group = microbatch_id % (pipeline_model_parallel_size * num_model_chunks)
            model_chunk_id = microbatch_id_in_group // pipeline_model_parallel_size
            if not forward:
                model_chunk_id = num_model_chunks - model_chunk_id - 1
            return model_chunk_id

        if pipeline_model_parallel_size > 1:
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                #forward_backward_pipelining_with_interleaving
                total_num_microbatches = num_microbatches * num_model_chunks
                if num_microbatches == pipeline_model_parallel_size:
                    num_warmup_microbatches = total_num_microbatches
                else:
                    num_warmup_microbatches = (pipeline_model_parallel_size - pipeline_parallel_rank - 1) * 2
                    num_warmup_microbatches += (num_model_chunks - 1) * pipeline_model_parallel_size
                    num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
                num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
                for k in range(num_warmup_microbatches):
                    cur_model_chunk_id = get_model_chunk_id(k, forward=True)
                    schedule.append(cur_model_chunk_id+1)
                for k in range(num_microbatches_remaining):
                    forward_k = k + num_warmup_microbatches
                    cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                    schedule.append(cur_model_chunk_id+1)
                    backward_model_chunk_id = get_model_chunk_id(k, forward=False)
                    schedule.append(-backward_model_chunk_id-1)
                for k in range(num_microbatches_remaining, total_num_microbatches):
                    backward_model_chunk_id = get_model_chunk_id(k, forward=False)
                    schedule.append(-backward_model_chunk_id-1)
            else:
                #forward_backward_pipelining_without_interleaving
                num_warmup_microbatches = (
                    pipeline_model_parallel_size
                    - pipeline_parallel_rank
                    - 1
                )
                num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
                num_microbatches_remaining = num_microbatches - num_warmup_microbatches
                schedule = [1]*num_warmup_microbatches + [1,-1]*num_microbatches_remaining + [-1]*num_warmup_microbatches
        else:
            #forward_backward_no_pipelining
            schedule = [1, -1]
        return schedule

    def cuda_graph_capture(self, trainer, cfg):
        trainer.model._optimizer.zero_grad()
        torch.distributed.barrier()
        start = time.time()
        torch.cuda.set_stream(torch.cuda.default_stream())
        schedule = self.get_microbatch_schedule(get_num_microbatches(), len(trainer.model.model) if isinstance(trainer.model.model, list) else None)
        print (f'SCHEDULE {schedule}')

        callables = []
        model_chunks = trainer.model.model if isinstance(trainer.model.model, list) else [trainer.model.model]
        # Collect layers for CUDA graph capture
        for m_no, model in enumerate(model_chunks):
            for l_no, layer in enumerate(model.module.decoder.layers):
                callables.append(layer)
        print (f'#layers {len(callables)}')
        if trainer.model.initialize_ub:
            trainer.model.initialize_ub_func()
            trainer.model.initialize_ub = False
        sequence_parallel = cfg.sequence_parallel
        tensor_model_parallel_size = cfg.tensor_model_parallel_size
        micro_batch_size = cfg.micro_batch_size
        slen = cfg.encoder_seq_length // tensor_model_parallel_size if sequence_parallel else cfg.encoder_seq_length

        fp8_recipe = None
        if cfg.fp8:
            if cfg.fp8_e4m3:
                fp8_format = recipe.Format.E4M3
            elif cfg.fp8_hybrid:
                fp8_format = recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=trainer.model.model.module.decoder.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, False),
            )

        sample_args = []
        for m_no, model in enumerate(model_chunks):
            for l_no, layer in enumerate(model.module.decoder.layers):
                for b in range(get_num_microbatches()):
                    graph_input = (torch.ones((slen, micro_batch_size, cfg.hidden_size), dtype=torch.bfloat16, requires_grad=True, device='cuda'),)
                    sample_args.append(graph_input)
        graphs = make_graphed_callables(tuple(callables), tuple(sample_args), _order=schedule, allow_unused_input=True, fp8_enabled=cfg.fp8, fp8_recipe=fp8_recipe if cfg.fp8 else None, fp8_weight_caching=True)

        for m_no, model in enumerate(model_chunks):
            for l_no, layer in enumerate(model.module.decoder.layers):
                model.module.decoder.cuda_graphs[l_no] = []
                for b in range(get_num_microbatches()):
                    model.module.decoder.cuda_graphs[l_no].append(graphs[m_no * get_num_microbatches() * len(model.module.decoder.layers) + b * len(model.module.decoder.layers) + l_no])

        torch.distributed.barrier()
        print (f'Time spent in cuda_graph_capture: {time.time() - start}s')

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.cfg.enable_cuda_graph:
            self.cuda_graph_capture (trainer, pl_module.cfg)


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    cg_callback = CudaGraphCallback(cfg)
    trainer = MegatronTrainerBuilder(cfg).create_trainer(callbacks=[cg_callback])
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronGPTModel(cfg.model, trainer)

    s = torch.cuda.Stream()
    torch.cuda.set_stream(s)
    trainer.fit(model)


if __name__ == '__main__':
    main()
