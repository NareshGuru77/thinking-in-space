import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer,  AutoModel, AutoImageProcessor


@register_model("EMU3")
class BAAIVideoXL2(lmms):
    """
    EMU3 Model
    https://github.com/baaivision/Emu3
    """

    def __init__(
        self,
        pretrained: str = "BAAI/Emu3-Chat",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        # load model
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained, trust_remote_code=True, device_map=device,quantization_config=None,
            attn_implementation="sdpa", torch_dtype=torch.float16, low_cpu_mem_usage=True) # sdpa
        self.model.config.enable_chunk_prefill = True
        prefill_config = {
            'chunk_prefill_mode': 'streaming',
            'chunk_size': 4,
            'step_size': 1,
            'offload': True,
            'chunk_size_for_vision_tower': 24,
        }
        self.model.config.prefill_config = prefill_config

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        # self.tokenizer.padding_side = "left"
        # self.tokenizer.pad_token_id = self.tokenizer.eod_id

        # TODO: check prompt template
        self.prompt = "<img>{}</img>{}"
        self._config = self._model.config
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            # self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eod_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        return res

    @staticmethod
    def flatten(inp):
        new_list = []
        for i in inp:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            assert len(visuals) == 1
            video_path = visuals[0]
            question1 = contexts

            # params
            max_num_frames = 1300
            sample_fps = None  # uniform sampling
            max_sample_fps = None

            gen_kwargs.update({"do_sample": False, "temperature": 0.01, "top_p": 0.001, "num_beams": 1,
                               "use_cache": True, "max_new_tokens": 128})
            with torch.inference_mode():
                response = self.model.chat(video_path, self.tokenizer, question1, chat_history=None,
                                           return_history=False, max_num_frames=max_num_frames, sample_fps=sample_fps,
                                           max_sample_fps=max_sample_fps, generation_config=gen_kwargs)

            peak_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Memory Peak: {peak_memory_allocated / (1024 ** 3):.2f} GB")
            print(response)
            res.append(response)
            pbar.update(1)

        pbar.close()
        return res
