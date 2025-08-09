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
from transformers import AutoModelForCausalLM, AutoTokenizer


@register_model("VideoChatQwen")
class VideoChatQwen(lmms):

    def __init__(
        self,
        pretrained: str = "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448",
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
            pretrained, trust_remote_code=True).to(torch.bfloat16).cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        # self.tokenizer.padding_side = "left"
        # self.tokenizer.pad_token_id = self.tokenizer.eod_id

        mm_llm_compress = False  # use the global compress or not
        if mm_llm_compress:
            self.model.config.mm_llm_compress = True
            self.model.config.llm_compress_type = "uniform0_attention"
            self.model.config.llm_compress_layer_list = [4, 18]
            self.model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
        else:
            self.model.config.mm_llm_compress = False

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

            # evaluation setting
            max_num_frames = 512
            generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1024,
                top_p=0.1,
                num_beams=1
            )

            with torch.inference_mode():
                # single-turn conversation
                response, chat_history = self.model.chat(
                    video_path=video_path, tokenizer=self.tokenizer, user_prompt=question1,
                    return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)
                print(response)

            res.append(response)
            pbar.update(1)

        pbar.close()
        return res
