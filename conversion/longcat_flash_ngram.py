from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Iterable

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("LongcatFlashNgramForCausalLM")
@ModelBase.example("meituan-longcat/LongCat-Flash-Lite")
class LongcatFlashNgramModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LONGCAT_FLASH_NGRAM
    supports_mtp_export = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # each HF layer holds two attention blocks, stored as two GGUF blocks; the NextN/MTP blocks follow them
        self.base_block_count = 2 * int(self.find_hparam(["num_layers", "num_hidden_layers", "n_layers", "n_layer"]))

        mtp_ids = {int(m.group(1)) for name in self.model_tensors if (m := re.match(r"model\.mtp\.layers\.(\d+)\.", name))}
        if mtp_ids != set(range(len(mtp_ids))):
            raise ValueError(f"MTP layer ids must be contiguous from 0, found {sorted(mtp_ids)}")
        if len(mtp_ids) > 1:
            raise ValueError(f"only one MTP layer is supported, found {len(mtp_ids)}")
        self.mtp_count = len(mtp_ids)

        self.block_count = self.base_block_count + self.mtp_count
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        is_mtp = name.startswith("model.mtp.")
        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            return None

        return name, gen

    def set_vocab(self):
        self._set_vocab_gpt2()

        # the tokenizer config appends EOS automatically, which llama.cpp prompt processing must not do
        self.gguf_writer.add_add_eos_token(False)

    def set_gguf_parameters(self):
        hparams = self.hparams

        # MLA is stored as MQA with one compressed KV head
        hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()

        if self.mtp_count > 0:
            self.gguf_writer.add_nextn_predict_layers(self.mtp_count)

        self.gguf_writer.add_feed_forward_length(hparams["ffn_hidden_size"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if hparams.get("q_lora_rank") is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])

        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(hparams["kv_lora_rank"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(hparams["expert_ffn_hidden_size"])
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_used_count(hparams["moe_topk"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_zero_count(hparams["zero_expert_num"])

        self.gguf_writer.add_ngram_neighbor_num(hparams["emb_neighbor_num"])
        self.gguf_writer.add_ngram_split_num(hparams["emb_split_num"])
        self.gguf_writer.add_ngram_vocab_size_ratio(hparams["ngram_vocab_size_ratio"])

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if (rope_mscale_all := self.rope_parameters.get("mscale_all_dim")) is not None:
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * rope_mscale_all)

    _experts: list[dict[str, Tensor]] | None = None

    def _remap_double_block(self, name: str, bid: int | None) -> tuple[str, int | None]:
        # the two attention blocks of HF layer i become GGUF blocks 2*i and 2*i + 1; the MoE and the first dense MLP
        # (stored as the shared expert) go with the even block, the second dense MLP with the odd block
        if m := re.match(r"model\.layers\.(\d+)\.self_attn\.(\d+)\.(.*)", name):
            new_bid = 2 * int(m.group(1)) + int(m.group(2))
            return f"model.layers.{new_bid}.self_attn.{m.group(3)}", new_bid

        if m := re.match(r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.(\d+)\.(.*)", name):
            new_bid = 2 * int(m.group(1)) + int(m.group(3))
            return f"model.layers.{new_bid}.{m.group(2)}.{m.group(4)}", new_bid

        if m := re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)", name):
            new_bid = 2 * int(m.group(1))
            return f"model.layers.{new_bid}.mlp.experts.{m.group(2)}.{m.group(3)}", new_bid

        if m := re.match(r"model\.layers\.(\d+)\.mlp\.router\.classifier\.(.*)", name):
            new_bid = 2 * int(m.group(1))
            return f"model.layers.{new_bid}.mlp.gate.{m.group(2)}", new_bid

        if m := re.match(r"model\.layers\.(\d+)\.mlp\.router\.e_score_correction(?:_bias|\.bias)", name):
            new_bid = 2 * int(m.group(1))
            return f"model.layers.{new_bid}.mlp.gate.e_score_correction.bias", new_bid

        if m := re.match(r"model\.layers\.(\d+)\.mlps\.0\.(.*)", name):
            new_bid = 2 * int(m.group(1))
            return f"model.layers.{new_bid}.mlp.shared_experts.{m.group(2)}", new_bid

        if m := re.match(r"model\.layers\.(\d+)\.mlps\.1\.(.*)", name):
            new_bid = 2 * int(m.group(1)) + 1
            return f"model.layers.{new_bid}.mlp.{m.group(2)}", new_bid

        return name, bid

    def _remap_mtp_tensor(self, name: str) -> tuple[str, int]:
        # the MTP embedding and final norm live outside model.mtp.layers.0 but belong to the first NextN block
        mtp_bid = self.base_block_count
        if name == "model.mtp.embed_tokens.weight":
            return f"model.layers.{mtp_bid}.embed_tokens.weight", mtp_bid
        if name == "model.mtp.norm.weight":
            return f"model.layers.{mtp_bid}.shared_head.norm.weight", mtp_bid

        m = re.match(r"model\.mtp\.layers\.(\d+)\.(.*)", name)
        if not m:
            raise ValueError(f"unsupported MTP tensor: {name}")

        bid = self.base_block_count + int(m.group(1))
        suffix = m.group(2)

        # the MTP FFN is wrapped in transformer_layer and the two NextN norms in an extra .m module
        if suffix.startswith("transformer_layer.mlp."):
            suffix = suffix.removeprefix("transformer_layer.")
        elif suffix in ("enorm.m.weight", "hnorm.m.weight"):
            suffix = suffix.replace(".m.", ".")

        return f"model.layers.{bid}.{suffix}", bid

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.mtp."):
            name, bid = self._remap_mtp_tensor(name)
        elif "ngram_embeddings" in name:
            yield from super().modify_tensors(data_torch, name, bid)
            return
        else:
            name, bid = self._remap_double_block(name, bid)

        # split the MLA kv_b projection into k_b and v_b
        if name.endswith("kv_b_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)

            yield from super().modify_tensors(k_b.transpose(1, 2), name.replace("kv_b_proj", "k_b_proj"), bid)
            yield from super().modify_tensors(v_b, name.replace("kv_b_proj", "v_b_proj"), bid)
            return

        # merge the routed experts into 3D tensors
        if re.search(r"mlp\.experts\.\d+\.", name):
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) < n_experts * 3:
                return

            for w_name in ("down_proj", "gate_proj", "up_proj"):
                datas = [self._experts[bid].pop(f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight") for xid in range(n_experts)]
                yield from super().modify_tensors(torch.stack(datas, dim=0), f"model.layers.{bid}.mlp.experts.{w_name}.weight", bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            if remaining := [name for block in self._experts for name in block]:
                raise ValueError(f"unprocessed experts: {remaining}")


@ModelBase.register("LongcatFlashSparseForCausalLM")
@ModelBase.example("meituan-longcat/LongCat-Flash-Lite-Sparse")
class LongcatFlashSparseModel(LongcatFlashNgramModel):
    model_arch = gguf.MODEL_ARCH.LONGCAT_FLASH_SPARSE

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # the original checkpoints name the n-gram tensors oe_embed_*, a save_pretrained round-trip names them ngram_embeddings.*
        name, gen = item

        if m := re.fullmatch(r"(?:model\.)?oe_embed_tokens(\d+)\.weight", name):
            name = f"model.ngram_embeddings.embedders.{int(m.group(1))}.weight"

        if m := re.fullmatch(r"(?:model\.)?oe_embed_proj(\d+)\.weight", name):
            name = f"model.ngram_embeddings.post_projs.{int(m.group(1))}.weight"

        return super().filter_tensors((name, gen))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the original config names the n-gram parameters oe_*
        aliases = {
            "emb_neighbor_num": "oe_neighbor_num",
            "emb_split_num": "oe_split_num",
            "ngram_vocab_size_ratio": "oe_vocab_size_ratio",
        }
        for name, alias in aliases.items():
            if self.hparams.get(name) is None:
                self.hparams[name] = self.hparams[alias]

        if (method := self.hparams.get("attention_method")) != "LSA":
            raise ValueError(f"unsupported attention_method: {method!r}")
        if (k_norm_type := self.hparams.get("index_k_norm_type", "rms")) != "rms":
            raise ValueError(f"unsupported index_k_norm_type: {k_norm_type!r}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_indexer_head_count(self.hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(self.hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(self.hparams["index_topk"])
        self.gguf_writer.add_indexer_init_tokens(self.hparams["index_init_tokens"])
        self.gguf_writer.add_indexer_local_tokens(self.hparams["index_local_tokens"])
        # the indexer key norm uses the RMSNorm default eps, not rms_norm_eps
        self.gguf_writer.add_indexer_k_norm_eps(self.hparams.get("index_k_norm_eps", 1e-6))
        self.gguf_writer.add_indexer_rope_interleave(self.hparams["indexer_rope_interleave"])

        # every cli_factor-th block owns an indexer, the others reuse its top-k
        cli_factor = self.hparams["cli_factor"]
        self.gguf_writer.add_indexer_types([block_id % cli_factor == 0 for block_id in range(self.base_block_count)])

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.endswith(".indexer.proj.weight"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
