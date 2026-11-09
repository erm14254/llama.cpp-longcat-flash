from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Iterable

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("LongcatFlashNgramForCausalLM")
@ModelBase.example("meituan-longcat/LongCat-Flash-Lite")
class LongcatFlashNgramModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LONGCAT_FLASH_NGRAM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Each logical HF layer has two sub-blocks in the GGUF representation.
        n_layers = int(
            self.find_hparam(
                ["num_layers", "num_hidden_layers", "n_layers", "n_layer"]
            )
        )
        self.base_block_count = 2 * n_layers
        self.mtp_count = self._detect_mtp_count()

        # LongCat-Flash-Lite currently ships one MTP decoder block. The llama.cpp
        # MTP runtime also currently expects a single appended NextN block.
        if self.mtp_count > 1:
            raise ValueError(
                "LongCat MTP export currently supports one MTP layer, "
                f"but found {self.mtp_count}"
            )

        # MTP/NextN blocks are stored after the main model blocks.
        self.block_count = self.base_block_count + self.mtp_count
        self.tensor_map = gguf.get_tensor_name_map(
            self.model_arch,
            self.block_count,
        )

    def _detect_mtp_count(self) -> int:
        """Detect appended LongCat MTP layers from the Safetensors index."""

        index_path = self.dir_model / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map", {})

            mtp_layer_ids = {
                int(match.group(1))
                for name in weight_map
                if (
                    match := re.match(
                        r"model\.mtp\.layers\.(\d+)\.",
                        name,
                    )
                )
            }

            if mtp_layer_ids:
                expected_ids = set(range(max(mtp_layer_ids) + 1))
                if mtp_layer_ids != expected_ids:
                    raise ValueError(
                        "LongCat MTP layer IDs must be contiguous from zero; "
                        f"found {sorted(mtp_layer_ids)}"
                    )
                return len(mtp_layer_ids)

            if any(name.startswith("model.mtp.") for name in weight_map):
                raise ValueError(
                    "Found LongCat MTP tensors, but no model.mtp.layers.<id> tensors"
                )

        # The published LongCat-Flash-Lite layout stores MTP tensors in this
        # auxiliary file. This fallback supports a directory whose index is absent.
        if (self.dir_model / "model-auxiliary.safetensors").exists():
            return 1

        return 0

    def set_vocab(self):
        self._set_vocab_gpt2()

        # The LongCat tokenizer config enables automatic EOS insertion. Disable it
        # for llama.cpp prompt processing.
        self.gguf_writer.add_add_eos_token(False)

    def set_gguf_parameters(self):
        hparams = self.hparams

        # MLA is represented as MQA with one compressed KV head in GGUF.
        hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()

        if self.mtp_count > 0:
            self.gguf_writer.add_nextn_predict_layers(self.mtp_count)

        self.gguf_writer.add_feed_forward_length(hparams["ffn_hidden_size"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if hparams.get("q_lora_rank") is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])

        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(
            hparams["kv_lora_rank"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(
            hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(
            hparams["expert_ffn_hidden_size"]
        )
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_used_count(hparams["moe_topk"])
        self.gguf_writer.add_expert_weights_scale(
            hparams["routed_scaling_factor"]
        )
        self.gguf_writer.add_expert_zero_count(hparams["zero_expert_num"])
        self.gguf_writer.add_leading_dense_block_count(0)

        self.gguf_writer.add_ngram_neighbor_num(hparams["emb_neighbor_num"])
        self.gguf_writer.add_ngram_split_num(hparams["emb_split_num"])
        self.gguf_writer.add_ngram_vocab_size_ratio(
            hparams["ngram_vocab_size_ratio"]
        )

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if (
            rope_mscale_all := self.rope_parameters.get("mscale_all_dim")
        ) is not None:
            self.gguf_writer.add_rope_scaling_yarn_log_mul(
                0.1 * rope_mscale_all
            )

    _experts: list[dict[str, Tensor]] | None = None

    def _remap_double_block(
        self, name: str, bid: int | None
    ) -> tuple[str, int | None]:
        """Map each logical LongCat layer to two GGUF blocks."""

        match = re.match(
            r"model\.layers\.(\d+)\.self_attn\.(\d+)\.(.*)", name
        )
        if match:
            new_bid = 2 * int(match.group(1)) + int(match.group(2))
            return (
                f"model.layers.{new_bid}.self_attn.{match.group(3)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\."
            r"(input_layernorm|post_attention_layernorm)\.(\d+)\.(.*)",
            name,
        )
        if match:
            new_bid = 2 * int(match.group(1)) + int(match.group(3))
            return (
                f"model.layers.{new_bid}.{match.group(2)}.{match.group(4)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)", name
        )
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.experts."
                f"{match.group(2)}.{match.group(3)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\.mlp\.router\.classifier\.(.*)", name
        )
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.gate.{match.group(2)}",
                new_bid,
            )

        match = re.match(
            r"model\.layers\.(\d+)\.mlp\.router\."
            r"e_score_correction(?:_bias|\.bias)",
            name,
        )
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.gate."
                "e_score_correction.bias",
                new_bid,
            )

        match = re.match(r"model\.layers\.(\d+)\.mlps\.0\.(.*)", name)
        if match:
            new_bid = 2 * int(match.group(1))
            return (
                f"model.layers.{new_bid}.mlp.shared_experts."
                f"{match.group(2)}",
                new_bid,
            )

        match = re.match(r"model\.layers\.(\d+)\.mlps\.1\.(.*)", name)
        if match:
            new_bid = 2 * int(match.group(1)) + 1
            return (
                f"model.layers.{new_bid}.mlp.{match.group(2)}",
                new_bid,
            )

        return name, bid

    def _remap_mtp_tensor(self, name: str) -> tuple[str, int]:
        """Map LongCat's auxiliary MTP tensors to one appended NextN block."""

        if self.mtp_count == 0:
            raise ValueError(
                f"Found MTP tensor {name!r}, but no MTP layer was detected"
            )

        # These tensors live outside model.mtp.layers.0 in the HF checkpoint,
        # but belong to the first appended MTP block in GGUF.
        mtp_bid = self.base_block_count
        if name == "model.mtp.embed_tokens.weight":
            return f"model.layers.{mtp_bid}.embed_tokens.weight", mtp_bid
        if name == "model.mtp.norm.weight":
            return f"model.layers.{mtp_bid}.shared_head.norm.weight", mtp_bid

        match = re.match(r"model\.mtp\.layers\.(\d+)\.(.*)", name)
        if not match:
            raise ValueError(f"Unsupported LongCat MTP tensor: {name}")

        mtp_layer_id = int(match.group(1))
        if mtp_layer_id >= self.mtp_count:
            raise ValueError(
                f"MTP tensor {name!r} references layer {mtp_layer_id}, "
                f"but only {self.mtp_count} layer(s) were detected"
            )

        bid = self.base_block_count + mtp_layer_id
        suffix = match.group(2)

        # LongCat wraps the MTP FFN in transformer_layer and stores its two
        # special RMSNorm weights under an extra `.m` module.
        if suffix.startswith("transformer_layer.mlp."):
            suffix = suffix.removeprefix("transformer_layer.")
        elif suffix == "enorm.m.weight":
            suffix = "enorm.weight"
        elif suffix == "hnorm.m.weight":
            suffix = "hnorm.weight"

        return f"model.layers.{bid}.{suffix}", bid

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.mtp"):
            name, bid = self._remap_mtp_tensor(name)
        else:
            # These names are already handled by the LongCat tensor map.
            if "ngram_embeddings" in name:
                yield from super().modify_tensors(data_torch, name, bid)
                return

            name, bid = self._remap_double_block(name, bid)

        # Split the MLA KV-B projection into separate K-B and V-B tensors.
        # This is required for both the main model and the appended MTP block.
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_heads = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            expected = n_heads * (v_head_dim + qk_nope_head_dim)
            assert data_torch.shape[0] == expected

            kv_b = data_torch.view(
                n_heads,
                v_head_dim + qk_nope_head_dim,
                data_torch.shape[-1],
            )
            k_b, v_b = torch.split(
                kv_b,
                [qk_nope_head_dim, v_head_dim],
                dim=1,
            )
            k_b = k_b.transpose(1, 2)

            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        # Merge individual expert weights into llama.cpp's 3-D tensors.
        if (
            "mlp.experts" in name
            and re.search(r"mlp\.experts\.\d+\.", name)
        ):
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) < n_experts * 3:
                return

            for weight_name in ("down_proj", "gate_proj", "up_proj"):
                tensors: list[Tensor] = []

                for expert_id in range(n_experts):
                    expert_name = (
                        f"model.layers.{bid}.mlp.experts."
                        f"{expert_id}.{weight_name}.weight"
                    )
                    tensors.append(self._experts[bid][expert_name])
                    del self._experts[bid][expert_name]

                merged = torch.stack(tensors, dim=0)
                merged_name = (
                    f"model.layers.{bid}.mlp.experts."
                    f"{weight_name}.weight"
                )
                yield from super().modify_tensors(
                    merged, merged_name, bid
                )
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            remaining = [
                name
                for block_experts in self._experts
                for name in block_experts
            ]
            if remaining:
                raise ValueError(f"Unprocessed experts: {remaining}")


@ModelBase.register("LongcatFlashSparseForCausalLM")
@ModelBase.example("meituan-longcat/LongCat-Flash-Lite-Sparse")
class LongcatFlashSparseModel(LongcatFlashNgramModel):
    model_arch = gguf.MODEL_ARCH.LONGCAT_FLASH_SPARSE

    @classmethod
    def filter_tensors(cls, item):
        # the original checkpoints name the n-gram tensors oe_embed_*, a save_pretrained round-trip
        # names them ngram_embeddings.*
        name, gen = item

        if match := re.fullmatch(r"(?:model\.)?oe_embed_tokens(\d+)\.weight", name):
            name = f"model.ngram_embeddings.embedders.{int(match.group(1))}.weight"

        if match := re.fullmatch(r"(?:model\.)?oe_embed_proj(\d+)\.weight", name):
            name = f"model.ngram_embeddings.post_projs.{int(match.group(1))}.weight"

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
        self.gguf_writer.add_indexer_k_norm_type(self.hparams.get("index_k_norm_type", "rms"))
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
