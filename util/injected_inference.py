import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
import transformers.models.llama.modeling_llama as llama_modeling

from transformers import modeling_utils

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

class Llama3_injected(nn.Module):
    def __init__(
            self, 
            llama3_model : nn.Module, 
            auto_encoder : nn.Module, # NOTICE: only on the given token index
            injected_layer_num : int,
        ):
        super(Llama3_injected, self).__init__()
        self.llama3_model = llama3_model
        self.config = self.llama3_model.config
        self.auto_encoder = auto_encoder.to(self.llama3_model.device)
        self.injected_layer_num = injected_layer_num
        if self.injected_layer_num > self.llama3_model.config.num_hidden_layers or self.injected_layer_num < 0:
            print("warning: injected_layer_num is out of range, will not be injected.")
        self.is_clean_run = False
        
        self.device = self.llama3_model.device

    def clean_run(self):
        self.is_clean_run = True

    def injected_run(self):
        self.is_clean_run = False

    
    def inner_forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        cache_position = None,
        use_cache = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.llama3_model.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.llama3_model.model.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.llama3_model.model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.llama3_model.model.rotary_emb(hidden_states, position_ids=position_ids)

        layer_count = 0

        for decoder_layer in self.llama3_model.model.layers[: self.config.num_hidden_layers]:
            if layer_count == self.injected_layer_num and not self.is_clean_run:
                hidden_states[0, -1, :] = self.auto_encoder(hidden_states[0, -1, :])[0]

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )[0]

            layer_count += 1

        hidden_states = self.llama3_model.model.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        cache_position = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        outputs: BaseModelOutputWithPast = self.inner_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.llama3_model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.llama3_model.loss_function(logits=logits, labels=labels, vocab_size=self.llama3_model.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )