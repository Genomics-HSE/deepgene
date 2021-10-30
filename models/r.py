import transformers.models.reformer.modeling_reformer


@add_start_docstrings(
    "The bare Reformer Model transformer outputting raw hidden-states" "without any specific head on top.",
    REFORMER_START_DOCSTRING,
)
transformers.models.reformer.modeling_reformer.ReformerModelWithLMHeadOutput
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        assert (
                self.config.num_hidden_layers > 0
        ), "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"
        
        self.embeddings = ReformerEmbeddings(config)
        self.encoder = ReformerEncoder(config)
        
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ReformerModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            num_hashes=None,
            past_buckets_states=None,
            use_cache=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # noqa: F841
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # noqa: F841
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        assert (
                len(input_shape) == 2
        ), f"`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {input_shape}"
        
        if past_buckets_states is not None:
            assert not self.training, "`past_buckets_states` can only be used for inference, not for training`."
        
        # prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers, is_attention_chunked=True)
        
        # original sequence length for padding
        orig_sequence_length = input_shape[-1]
        
        # if needs padding
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(self.config)
        min_chunk_length = _get_min_chunk_len(self.config)
        
        must_pad_to_match_chunk_length = (
                input_shape[-1] % least_common_mult_chunk_length != 0
                and input_shape[-1] > min_chunk_length
                and past_buckets_states is None
        )
        
        if must_pad_to_match_chunk_length:
            padding_length = least_common_mult_chunk_length - input_shape[-1] % least_common_mult_chunk_length
            
            if self.training is True:
                raise ValueError(
                    f"If training, sequence length {input_shape[-1]} has to be a multiple of least common multiple "
                    f"chunk_length {least_common_mult_chunk_length}. Please consider padding the input to a length "
                    f"of {input_shape[-1] + padding_length}."
                )
            
            # pad input
            input_ids, inputs_embeds, attention_mask, position_ids, input_shape = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_shape=input_shape,
                padding_length=padding_length,
                padded_seq_length=least_common_mult_chunk_length,
                device=device,
            )
        
        # start index for position encoding depends on incremental decoding
        if past_buckets_states is not None:
            start_idx_pos_encodings = past_buckets_states[0][1].shape[1]
        else:
            start_idx_pos_encodings = 0
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            start_idx_pos_encodings=start_idx_pos_encodings,
        )
        
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.hidden_states
        
        # if padding was applied
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]
        
        past_buckets_states = encoder_outputs.past_buckets_states if use_cache else None
        hidden_states = encoder_outputs.all_hidden_states if output_hidden_states else None
        attentions = encoder_outputs.all_attentions if output_attentions else None
        
        if not return_dict:
            return tuple(v for v in [sequence_output, past_buckets_states, hidden_states, attentions] if v is not None)
        return ReformerModelOutput(
            last_hidden_state=sequence_output,
            past_buckets_states=past_buckets_states,
            hidden_states=hidden_states,
            attentions=attentions,
        )
    
    def _pad_to_mult_of_chunk_length(
            self,
            input_ids,
            inputs_embeds=None,
            attention_mask=None,
            position_ids=None,
            input_shape=None,
            padding_length=None,
            padded_seq_length=None,
            device=None,
    ):
        logger.info(
            f"Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a "
            f"multiple of `config.chunk_length`: {padded_seq_length}"
        )
        
        padded_input_ids = torch.full(
            (input_shape[0], padding_length),
            self.config.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        
        # Extend `attention_mask`
        if attention_mask is not None:
            pad_attention_mask = torch.zeros(input_shape[0], padding_length, device=device, dtype=attention_mask.dtype)
            
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=-1)
        else:
            attention_mask = torch.cat(
                [
                    torch.ones(input_shape, device=device, dtype=torch.uint8),
                    torch.zeros((input_shape[0], padding_length), device=device, dtype=torch.uint8),
                ],
                dim=-1,
            )
        
        # Extend `input_ids` with padding to match least common multiple chunk_length
        if input_ids is not None:
            input_ids = torch.cat([input_ids, padded_input_ids], dim=-1)
            input_shape = input_ids.size()
            
            # Pad position ids if given
            if position_ids is not None:
                padded_position_ids = torch.arange(input_shape[-1], padded_seq_length, dtype=torch.long, device=device)
                padded_position_ids = position_ids.unsqueeze(0).expand(input_shape[0], padding_length)
                position_ids = torch.cat([position_ids, padded_position_ids], dim=-1)
        
        # Extend `inputs_embeds` with padding to match least common multiple chunk_length
        if inputs_embeds is not None:
            padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
            inputs_embeds = torch.cat([inputs_embeds, padded_inputs_embeds], dim=-2)
            input_shape = inputs_embeds.size()
        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape