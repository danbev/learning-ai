### llama-server checkpoints
In llama-context.cpp, update_slots we have the following:
```console
(gdb) (gdb) f
#0  server_context_impl::update_slots (this=0x555555eda220)
    at /home/danbev/work/ai/llama.cpp-debug/tools/server/server-context.cpp:2491
2491                        do_checkpoint = do_checkpoint && (
(gdb) l
2486                        // checkpoints are created only if:
2487                        // - the model uses SWA and we are not using `swa_full`
2488                        // - the model architecture is marked as recurrent or hybrid
2489                        //
2490                        // TODO: try to make this conditional on the context or the memory module, instead of the model type
2491                        do_checkpoint = do_checkpoint && (
2492                                llama_model_is_recurrent(model) ||
2493                                llama_model_is_hybrid(model) ||
2494                                (llama_model_n_swa(model) > 0 && !params_base.swa_full)
2495                                );
```
An example of memory that cannot be rolled back is the hidden state of a
recurrent model like Mamba. The state is moved forward and if a token had been
decoded it has been incorporated into the hidden state. In contrast, for
transformer models that use a KV-cache it is possible to remove that last token
from the sequence in question and then replay that token to get the logits.

So for a recurrent model do_checkpoints will be true
```c++
                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }
```
TODO: do a separate walkthough of LoRA and Asymmetric LoRA (Alora).

Next we will add each token to the batch using following code where the
input_tokens are:
```console
(gdb) p input_tokens 
$11 = (const server_tokens &) @0x555555edc538: {has_mtmd = false, map_idx_to_media = std::map with 0 elements, 
  tokens = std::vector of length 16, capacity 16 = {1, 6, 6423, 708, 3493, 856, 779, 5706, 803, 18859, 540, 7, 708, 6, 64015, 708}}

(gdb) p this->model.vocab.pimpl.get()->id_to_token[1]
$17 = {text = "<|startoftext|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[6]
$18 = {text = "<|im_start|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[6423]
$19 = {text = "user", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[708]
$20 = {text = "Ċ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[3493]
$21 = {text = "What", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[856]
$22 = {text = "Ġis", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[779]
$23 = {text = "Ġthe", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[5706]
$24 = {text = "Ġcapital", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[803]
$25 = {text = "Ġof", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[18859]
$26 = {text = "ĠSweden", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[540]
$27 = {text = "?", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[7]
$28 = {text = "<|im_end|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[708]
$29 = {text = "Ċ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[6]
$30 = {text = "<|im_start|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[64015]
$31 = {text = "assistant", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p this->model.vocab.pimpl.get()->id_to_token[708]
$32 = {text = "Ċ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
```c++
                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.task->need_embd());
```
And each token is added to slot.promp.tokens:
```c++

                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;
```
```c++
                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        if (do_checkpoint && slot.task->n_tokens() - slot.prompt.n_tokens() == 64) {
                            break;
                        }
                    }
```
After this, after the complete prompt has been processed we then have:
```c+
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();

                        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);

                        // no need for empty or small checkpoints
                        do_checkpoint = do_checkpoint && (pos_min >= 0 && pos_max >= 64);

                        // no need to create checkpoints that are too close together
                        do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || pos_max > slot.prompt.checkpoints.back().pos_max + 64);
```
```c++
                        if (do_checkpoint) {
                            while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
                                // make room for the new checkpoint, if needed
                                const auto & cur = slot.prompt.checkpoints.front();

                                SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n",
                                        cur.pos_min, cur.pos_max, (float) cur.data.size() / 1024 / 1024);

                                slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
                            }
```
The above is making room for the new checkpoint, if needed. 
Then it creates a new checkpoint with:
```c++
                            const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                            auto & cur = slot.prompt.checkpoints.emplace_back(server_prompt_checkpoint{
                                /*.pos_min = */ pos_min,
                                /*.pos_max = */ pos_max,
                                /*.data    = */ std::vector<uint8_t>(checkpoint_size),
                            });

                            llama_state_seq_get_data_ext(ctx, cur.data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        }
                    }
```
```c++
size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}
```
```c++
size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}
```
And the type of memory in this case is hybrid memory:
```c++
(gdb) p *ctx.memory.get()
$9 = {_vptr.llama_memory_i = 0x7ffff7a55ff0 <vtable for llama_memory_hybrid+16>}
```
Which has its state_write methon defined in src/llama-memory-hybrid.cpp:
```c++
void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}
```
And here we can see that it is using the LLAMA_STATE_SEQ_FLAG_PARTIAL_ONLY flag
so we are skipping the attnention memory and only writing the recurrent memory.

`llama_state_seq_get_data_ext`:
```console
size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}
```
This will land us in llama-memory-recurrent.cpp:
```c++
void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges);
}
```
The memory of a recurrent/hybrid models is a ring buffer (circular cache) that
stores the hidden states from recurrent layers.
```console
Ring Buffer (size = capacity, e.g., 256 cells)
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  │  0  │  1  │  2  │  3  │ ... │ 254 │ 255 │
  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
     ↑                 ↑
   index             index
```
Each entry in this ring buffer is a mem_cell:
```c++
    std::vector<mem_cell> cells;
```
A mem_cell stores the hidden state for one token position in the sequence. For
example:
```console
(gdb) p cells[0]
$30 = {pos = 181, src = 0, src0 = 0, tail = -1, seq_id = std::set with 1 element = {[0] = 3}}
```
So this cell represents the token in position 181 of sequence 3. And it stores
the hidden state for that token.
Notice that the first thing written is the cell_count:
```console
(gdb) p cell_count
$38 = 1
```
Followed by the meta data:
```c++
void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            // if seq_id = -1 we are writing all sequences and we need to write
            // the number of sequences that follow this one. For a single sequence
            // we are only storing one entry so zero follow.
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));   // writes the position
            io.write(&n_seq_id, sizeof(n_seq_id)); // writes the number of sequences that follow this one

            // if we are writing all sequences, then we need to write the sequence IDs
            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}
```
```console
737	    state_write_meta(io, cell_ranges, seq_id);
(gdb) p cell_ranges
$39 = std::vector of length 1, capacity 1 = {{first = 0, second = 1}}
```
And the we have the writing of the actual tensor data:
```c++
    state_write_data(io, cell_ranges);
```
Now, recall that recurrent memory stores two types of tensors per layer which
are r_l[il] and s_l[il]. R is the recurrent state like the hidden state in 
LSTM/Mamba and S is the state in LSTM or SSM state in Mamba.
```c++
void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer;

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;
```
```console
(gdb) p s_trans
$43 = 0
(gdb) p n_layer
$42 = 16
```
So we first write the s_trans and n_layer. Then we iterate through the layers:
```c++
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }
}
```
Lets take a look at one of the r_l[il] tensors:
```console
(gdb) p *r_l[il]
$45 = {type = GGML_TYPE_F32, buffer = 0x555555f16d50, ne = {16384, 1, 1, 1}, nb = {4, 65536, 65536, 65536}, op = GGML_OP_NONE,
  op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0,
  view_offs = 0, data = 0x55555779ce00, name = "cache_r_l0", '\000' <repeats 53 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```
So we start by writing the type of the tensor, followed by the row size in bytes.
And then the R tensor for this layer is written.

After that we have the S tensor which has a different flow for if the S tensor
is transposed or not:
```c++
    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Read each range of cells of s_size_row length
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write value type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                // Read each range of cells of v_size_el length
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
```
