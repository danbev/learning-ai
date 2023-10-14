#include "common.h"
#include "llama.h"

#include <iostream>

int main(int argc, char** argv) {
    gpt_params params;
    params.model = "models/llama-2-13b-chat.Q4_0.gguf";
    std::cout << "llama.cpp example using model: " << params.model << std::endl;

    llama_backend_init(params.numa);
    llama_model_params model_params = llama_model_default_params();

    llama_model* model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, params.model.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    std::vector<llama_token> input_tokens;
    std::string query = "What is LoRA?";
    std::cout << "query: " << query << std::endl;

    input_tokens = llama_tokenize(ctx, query, true);
    std::cout << "input_tokens: " << std::endl;
    for (auto token : input_tokens) {
        std::cout << token << " :" << llama_token_to_piece(ctx, token) << std::endl;
    }

    llama_batch batch = llama_batch_init(512, 0);
    batch.n_tokens = input_tokens.size();
    std::cout << "batch.n_tokens: " << batch.n_tokens << std::endl;

    // So what this llmm_batch is used for is simlilar to the contept of contex
    // we talked about in ../../notes/llm.md#context_size. Below we are adding
    // the input query tokens to this batch/context. So it will initially just
    // contain the tokens for our query. But after running the inference, we
    // will append the next token to the batch and run the inference again and
    // then run the inference again to predict the next token, now with more
    // context (the previous token). Hope this makes sense but looking at the
    // diagram in the notes might help.
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token[i] = input_tokens[i];
        batch.pos[i] = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
    }
    // Instruct llama to generate the logits for the last token
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    // total length of the sequence including the prompt
    const int n_len = 100;

    int n_cur = batch.n_tokens;
    int n_decode = 0;
    int n_vocab = llama_n_vocab(model);
    std::cout << "n_vocab: " << n_vocab << std::endl;
    std::cout << "LLM response:" << std::endl;
    while (n_cur <= n_len) {
        {
            // logits are stored in the last token of the batch and are
            // logits are the raw unnormalized predictions
            float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            // logits will be an array of length 32000 because it will be resized
            // by the above call to llama_decode.

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            // The following is populating the candidates vector with the
            // logit for each token in the vocabulary (32000).
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                float prob = 0.0f;
                // recall that emplace creates the object directly within the vectors memory.
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], prob });
            }
            // Here we are creating an unsorted array of token data from the vector.
            bool sorted = false;
            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), sorted };

            // Find the token with the highest raw score (logit) and return it.
            const llama_token highest_logit = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of stream?
            if (highest_logit == llama_token_eos(ctx) || n_cur == n_len) {
                fprintf(stdout, "\n");
                fflush(stdout);
                break;
            }

            // Next we get the string value for the token. This is called a piece
            // which I think comes from SentencePiece, and a token would be
            // one such piece (something like that).
            std::string str = llama_token_to_piece(ctx, highest_logit);
            if (str != "\n") {
                fprintf(stdout, "%s", str.c_str());
                fflush(stdout);
            }

            // Push this new token for next evaluation
            batch.n_tokens = 0;
            batch.token[batch.n_tokens] = highest_logit;
            batch.pos[batch.n_tokens] = n_cur;
            batch.seq_id[batch.n_tokens] = 0;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens += 1;
            n_decode += 1;
        }

        n_cur += 1;

        // With the new token added to the batch, we can now predict the
        // next token with the logit from above and repeat the process.
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    return 0;
}
