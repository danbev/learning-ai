#include "llama.h"
//#include "common.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    llama_model_params model_params = llama_model_default_params();
    std::string model_path = "/home/danielbevenius/work/ai/llama.cpp/models/llama-2-13b-chat.Q4_0.gguf";
    fprintf(stdout, "llama.cpp example using model: %s\n", model_path.c_str());

    std::string prompt = "What is LoRA?\n";

    bool numa = false;
    llama_backend_init(numa);

    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = 6;
    ctx_params.n_threads_batch = 6;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const int add_bos_token = llama_add_bos_token(model);
    const bool add_bos  = add_bos_token != -1 ? bool(add_bos_token) :
        (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM); // SPM = SentencePiece Model

    int n_tokens = prompt.length() + add_bos;
    std::vector<llama_token> input_tokens(n_tokens);
    n_tokens = llama_tokenize(model,
                              prompt.data(),
                              prompt.length(),
                              input_tokens.data(),
                              input_tokens.size(),
                              true,
                              false);
    // llama_tokenize will return the negative length of the token if
    // it is longer that the passed in result.length. If that is the case
    // then we need to resize the result vector to the length of the token
    // and call llama_tokenize again.
    if (n_tokens < 0) {
        input_tokens.resize(-n_tokens);
        int new_len = llama_tokenize(model, prompt.data(), prompt.length(), input_tokens.data(), input_tokens.size(), add_bos, false);
    } else {
        input_tokens.resize(n_tokens);
    }
    fprintf(stderr, "\n");

    //std::string token_str = std::string(input_tokens.data(), input_tokens.size());

    //fprintf(stdout, "input_tokens[%d]: %s\n", i, token_str.c_str());
    //fprintf(stdout, "input_tokens size: %ld\n", input_tokens.size());

    llama_batch batch = llama_batch_init(512, 0, 1);

    // So llama_batch is used for is similar to the concept of context
    // we talked about in ../../notes/llm.md#context_size. Below we are adding
    // the input query tokens to this batch/context.
    const std::vector<llama_seq_id>& seq_ids = { 0 };

    for (size_t i = 0; i < input_tokens.size(); i++) {
        // the token of this batch entry.
        batch.token[batch.n_tokens] = input_tokens[i];
        // the position in the sequence of this batch entry.
        batch.pos[batch.n_tokens] = i,
        // the sequence id (if any) of this batch entry.
        batch.n_seq_id[batch.n_tokens] = seq_ids.size();
        for (size_t s = 0; s < seq_ids.size(); ++s) {
            batch.seq_id[batch.n_tokens][s] = seq_ids[s];
        }
        // Determins if the logits for this token should be generated or not.
        batch.logits[batch.n_tokens] = false;
        // Increment the number of tokens in the batch.
        batch.n_tokens++;
    }

    // Instruct llama to generate the logits for the last token
    batch.logits[batch.n_tokens - 1] = true;

    fprintf(stderr, "prompt: %s", prompt.c_str());
    fflush(stderr);

    // Now we run the inference on the batch. This will populate the logits
    // for the last token in the batch.
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    // This is the total number of tokens that we will generate, which recall
    // includes our query tokens (they are all in the llm_batch).
    const int n_len = 32;

    int n_cur = batch.n_tokens;
    int n_decode = 0;
    //std::cout << "n_vocab: " << n_vocab << std::endl;
    while (n_cur <= n_len) {
        {
            int n_vocab = llama_n_vocab(model);
            // logits are stored in the last token of the batch and are the 
            // raw unnormalized predictions.
            float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            // The following is populating the candidates vector with the
            // logit for each token in the vocabulary (32000).
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }
            // Here we are creating an unsorted array of token data from the vector.
            bool sorted = false;
            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), sorted };

            // Find the token with the highest raw score (logit) and return it.
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of stream?
            if (new_token_id == llama_token_eos(model) || n_cur == n_len) {
                fprintf(stderr, "\n");
                fflush(stderr);
                break;
            }

            // Next we get the string value for the token. This is called a piece
            // which I think comes from SentencePiece, and a token would be
            // one such piece (something like that).
            std::vector<char> result(8, 0);
            int n_tokens = llama_token_to_piece(model, new_token_id, result.data(), result.size());
            // llama_token_to_piece will return the negative length of the token if
            // it is longer that the passed in result.length. If that is the case
            // then we need to resize the result vector to the length of the token
            // and call llama_token_to_piece again.
            if (n_tokens < 0) {
                result.resize(-n_tokens);
                int new_len = llama_token_to_piece(model, new_token_id, result.data(), result.size());
            } else {
                result.resize(n_tokens);
            }
            std::string token_str = std::string(result.data(), result.size());
            fprintf(stderr, "%s", token_str.c_str());
            // stdout is line buffered and we are not printing a newline so we
            // above so we need to call flush.
            fflush(stderr);

            // Update the number of tokens in the batch to 0.
            batch.n_tokens = 0;

            //fprintf(stderr, "batch.pos: %d\n", batch.pos[batch.n_tokens]);
            const std::vector<llama_seq_id>& seq_ids = { 0 };
            // Update the token to the new predicted token id.
            batch.token[batch.n_tokens] = new_token_id;
            // Update the position in the sequence of this batch entry.
            batch.pos[batch.n_tokens] = n_cur,
            batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            for (size_t s = 0; s < seq_ids.size(); ++s) {
                batch.seq_id[batch.n_tokens][s] = seq_ids[s];
            }
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens++;
            //llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

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
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
