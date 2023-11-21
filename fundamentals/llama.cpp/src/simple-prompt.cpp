#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    llama_model_params model_params = llama_model_default_params();
    std::string model_path = "models/llama-2-13b-chat.Q4_0.gguf";
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

    // Tokenize the prompt.
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

    // Create a new batch
    llama_batch batch = llama_batch_init(512,/*embd*/ 0, /*n_seq_max*/ 1);
    // batch.token will be a pointer to llam_token with a bytes size of 2048
    // sizeof(llama_token) = 4, 4 * n_tokens = 2048, so it will be able to
    // store 512 tokens.
    // batch.pos is similarly a pointer to llama_pos with a bytes size of 2048
    // sizeof(llama_pos) = 4, 4 * n_tokens = 2048, so it will be able to
    // store 512 positions.
    // n_seq_max is the max number of sequences in the batch.
    // batch.logits is an array of bools with a bytes size of 512.

    // Next we are doing to populate the batch we created above. For each token
    // of the tokenized prompt we are going to add it to the the batch.
    const std::vector<llama_seq_id>& seq_ids = { 0 }; // seq_ids.length = 1
    for (size_t i = 0; i < input_tokens.size(); i++) {
        // the token of this batch entry.
        batch.token[batch.n_tokens] = input_tokens[i];
        // the position in the sequence of this batch entry.
        batch.pos[batch.n_tokens] = i,
        // the number of sequence id's (if any) of this batch entry.
        batch.n_seq_id[batch.n_tokens] = seq_ids.size();
        for (size_t s = 0; s < seq_ids.size(); ++s) {
            batch.seq_id[batch.n_tokens][s] = seq_ids[s];
        }
        // Determins if the logits for this token should be generated or not.
        batch.logits[batch.n_tokens] = false;
        // Increment the number of tokens in the batch.
        batch.n_tokens++;
    }
    //batch.n_seq_id = nullptr;
    //batch.seq_id = nullptr;

    // Instruct llama to generate the logits for the last token
    batch.logits[batch.n_tokens - 1] = true;
    fprintf(stderr, "batch.n_tokens: %d\n", batch.n_tokens);

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
    int n_vocab = llama_n_vocab(model);
    while (n_cur <= n_len) {
        {
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
            // This is the token id that the model predicted.

            // is it an end of stream?
            if (new_token_id == llama_token_eos(model) || n_cur == n_len) {
                fprintf(stderr, "\n");
                fflush(stderr);
                break;
            }

            // Next we get the string value for the token id. This is called a
            // piece // which I think comes from SentencePiece.
            // We don't know that actual length of the token so we are using 
            // 8 here a "guess". If the token is longer than 8 bytes then we
            // will resize the result vector and call llama_token_to_piece again.
            std::vector<char> piece(8, 0);
            int n_tokens = llama_token_to_piece(model, new_token_id, piece.data(), piece.size());
            // llama_token_to_piece will return the negative length of the token if
            // it is longer that the passed in result.length. If that is the case
            // then we need to resize the result vector to the length of the token
            // and call llama_token_to_piece again.
            if (n_tokens < 0) {
                piece.resize(-n_tokens);
                int new_len = llama_token_to_piece(model, new_token_id, piece.data(), piece.size());
            } else {
                piece.resize(n_tokens);
            }
            std::string piece_str = std::string(piece.data(), piece.size());
            fprintf(stderr, "%s", piece_str.c_str());
            // stdout is line buffered and we are not printing a newline so we
            // above so we need to call flush.
            fflush(stderr);

            // So we initially had a batch of size equal to the number of tokens
            // of the prompt. Now, we want to pass in the token that we just
            // predicted to the model and get the logits for the next token.

            // Update the number of tokens in the batch to 0.
            batch.n_tokens = 0;
            batch.token[0] = new_token_id;
            batch.pos[0] = n_cur,
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            batch.n_tokens++;

            n_decode += 1;
        }

        n_cur += 1;

        // With the new token added to the batch, we can now predict the
        // next token.
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }
    fprintf(stdout, "Decoded %d tokens\n", n_decode);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
