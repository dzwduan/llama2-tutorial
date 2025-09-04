#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#define TESTING
#include "run.c"

void test_rmsnorm() {
    printf("\n--- test_rmsnorm ---\n");
    float x[3] = {1.0f, 2.0f, 3.0f};
    float w[3] = {1.0f, 2.0f, 0.5f};
    float actual[3];
    float expected[] = {0.462910f, 1.851638f, 0.694364f};

    rmsnorm(actual, x, w, 3);

    for (int j = 0; j < 3; j++)
        assert(fabs(actual[j] - expected[j]) < 1e-5f);

    printf("rmsnorm test passed.\n");
}

void test_softmax() {
    printf("\n--- test_softmax ---\n");
    float x[3] = {1.0f, 2.0f, 3.0f};
    float expected[3] = {0.09003057f, 0.24472847f, 0.66524096f};

    softmax(x, 3);

    for (int i = 0; i < 3; i++)
        assert(fabs(x[i] - expected[i]) < 1e-5f);

    printf("softmax test passed.\n");
}

void test_matmul() {
    printf("\n--- test_matmul ---\n");
    float x[3] = {1.0f, 2.0f, 3.0f};
    float w[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float actual[2];
    float expected[2] = {1.0f, 2.0f};

    matmul(actual, x, w, 3, 2);

    for (int i = 0; i < 2; i++)
        assert(fabs(actual[i] - expected[i]) < 1e-5f);

    printf("matmul test passed.\n");
}

void test_random_f32() {
    printf("\n--- test_random_f32 ---\n");
    unsigned long long state = 123456789;
    float actual = random_f32(&state);

    assert(actual >= 0.0f && actual < 1.0f);

    printf("random_f32 test passed.\n");
}

void test_random_u32() {
    printf("\n--- test_random_u32 ---\n");
    unsigned long long state = 123456789;
    unsigned int actual = random_u32(&state);

    assert(actual <= UINT32_MAX);

    printf("random_u32 test passed.\n");
}

void test_safe_printf() {
    printf("\n--- test_safe_printf ---\n");
    char *test_str = "Hello, World!";
    safe_printf(test_str);

    char non_printable[2] = {0x01, '\0'};
    safe_printf(non_printable);

    printf("safe_printf test passed.\n");
}

void test_time_in_ms() {
    printf("\n--- test_time_in_ms ---\n");
    long t1 = time_in_ms();
    usleep(1000);
    long t2 = time_in_ms();

    assert(t2 >= t1);

    printf("time_in_ms test passed.\n");
}

void test_compare_functions() {
    printf("\n--- test_compare_functions ---\n");
    TokenIndex a = {.str = "hello", .id = 1};
    TokenIndex b = {.str = "world", .id = 2};
    TokenIndex c = {.str = "hello", .id = 3};

    assert(compare_tokens(&a, &b) < 0);
    assert(compare_tokens(&a, &c) == 0);

    ProbIndex d = {.prob = 0.5f, .index = 1};
    ProbIndex e = {.prob = 0.3f, .index = 2};
    ProbIndex f = {.prob = 0.5f, .index = 3};

    assert(compare(&d, &e) < 0);
    assert(compare(&d, &f) == 0);
    assert(compare(&e, &f) > 0);

    printf("compare test passed.\n");
}

void test_str_lookup() {
    printf("\n--- test_str_lookup ---\n");
    TokenIndex vocab[] = {{"hello", 0}, {"world", 1}, {"!", 2}};
    int actual = str_lookup("world", vocab, 3);
    int expected = 1;

    assert(actual == expected);
    assert(str_lookup("notfound", vocab, 3) == -1);

    printf("str_lookup test passed.\n");
}

void test_read_stdin() {
    printf("\n--- test_read_stdin ---\n");
    FILE *test_input = fopen("test_input.txt", "w");
    fprintf(test_input, "test input\n");
    fclose(test_input);

    int old_stdin = dup(STDIN_FILENO);
    int new_stdin = open("test_input.txt", O_RDONLY);
    dup2(new_stdin, STDIN_FILENO);

    char actual[100];
    char expected[] = "test input";
    read_stdin("Test prompt: ", actual, sizeof(actual));

    assert(strcmp(actual, expected) == 0);

    dup2(old_stdin, STDIN_FILENO);
    close(new_stdin);
    close(old_stdin);
    unlink("test_input.txt");

    printf("read_stdin test passed.\n");
}

void test_sample_argmax() {
    printf("\n--- test_sample_argmax ---\n");
    float probabilities[] = {0.1f, 0.5f, 0.4f};
    int actual = sample_argmax(probabilities, 3);
    int expected = 1;

    assert(actual == expected);

    printf("sample_argmax test passed.\n");
}

void test_sample_mult() {
    printf("\n--- test_sample_mult ---\n");
    float probabilities[] = {0.1f, 0.5f, 0.4f};
    float coin = 0.5f;
    int actual = sample_mult(probabilities, 3, coin);
    int expected = 1;

    assert(actual == expected);

    float probabilities2[] = {0.33333333f, 0.33333333f, 0.33333334f};
    float coin2 = 1.0f;
    int actual2 = sample_mult(probabilities2, 3, coin2);
    int expected2 = 2;

    assert(actual2 == expected2);

    float probabilities3[] = {0.1f, 0.2f, 0.3f};
    float coin3 = 0.8f;
    int actual3 = sample_mult(probabilities3, 3, coin3);
    int expected3 = 2;

    assert(actual3 == expected3);

    printf("sample_mult test passed.\n");
}

void test_sample_topp() {
    printf("\n--- test_sample_topp ---\n");
    float probabilities[] = {0.1f, 0.5f, 0.4f};
    ProbIndex probindex[3] = {{0.1f, 0}, {0.5f, 1}, {0.4f, 2}};
    float topp = 0.8f;
    float coin = 0.4f;
    int actual = sample_topp(probabilities, 3, topp, probindex, coin);
    int expected = 1;

    assert(actual == expected);

    printf("sample_topp test passed.\n");
}

void test_sample() {
    printf("\n--- test_sample ---\n");
    Sampler sampler1;
    build_sampler(&sampler1, 3, 1.0f, 0.8f, 123456789);
    float logits1[] = {0.1f, 0.5f, 0.4f};
    int actual1 = sample(&sampler1, logits1);

    assert(actual1 >= 0 && actual1 < sampler1.vocab_size);
    free_sampler(&sampler1);

    Sampler sampler2;
    build_sampler(&sampler2, 3, 0.0f, 0.8f, 123456789);
    float logits2[] = {0.1f, 0.5f, 0.4f};
    int actual2 = sample(&sampler2, logits2);

    assert(actual2 >= 0 && actual2 < sampler2.vocab_size);
    free_sampler(&sampler2);

    Sampler sampler3;
    build_sampler(&sampler3, 3, 1.0f, 1.0f, 123456789);
    float logits3[] = {0.1f, 0.5f, 0.4f};
    int actual3 = sample(&sampler3, logits3);

    assert(actual3 >= 0 && actual3 < sampler3.vocab_size);
    free_sampler(&sampler3);

    printf("sample test passed.\n");
}

void test_malloc_run_state_and_free_run_state_unit() {
    printf("\n--- test_malloc_run_state_and_free_run_state_unit ---\n");
    Config config = {.dim = 8,
                     .hidden_dim = 16,
                     .n_layers = 2,
                     .n_heads = 4,
                     .n_kv_heads = 2,
                     .vocab_size = 20,
                     .seq_len = 10};
    RunState s = {0};

    malloc_run_state(&s, &config);

    assert(s.x != NULL);
    assert(s.xb != NULL);
    assert(s.xb2 != NULL);
    assert(s.hb != NULL);
    assert(s.hb2 != NULL);
    assert(s.q != NULL);
    assert(s.key_cache != NULL);
    assert(s.value_cache != NULL);
    assert(s.att != NULL);
    assert(s.logits != NULL);

    free_run_state(&s);

    printf("malloc_run_state_and_free_run_state test passed.\n");
}

void create_test_checkpoint(const char *filename) {
    FILE *file = fopen(filename, "wb");

    Config config = {.dim = 64,
                     .hidden_dim = 128,
                     .n_layers = 2,
                     .n_heads = 4,
                     .n_kv_heads = 4,
                     .vocab_size = 1000,
                     .seq_len = 256};
    fwrite(&config, sizeof(Config), 1, file);

    int head_size = config.dim / config.n_heads;
    size_t weights_size =
        config.vocab_size * config.dim + config.n_layers * config.dim +
        config.n_layers * config.dim * config.dim +
        config.n_layers * config.dim * config.dim +
        config.n_layers * config.dim * config.dim +
        config.n_layers * config.dim * config.dim +
        config.n_layers * config.dim +
        config.n_layers * config.dim * config.hidden_dim +
        config.n_layers * config.hidden_dim * config.dim +
        config.n_layers * config.dim * config.hidden_dim + config.dim +
        config.seq_len * head_size / 2 + config.seq_len * head_size / 2 +
        config.vocab_size * config.dim;

    float *dummy_weights = calloc(weights_size, sizeof(float));
    for (int i = 0; i < 100 && i < weights_size; i++) {
        dummy_weights[i] = (float)(i % 10) / 10.0f;
    }
    fwrite(dummy_weights, sizeof(float), weights_size, file);

    free(dummy_weights);
    fclose(file);
}

void create_test_tokenizer(const char *filename) {
    FILE *file = fopen(filename, "wb");

    int max_token_length = 32;
    fwrite(&max_token_length, sizeof(int), 1, file);

    int vocab_size = 1000;
    for (int i = 0; i < vocab_size; i++) {
        float score;
        if (i == 7) {
            score = 0.99f;
        } else if (i == 8) {
            score = 0.98f;
        } else {
            score = (float)i / vocab_size;
        }
        fwrite(&score, sizeof(float), 1, file);

        char token[64];
        if (i == 0)
            strcpy(token, "<unk>");
        else if (i == 1)
            strcpy(token, "<s>");
        else if (i == 2)
            strcpy(token, "</s>");
        else if (i == 3)
            strcpy(token, " ");
        else if (i == 4)
            strcpy(token, "Hello");
        else if (i == 5)
            strcpy(token, "world");
        else if (i == 6)
            strcpy(token, "test");
        else if (i == 7)
            strcpy(token, "ab");
        else if (i == 8)
            strcpy(token, "HelloWorld");
        else if (i == 9)
            strcpy(token, "a");
        else if (i == 10)
            strcpy(token, "b");
        else if (i == 11)
            strcpy(token, "c");
        else if (i < 256 + 3) {
            sprintf(token, "<0x%02X>", i - 3);
        } else {
            sprintf(token, "token_%d", i);
        }

        int len = strlen(token);
        fwrite(&len, sizeof(int), 1, file);
        fwrite(token, len, 1, file);
    }

    fclose(file);
}

void test_tokenizer() {
    printf("\n--- test_tokenizer ---\n");
    create_test_tokenizer("test_tokenizer.bin");

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "test_tokenizer.bin", 1000);

    int expected_vocab_size = 1000;
    assert(tokenizer.vocab_size == expected_vocab_size);
    assert(tokenizer.vocab != NULL);
    assert(tokenizer.vocab_scores != NULL);

    char *test_text = "Hello world";
    int tokens[100];
    int n_tokens = 0;

    encode(&tokenizer, test_text, 1, 0, tokens, &n_tokens);
    assert(n_tokens > 0);
    assert(tokens[0] == 1);

    encode(&tokenizer, test_text, 1, 1, tokens, &n_tokens);
    assert(tokens[n_tokens - 1] == 2);

    char *actual_decoded = decode(&tokenizer, 0, 1);
    assert(actual_decoded != NULL);

    actual_decoded = decode(&tokenizer, 1, 259);
    assert(actual_decoded != NULL);

    actual_decoded = decode(&tokenizer, 0, 65);
    assert(actual_decoded != NULL);

    encode(&tokenizer, "", 1, 0, tokens, &n_tokens);
    int expected_empty_tokens = 1;
    assert(n_tokens == expected_empty_tokens);

    char *simple_bpe_text = "ab";
    encode(&tokenizer, simple_bpe_text, 0, 0, tokens, &n_tokens);
    assert(n_tokens > 0);

    char *bpe_test_text = "HelloWorld";
    encode(&tokenizer, bpe_test_text, 0, 0, tokens, &n_tokens);
    assert(n_tokens > 0);

    char *shift_test_text = "abc";
    encode(&tokenizer, shift_test_text, 0, 0, tokens, &n_tokens);
    assert(n_tokens > 0);

    char utf8_test[] = {0xE4, 0xB8, 0xAD, 0x00};
    encode(&tokenizer, utf8_test, 0, 0, tokens, &n_tokens);
    assert(n_tokens > 0);

    char utf8_emoji[] = {0xF0, 0x9F, 0x98, 0x80, 0x00};
    encode(&tokenizer, utf8_emoji, 0, 0, tokens, &n_tokens);
    assert(n_tokens > 0);

    char orphan_bytes[] = {0x80, 0x81, 0x00};
    encode(&tokenizer, orphan_bytes, 0, 0, tokens, &n_tokens);
    assert(n_tokens > 0);

    char unknown_char[] = {0xF4, 0x8F, 0xBF, 0xBF, 0x00};
    encode(&tokenizer, unknown_char, 0, 0, tokens, &n_tokens);
    assert(n_tokens > 0);

    free_tokenizer(&tokenizer);
    unlink("test_tokenizer.bin");

    printf("tokenizer test passed.\n");
}

void test_read_checkpoint() {
    printf("\n--- test_read_checkpoint ---\n");
    const char *filename = "test_ckpt1.bin";
    create_test_checkpoint(filename);

    Config actual_config;
    TransformerWeights actual_weights;
    int actual_fd;
    float *actual_data;
    ssize_t actual_file_size;

    read_checkpoint((char *)filename, &actual_config, &actual_weights,
                    &actual_fd, &actual_data, &actual_file_size);

    int expected_vocab_size = 1000;
    assert(actual_config.vocab_size == expected_vocab_size);
    assert(actual_file_size > sizeof(Config));
    assert(actual_data != NULL);
    assert(actual_fd != -1);

    munmap(actual_data, actual_file_size);
    close(actual_fd);
    remove(filename);

    printf("read_checkpoint test passed.\n");
}

void test_build_transformer() {
    printf("\n--- test_build_transformer ---\n");
    const char *filename = "test_transformer.bin";
    create_test_checkpoint(filename);

    Transformer transformer;
    build_transformer(&transformer, (char *)filename);

    int expected_dim = 64;
    int expected_vocab_size = 1000;
    assert(transformer.config.dim == expected_dim);
    assert(transformer.config.vocab_size == expected_vocab_size);
    assert(transformer.weights.token_embedding_table != NULL);
    assert(transformer.state.x != NULL);

    free_transformer(&transformer);
    remove(filename);

    printf("build_transformer test passed.\n");
}

void test_transformer_integration() {
    printf("\n--- test_transformer_integration ---\n");
    create_test_checkpoint("test_model.bin");
    create_test_tokenizer("test_tokenizer.bin");

    Transformer transformer;
    build_transformer(&transformer, "test_model.bin");

    int expected_dim = 64;
    assert(transformer.config.dim == expected_dim);
    assert(transformer.weights.token_embedding_table != NULL);
    assert(transformer.state.x != NULL);

    float *actual_logits = forward(&transformer, 10, 0);
    assert(actual_logits != NULL);

    actual_logits = forward(&transformer, 20, 1);
    assert(actual_logits != NULL);

    free_transformer(&transformer);
    unlink("test_model.bin");
    unlink("test_tokenizer.bin");

    printf("transformer_integration test passed.\n");
}

void test_generation_integration() {
    printf("\n--- test_generation_integration ---\n");
    create_test_checkpoint("test_model.bin");
    create_test_tokenizer("test_tokenizer.bin");

    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;

    build_transformer(&transformer, "test_model.bin");
    build_tokenizer(&tokenizer, "test_tokenizer.bin",
                    transformer.config.vocab_size);
    build_sampler(&sampler, tokenizer.vocab_size, 1.0f, 0.9f, 12345);

    generate(&transformer, &tokenizer, &sampler, "test", 10);
    generate(&transformer, &tokenizer, &sampler, NULL, 5);
    generate(&transformer, &tokenizer, &sampler, "", 5);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    unlink("test_model.bin");
    unlink("test_tokenizer.bin");

    printf("generation_integration test passed.\n");
}

void test_chat_function() {
    printf("\n--- test_chat_function ---\n");
    create_test_checkpoint("test_chat_model.bin");
    create_test_tokenizer("test_chat_tokenizer.bin");

    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;

    build_transformer(&transformer, "test_chat_model.bin");
    build_tokenizer(&tokenizer, "test_chat_tokenizer.bin",
                    transformer.config.vocab_size);
    build_sampler(&sampler, tokenizer.vocab_size, 1.0f, 0.9f, 12345);

    FILE *input_file = fopen("chat_input.txt", "w");
    fprintf(input_file, "Test system prompt\nTest user prompt\n");
    fclose(input_file);

    int old_stdin = dup(STDIN_FILENO);
    int new_stdin = open("chat_input.txt", O_RDONLY);
    dup2(new_stdin, STDIN_FILENO);

    chat(&transformer, &tokenizer, &sampler, NULL, NULL, 2);

    dup2(old_stdin, STDIN_FILENO);
    close(new_stdin);
    close(old_stdin);
    unlink("chat_input.txt");

    chat(&transformer, &tokenizer, &sampler, "test_system", "test_user", 2);
    chat(&transformer, &tokenizer, &sampler, "", "", 50);
    chat(&transformer, &tokenizer, &sampler, "test_system", "", 2);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    unlink("test_chat_model.bin");
    unlink("test_chat_tokenizer.bin");

    printf("chat_function test passed.\n");
}

int main() {
    test_rmsnorm();
    test_softmax();
    test_matmul();
    test_random_f32();
    test_random_u32();
    test_safe_printf();
    test_time_in_ms();
    test_compare_functions();
    test_str_lookup();
    test_read_stdin();
    test_sample_argmax();
    test_sample_mult();
    test_sample_topp();
    test_sample();
    test_malloc_run_state_and_free_run_state_unit();
    test_tokenizer();
    test_read_checkpoint();
    test_build_transformer();
    test_transformer_integration();
    test_generation_integration();
    test_chat_function();

    return 0;
}