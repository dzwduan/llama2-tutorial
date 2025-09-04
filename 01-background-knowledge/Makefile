CC = gcc
CFLAGS = -Wall -Wextra -std=c11

EXEC_TEST_NEURON = test_neuron
EXEC_TEST_NET = test_neural_network

.PHONY: all clean run_test_neuron run_test_neural_network

all: $(EXEC_TEST_NEURON) $(EXEC_TEST_NET)

$(EXEC_TEST_NEURON): test_neuron.o neuron.o
	$(CC) $(CFLAGS) $^ -o $@ -lm

$(EXEC_TEST_NET): test_neural_network.o neural_network.o neuron.o
	$(CC) $(CFLAGS) $^ -o $@ -lm

neuron.o: neuron.c neuron.h
	$(CC) $(CFLAGS) -c $< -o $@

test_neuron.o: test_neuron.c neuron.h
	$(CC) $(CFLAGS) -c $< -o $@

neural_network.o: neural_network.c neural_network.h neuron.h
	$(CC) $(CFLAGS) -c $< -o $@

test_neural_network.o: test_neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXEC_TEST_NEURON) $(EXEC_TEST_NET) *.o

run_test_neuron: $(EXEC_TEST_NEURON)
	./$(EXEC_TEST_NEURON)

run_test_neural_network: $(EXEC_TEST_NET)
	./$(EXEC_TEST_NET)
