# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc


.PHONY: test
test: test_llama2.c
	$(CC) -fprofile-arcs -ftest-coverage -DTESTING test_llama2.c -o test_llama2 -lm


.PHONY: coverage
coverage: test_llama2
	./test_llama2
	lcov --capture --directory . --output-file coverage.info
	genhtml coverage.info --output-directory out
	@echo "HTML coverage report generated in ./out/index.html"


.PHONY: clean
clean:
	rm -f run
	rm -f runq
	rm -f test_llama2
	rm -f *.gcda *.gcno coverage.info
	rm -rf out