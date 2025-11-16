.PHONY: all
all: generate


# test name
test-name = num-seq_$(1)
get-num-seq = $(lastword $(subst _, ,$(filter num-seq_%,$(subst __, ,$(1)))))

# tests;
TESTS =
include tests.mk

TESTS_DIRS = $(TESTS)


$(addsuffix /parameters.mk,$(TESTS_DIRS)): %/parameters.mk:
	@echo Creating $@
	@mkdir -p $(dir $@)
	@touch $@
	@echo test-name  = $* >> $@
	@echo num-seq = $(call get-num-seq,$*) >> $@


include $(HB_HAMMERBENCH_PATH)/mk/testbench_common.mk

clean:
	rm -rf num-seq*

