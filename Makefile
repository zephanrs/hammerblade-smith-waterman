.PHONY: all
all: generate


# test name
test-name = seq-len_$(1)__num-seq_$(2)
get-num-seq = $(lastword $(subst _, ,$(filter num-seq_%,$(subst __, ,$(1)))))
get-seq-len = $(lastword $(subst _, ,$(filter seq-len_%,$(subst __, ,$(1)))))


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
	@echo seq-len = $(call get-seq-len,$*) >> $@


include $(HB_HAMMERBENCH_PATH)/mk/testbench_common.mk

clean:
	rm -rf seq-len*

