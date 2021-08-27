
export TOP_DIR = $(shell pwd)
export INC_PATH =
# export LIBS := -L$(TOP_DIR)/lua/lib -llua $(TOP_DIR)/mjson/mjson.a
export LIBS := $(TOP_DIR)/lua/lib/liblua.a $(TOP_DIR)/mjson/mjson.a

include architecture.h 


all: liblua $(ADDITIONAL_TARGETS) libmjson LSMS
# all: liblua LSMS Documentation

clean:
	cd lua && $(MAKE) clean
	cd mjson && $(MAKE) clean
	cd src && $(MAKE) clean
	cd lib && $(MAKE) clean
	cd doc && $(MAKE) clean
	cd CBLAS && $(MAKE) clean

LSMS: liblua $(ADDITIONAL_TARGETS)
	cd src && $(MAKE)

Documentation:
	cd doc && $(MAKE)

liblua:
	cd lua; $(MAKE); $(MAKE) local

libmjson:
	cd mjson && $(MAKE)

CBLAS_target:
	cd CBLAS && $(MAKE) alllib

test: liblua $(ADDITIONAL_TARGETS) libmjson
	cd src && $(MAKE) test
