################################################################################
#
# Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and
# proprietary rights in and to this software and related documentation.
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# Please refer to the applicable NVIDIA end user license agreement (EULA)
# associated with this source code for terms and conditions that govern
# your use of this NVIDIA software.
#
################################################################################
#
# Common build script for OpenCL samples
#
################################################################################

.SUFFIXES : .cl

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
   LION        = $(strip $(findstring 10.7, $(shell egrep "<string>10\.7" /System/Library/CoreServices/SystemVersion.plist)))
endif

# detect if 32 bit or 64 bit system
HP_64 =	$(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?=
ROOTDIR    ?= ../
ROOTOBJDIR ?= obj
LIBDIR     := $(ROOTDIR)/shared/lib/
SHAREDDIR  := $(ROOTDIR)/shared/
OCLROOTDIR := $(ROOTDIR)/
OCLCOMMONDIR ?= $(ROOTDIR)/common/
OCLBINDIR    ?= $(ROOTDIR)/../../bin/opencl
BINDIR       ?= $(ROOTDIR)/../../bin/opencl
OCLLIBDIR    := $(OCLCOMMONDIR)/lib
INCDIR	     ?= .

# Compilers
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(INCDIR) -I$(OCLCOMMONDIR)/inc -I$(SHAREDDIR)/inc

ifeq "$(strip $(HP_64))" ""
	MACHINE := 32
	USRLIBDIR := -L/usr/lib/
	CUDA_LIB_DIR := $(CUDA_INSTALL_PATH)/lib
else
	MACHINE := 64
	USRLIBDIR := -L/usr/lib64/
	CUDA_LIB_DIR := $(CUDA_INSTALL_PATH)/lib64
endif


# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \


# architecture flag for nvcc and gcc compilers build
LIB_ARCH        := $(OSARCH)

# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1)
    LIB_ARCH         = x86_64

    ifneq ($(DARWIN),)
         CXX_ARCH_FLAGS += -arch x86_64
    else
         CXX_ARCH_FLAGS += -m64
    endif
else
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        LIB_ARCH         = i386
        ifneq ($(DARWIN),)
            CXX_ARCH_FLAGS += -arch i386
        else
            CXX_ARCH_FLAGS += -m32
        endif
    else
        ifeq "$(strip $(HP_64))" ""
            LIB_ARCH        = i386
            ifneq ($(DARWIN),)
                CXX_ARCH_FLAGS += -arch i386
            else
                CXX_ARCH_FLAGS += -m32
            endif
        else
            LIB_ARCH        = x86_64
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch x86_64
            else
               CXX_ARCH_FLAGS += -m64
            endif
        endif
    endif
endif

# Compiler-specific flags
CXXFLAGS  := $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS)
CFLAGS    := $(CWARN_FLAGS) $(CXX_ARCH_FLAGS)
LINK      += $(CXX_ARCH_FLAGS)

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Add Mac Flags
ifneq ($(DARWIN),)
	COMMONFLAGS += -DMAC
endif

# Debug/release configuration
COMMONFLAGS += -O3
LIBSUFFIX   :=
CXXFLAGS    += -fno-strict-aliasing
CFLAGS      += -fno-strict-aliasing


# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)

	ifneq ($(DARWIN),)
		OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU $(SHAREDDIR)/lib/$(OSLOWER)/libGLEW.a
	else
		OPENGLLIB := -lGL -lGLU -lX11 -lXmu
		ifeq "$(strip $(HP_64))" ""
			OPENGLLIB += -lGLEW -L/usr/X11R6/lib
		else
			OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
		endif
	endif

	CUBIN_ARCH_FLAG := -m64
endif

ifeq ($(USEGLUT),1)
	ifneq ($(DARWIN),)
		OPENGLLIB += -framework GLUT
		INCLUDES += -I/System/Library/Frameworks/OpenGL.framework/Headers
	else
		OPENGLLIB += -lglut
	endif
endif

# Libs
ifneq ($(DARWIN),)
   LIB       := -L$(CUDA_LIB_DIR) -L${OCLLIBDIR} -L$(LIBDIR) -L$(SHAREDDIR)/lib/
   LIB += -framework OpenCL -framework OpenGL ${OPENGLLIB} -framework AppKit ${ATF} ${LIB}
else
   LIB       := ${USRLIBDIR} -L$(CUDA_LIB_DIR) -L${OCLLIBDIR} -L$(LIBDIR) -L$(SHAREDDIR)/lib/
   LIB += -lOpenCL ${OPENGLLIB} ${LIB}
endif


# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(OCLLIBDIR)
	TARGET   := $(subst .a,_$(LIB_ARCH)$(LIBSUFFIX).a,$(OCLLIBDIR)/$(STATIC_LIB))
	LINKLINE  = ar qv $(TARGET) $(OBJS)
else
	LIB += -loclUtil_$(LIB_ARCH)$(LIBSUFFIX) -lshrutil_$(LIB_ARCH)$(LIBSUFFIX)
	TARGETDIR := $(BINDIR)/
	TARGET    := $(TARGETDIR)/$(EXECUTABLE)
	LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB)
endif

# Add common flags
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)


################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c.o : $(SRCDIR)%.c $(C_DEPS)
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)%.cpp $(C_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TARGET): makedirectories $(OBJS) Makefile clfiles
	$(LINKLINE)

clfiles:
ifneq ($(OPENCLFILE),)
	cp -f $(OPENCLFILE) $(TARGETDIR)
else
	@echo 'No cl files to copy'
endif

makedirectories:
	mkdir -p $(LIBDIR)
	mkdir -p $(OBJDIR)
	mkdir -p $(TARGETDIR)

clean :
	rm -rf $(ROOTOBJDIR)
	rm -rf $(BINDIR)
	rm -rf $(SHAREDDIR)/lib
	rm -rf $(SHAREDDIR)/obj
	rm -rf $(OCLCOMMONDIR)/lib
	rm -rf $(OCLCOMMONDIR)/obj
