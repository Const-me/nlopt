/*==============================================================================
# NLOPT CMake configuration file
#
# NLopt is a free/open-source library for nonlinear optimization, providing
# a common interface for a number of different free optimization routines
# available online as well as original implementations of various other
# algorithms
# WEBSITE: http://ab-initio.mit.edu/wiki/index.php/NLopt
# AUTHOR: Steven G. Johnson
#
# This config.cmake.h.in file was created to compile NLOPT with the CMAKE utility.
# Benoit Scherrer, 2010 CRL, Harvard Medical School
# Copyright (c) 2008-2009 Children's Hospital Boston
#
# Minor changes to the source was applied to make possible the compilation with
# Cmake under Linux/Win32
#============================================================================*/

/* Bugfix version number. */
#define BUGFIX_VERSION 0

/* Define to enable extra debugging code. */
#undef DEBUG

/* Define to 1 if you have the `BSDgettimeofday' function. */
#undef HAVE_BSDGETTIMEOFDAY

/* Define if the copysign function/macro is available. */
#define HAVE_COPYSIGN

/* Define if the fpclassify() function/macro is available. */
#define HAVE_FPCLASSIFY

/* Define to 1 if you have the <getopt.h> header file. */
/* #undef HAVE_GETOPT_H */

/* Define to 1 if you have the getopt function in your standard library. */
/* #undef HAVE_GETOPT */

/* Define to 1 if you have the `getpid' function. */
#define HAVE_GETPID

/* Define if syscall(SYS_gettid) available. */
#undef HAVE_GETTID_SYSCALL

/* Define to 1 if you have the `gettimeofday' function. */
/* #undef HAVE_GETTIMEOFDAY */

/* Define if the isinf() function/macro is available. */
#define HAVE_ISINF

/* Define if the isnan() function/macro is available. */
#define HAVE_ISNAN

/* Define to 1 if you have the `m' library (-lm). */
#undef HAVE_LIBM

/* Define to 1 if you have the `qsort_r' function. */
/* #undef HAVE_QSORT_R */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H

/* Define to 1 if you have the <sys/types.h> header file. */
/* #undef HAVE_SYS_TIME_H */

/* Define to 1 if you have the `time' function. */
#define HAVE_TIME

/* Define to 1 if the system has the type `uint32_t'. */
#define HAVE_UINT32_T

/* Define to 1 if you have the <unistd.h> header file. */
/* #undef HAVE_UNISTD_H */

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#undef LT_OBJDIR

/* Major version number. */
#define MAJOR_VERSION 2

/* Minor version number. */
#define MINOR_VERSION 10

/* Name of package */
#undef PACKAGE

/* Define to the address where bug reports for this package should be sent. */
#undef PACKAGE_BUGREPORT

/* Define to the full name of this package. */
#undef PACKAGE_NAME

/* Define to the full name and version of this package. */
#undef PACKAGE_STRING

/* Define to the one symbol short name of this package. */
#undef PACKAGE_TARNAME

/* Define to the home page for this package. */
#undef PACKAGE_URL

/* Define to the version of this package. */
#undef PACKAGE_VERSION

/* replacement for broken HUGE_VAL macro, if needed */
#undef REPLACEMENT_HUGE_VAL

/* The size of `unsigned int', as computed by sizeof. */
#define SIZEOF_UNSIGNED_INT 4

/* The size of `unsigned long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG 4

/* Define to 1 if you have the ANSI C header files. */
#undef STDC_HEADERS

/* Define to C thread-local keyword, or to nothing if this is not supported in
   your compiler. */
#define THREADLOCAL __declspec(thread)

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
/* #undef TIME_WITH_SYS_TIME */

/* Version number of package */
#undef VERSION

/* Define if compiled including C++-based routines */
#define NLOPT_CXX

/* Define to empty if `const' does not conform to ANSI C. */
#undef const

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
#undef inline
#endif
