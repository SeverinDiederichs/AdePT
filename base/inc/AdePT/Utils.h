// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Utils.h
 * @brief General utility functions.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_UTILS_H_
#define ADEPT_UTILS_H_

#include <string.h> // For memset and memcpy

namespace adept {
namespace utils {
/**
 * @brief Rounds up a value to upper aligned version
 * @param value Value to round-up
 */
template <typename Type>
VECCORE_ATT_HOST_DEVICE Type round_up_align(Type value, size_t padding)
{
  size_t remainder = ((size_t)value) % padding;
  if (remainder == 0) return value;
  return (value + padding - remainder);
}

/** @brief CPP/CUDA Portable memset operation */
VECCORE_ATT_HOST_DEVICE
VECCORE_FORCE_INLINE
void memset(void *ptr, int value, size_t num)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  memset(ptr, value, num);
#else
  cudaMemset(ptr, value, num);
#endif
}

/** @brief CPP/CUDA Portable memcpy operation */
VECCORE_ATT_HOST_DEVICE
VECCORE_FORCE_INLINE
void memcpy(void *destination, const void *source, size_t num, int type = 0)
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  memcpy(destination, source, num);
#else
  cudaMemcpy(destination, source, num, (cudaMemcpyKind)type);
#endif
}

} // End namespace utils
} // End namespace adept

#endif // ADEPT_UTILS_H_
