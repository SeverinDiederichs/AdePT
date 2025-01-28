// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRACK_CUH
#define ASYNC_ADEPT_TRACK_CUH

#include <AdePT/base/MParray.h>
#include <AdePT/copcore/SystemOfUnits.h>
#include <AdePT/copcore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

// TODO: This needs to be unified with the other Track struct, however due to the slot manager
// approach, this can't be done before introducing the SlotManager for all kernels

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  using Precision = vecgeom::Precision;
  RanluxppDouble rngState;
  double eKin              = 0;
  float numIALeft[4]       = {-1.f, -1.f, -1.f, -1.f};
  float initialRange       = -1.f; // Only for e-?
  float dynamicRangeFactor = -1.f; // Only for e-?
  float tlimitMin          = -1.f; // Only for e-?

  double globalTime = 0.;
  float localTime   = 0.f; // Only for e-?
  float properTime  = 0.f; // Only for e-?

  vecgeom::Vector3D<Precision> pos;
  vecgeom::Vector3D<Precision> dir;
  vecgeom::Vector3D<float> safetyPos; ///< last position where the safety was computed
  float safety{0.f};                  ///< last computed safety value
  vecgeom::NavigationState navState;
  unsigned int eventId{0};
  int parentId{-1};
  short threadId{-1};
  unsigned short stepCounter{0};
  unsigned short looperCounter{0};

  __host__ __device__ double Uniform() { return rngState.Rndm(); }

  /// Construct a new track for GPU transport.
  /// NB: The navState remains uninitialised.
  __device__ Track(uint64_t rngSeed, double eKin, double globalTime, float localTime, float properTime,
                   double const position[3], double const direction[3], unsigned int eventId, int parentId,
                   short threadId)
      : eKin{eKin}, globalTime{globalTime}, localTime{localTime}, properTime{properTime}, eventId{eventId},
        parentId{parentId}, threadId{threadId}
  {
    rngState.SetSeed(rngSeed);

    pos = {position[0], position[1], position[2]};
    dir = {direction[0], direction[1], direction[2]};
  }

  /// Construct a secondary from a parent track.
  /// NB: The caller is responsible to branch a new RNG state.
  __device__ Track(RanluxppDouble const &rngState, double eKin, const vecgeom::Vector3D<Precision> &parentPos,
                   const vecgeom::Vector3D<Precision> &newDirection, const vecgeom::NavigationState &newNavState,
                   const Track &parentTrack)
      : rngState{rngState}, eKin{eKin}, globalTime{parentTrack.globalTime}, pos{parentPos}, dir{newDirection},
        navState{newNavState}, eventId{parentTrack.eventId}, parentId{parentTrack.parentId},
        threadId{parentTrack.threadId}
  {
  }

  Track const &operator=(Track const &other) = delete;

  /// @brief Get recomputed cached safety ay a given track position
  /// @param new_pos Track position
  /// @param accurate_limit Only return non-zero if the recomputed safety if larger than the accurate_limit
  /// @return Recomputed safety.
  __host__ __device__ VECGEOM_FORCE_INLINE float GetSafety(vecgeom::Vector3D<Precision> const &new_pos,
                                                           float accurate_limit = 0.f) const
  {
    float dsafe = safety - accurate_limit;
    if (dsafe <= 0.f) return 0.f;
    float distSq = (vecgeom::Vector3D<float>(new_pos) - safetyPos).Mag2();
    if (dsafe * dsafe < distSq) return 0.f;
    return (safety - vecCore::math::Sqrt(distSq));
  }

  /// @brief Set Safety value computed in a new point
  /// @param new_pos Position where the safety is computed
  /// @param safe Safety value
  __host__ __device__ VECGEOM_FORCE_INLINE void SetSafety(vecgeom::Vector3D<Precision> const &new_pos, float safe)
  {
    safetyPos.Set(static_cast<float>(new_pos[0]), static_cast<float>(new_pos[1]), static_cast<float>(new_pos[2]));
    safety = vecCore::math::Max(safe, 0.f);
  }
};
#endif
