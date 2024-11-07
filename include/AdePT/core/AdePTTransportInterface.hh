// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_INTERFACE_H
#define ADEPT_TRANSPORT_INTERFACE_H

#include <memory>
#include <string>
#include <vector>

class AdePTTransportInterface {
public:
  virtual ~AdePTTransportInterface() {}

  /// @brief Adds a track to the buffer
  virtual void AddTrack(int pdg, int parentID, double energy, double x, double y, double z, double dirx, double diry,
                        double dirz, double globalTime, double localTime, double properTime, int threadId,
                        unsigned int eventId, unsigned int trackIndex) = 0;

  /// @brief Set capacity of on-GPU track buffer.
  virtual void SetTrackCapacity(size_t capacity) = 0;
  /// @brief Set Hit buffer capacity on GPU and Host
  virtual void SetHitBufferCapacity(size_t capacity) = 0;
  /// @brief Set maximum batch size
  virtual void SetMaxBatch(int npart) = 0;
  /// @brief Set buffer threshold
  virtual void SetBufferThreshold(int limit) = 0;
  /// @brief Set debug level for transport
  virtual void SetDebugLevel(int level) = 0;
  /// @brief Set whether AdePT should transport particles across the whole geometry
  virtual void SetTrackInAllRegions(bool trackInAllRegions) = 0;
  /// @brief Check whether AdePT should transport particles across the whole geometry
  virtual bool GetTrackInAllRegions() const = 0;
  /// @brief Set Geant4 region to which it applies
  virtual void SetGPURegionNames(std::vector<std::string> const *regionNames) = 0;
  virtual std::vector<std::string> const *GetGPURegionNames()                 = 0;
  /// @brief Initialize service and copy geometry & physics data on device
  virtual void Initialize(bool common_data = false) = 0;
  /// @brief Interface for transporting a buffer of tracks in AdePT.
  virtual void Shower(int event, int threadId) = 0;
  virtual void Cleanup()                       = 0;
};

/// @brief Factory function to create AdePT instances.
/// Every AdePT transport implementation needs to provide this function to create
/// instances of the transport implementation. These might either be one instance
/// per thread, or share one instance across many threads. This is up to the
/// transport implementation.
extern std::shared_ptr<AdePTTransportInterface> AdePTTransportFactory(unsigned int nThread, unsigned int nTrackSlot,
                                                                      unsigned int nHitSlot, int verbosity,
                                                                      std::vector<std::string> const *GPURegionNames,
                                                                      bool trackInAllRegions);

#endif