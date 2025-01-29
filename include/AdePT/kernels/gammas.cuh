// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>
// Pull in implementation.
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>
#include <G4HepEmGammaInteractionPhotoelectric.icc>

using VolAuxData = adeptint::VolAuxData;

#ifdef ASYNC_MODE
namespace AsyncAdePT {
// Asynchronous TransportGammas Interface
template <typename Scoring>
__global__ void __launch_bounds__(256, 1)
    TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries, adept::MParray *activeQueue,
                    adept::MParray *leakedQueue, Scoring *userScoring)
{
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  int activeSize                     = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];
    auto navState       = currentTrack.navState;

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = navState.GetLogicalId();
#endif
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
#else

// Synchronous TransportGammas Interface
template <typename Scoring>
__global__ void TransportGammas(adept::TrackManager<Track> *gammas, Secondaries secondaries, MParrayTracks *leakedQueue,
                                Scoring *userScoring, VolAuxData const *auxDataArray)
{
  using namespace adept_impl;
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  constexpr int Pdg                  = 22;
  int activeSize                     = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*gammas->fActiveTracks)[i];
    adeptint::TrackData trackdata;
    Track &currentTrack = (*gammas)[slot];
    auto navState       = currentTrack.navState;

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = navState.GetLogicalId();
#endif
    VolAuxData const &auxData = auxDataArray[lvolID]; // FIXME unify VolAuxData

#endif

    auto eKin          = currentTrack.eKin;
    auto preStepEnergy = eKin;
    auto pos           = currentTrack.pos;
    vecgeom::Vector3D<Precision> preStepPos(pos);
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<Precision> preStepDir(dir);
    double globalTime = currentTrack.globalTime;
    double localTime  = currentTrack.localTime;
    double properTime = currentTrack.properTime;
    // the MCC vector is indexed by the logical volume id

    // Write local variables back into track and enqueue
    auto survive = [&](bool leak = false) {
      currentTrack.eKin       = eKin;
      currentTrack.pos        = pos;
      currentTrack.dir        = dir;
      currentTrack.globalTime = globalTime;
      currentTrack.localTime  = localTime;
      currentTrack.properTime = properTime;
      currentTrack.navState   = navState;
#ifdef ASYNC_MODE
      if (leak)
        leakedQueue->push_back(slot);
      else
        activeQueue->push_back(slot);
#else
      currentTrack.CopyTo(trackdata, Pdg);
      if (leak)
        leakedQueue->push_back(trackdata);
      else
        gammas->fNextTracks->push_back(slot);
#endif
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);

    // Re-sample the `number-of-interaction-left` (if needed, otherwise use stored numIALeft) and put it into the
    // G4HepEmTrack. Use index 0 since numIALeft for gammas is based only on the total macroscopic cross section. The
    // currentTrack.numIALeft[0] are updated later
    if (currentTrack.numIALeft[0] <= 0.0) {
      theTrack->SetNumIALeft(-std::log(currentTrack.Uniform()), 0);
    } else {
      theTrack->SetNumIALeft(currentTrack.numIALeft[0], 0);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();

    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    vecgeom::NavigationState nextState;
    double geometryStepLength;
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, hitsurf_index);
#else
    geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState, nextState);
#endif
    //  printf("pvol=%d  step=%g  onboundary=%d  pos={%g, %g, %g}  dir={%g, %g, %g}\n", navState.TopId(),
    //  geometryStepLength,
    //         nextState.IsOnBoundary(), pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);
    pos += geometryStepLength * dir;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Update the flight times of the particle
    double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    globalTime += deltaTime;
    localTime += deltaTime;

    int winnerProcessIndex;
    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.

      // Kill the particle if it left the world.
      if (!nextState.IsOutside()) {

        G4HepEmGammaManager::UpdateNumIALeft(theTrack);

        // Save the `number-of-interaction-left` in our track.
        // Use index 0 since numIALeft stores for gammas only the total macroscopic cross section
        double numIALeft          = theTrack->GetNumIALeft(0);
        currentTrack.numIALeft[0] = numIALeft;

#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
        if (nextState.IsOutside()) continue;
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
#endif
        // Move to the next boundary.
        navState = nextState;
        // printf("  -> pvol=%d pos={%g, %g, %g} \n", navState.TopId(), pos[0], pos[1], pos[2]);
        //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
#ifndef ADEPT_USE_SURF
        const int nextlvolID = navState.Top()->GetLogicalVolume()->id();
#else
        const int nextlvolID = navState.GetLogicalId();
#endif

#ifdef ASYNC_MODE // FIXME unify VolAuxData
        VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
#else
        VolAuxData const &nextauxData = auxDataArray[nextlvolID];
#endif

        if (nextauxData.fGPUregion > 0)
          survive();
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          pos += kPushOutRegion * dir;
          survive(/*leak*/ true);
        }
      } else {
        // release slot for particle that has left the world
#ifdef ASYNC_MODE
        slotManager.MarkSlotForFreeing(slot);
#endif
      }
      continue;
    } else {

      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());
      winnerProcessIndex = theTrack->GetWinnerProcessIndex();
      // NOTE: no simple re-drawing is possible for gamma-nuclear, since HowFar returns now smaller steps due to the
      // gamma-nuclear reactions in comparison to without gamma-nuclear reactions. Thus, an empty step without a
      // reaction is needed to compensate for the smaller step size returned by HowFar.

      // Reset number of interaction left for the winner discrete process also in the currentTrack (SampleInteraction()
      // resets it for theTrack), will be resampled in the next iteration.
      currentTrack.numIALeft[0] = -1.0;
    }

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (eKin < 2 * copcore::units::kElectronMassC2) {
        survive();
        break;
      }

      double logEnergy = std::log(eKin);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, eKin, logEnergy, auxData.fMCIndex, elKinEnergy,
                                                           posKinEnergy, &rnge);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

      // Check the cuts and deposit energy in this volume if needed
      double edep = 0;

      if (ApplyCuts && elKinEnergy < theElCut) {
        // Deposit the energy here and kill the secondary
        edep = elKinEnergy;
      } else {
#ifdef ASYNC_MODE
        secondaries.electrons.NextTrack(
            newRNG, elKinEnergy, pos,
            vecgeom::Vector3D<Precision>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]}, navState,
            currentTrack);
#else
        Track &electron = secondaries.electrons->NextTrack();
        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentId = currentTrack.parentId;
        electron.rngState = newRNG;
        electron.eKin     = elKinEnergy;
        electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);
#endif
      }

      if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut && posKinEnergy < theElCut)) {
        // Deposit: posKinEnergy + 2 * copcore::units::kElectronMassC2 and kill the secondary
        edep += posKinEnergy + 2 * copcore::units::kElectronMassC2;
      } else {
#ifdef ASYNC_MODE
        secondaries.positrons.NextTrack(
            currentTrack.rngState, posKinEnergy, pos,
            vecgeom::Vector3D<Precision>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]}, navState,
            currentTrack);
#else
        Track &positron = secondaries.positrons->NextTrack();
        positron.InitAsSecondary(pos, navState, globalTime);
        // Reuse the RNG state of the dying track.
        positron.parentId = currentTrack.parentId;
        positron.rngState = currentTrack.rngState;
        positron.eKin     = posKinEnergy;
        positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);
#endif
      }

      // If there is some edep from cutting particles, record the step
      if (edep > 0) {
        if (auxData.fSensIndex >= 0)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentId,  // Track ID
                                   2,                      // Particle type
                                   geometryStepLength,     // Step length
                                   edep,                   // Total Edep
                                   &navState,              // Pre-step point navstate
                                   &preStepPos,            // Pre-step point position
                                   &preStepDir,            // Pre-step point momentum direction
                                   nullptr,                // Pre-step point polarization
                                   preStepEnergy,          // Pre-step point kinetic energy
                                   0,                      // Pre-step point charge
                                   &nextState,             // Post-step point navstate
                                   &pos,                   // Post-step point position
                                   &dir,                   // Post-step point momentum direction
                                   nullptr,                // Post-step point polarization
                                   0,                      // Post-step point kinetic energy
                                   0,                      // Post-step point charge
                                   currentTrack.eventId,   // Event Id
                                   currentTrack.threadId); // thread ID
      }
      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
#ifdef ASYNC_MODE
      slotManager.MarkSlotForFreeing(slot);
#endif
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (eKin < LowEnergyThreshold) {
        survive();
        break;
      }
      const double origDirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirPrimary[3];
      double newEnergyGamma =
          G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(eKin, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = eKin - newEnergyGamma;

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

      // Check the cuts and deposit energy in this volume if needed
      double edep = 0;

      if (ApplyCuts ? energyEl > theElCut : energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
#ifdef ASYNC_MODE
        Track &electron = secondaries.electrons.NextTrack(
            newRNG, energyEl, pos, eKin * dir - newEnergyGamma * newDirGamma, navState, currentTrack);
#else
        Track &electron = secondaries.electrons->NextTrack();

        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentId = currentTrack.parentId;
        electron.rngState = newRNG;
        electron.eKin     = energyEl;
        electron.dir      = eKin * dir - newEnergyGamma * newDirGamma;
#endif

        electron.dir.Normalize();
      } else {
        edep = energyEl;
      }

      // Check the new gamma energy and deposit if below threshold.
      // Using same hardcoded very LowEnergyThreshold as G4HepEm
      if (newEnergyGamma > LowEnergyThreshold) {
        eKin = newEnergyGamma;
        dir  = newDirGamma;
        survive();
      } else {
        edep += newEnergyGamma;
        newEnergyGamma = 0.;
        // The current track is killed by not enqueuing into the next activeQueue and the slot is released
#ifdef ASYNC_MODE
        slotManager.MarkSlotForFreeing(slot);
#endif
      }

      // If there is some edep from cutting particles, record the step
      if (edep > 0) {
        if (auxData.fSensIndex >= 0)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentId,                        // Track ID
                                   2,                                            // Particle type
                                   geometryStepLength,                           // Step length
                                   edep,                                         // Total Edep
                                   &navState,                                    // Pre-step point navstate
                                   &preStepPos,                                  // Pre-step point position
                                   &preStepDir,                                  // Pre-step point momentum direction
                                   nullptr,                                      // Pre-step point polarization
                                   preStepEnergy,                                // Pre-step point kinetic energy
                                   0,                                            // Pre-step point charge
                                   &nextState,                                   // Post-step point navstate
                                   &pos,                                         // Post-step point position
                                   &dir,                                         // Post-step point momentum direction
                                   nullptr,                                      // Post-step point polarization
                                   newEnergyGamma,                               // Post-step point kinetic energy
                                   0,                                            // Post-step point charge
                                   currentTrack.eventId, currentTrack.threadId); // event and thread ID
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process.
      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), eKin, &rnge);

      double edep             = bindingEnergy;
      const double photoElecE = eKin - edep;
      if (ApplyCuts ? photoElecE > theElCut : photoElecE > theLowEnergyThreshold) {

        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        double dirGamma[] = {dir.x(), dir.y(), dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        // Create a secondary electron and sample directions.
#ifdef ASYNC_MODE
        secondaries.electrons.NextTrack(newRNG, photoElecE, pos,
                                        vecgeom::Vector3D<Precision>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]},
                                        navState, currentTrack);
#else
        Track &electron = secondaries.electrons->NextTrack();
        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentId = currentTrack.parentId;
        electron.rngState = newRNG;
        electron.eKin     = photoElecE;
        electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
#endif

      } else {
        // If the secondary electron is cut, deposit all the energy of the gamma in this volume
        edep = eKin;
      }
      if (auxData.fSensIndex >= 0)
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.parentId,                        // Track ID
                                 2,                                            // Particle type
                                 geometryStepLength,                           // Step length
                                 edep,                                         // Total Edep
                                 &navState,                                    // Pre-step point navstate
                                 &preStepPos,                                  // Pre-step point position
                                 &preStepDir,                                  // Pre-step point momentum direction
                                 nullptr,                                      // Pre-step point polarization
                                 preStepEnergy,                                // Pre-step point kinetic energy
                                 0,                                            // Pre-step point charge
                                 &nextState,                                   // Post-step point navstate
                                 &pos,                                         // Post-step point position
                                 &dir,                                         // Post-step point momentum direction
                                 nullptr,                                      // Post-step point polarization
                                 0,                                            // Post-step point kinetic energy
                                 0,                                            // Post-step point charge
                                 currentTrack.eventId, currentTrack.threadId); // event and thread ID
        // The current track is killed by not enqueuing into the next activeQueue and the slot is released
#ifdef ASYNC_MODE
      slotManager.MarkSlotForFreeing(slot);
#endif
      break;
    }
    case 3: {
      // Invoke gamma nuclear needs to be handled by Geant4 directly, to be implemented
      // Just keep particle alive
      survive();
    }
    }
  }
}

#ifdef ASYNC_MODE
} // namespace AsyncAdePT
#endif
