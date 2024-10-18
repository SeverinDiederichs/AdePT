// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "Run.hh"

#include <AdePT/benchmarking/TestManager.h>
#include <AdePT/benchmarking/TestManagerStore.h>
#include <cmath>

#define STDEV(N, MEAN, SUM_SQUARES) N > 1 ? sqrt((SUM_SQUARES - N * MEAN * MEAN) / N) : 0

Run::Run()
{
  fTestManager = new TestManager<TAG_TYPE>();
}

Run::~Run() {}

void Run::Merge(const G4Run *run)
{
  const Run *localRun = static_cast<const Run *>(run);

  TestManager<TAG_TYPE> *aTestManager = localRun->GetTestManager();

  fTestManager->addToAccumulator(accumulators::EVENT_SUM, aTestManager->getAccumulator(accumulators::EVENT_SUM));
  fTestManager->addToAccumulator(accumulators::EVENT_SQ, aTestManager->getAccumulator(accumulators::EVENT_SQ));
  fTestManager->addToAccumulator(accumulators::NONEM_SUM, aTestManager->getAccumulator(accumulators::NONEM_SUM));
  fTestManager->addToAccumulator(accumulators::NONEM_SQ, aTestManager->getAccumulator(accumulators::NONEM_SQ));
  fTestManager->addToAccumulator(accumulators::ECAL_SUM, aTestManager->getAccumulator(accumulators::ECAL_SUM));
  fTestManager->addToAccumulator(accumulators::ECAL_SQ, aTestManager->getAccumulator(accumulators::ECAL_SQ));
  fTestManager->addToAccumulator(accumulators::NUM_PARTICLES,
                                  aTestManager->getAccumulator(accumulators::NUM_PARTICLES));

  // TEMP: DELETE THIS
  fTestManager->addToAccumulator(accumulators::EVENT_HIT_COPY_SIZE,
                                  aTestManager->getAccumulator(accumulators::EVENT_HIT_COPY_SIZE));
  fTestManager->addToAccumulator(accumulators::EVENT_HIT_COPY_SIZE_SQ,
                                  aTestManager->getAccumulator(accumulators::EVENT_HIT_COPY_SIZE_SQ));
  

  G4Run::Merge(run);
}

void Run::EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilename)
{
  // Export the results per event
  std::vector<std::map<int, double>> *aBenchmarkStates = TestManagerStore<int>::GetInstance()->GetStates();
  TestManager<std::string> aOutputTestManager;

  for (int i = 0; i < aBenchmarkStates->size(); i++) {
    // Recover the results from each event and output them to the specified file
    double eventTime = (*aBenchmarkStates)[i][Run::timers::EVENT];
    aOutputTestManager.setAccumulator("Event", eventTime);

    aOutputTestManager.setOutputDirectory(aOutputDirectory);
    aOutputTestManager.setOutputFilename(aOutputFilename);
    aOutputTestManager.exportCSV();

    aOutputTestManager.reset();
  }
  TestManagerStore<int>::GetInstance()->Reset();

  // Export global results

  aOutputTestManager.setAccumulator("Totaltime", fTestManager->getDurationSeconds(timers::TOTAL));
  aOutputTestManager.setAccumulator("NumParticles", fTestManager->getAccumulator(accumulators::NUM_PARTICLES));

  aOutputTestManager.setOutputDirectory(aOutputDirectory);
  aOutputTestManager.setOutputFilename(aOutputFilename + "_global");
  aOutputTestManager.exportCSV();
}