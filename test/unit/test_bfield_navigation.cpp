// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// Host-side GTest coverage for the fieldPropagatorRungeKutta handoff between
// RK advancement, geometric safety, and VecGeom navigation. The tests use fake
// drivers and navigators so they can pin boundary-navigation contracts without
// requiring a real geometry or launching CUDA kernels.

// The production headers annotate many small functions for CUDA. Defining the
// qualifiers away lets this test compile the templated logic as ordinary host
// C++, which keeps the checks fast and runnable through GTest.
#define __device__
#define __host__

#include <AdePT/transport/support/PhysicalConstants.h>
#include <AdePT/transport/magneticfield/DormandPrinceRK45.h>
#include <AdePT/transport/magneticfield/MagneticFieldEquation.h>
#include <AdePT/transport/magneticfield/OffsetMagneticField.h>
#include <AdePT/transport/magneticfield/RkIntegrationDriver.h>
#include <AdePT/transport/magneticfield/UniformMagneticField.cuh>
#include <AdePT/transport/magneticfield/fieldPropagatorRungeKutta.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <utility>

namespace {

constexpr unsigned int kNvar              = 6;
constexpr unsigned long kParentNavIndex   = 1;
constexpr unsigned long kDaughterNavIndex = 2;
constexpr unsigned long kOtherNavIndex    = 3;

struct DummyField {};

// Fake drivers report the derivative at their endpoint as the real driver does: the propagator takes the
// curvature of the next arc from its momentum part, |dp/ds| = curvature * |p|.
void SetEndDerivative(double dydx[kNvar], const vecgeom::Vector3D<double> &momentum, double curvature)
{
  const auto direction = momentum.Unit();
  const vecgeom::Vector3D<double> normal{-direction[1], direction[0], 0.0};
  for (int i = 0; i < 3; ++i) {
    dydx[i]     = direction[i];
    dydx[3 + i] = curvature * momentum.Mag() * normal[i];
  }
}

template <typename T>
void ExpectVectorNear(const vecgeom::Vector3D<T> &actual, const vecgeom::Vector3D<T> &expected, double tolerance)
{
  const double delta = (actual - expected).Length();
  EXPECT_LE(delta, tolerance) << "actual={" << actual[0] << ", " << actual[1] << ", " << actual[2] << "} expected={"
                              << expected[0] << ", " << expected[1] << ", " << expected[2] << "}";
}

vecgeom::NavigationState MakeState(unsigned long navIndex, bool onBoundary)
{
  vecgeom::NavigationState state;
  state.SetNavIndex(navIndex);
  state.SetBoundaryState(onBoundary);
  return state;
}

// Simulates the all-trials-failed path in RkIntegrationDriver::Advance. It
// leaves the endpoint unchanged, which is the case that used to risk normalizing
// a zero-length chord.
struct FailingPropagatorDriver {
  static double Advance(vecgeom::Vector3D<double> &, vecgeom::Vector3D<double> &, int, double, const DummyField &,
                        double[kNvar], double &, unsigned int, int)
  {
    return 0.0;
  }
};

// Deterministic successful RK substitute: advance exactly along the incoming
// momentum direction and report the full trial step as the last good step.
struct StraightLinePropagatorDriver {
  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned int, int)
  {
    position += step * momentum.Unit();
    lastGoodStep = step;
    SetEndDerivative(dydx, momentum, 0.0);
    return step;
  }
};

// Exact circular motion keeps this carry-over regression independent of RK
// tolerances while recording the arc lengths requested by the propagator.
struct RecordingCircularPropagatorDriver {
  static constexpr double kRadius = 8.0;
  inline static std::array<double, 16> requestedSteps{};
  inline static unsigned int calls = 0;

  static void Reset()
  {
    calls = 0;
    requestedSteps.fill(0.0);
  }

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned int, int)
  {
    if (calls < requestedSteps.size()) requestedSteps[calls] = step;
    ++calls;
    const double angle = step / kRadius;
    const auto tangent = momentum.Unit();
    const vecgeom::Vector3D<double> normal{-tangent[1], tangent[0], 0.0};
    position += kRadius * (std::sin(angle) * tangent + (1.0 - std::cos(angle)) * normal);
    momentum     = momentum.Mag() * (std::cos(angle) * tangent + std::sin(angle) * normal);
    lastGoodStep = step;
    SetEndDerivative(dydx, momentum, 1.0 / kRadius);
    return step;
  }
};

// Reports that the requested RK advance did not fully finish, while still
// returning a nonzero accepted endpoint. RkIntegrationDriver can do this after
// exhausting its trial budget with partial accepted progress, and the propagator
// must bookkeep only that accepted distance.
struct PartialThenCompletePropagatorDriver {
  inline static int calls = 0;

  static void Reset() { calls = 0; }

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned int, int)
  {
    ++calls;
    const double acceptedStep = (calls == 1) ? 2.0 : step;
    position += acceptedStep * momentum.Unit();
    lastGoodStep = acceptedStep;
    SetEndDerivative(dydx, momentum, 0.0);
    return acceptedStep;
  }
};

// Produces a chord whose direction differs from the physical direction kept in
// the momentum vector. This models the boundary-ambiguity shape where the RK
// chord can touch one side of a tolerance surface while the transported
// direction points somewhere else.
struct BentChordPropagatorDriver {
  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned int, int)
  {
    position += step * vecgeom::Vector3D<double>{1.0, 0.0, -1.0e-5}.Unit();
    lastGoodStep = step;
    SetEndDerivative(dydx, momentum, 0.0);
    return step;
  }
};

// Produces a first RK chord shorter than the ambiguity band. This protects the
// classifier from accepting a physical-direction crossing that lies beyond the
// endpoint actually integrated by the RK driver.
struct TinyBentChordPropagatorDriver {
  static constexpr double kAcceptedStep = 4.0e-8;

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned int, int)
  {
    const double acceptedStep = (step < kAcceptedStep) ? step : kAcceptedStep;
    position += acceptedStep * vecgeom::Vector3D<double>{1.0, 0.0, -1.0e-5}.Unit();
    lastGoodStep = acceptedStep;
    SetEndDerivative(dydx, momentum, 0.0);
    return acceptedStep;
  }
};

// Navigator used when a test expects the propagator to return before geometry
// stepping. Any ComputeStepAndNextVolume call is therefore a test failure.
struct ThrowingNavigator {
  static constexpr double kBoundaryPush = 0.0;
  inline static int stepCalls           = 0;
  inline static int safetyCalls         = 0;

  static void Reset()
  {
    stepCalls   = 0;
    safetyCalls = 0;
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    ++safetyCalls;
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &, double,
                                         const vecgeom::NavigationState &, vecgeom::NavigationState &)
  {
    ++stepCalls;
    ADD_FAILURE() << "navigation should not be called";
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = -1;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

// Fake navigator with a boundary at one quarter of the proposed chord. This
// verifies that fieldPropagatorRungeKutta shortens the accepted chord according
// to the navigator result and marks the boundary state.
struct BoundaryNavigator {
  static constexpr double kBoundaryPush = 0.0;
  inline static int stepCalls           = 0;

  static void Reset() { stepCalls = 0; }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &,
                                         double step, const vecgeom::NavigationState &,
                                         vecgeom::NavigationState &nextState)
  {
    ++stepCalls;
    nextState.SetBoundaryState(true);
    return 0.25 * step;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = 0;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

// Navigator stub with a configurable finite safety and an optional boundary at
// a fixed distance on a chosen navigation call. Counts safety refreshes and
// navigation calls so tests can assert how many chords were integrated.
struct FiniteSafetyNavigator {
  static constexpr double kBoundaryPush        = 1.0e-8;
  inline static int safetyCalls                = 0;
  inline static int stepCalls                  = 0;
  inline static double safety                  = 0.0;
  inline static double maxRequestedChord       = 0.0;  // longest chord passed to navigation
  inline static double boundaryDistance        = -1.0; // < 0: never report a boundary
  inline static int boundaryOnCall             = -1;   // 1-based navigation call that reports it
  inline static unsigned long boundaryNavIndex = kDaughterNavIndex;

  static void Reset(double safetyValue, double boundaryDist, int boundaryCall)
  {
    safetyCalls       = 0;
    stepCalls         = 0;
    safety            = safetyValue;
    maxRequestedChord = 0.0;
    boundaryDistance  = boundaryDist;
    boundaryOnCall    = boundaryCall;
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    ++safetyCalls;
    return safety;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &,
                                         double step, const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState)
  {
    ++stepCalls;
    maxRequestedChord = std::max(maxRequestedChord, step);
    if (stepCalls == boundaryOnCall && boundaryDistance >= 0.0 && boundaryDistance < step) {
      nextState = MakeState(boundaryNavIndex, true);
      return boundaryDistance;
    }
    currentState.CopyTo(&nextState);
    nextState.SetBoundaryState(false);
    return step + 1.0; // no boundary before the proposed step
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = 0;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

struct NavigationResponse {
  double move;
  unsigned long navIndex;
  bool onBoundary;
};

// Scripted navigator for boundary-ambiguity algorithm tests. A negative move,
// or a positive move beyond the requested maximum step, is interpreted as "no
// boundary before the proposed step". This models the VecGeom max-step contract
// used by the physical-direction probe.
struct ScriptedNavigator {
  static constexpr double kBoundaryPush                                     = 1.0e-8;
  static constexpr int kMaxScriptedCalls                                    = 256;
  inline static std::array<NavigationResponse, kMaxScriptedCalls> responses = {};
  inline static std::array<double, kMaxScriptedCalls> requestedSteps        = {};
  inline static int responseCount                                           = 0;
  inline static int stepCalls                                               = 0;

  static void Reset(std::initializer_list<NavigationResponse> script)
  {
    responseCount = 0;
    stepCalls     = 0;
    requestedSteps.fill(0.0);
    for (const NavigationResponse &response : script) {
      responses[responseCount++] = response;
    }
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &,
                                         double step, const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState)
  {
    const int responseIndex = (stepCalls < responseCount) ? stepCalls : responseCount - 1;
    if (stepCalls < kMaxScriptedCalls) requestedSteps[stepCalls] = step;
    ++stepCalls;
    const NavigationResponse &response = responses[responseIndex];
    currentState.CopyTo(&nextState);
    if (response.move >= 0.0 && response.move <= step) {
      nextState.SetNavIndex(response.navIndex);
      nextState.SetBoundaryState(response.onBoundary);
      return response.move;
    }
    nextState.SetBoundaryState(false);
    return step;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = stepCalls;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

} // namespace

// Existing fieldPropagatorRungeKutta navigation contracts. These keep the RK
// driver fake and focus on failed RK advance, safety shortcuts, and
// navigator-limited chord acceptance.

// If the RK driver makes no accepted progress, the propagator must stop before
// normalizing a zero-length chord or asking geometry to navigate a bogus step.
TEST(FieldPropagatorRungeKuttaNavigation, FailedAdvanceStopsBeforeZeroChordNormalization)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, FailingPropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = true;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{1.0, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// The RK driver may return partial accepted progress after exhausting its trial
// budget. The propagator must bookkeep that accepted arc length and continue.
TEST(FieldPropagatorRungeKuttaNavigation, UsesAcceptedArcLengthWhenDriverReportsIncomplete)
{
  using Propagator =
      fieldPropagatorRungeKutta<DummyField, PartialThenCompletePropagatorDriver, double, ThrowingNavigator>;

  PartialThenCompletePropagatorDriver::Reset();
  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 100.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(PartialThenCompletePropagatorDriver::calls, 2);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// A large cached geometric safety should allow the whole chord to be accepted
// without calling VecGeom navigation. This protects the fast safety shortcut.
TEST(FieldPropagatorRungeKuttaNavigation, UsesSafetyToAcceptFullChordWithoutNavigation)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 100.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// Geometry safety may enlarge one arc without enlarging the curvature
// allowance for subsequent chords. The first 10 mm arc has a 9.362 mm chord,
// wholly inside the initial safety sphere. That safety is exhausted later.
// Reusing its 10 mm arc length would incorrectly request the remaining 5 mm
// in one go; the next arc must instead follow the curvature allowance at the
// chord end, which the driver reports through its end derivative.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotCarrySafetyEnlargedArcIntoNextChord)
{
  using Driver     = RecordingCircularPropagatorDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, FiniteSafetyNavigator>;
  Driver::Reset();
  FiniteSafetyNavigator::Reset(/*safetyValue=*/0.0, /*boundaryDist=*/-1.0, /*boundaryCall=*/-1);

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  const auto currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 10.0);

  constexpr double safeLength  = 2.0;
  constexpr double physicsStep = 15.0;
  const double moved           = Propagator::ComputeStepAndNextVolume(
      DummyField{}, 10.0, 1.0, 1, physicsStep, safeLength, position, direction, currentState, nextState,
      hitSurfaceIndex, propagated, safetyCache, 20, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, physicsStep);
  EXPECT_TRUE(propagated);
  EXPECT_FALSE(nextState.IsOnBoundary());
  EXPECT_TRUE(nextState.HasSamePathAsOther(currentState));
  // Arc whose chord on this circle has the target sagitta (sagitta = arc^2 / (8 radius)).
  const double curvatureArc = std::sqrt(8.0 * Propagator::kChordTarget * Driver::kRadius);
  ASSERT_LT(curvatureArc, physicsStep - 10.0);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[0], 10.0);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[1], curvatureArc);
  EXPECT_EQ(Driver::calls, 3u);
  EXPECT_NEAR(Driver::requestedSteps[2], physicsStep - 10.0 - curvatureArc, 1.0e-12);
  EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 2);

  const double angle = physicsStep / Driver::kRadius;
  ExpectVectorNear(
      position,
      vecgeom::Vector3D<double>{Driver::kRadius * std::sin(angle), Driver::kRadius * (1.0 - std::cos(angle)), 0.0},
      1.0e-12);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{std::cos(angle), std::sin(angle), 0.0}, 1.0e-12);
}

// The cached safety has to be reduced to the current chord start, not to the
// proposed endpoint. Otherwise a safe chord can be rejected and sent to geometry.
TEST(FieldPropagatorRungeKuttaNavigation, UsesCurrentSafetyBeforeTestingCandidateChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 6.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// The stored cache value belongs to its origin, not necessarily to the current
// position. A cache from an older point must be reduced before accepting a chord.
TEST(FieldPropagatorRungeKuttaNavigation, ReducesSafetyFromCacheOriginBeforeTestingChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, BoundaryNavigator>;

  BoundaryNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(vecgeom::Vector3D<double>{0.0, 2.0, 3.0}, 5.5);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 1.25);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(BoundaryNavigator::stepCalls, 1);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{2.25, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// When the safety shortcut is not enough, the propagator must defer to the
// navigator and stop at the boundary distance reported along the chord.
TEST(FieldPropagatorRungeKuttaNavigation, StopsAtNavigatorBoundaryAlongChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, BoundaryNavigator>;

  BoundaryNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 8.0, 8.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 2.0);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(BoundaryNavigator::stepCalls, 1);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{3.0, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// Boundary-ambiguity tests for the Geant4-inspired direction-resolved
// classifier in the RK field propagator. These use a bent RK chord so the
// chord navigator and physical-direction probe can deliberately disagree.

// Chord artifact case: the chord clips a daughter boundary inside the ambiguity
// band, but the physical-direction probe finds no boundary before the ambiguity
// limit. The propagator must reduce the arc and stay in the current volume.
TEST(FieldPropagatorRungeKuttaNavigation, ReducesTinyChordArtifactWhenPhysicalDirectionStaysInCurrentVolume)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {2.5e-8, kDaughterNavIndex, true},
      {-1.0, kParentNavIndex, false},
      {-1.0, kParentNavIndex, false},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 100, itersDone, 0, false);

  EXPECT_GT(moved, 2.5e-8);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(nextState.GetNavIndex(), currentState.GetNavIndex());
  EXPECT_FALSE(nextState.IsOnBoundary());
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[1], 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_GE(ScriptedNavigator::stepCalls, 2);
}

// The ambiguity classifier is only for tracks already located on a boundary.
// Away from a boundary, even a push-scale chord crossing is a normal navigator
// result and must not trigger the physical-direction probe.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotProbePhysicalDirectionWhenStartIsNotOnBoundary)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState  = MakeState(kParentNavIndex, false);
  const vecgeom::NavigationState daughterState = MakeState(kDaughterNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-9, kDaughterNavIndex, true},
      {-1.0, kParentNavIndex, false},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-9);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(daughterState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(ScriptedNavigator::stepCalls, 1);
}

// The special classifier is restricted to the tiny ambiguity band. A chord hit
// just outside 10*kBoundaryPush is already a finite boundary crossing and should
// be accepted without any physical-direction probe.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotProbePhysicalDirectionOutsideTinyBoundaryBand)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState  = MakeState(kParentNavIndex, true);
  const vecgeom::NavigationState daughterState = MakeState(kDaughterNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {1.1e-7, kDaughterNavIndex, true},
      {-1.0, kParentNavIndex, false},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 1.1e-7);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(daughterState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_GT(moved, 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 1);
}

// Immediate physical return/backscatter: both the chord and physical direction
// find the previous volume at push scale. This remains a zero-length relocation
// so the navigator can re-establish the boundary state without fabricated motion.
TEST(FieldPropagatorRungeKuttaNavigation, AcceptsImmediatePhysicalReturnCrossing)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-9, kParentNavIndex, true},
      {5.0e-9, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  ExpectVectorNear(position, vecgeom::Vector3D<double>{0.0, 0.0, 0.0}, 0.0);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Review regression: the chord clips the tolerance surface at zero, but the
// physical direction exits at a real finite distance inside the ambiguity band.
// The finite direction crossing must win over the chord's zero-distance result.
TEST(FieldPropagatorRungeKuttaNavigation, UsesFiniteDirectionCrossingWhenChordHitIsPushScale)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {0.0, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-8);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_GT(moved, ScriptedNavigator::kBoundaryPush);
  EXPECT_LT(moved, 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[1], 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Position consistency regression: when a finite crossing is accepted from the
// physical-direction probe, the stored position must lie on that physical ray,
// not on the different RK chord ray that produced the original push-scale hit.
TEST(FieldPropagatorRungeKuttaNavigation, PlacesDirectionResolvedFiniteCrossingOnPhysicalRay)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {0.0, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-8);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  ExpectVectorNear(position, vecgeom::Vector3D<double>{0.0, 0.0, 5.0e-8}, 0.0);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Short integrated chord: the physical-direction probe must not be allowed to
// accept a crossing farther away than the endpoint produced by the current RK
// advance, even if that crossing is still inside the ambiguity band.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotAcceptDirectionCrossingBeyondIntegratedChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, TinyBentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {0.0, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved =
      Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction, currentState,
                                           nextState, hitSurfaceIndex, propagated, safetyCache, 1, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_EQ(nextState.GetNavIndex(), currentState.GetNavIndex());
  EXPECT_FALSE(nextState.IsOnBoundary());
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[0], TinyBentChordPropagatorDriver::kAcceptedStep);
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[1], TinyBentChordPropagatorDriver::kAcceptedStep);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Finite tiny crossing: when the boundary is inside the ambiguity band but
// farther than the push distance, preserve the navigator distance instead of
// rounding it to zero or to the full ambiguity limit.
TEST(FieldPropagatorRungeKuttaNavigation, KeepsFiniteTinyDirectionCrossingAtNavigatorDistance)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-8, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-8);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_LT(moved, 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// End-to-end mixed-precision test with the real float RK driver: at large
// coordinates (r ~ 5 m) a very short physics step must move the track by
// exactly the length the propagator reports. Before this fix the float state
// carried the absolute position, and such steps are smaller than float can
// represent there: the propagator reported an advance but the position did
// not follow, producing stuck tracks.
TEST(FieldPropagatorRungeKuttaNavigation, MixedPrecisionMicronStepMovesTrackAtLargeRadius)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;
  using Propagator    = fieldPropagatorRungeKutta<Field, Driver, float, ThrowingNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(2.0 * tesla)});

  // 1e-7 mm matches the anti-stuck push (100 * vecgeom::kTolerance) in the
  // electron kernels, which was lost in float at this radius before the fix.
  for (const double physicsStep : {1.0e-3, 1.0e-4, 1.0e-7}) {
    SCOPED_TRACE(physicsStep);
    const vecgeom::Vector3D<double> start{3000.0, 3000.0, 3000.0};
    vecgeom::Vector3D<double> position = start;
    vecgeom::Vector3D<double> direction{0.3, -0.9, 0.3086067};
    direction = direction.Unit();

    const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, false);
    vecgeom::NavigationState nextState;
    long hitSurfaceIndex = -1;
    bool propagated      = false;
    int itersDone        = 0;
    SafetyCache safetyCache;
    safetyCache.Refresh(position, 1.0e3); // far from any boundary: navigation must not be consulted

    const double moved = Propagator::ComputeStepAndNextVolume(
        field, 100.0 * MeV, copcore::units::kElectronMassC2, -1, physicsStep, /*safeLength=*/10.0, position, direction,
        currentState, nextState, hitSurfaceIndex, propagated, safetyCache, 10, itersDone, 0, false);

    EXPECT_TRUE(propagated);
    EXPECT_NEAR(moved, physicsStep, 1.0e-5 * physicsStep);
    EXPECT_NEAR((position - start).Length(), moved, 1.0e-2 * physicsStep);
  }
}

// Multi-chord mixed-precision propagation at detector-scale coordinates: the
// physics step is split into several chords, so the anchor moves with the
// track between Advance calls. The accumulated step must match the requested
// physics step and the actual displacement.
TEST(FieldPropagatorRungeKuttaNavigation, MixedPrecisionMultiChordAdvancesWithMovingAnchor)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;
  using Propagator    = fieldPropagatorRungeKutta<Field, Driver, float, FiniteSafetyNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(2.0 * tesla)});

  const vecgeom::Vector3D<double> start{3000.0, 3000.0, 3000.0};
  vecgeom::Vector3D<double> position = start;
  vecgeom::Vector3D<double> direction{0.3, -0.9, 0.3086067};
  direction = direction.Unit();

  const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0); // no cached safety: chords are limited by safeLength

  // Safety refreshes return just above the chord length, so every chord after
  // the first is validated against the refreshed safety and no boundary is hit.
  FiniteSafetyNavigator::Reset(/*safetyValue=*/2.5, /*boundaryDist=*/-1.0, /*boundaryCall=*/-1);

  // The first chord uses the 2 mm input allowance; later chords follow the curvature at their start
  // (about 17 mm for R ~ 175 mm), so a 60 mm step needs several chords.
  const double physicsStep = 60.0;
  const double moved       = Propagator::ComputeStepAndNextVolume(
      field, 100.0 * MeV, copcore::units::kElectronMassC2, -1, physicsStep, /*safeLength=*/2.0, position, direction,
      currentState, nextState, hitSurfaceIndex, propagated, safetyCache, 10, itersDone, 0, false);

  EXPECT_TRUE(propagated);
  EXPECT_NEAR(moved, physicsStep, 1.0e-4 * physicsStep);
  // Several chords were integrated, each starting from the advanced position.
  EXPECT_GE(FiniteSafetyNavigator::safetyCalls, 3);
  // With R ~ 175 mm the 60 mm arc stays close to its chord.
  const double displacement = (position - start).Length();
  EXPECT_LE(displacement, moved + 1.0e-6);
  EXPECT_GE(displacement, 0.97 * moved);
}

// Mixed-precision boundary crossing on a later chord: the first chord advances
// fully (moving the anchor), the second chord intersects a boundary partway.
// The accepted step must be the first chord plus the navigator-reported
// distance, with the end state flagged as on-boundary.
TEST(FieldPropagatorRungeKuttaNavigation, MixedPrecisionChordBoundaryCrossingAtLargeRadius)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;
  using Propagator    = fieldPropagatorRungeKutta<Field, Driver, float, FiniteSafetyNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(2.0 * tesla)});

  const vecgeom::Vector3D<double> start{3000.0, 3000.0, 3000.0};
  vecgeom::Vector3D<double> position = start;
  vecgeom::Vector3D<double> direction{0.3, -0.9, 0.3086067};
  direction = direction.Unit();

  const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  // Small refreshed safety forces navigation on every chord; the second
  // navigation call reports a boundary 0.3 mm into the chord.
  FiniteSafetyNavigator::Reset(/*safetyValue=*/0.5, /*boundaryDist=*/0.3, /*boundaryCall=*/2);

  const double physicsStep = 6.0;
  const double moved       = Propagator::ComputeStepAndNextVolume(
      field, 100.0 * MeV, copcore::units::kElectronMassC2, -1, physicsStep, /*safeLength=*/2.0, position, direction,
      currentState, nextState, hitSurfaceIndex, propagated, safetyCache, 10, itersDone, 0, false);

  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 2);
  // First chord: full 2 mm arc; second chord: 0.3 mm to the boundary.
  EXPECT_NEAR(moved, 2.3, 2.0e-3);
  const double displacement = (position - start).Length();
  EXPECT_NEAR(displacement, moved, 2.0e-3);
}

// Persistent same-state ambiguity: if both chord and physical direction remain
// in the same volume at push scale, this is not a backscatter/return crossing.
// Leave it unresolved so the reduced-step retry path handles it.
TEST(FieldPropagatorRungeKuttaNavigation, KeepsPersistentTinySameStateSeparateFromBackscatter)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kOtherNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = true;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-9, kOtherNavIndex, true},
      {5.0e-9, kOtherNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(currentState));
  EXPECT_GE(ScriptedNavigator::stepCalls, 2);
}

// ----------------------------------------------------------------------------
// Chord check outside geometry safety: curvature prediction, acceptance and containment.

namespace {
// Counts the Advance calls of a wrapped driver, to check how often the propagator integrates.
template <class Driver>
struct CountingDriver {
  inline static unsigned calls = 0;
  template <class... Args>
  static auto Advance(Args &&...args)
  {
    ++calls;
    return Driver::Advance(std::forward<Args>(args)...);
  }
};

// Bends on an exact circle, but reports a four times smaller curvature in its end derivative, as in a field that
// grows along the path: every predicted arc is too long and needs a shorter retry from its start.
struct UnderreportingCircularDriver {
  static constexpr double kRadius = 8.0;
  inline static unsigned calls    = 0;

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned, int)
  {
    ++calls;
    const double angle = step / kRadius;
    const auto tangent = momentum.Unit();
    const vecgeom::Vector3D<double> normal{-tangent[1], tangent[0], 0.0};
    position += kRadius * (std::sin(angle) * tangent + (1.0 - std::cos(angle)) * normal);
    momentum     = momentum.Mag() * (std::cos(angle) * tangent + std::sin(angle) * normal);
    lastGoodStep = step;
    SetEndDerivative(dydx, momentum, 0.25 / kRadius);
    return step;
  }
};

// Reports the requested arc length, but its straight endpoint lies 1e-5 mm farther: a numerical endpoint error.
struct SlightlyOvershootingDriver {
  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned, int)
  {
    position += (step + 1.0e-5) * momentum.Unit();
    lastGoodStep = step;
    SetEndDerivative(dydx, momentum, 0.0);
    return step;
  }
};

// Moves exactly along +x, without normalizing a vector, so that the endpoint equals the requested length.
struct ExactEndpointDriver {
  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned, int)
  {
    position[0] += step;
    lastGoodStep = step;
    SetEndDerivative(dydx, momentum, 0.0);
    return step;
  }
};

// Overshoots its straight endpoint like SlightlyOvershootingDriver while turning the momentum by 1.5 rad per mm:
// the chord is too curved for the acceptance although the arc length equals the safety.
struct CurvedOvershootingDriver {
  static constexpr double kCurvature = 1.5;
  inline static std::array<double, 8> requested{};
  inline static unsigned calls = 0;
  static void Reset()
  {
    calls = 0;
    requested.fill(0.0);
  }

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double dydx[kNvar], double &lastGoodStep, unsigned, int)
  {
    if (calls < requested.size()) requested[calls] = step;
    ++calls;
    const auto tangent = momentum.Unit();
    const vecgeom::Vector3D<double> normal{-tangent[1], tangent[0], 0.0};
    position += (step + 1.0e-5) * tangent;
    const double angle = kCurvature * step;
    momentum           = momentum.Mag() * (std::cos(angle) * tangent + std::sin(angle) * normal);
    lastGoodStep       = step;
    SetEndDerivative(dydx, momentum, kCurvature);
    return step;
  }
};

// Exact planar geometry: the current volume is x <= plane, and the safety is the distance to the plane.
struct ContainmentPlaneNavigator {
  static constexpr double kBoundaryPush = 1.0e-8;
  inline static double plane            = 1.0;
  inline static int calls               = 0;

  static void Reset(double boundary)
  {
    plane = boundary;
    calls = 0;
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &position, const vecgeom::NavigationState &, double)
  {
    return std::max(0.0, plane - position[0]);
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &current, vecgeom::NavigationState &next)
  {
    ++calls;
    current.CopyTo(&next);
    next.SetBoundaryState(false);
    if (direction[0] > 0.0) {
      const double distance = (plane - position[0]) / direction[0];
      if (distance >= 0.0 && distance <= step) {
        next.SetNavIndex(kDaughterNavIndex);
        next.SetBoundaryState(true);
        return distance;
      }
    }
    return step;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &current, vecgeom::NavigationState &next,
                                         long &surface)
  {
    surface = 0;
    return ComputeStepAndNextVolume(position, direction, step, current, next);
  }
};

struct PropagationResult {
  double moved;
  bool propagated;
  vecgeom::NavigationState next;
};

// Propagates from the origin along +x with an initial cached safety, through a fake driver and navigator.
template <class Propagator>
PropagationResult Propagate(vecgeom::Vector3D<double> &position, double physicsStep, double safeLength, double safety,
                            int maxIterations = 40)
{
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  const auto current = MakeState(kParentNavIndex, false);
  PropagationResult result{0.0, false, {}};
  long surface   = -1;
  int iterations = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, safety);
  result.moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, physicsStep, safeLength, position,
                                                      direction, current, result.next, surface, result.propagated,
                                                      safetyCache, maxIterations, iterations, 0, false);
  return result;
}
} // namespace

// In a helix the radius of the transverse projection is not the curvature radius of the three-dimensional path.
// The predicted arc must give the target sagitta for every pitch, checked against the analytical helix.
TEST(FieldPropagatorRungeKuttaNavigation, SafeLengthUsesThreeDimensionalHelixCurvature)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, ThrowingNavigator>;
  constexpr double momentumMagnitude = 10.0;
  vecgeom::Vector3D<double> field{0.0, 0.0, 0.0038};
  const double omega = std::fabs(fieldConstants::kB2C) * field.Mag() / momentumMagnitude;

  for (const double pitch : {0.1, 0.5, 1.5707963267948966}) {
    SCOPED_TRACE(pitch);
    vecgeom::Vector3D<double> momentum{momentumMagnitude * std::sin(pitch), 0.0, momentumMagnitude * std::cos(pitch)};
    const double length = Propagator::ComputeSafeLength(momentum, field, 1);
    const auto helix    = [=](double arc) {
      return vecgeom::Vector3D<double>{std::sin(pitch) * std::sin(omega * arc) / omega,
                                       std::sin(pitch) * (1.0 - std::cos(omega * arc)) / omega, std::cos(pitch) * arc};
    };
    const auto chordDirection = helix(length).Unit();
    double maximumDeviation   = 0.0;
    for (int sample = 0; sample <= 128; ++sample) {
      const auto point = helix(length * sample / 128.0);
      maximumDeviation = std::max(maximumDeviation, (point - point.Dot(chordDirection) * chordDirection).Mag());
    }
    EXPECT_LE(maximumDeviation, Propagator::kChordTarget);
    EXPECT_GT(maximumDeviation, 0.95 * Propagator::kChordTarget);
  }
  EXPECT_LT(Propagator::kChordTarget, Propagator::kChordAcceptance);
}

// Without bending (momentum along B, no field, or no charge) the predicted arc is very long but finite.
TEST(FieldPropagatorRungeKuttaNavigation, SafeLengthRemainsFiniteForNoMagneticBending)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, ThrowingNavigator>;
  vecgeom::Vector3D<double> momentum{0.0, 0.0, 10.0};
  vecgeom::Vector3D<double> alongMomentum{0.0, 0.0, 0.0038}, noField{0.0, 0.0, 0.0}, transverse{0.0038, 0.0, 0.0};
  for (const double length :
       {Propagator::ComputeSafeLength(momentum, alongMomentum, 1), Propagator::ComputeSafeLength(momentum, noField, 1),
        Propagator::ComputeSafeLength(momentum, transverse, 0)}) {
    EXPECT_TRUE(std::isfinite(length));
    EXPECT_GT(length, 1.0e10);
  }
}

// Chords are accepted up to deltaChord, as in Geant4's G4ChordFinder: an exact circular chord whose sagitta lies
// between the former 0.3 deltaChord budget and deltaChord is navigated without a retry; a larger one is shortened
// before navigation.
TEST(FieldPropagatorRungeKuttaNavigation, AcceptsChordsUpToDeltaChordLikeGeant4)
{
  using Driver     = RecordingCircularPropagatorDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, FiniteSafetyNavigator>;
  for (const double sagittaFraction : {0.9, 1.2}) {
    SCOPED_TRACE(sagittaFraction);
    Driver::Reset();
    FiniteSafetyNavigator::Reset(0.0, -1.0, -1);
    // Arc with the sagitta L^2 / (8 R) to leading order; the exact circular sagitta is slightly smaller.
    const double arc          = std::sqrt(8.0 * Driver::kRadius * sagittaFraction * fieldConstants::deltaChord);
    const double exactSagitta = Driver::kRadius * (1.0 - std::cos(arc / (2.0 * Driver::kRadius)));
    vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
    const auto result = Propagate<Propagator>(position, arc, arc, 0.0);
    EXPECT_NEAR(result.moved, arc, 1.0e-12);
    EXPECT_TRUE(result.propagated);
    if (sagittaFraction < 1.0) {
      ASSERT_GT(exactSagitta, 0.3 * fieldConstants::deltaChord);
      ASSERT_LT(exactSagitta, Propagator::kChordAcceptance);
      EXPECT_EQ(Driver::calls, 1u);
      EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 1);
    } else {
      ASSERT_GT(exactSagitta, Propagator::kChordAcceptance);
      EXPECT_GT(Driver::calls, static_cast<unsigned>(FiniteSafetyNavigator::stepCalls));
      const double halfChord = 0.5 * FiniteSafetyNavigator::maxRequestedChord;
      EXPECT_LE(Driver::kRadius - std::sqrt(Driver::kRadius * Driver::kRadius - halfChord * halfChord),
                Propagator::kChordAcceptance);
    }
  }
}

// Geometry safety certifies the whole arc even when its chord would be too inaccurate for navigation: an arc no
// longer than the safety is accepted without a retry and without navigation.
TEST(FieldPropagatorRungeKuttaNavigation, KeepsOversizedArcInsideGeometrySafety)
{
  using Driver     = RecordingCircularPropagatorDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, ThrowingNavigator>;
  Driver::Reset();
  ThrowingNavigator::Reset();
  constexpr double physicsStep = 5.0;
  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  const auto result = Propagate<Propagator>(position, physicsStep, physicsStep, physicsStep);
  EXPECT_DOUBLE_EQ(result.moved, physicsStep);
  EXPECT_TRUE(result.propagated);
  EXPECT_EQ(Driver::calls, 1u);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  EXPECT_GT(Driver::kRadius * (1.0 - std::cos(physicsStep / (2.0 * Driver::kRadius))), Propagator::kChordAcceptance);
}

// A numerical endpoint beyond the certified safety is navigated even though the reported arc length fits.
TEST(FieldPropagatorRungeKuttaNavigation, NavigatesNumericalEndpointOutsideCertifiedSafety)
{
  using Propagator =
      fieldPropagatorRungeKutta<DummyField, SlightlyOvershootingDriver, double, ContainmentPlaneNavigator>;
  ContainmentPlaneNavigator::Reset(1.000005);
  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  const auto result = Propagate<Propagator>(position, 1.0, 1.0, /*safety=*/1.0, 10); // plane is 1.000005 mm away
  EXPECT_EQ(ContainmentPlaneNavigator::calls, 1);
  EXPECT_TRUE(result.next.IsOnBoundary());
  EXPECT_EQ(result.next.GetNavIndex(), kDaughterNavIndex);
  EXPECT_NEAR(position[0], 1.000005, 1.0e-14);
  EXPECT_GT(result.moved, 0.0);
}

// The numerical endpoint must lie strictly inside the safety sphere; an endpoint exactly on it is navigated.
TEST(FieldPropagatorRungeKuttaNavigation, NavigatesNumericalEndpointExactlyAtSafetyBoundary)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, ExactEndpointDriver, double, ContainmentPlaneNavigator>;
  ContainmentPlaneNavigator::Reset(1.0);
  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  const auto result = Propagate<Propagator>(position, 1.0, 1.0, /*safety=*/1.0, 10);
  EXPECT_EQ(ContainmentPlaneNavigator::calls, 1);
  EXPECT_DOUBLE_EQ(position[0], 1.0);
  EXPECT_DOUBLE_EQ(result.moved, 1.0);
}

// A too curved arc whose numerical endpoint overshoots the safety is retried shorter, inside the safety sphere,
// instead of being navigated.
TEST(FieldPropagatorRungeKuttaNavigation, ShortensCurvedNumericalOvershootBelowSafetyToMakeProgress)
{
  using Driver     = CurvedOvershootingDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, ThrowingNavigator>;
  Driver::Reset();
  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  const auto result = Propagate<Propagator>(position, 1.0, 1.0, /*safety=*/1.0, 2);
  ASSERT_EQ(Driver::calls, 2u);
  EXPECT_DOUBLE_EQ(Driver::requested[0], 1.0);
  EXPECT_GT(Driver::requested[1], 0.0);
  EXPECT_LT(Driver::requested[1], Driver::requested[0]);
  EXPECT_GT(result.moved, 0.0);
  EXPECT_LT(result.moved, 1.0);
  EXPECT_FALSE(result.next.IsOnBoundary());
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
}

// A curvature retry restarts from the same chord start, so the safety refreshed for the rejected arc still holds
// and must not be computed again.
TEST(FieldPropagatorRungeKuttaNavigation, RefreshesSafetyOncePerChordStartAcrossCurvatureRetries)
{
  using Driver     = UnderreportingCircularDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, FiniteSafetyNavigator>;
  Driver::calls    = 0;
  FiniteSafetyNavigator::Reset(/*safetyValue=*/0.0, /*boundaryDist=*/-1.0, /*boundaryCall=*/-1);
  constexpr double physicsStep = 12.0;
  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  const auto result = Propagate<Propagator>(position, physicsStep, /*safeLength=*/1.0, /*safety=*/0.0);
  EXPECT_NEAR(result.moved, physicsStep, 1.0e-12);
  EXPECT_TRUE(result.propagated);
  // Chords after the first needed shorter retries ...
  EXPECT_GE(FiniteSafetyNavigator::stepCalls, 3);
  EXPECT_GE(Driver::calls, static_cast<unsigned>(FiniteSafetyNavigator::stepCalls) + 2u);
  // ... but each chord start refreshed its safety once. The first chord starts with the caller's fresh safety.
  EXPECT_EQ(FiniteSafetyNavigator::safetyCalls, FiniteSafetyNavigator::stepCalls - 1);
}

// The circular sagitta estimate stays exact at large turning angles, so correctly predicted arcs of low-momentum
// electrons near boundaries are navigated without shorter retries. The first-order estimate L tan(phi) / 4
// overestimates there.
TEST(FieldPropagatorRungeKuttaNavigation, LowMomentumHelixChordsNeedNoRetry)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field      = UniformMagneticField;
  using Equation   = MagneticFieldEquation<Field>;
  using Stepper    = DormandPrinceRK45<Equation, Field, kNvar, double>;
  using Driver     = CountingDriver<RkIntegrationDriver<Stepper, double, int, Equation, Field>>;
  using Propagator = fieldPropagatorRungeKutta<Field, Driver, double, FiniteSafetyNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(3.8 * tesla)});
  const double mass = copcore::units::kElectronMassC2;
  for (const double kinE : {0.02 * MeV, 0.05 * MeV, 0.1 * MeV, 0.2 * MeV}) {
    SCOPED_TRACE(kinE);
    vecgeom::Vector3D<double> position{300.0, 0.0, 0.0};
    vecgeom::Vector3D<double> direction = vecgeom::Vector3D<double>{1.0, 0.2, 0.0}.Unit();
    vecgeom::Vector3D<double> momentum  = std::sqrt(kinE * (kinE + 2.0 * mass)) * direction;
    vecgeom::Vector3D<double> fieldValue{0.0, 0.0, 3.8 * tesla};
    const double safeLength  = Propagator::ComputeSafeLength(momentum, fieldValue, -1);
    const double physicsStep = 4.0 * safeLength;
    Driver::calls            = 0;
    FiniteSafetyNavigator::Reset(0.0, -1.0, -1);
    const auto currentState = MakeState(kParentNavIndex, false);
    vecgeom::NavigationState nextState;
    long hitSurfaceIndex = -1;
    bool propagated      = false;
    int itersDone        = 0;
    SafetyCache safetyCache;
    safetyCache.Refresh(position, 0.0);
    const double moved = Propagator::ComputeStepAndNextVolume(field, kinE, mass, -1, physicsStep, safeLength, position,
                                                              direction, currentState, nextState, hitSurfaceIndex,
                                                              propagated, safetyCache, 20, itersDone, 0, false);
    EXPECT_TRUE(propagated);
    // The propagator stops once the remainder is below 1e-7 of the step.
    EXPECT_NEAR(moved, physicsStep, 1.0e-6 * physicsStep);
    EXPECT_EQ(Driver::calls, static_cast<unsigned>(FiniteSafetyNavigator::stepCalls));
  }
}

// Mixed precision requests a safety-limited arc in float. When float rounds the safety up, the integrated arc is
// one float ulp longer than the safety and must still be accepted as contained in the safety sphere. Comparing the
// float length with the safety rejected it, and the retry requested the same rounded length again, so the track
// made no progress for about half of all safety values.
TEST(FieldPropagatorRungeKuttaNavigation, MixedPrecisionSafetyLimitedArcProgressesWhenFloatRoundsAboveSafety)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;
  using Propagator    = fieldPropagatorRungeKutta<Field, Driver, float, FiniteSafetyNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(3.8 * tesla)});
  const double kinE                              = 2.0 * MeV;
  const double mass                              = copcore::units::kElectronMassC2;
  const vecgeom::Vector3D<double> start          = {1234.5678, -321.0987, 456.789};
  const vecgeom::Vector3D<double> startDirection = vecgeom::Vector3D<double>{1.0, 0.2, 0.0}.Unit();
  vecgeom::Vector3D<double> momentum             = std::sqrt(kinE * (kinE + 2.0 * mass)) * startDirection;
  vecgeom::Vector3D<float> fieldValue{0.0f, 0.0f, float(3.8 * tesla)};
  const double safeLength  = Propagator::ComputeSafeLength(momentum, fieldValue, -1);
  const double physicsStep = 10.0;

  int checked = 0;
  for (int i = 0; i < 400; ++i) {
    // Safeties well above the curvature-limited arc, so a rejected containment would request a retry.
    const double safety = safeLength * (2.7 + 1.0e-3 * i);
    if (!(double(float(safety)) > safety)) continue;
    ++checked;
    SCOPED_TRACE(safety);
    FiniteSafetyNavigator::Reset(safety, -1.0, -1);
    vecgeom::Vector3D<double> position  = start;
    vecgeom::Vector3D<double> direction = startDirection;
    const auto currentState             = MakeState(kParentNavIndex, false);
    vecgeom::NavigationState nextState;
    long hitSurfaceIndex = -1;
    bool propagated      = false;
    int itersDone        = 0;
    SafetyCache safetyCache;
    safetyCache.Refresh(position, safety);
    const double moved = Propagator::ComputeStepAndNextVolume(field, kinE, mass, -1, physicsStep, safeLength, position,
                                                              direction, currentState, nextState, hitSurfaceIndex,
                                                              propagated, safetyCache, 10, itersDone, 0, false);
    EXPECT_TRUE(propagated);
    EXPECT_NEAR(moved, physicsStep, 1.0e-4 * physicsStep);
    EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 0);
  }
  EXPECT_GE(checked, 100);
}
