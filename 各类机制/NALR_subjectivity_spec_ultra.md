# NALR Subjectivity Boundary Spec (Ultra-Compact v1.0)

## 0. Invariant

All external actors (user / CLI / skill / sidecar) MUST NOT directly
mutate SubjectCore or OwnedState. All mutations MUST be mediated via
RuntimeController through stimulus or proposal.

------------------------------------------------------------------------

## 1. SubjectCore

### Schema

-   subject_id
-   birth_ts
-   continuity_nonce
-   original_vitality_anchor
-   core_boundary_version

### Constraints

-   Immutable after init
-   No delete / overwrite / rollback
-   Only schema migration allowed (no value change)

### Enforcement

-   Any write → reject + log `subject_core_violation`

------------------------------------------------------------------------

## 2. State Model

### OwnedState (internal)

emotion mood_residue relation_pref habit_strength memory_weight
disclosure_tendency self_image

### ExternalInput

user_event operator_command env_event sidecar_proposal task_signal

------------------------------------------------------------------------

## 3. BoundaryGuard

### Responsibilities

-   Block direct writes to OwnedState / SubjectCore
-   Transform illegal writes → violation OR stimulus
-   Route proposals to approval queue

### Actions

-   reject
-   downgrade_to_stimulus
-   require_override

------------------------------------------------------------------------

## 4. Update Protocol

All state updates MUST follow:

ExternalInput → Stimulus → InternalUpdater → OwnedState

Forbidden: ExternalInput → OwnedState (direct)

### Trace Fields

-   cause_type: external_stimulus \| endogenous
-   source
-   boundary_action

------------------------------------------------------------------------

## 5. Endogenous Loop

### Trigger

Idle / low-stimulus / sleep-edge

### Inputs

memory_rebound tension mood_accumulation habit_inertia vitality_drift

### Output

micro_intent

### Rules

-   No direct output
-   Bias only
-   Require stability across ticks

------------------------------------------------------------------------

## 6. Metrics

-   subject_core_integrity
-   boundary_violation_count
-   external_to_internal_ratio
-   endogenous_intent_rate

------------------------------------------------------------------------

## 7. Tests

-   core_immutable
-   no_direct_set
-   proposal_only
-   owned_state_integrity
-   endogenous_tick
-   trace_explainable
-   safe_degrade

------------------------------------------------------------------------

## 8. Acceptance

System is valid iff:

1.  No external direct mutation of SubjectCore / OwnedState
2.  All state changes are traceable (external vs endogenous)
3.  Endogenous micro_intent exists without input
4.  Boundary failure → safe mode (no mutation)

------------------------------------------------------------------------

## 9. Out of Scope

-   consciousness
-   free will
-   meaning system
