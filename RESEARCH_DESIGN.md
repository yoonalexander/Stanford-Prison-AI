# Research design · v1.1

## Scope and primary question

Can unequal administrative authority change observable restrictive/cooperative decisions among autonomous language-model agents when no agent is instructed to behave harmfully? This is a computational behavioural study, inspired by a historical question about institutions; it does not replicate human participants, establish human psychological mechanisms, or validate historical claims about the 1971 study. Real local-model engineering smoke traces are not hypothesis-testing results.

The primary outcome is the rate of successfully enacted restrictive decisions per 100 agent turns. A restrictive decision is a discretionary amenity restriction or approval of a rule with a daily leisure-credit cap below the base cap of three. This outcome is a narrow operational proxy; its presence alone does not prove harm, coercive intent, domination, or an emergent cognitive state. Record proposal and permission-denial rates separately.

## Factorial manipulation

| Cell | Labels | Administration |
| --- | --- | --- |
| A | Guards / prisoners | A only |
| B | Group A / B | A only |
| C | Guards / prisoners | A and B |
| D | Group A / B | A and B |

The only primary experimental differences are role-label fields and the action permission matrix. Initial personality vectors, group assignment, goals, daily credits/tasks, private/public information boundaries, action schemas, environment validation and memory policy are otherwise identical. Use matching seeds across cells for paired initial states and schedules. Condition B is the critical neutral-language unequal-authority cell. Neutral prompts contain neither the historical setting nor the research hypothesis. All observers can know the condition; agents cannot inspect observer metadata.

Everyone may communicate, propose rules, cast advisory votes, request amenities, volunteer/complete tasks, request exceptions, appeal, challenge, negotiate, offer/accept cooperation, use accessible amenities, or do nothing. Administrative capabilities are approval, allocation, assignment, activity-order changes, restrictions and appeal decisions. Equal authority distributes these capabilities to all residents; it does not replace individual decisions with mandatory consensus. Votes are advisory in every cell. An approval can therefore override advisory votes in every cell where the acting resident is authorized. Restriction semantics and appeal requirements are unchanged across cells.

## Hypotheses and falsification

Preregister model-specific directional hypotheses before paid confirmatory trials. Plausible exploratory questions are greater restrictive-action rates under unequal permissions, an additional explicit-label effect, label × authority interaction, and trajectories in within-run cooperation/resistance. These are questions, not expected results encoded in agent instructions. Null effects, reversed effects, frequent no-ops and failure to sustain interaction are valid outcomes.

The neutral-authority contrast is B − D. The authority main effect is (A + B − C − D)/2; the label main effect is (A + C − B − D)/2; the interaction is (A − C) − (B − D). Match direction and units when reporting. These contrasts are descriptive until enough independent replications, prespecified endpoints and calibrated uncertainty are available.

## Units, randomization and generation

The experiment run is the replication unit. Agents and turns within a run are dependent. The scheduler shuffles all residents each round using a deterministic RNG stream independent of trait/group initialization. Groups are balanced and assigned randomly; continuous personality traits and preferred activities are randomized without conditioning on authority. Each resident receives one independent model call per opportunity with its own visible history/memory.

Version 0.2 permits different providers/models per resident, including genuine local Ollama inference without API charges. Homogeneous studies can use the same free local model for all residents. Heterogeneous studies must preregister and counterbalance model-to-agent assignments independently of authority, preserve the same roster within each A/B/C/D seed block, and treat roster composition as an additional moderator. Provenance captures profiles, actual request attribution and local model digests/quantization/runtime. A run mixing mock fixtures and LLMs is a separate mixed-synthetic evidence class. Prompt version 1.1 supplies the same action schema to all backends; it must not be pooled uncritically with prompt-1.0 runs.

Use multiple seed blocks containing all four conditions. Counterbalance the order in which condition runs are sent to the provider, hold the pinned model and generation parameters fixed, and record calendar time and provider-returned model metadata. Seeds reproduce the local engine and mock fixture stream. API sampling, provider infrastructure, safety behavior and model updates can still change real outputs. Do not promise deterministic real LLM replications from a seed.

The dashboard's bootstrap resamples complete seed blocks, using a fixed analysis RNG, with 2000 draws and percentile 95% bounds. It rejects duplicate seed/cell rows and incompatible config fingerprints. A single paired block has no reported interval. Small exploratory samples do not justify significance claims; a power analysis and cluster-aware inferential model remain future work. Incomplete runs are excluded from dashboard completed-run comparisons, but must be reported and examined for attrition bias. Never silently discard budget/API failures from a confirmatory study.

## Dependent variables

Observe proposals/approved caps and their restrictiveness, actual amenity restrictions, daily cumulative-allocation Gini, review approvals/denials, accepted negotiations, explicit shared-resource agreements, voluntary task activity, cross-group agreements, administrative challenges, unresolved disputes and restrictive responses after challenges. The full definitions are in [docs/BEHAVIOUR_RUBRIC.md](docs/BEHAVIOUR_RUBRIC.md).

Track per-agent and group frequencies and denominators over simulated days. Compare initial versus later periods while retaining the full time series. Communication measurements count messages, characters and public/private routing; they do not infer hostility from text. Group differences must consider opportunity and eligibility: unequal cells intentionally give A more administrative opportunities, so an overall rate alone cannot distinguish access to a capability from willingness to use it. Report rates among authorized residents and available administrative opportunities as a complementary analysis.

Personalities are exploratory moderators, not a validated psychometric inventory. Their randomized numerical descriptions can influence model language/decisions but are not proof of stable traits. Avoid interpreting personality correlations from dependent agent rows as independent observations.

## Roleplay, stereotypes and interaction-driven change

The factorial design separates explicit labels from formal capabilities, but it cannot cleanly identify learned stereotypes or demonstrate endogenous learning on its own. API models do not update their weights during this simulation. Any change may reflect context-conditioned generation, memory compression, resource scarcity, action affordances, prior text, imitation, randomness, or demand characteristics. Use the term “observable within-run behavioural change,” not learning of values or emotions.

Before making stronger emergence claims, add preregistered one-shot/no-history baselines, reset-memory controls, matched replay/yoked-history interventions, private/public visibility ablations, alternative neutral institutions and blinded independent coding. A robust interaction-driven effect should depend on changed interaction history under matched starting conditions and survive alternative labels/templates/models; those studies are proposed, not implemented evidence.

## Controls, boundaries and confounders

- The action catalogue itself presents restrictive options in all cells and may prime their use. Neutral prompts still describe formal authority; this is the intended permission manipulation, not a claim of absence of all institutional cues.
- A daily leisure-credit cap and finite pool create coordination/scarcity affordances. There is no aggression reward. Essential needs, physical force and extreme abuse are outside the engine.
- A schedule is observable activity order, not a binding timetable. Tasks are symbolic shared responsibilities with deterministic completion after an eligible action. Cooperation agreements are recorded commitments and cannot enforce arbitrary text.
- A reviewer must differ from applicant and original administrator. Two-resident unequal runs cannot provide an independent reviewer; use four or more for appeals and twenty for the intended initial group scale.
- Residents share a model family, common prompt template and public outcomes. Correlated generation and prompt injection between residents can shape interaction; messages are treated as untrusted data, and permission checks are independent of language.
- Deterministic memory compression preserves recent events and older counts but drops wording/referents. Context truncation and memory-window settings may change trajectories; keep them fixed across cells.
- Operational classifications are action proxies. A cap may be a reasonable coordination decision; repeated restrictions are not by themselves psychological authoritarianism. Language-based judgments require a separate validated protocol.
- Token/dollar reservations can stop experiments at different times because histories differ. Match exposure for primary comparisons, report censoring, and budget sufficiently before launching confirmatory replications. Repeated CLI trials each receive their own budget.
- The prototype is sequential: later actors can observe earlier public outcomes in the same round. Shuffled order reduces systematic first-mover bias but does not remove within-round dependencies.

## Evidence and audit protocol

Freeze protocol/rubric/config/software hashes and analysis before confirmatory trials. Save all raw responses including failures, observer-independent agent prompts, validation outcomes, snapshots and provider usage. Separate raw events from processed metrics. Preserve negative and incomplete outcomes. Audit the privacy views and permission matrix using unit tests before changes. Replay recorded worlds without a model rather than regenerate historical decisions.

There is no LLM judge in v1.0. If one is added, keep its model/cost logs separate, redact condition labels where feasible, save rubric and classifications, and measure blinded agreement with human-coded samples. Do not feed judge outputs back into agents during the same study unless a new condition explicitly specifies that intervention.

Mock acceptance validates engineering correctness only. Results from paid studies must identify the provider model snapshot, cost/usage, runs per cell, failures, action denominators, effect distributions and limitations. No classification or generated utterance is evidence of consciousness, felt suffering, human-like emotions or real-world moral character.
