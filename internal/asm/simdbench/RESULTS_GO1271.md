# Go 1.27.1 SIMD results and limits

## Summary

- SIMD kernels are still comparison candidates.
- Ordinary AMD64 dispatch keeps assembly, with isolated correctness repairs.
- On the measured host, the evidence supports specific improvements and
  explicit cost tradeoffs.
- The evidence does not show that every routine is faster than assembly.

## Evidence location

- Paths below are relative to `benchmark-results/simd-next-c7adb87b/` in the
  evidence archive.
- `final-go1271-results-scope-addendum-v3/EVIDENCE_INDEX.json` binds the
  detailed records by hash.
- This source file does not need to embed benchmark binaries and caches.

## Setup

Toolchain:

- The main nine binaries used stock **Go1.27.1**, `GOEXPERIMENT=simd`,
  AMD64v1, PGO off and CGO1.
- Metadata reports `go1.27.1-X:simd`. See `h2-go1271-toolchain-evidence-v1/`.
- The stock toolchain path is
  `/home/arjunanj/.local/share/mise/installs/go/1.27.1`. Archived metadata and
  build argv bind its use, not a custom compiler.
- Go1.24 supplies compatibility evidence.
- Custom-compiler experiments are separate and not adopted.

Host:

- Intel Core i7-11850H (8 performance cores/16 threads), with AVX2/AVX512.
- Benchmark processes use CPU2. Its SMT sibling is CPU10.
- The powersave governor stays unchanged.
- The records keep the exact process order, environment and host context.
- Software feature overrides and ARM64 compilation do not prove execution on
  all physical hardware.

## Historical main comparison

The historical main comparison is source790, with **53,556 usable
observations**:

- 49,452 completed direct/BLAS observations from the failed first session;
- 4,104 fresh complete matrix observations.

Limits of this data:

- The single set of 114 earlier matrix observations stays excluded as
  incomplete-cohort evidence.
- The failed session is not presented as a successful run. Missing in-memory
  timestamps were not fabricated.
- These are not measurements of the later combined source.

## Routine ledger

`h2-final-source-decision-ledger-preparation-v1/ROUTINE_SURFACE_58.json`
indexes all 58 routines:

- 57 manifest routines;
- the separate 48-observation math32.Sqrt supplement. It did not pass its
  both-width direct significance gate.

The ledger keeps:

- **2,484 positive costs**: 1,413 versus ASM, 995 versus c7, 76 versus
  production;
- all 1,185 pointwise-material IDs;
- the overlapping historical 32/139/1,379 cohorts.

Positive medians are not necessarily significant. Material/Holm tags are
review signals, not automatic selection rules.

## Retained changes and explicit costs

### Unitary L1 ASM infinity repair (FBB)

- 35 finite identities/560 observations all had lower medians, 0 positive
  costs.
- This compares repaired and original ASM. It is not a Go SIMD gain.
- Authority: `l1norm-asm-repair-root-retention-v1/DISPOSITION.json`.

### Strided L1 ASM infinity repair (MH)

- 76 identities/1,216 observations had 70 lower and 6 positive medians, 0
  material costs.
- The records keep all six and the actual original repeated-infinity failures.
- R6 supersedes `l1norminc-repair-root-retention-v1/RETENTION.json` only for
  the old zero-stride expectations.

### Zero-stride contract repair (R6)

The change: return +0 without reading input. Results versus MH-repaired ASM:

- 76 identities keep 55 positive, 14 pointwise-material and 9 Holm costs, with
  0 Holm gains.
- Worst direct case, n100000/inc1: +2,824ns (+10.5679%).
- Dlangb MaxColumnSum m=n256/kl15/ku16/ldab34: +177ns (+4.9663%). This is
  Holm-significant although it is below 5%.
- Ten of 12 consumer medians are positive.

This is a required correctness tradeoff, not a speed win. Authorities:

- `l1norminc-zero-stride-root-retention-v1/RETENTION.json`;
- `l1norminc-zero-guard-finite-analysis-v1/ALL_POSITIVE_COSTS.json`.

### Four-chain L1 vector remainder

- 10,944 observations/1,140 declared comparisons keep 286 positive costs.
- Versus current: 26 positive, 0 pointwise-material, 1 Holm cost and 171 Holm
  gains.
- Versus repaired ASM: 46 positive, 13 material, 15 Holm costs and 170 Holm
  gains.
- The retained n32/spread/512 control costs +0.295ns (+4.592869%). The
  unchanged short source does not explain or remove this cost.
- Authority: `l1norm-long-vector-remainder-root-retention-v1/`.

### f32 short Sum pointer plus shared zero initializers

- Of 69 changed n4..31 identities, 67 medians are lower and 52 are Holm gains
  versus current.
- All 126 identities keep 10 current costs. These include material/Holm n64
  packed controls:
  - width256: +0.6655ns (+10.889307%);
  - width512: +0.673ns (+11.463124%).
- Versus ASM, 30 costs/22 material remain. All three contrasts keep 122 costs.
- Both measured arms include the zero workaround, so this does not measure its
  isolated speed effect.
- There is no claim of a production f32 Sum consumer gain or an
  assembly-dispatch replacement.
- Authority: `f32-sum-short-pointer-root-retention-v1/RETENTION.json`.

### K67 c128 stride-pair (rejected)

- The completed 258-identity/9,288-observation trial and a separate grouped
  audit keep 292 positive/139 material costs across all 774 contrasts.
- Of 72 changed pair-versus-current contexts, 64 cost more (31 Holm costs, 28
  material). 8 have lower medians (3 Holm gains).
- The 186 controls keep 63 costs (5 Holm costs, 2 material) and 123 lower
  medians (17 Holm gains).
- The instruction-count hypothesis did not justify selection. Keep the existing
  c128 source.
- Authority: `c128-zscal-stride-pair-root-disposition-v1/DISPOSITION.json`.
  Full exact rows: `c128-zscal-stride-pair-complete-decision-ledger-v1/`.

### Changes already in source790

Source790 already contains:

- the Q3 valid-unaligned c128 AXPY repair;
- checked dot-span/widened-Sdsdot correctness changes;
- the restored c7 DscalUnitary SIMD entry.

The separate Q3 aligned-cost trial keeps 47 positive, 3 Holm and 0 material
costs (maximum +4.636%). Dot evidence separates AMD64 full-width products from
386 helper calls. Source equality does not waive any correctness or rollback
cost.

## Five historical material consumer costs

| Actual caller / width | Cost |
|---|---:|
| Caxpy n5, strides1/1 /512 | +0.810ns, +7.479% |
| Sger n8, inc1 /512 | +1.810ns, +7.594% |
| Dger m8/n32/pad0 /512 | +2.745ns, +5.987% |
| Sger m64/n129/inc7 /256 | +71ns, +6.592% |
| Zscal n33/inc7 /256 | +1.305ns, +7.018% |

- Exact rows: `h2-final-source-decision-ledger-preparation-v1/FIVE_CONSUMER_COSTS.json`.
- Routes: `h2-five-consumer-cost-source-review-v1/`.
- The later RQPAGI GER mask proposal failed actual codegen. That rejection does
  not prove a hardware speed limit.

Later K67 measurement of Zscal33/inc7/256:

- current 19.85ns, pair 21.51ns, production 21.795ns;
- pair/current +8.363% is a Holm cost;
- pair/production is inconclusive.

That separate session neither removes the original +7.018% cost nor explains
its change.

## Nineteen historical ASM-losing classes

- These classes contain 46 positive medians among four manifest cases each.
  They are not 19 routines that are always slow or significantly slow.
- Exact IDs, prior trials and source-only hypotheses:
  `h2-final-source-decision-ledger-preparation-v1/class-evidence.json`.
- Updated machine index:
  `final-go1271-results-scope-addendum-v3/CLASS_STATUS_19.json`.

| Routine | Positive /4 | Investigation and remaining limit |
|---|---:|---|
| c128.DotcInc | 2 | Clear/return trial rejected; custom-compiler extraction measured but unadopted; full-span correctness retained. Short extraction/ABI costs remain. |
| c128.DotuInc | 2 | Same strided-return and exact public-span investigations. No trusted-caller bypass; short costs remain. |
| c128.DscalInc | 3 | Tiny/layout trial rejected; c7 unitary entry restored;32 actual Dscal/mat routes and554-ID link diagnostic reviewed. Source/code equality does not waive costs. |
| c64.AxpyInc | 4 | Equal-stride two-pointer trial retained with95 controls per width. Mixed/reverse/sparse entry costs remain. |
| c64.AxpyIncTo | 4 | Same equal-stride two-pointer trial with destination semantics preserved. All four historical manifest medians remain positive. |
| c64.DotuInc | 2 | Entry flattening retained; paired-span and sign-first candidates rejected for actual extra hot reloads, before correctness/timing. |
| c64.DotuUnitary | 2 | Width/cache routes and actual unitary/public handoff inspected. Unit stride bypasses new full-span work; no new alignment-fault claim. |
| f32.DdotInc | 2 | AVX2 admission/cache retained; accumulator seeding rejected; widened Sdsdot bias fixed separately. Conversion/reduction/ABI costs remain. |
| f32.DdotUnitary | 2 | Same widened-reduction investigations; custom compiler unadopted. No universal conversion or hardware ceiling proved. |
| f32.DotInc | 2 | Pair packing retained; five-argument raw-leaf trial rejected on callers/controls. Nonunit public callers retain checked spans. |
| f32.Ger | 2 | Register tails retained; larger entries rejected; RQPAGI n&7 screen rejected at frame280 to288. Awkward shapes remain. |
| f32.Sum | 2 | Pointer plus shared zero initialization retained after4536 rows.10 current costs including2 material n64 controls remain; no production consumer gain. |
| f64.AxpyInc | 2 | Gather/address implementation retained; unrolled-tail consumer trial rejected. Sparse/short costs remain. |
| f64.AxpyIncTo | 2 | Same address/tail caller investigation, preserving destination semantics. Further hypotheses need actual caller evidence. |
| f64.Div | 2 | Vector division and actual42-shape floats caller trial completed. Reset/alias costs retained; no divider/bandwidth bound proved. |
| f64.L1Norm | 1 | Four-chain vector remainder selected after10944 rows; separate ASM correctness repairs.46 positive ASM comparisons/13 material;7RSCCJ limit remains. |
| f64.ScalInc | 4 | Running addresses retained; RN6 measured/rejected;3WE pretest codegen rejected; actual ScaleVec/link-layout controls reviewed. Tiny/nonunit costs remain. |
| f64.ScalIncTo | 4 | Same running-address, tail and consumer controls. No global alignment selection or proof all alternatives exhausted. |
| f64.Sum | 2 | Actual floats callers complete; static-length entry rejected after4032 rows/504 contrasts. All215 positive/62 material trial costs remain. |

Rejected changes stay as evidence, not deliverables:

- RN6 tiny Scal;
- 3WE loop entry;
- WINU64 Sum entry;
- RQPAGI GER mask;
- both c64 shared-span candidates;
- the K67 c128 pair;
- custom-compiler extraction;
- global funcalign64.

Specific frame/reload/caller-cost failures do not prove that every alternative
is exhausted. The separate floats trial keeps 119 positive/41 material
comparisons. Its Norm1 is an unchanged scalar control, not long L1Norm.

## Compatibility

### AVX-only route

- The original f32 pointer gate stays rejected. An existing 128-bit AVX-only
  route emitted an AVX2 broadcast.
- Common zero-value declarations remove this broadcast in the separately
  reviewed revised f32 routes.
- The **separate 7RSCCJ L1 portable zero/Abs lowering issue remains open**.
- Tested AVX2/AVX512 eligibility does not give global physical AVX-only
  acceptance.
- Archived codegen does not show a physical crash.

### checkptr

- Full f32 checkptr2 keeps four named allocation failures.
- The matched current-zero control reproduces 2/1/2 allocations.
- The exact classification keeps the failing suite/raw outcomes. It is not a
  full checkptr PASS or a generic waiver.
- Other ordinary/race/safe/noasm checks are separate.
- Temperature/throttle snapshots and repeated-round diagnostics neither make
  the samples independent nor explain away costs.

## Final-source evidence

### Composition

- Records: `final-go1271-composition-v1/{INITIAL_COMPONENTS,R6_COMPONENTS,F32_COMPONENTS}.json`.
- They map the ASM repairs, zero-stride guard/regressions, selected L1
  entry/helper and f32 entry/common initializers.
- Two fixture changes are explicit integration adaptations:
  - optional L1 environment expectations;
  - removal of the experiment-only mode assertion in f32.
- Numerical/guard assertions stay intact.
- The Level1 generator and generated Sdsdot source require parity.

### Validation authority

The authority for the combined tree of the delivered bundle is:

- its actual final validation receipts;
- the linked codegen/N1 records;
- the source/patch replay identities.

The validation plan requires one full SIMD ./... run. Ordinary/safe/noasm/Go1.24
runs cover all chosen compatibility packages, plus focused Dlangb/L1/R6 LAPACK
checks.

Limits:

- Historical broad LAPACK and standalone candidate evidence keeps its original
  identity.
- This document does not relabel historical timing as final-source timing.
  It does not assert future validation success.
- The old 790 exporter is not valid for the changed tree.

Delivery records must:

- keep excluded data, failed attempts and all costs;
- omit build caches;
- verify exact-origin patch replay and device readback.
