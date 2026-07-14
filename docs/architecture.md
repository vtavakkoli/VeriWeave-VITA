# VeriWeave-VITA-BPA Architecture

VITA is the **Verification-Impact, Temporal, and Argumentation** layer that performs bounded coalitional stress testing, precedence-grounded argumentation, and policy-version replay.

```text
question + provisional decision
          |
          v
initial retriever ---------> initial policy subgraph
          |                           |
          v                           v
candidate LLM answer ------------> atomic claims
                                      |
                                      v
                     omitted policy-clause candidates
                                      |
                                      v
                 Boltzmann Policy Attention (BPA)
           unary energies + pairwise policy couplings
              adaptive temperature + round annealing
                                      |
                 marginal attention / mass certificate
                                      |
                                      v
                  bounded coalitional VITA tests
                                      |
             +------------------------+---------------------+
             |                        |                     |
             v                        v                     v
     decision space      grounded argumentation    temporal replay
             |                        |                     |
             +------------------------+---------------------+
                                      v
                       multi-certificate envelope
```

## Policy-set attention

BPA first receives a bounded high-recall candidate universe. It computes unary clause energies from relevance, decision risk, authority, currency, graph structure, and novelty. It then computes pairwise couplings from typed policy interactions. Exact coalition probabilities produce marginal attention and a selected clause set subject to a probability-mass target and hard budget.

The selected probability mass and residual mass replace the previous practice of reporting only residual raw retrieval score. This does not make the search complete; it makes the budget boundary explicit under the specified energy model.

## VITA closure

The selected pool is tested as singletons and bounded coalitions. Restrictive or synergistic witnesses can be added to the effective evidence set. BPA is rerun at the next round with an annealed temperature, allowing early exploration and later concentration.

The process stops when:

- no candidates remain;
- no new restrictive or synergistic effect is found;
- no selected evidence can be added; or
- the round budget is exhausted.

## Certificates

The Verification Envelope can include:

1. claim validation and precedence resolution;
2. Evidence Horizon Certificate;
3. VITA Decision-Space Certificate;
4. Argumentation Certificate;
5. Temporal Drift Certificate;
6. **Boltzmann Policy Attention Certificate**.

## Complexity

For `n` attention candidates and maximum policy-set size `m`, BPA exactly evaluates

```text
Σ C(n, k), k = 1..m
```

coalitions. The VITA layer then tests the selected evidence pool under its own bounded coalition budget. Defaults keep exact inference tractable. Large policy graphs require approximate inference and must report approximation error or calibration separately.
