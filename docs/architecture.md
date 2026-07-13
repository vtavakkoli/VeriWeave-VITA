# ClaimGraph architecture

ClaimGraph separates language generation from policy verification:

```text
Natural-language request
        |
        v
Text or graph retrieval
        |
        v
Candidate answer -> atomic claim extraction
        |
        v
+---------------- typed property graph ----------------+
| Document --HAS_VERSION--> Version                    |
| Clause --DERIVED_FROM--> Document                    |
| Clause --APPLIES_TO--> Concept                       |
| Clause --SUPPORTS/CONTRADICTS--> Clause              |
| Document --OVERRIDES--> older Document               |
+-------------------------------------------------------+
        |
        v
Support | contradiction | applicability | temporal validity | provenance
        |
        v
Verified answer + claim-evidence subgraph + review routing
```

The repository uses an in-memory graph so experiments remain reproducible with only the Python standard library. `data/graph/property_graph.json` is generated at runtime and can be imported into a dedicated graph database.
