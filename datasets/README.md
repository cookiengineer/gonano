# Domain replay / router test corpus

Minimal, human-editable corpora for exercising the domain meta-router and the
domain model bank end to end. Each subdirectory is one domain label; every
`.md` file is one document.

| Directory | Label | Focus |
|:----------|:------|:------|
| `physics/` | 0 | mechanics, thermodynamics, waves, energy |
| `math/` | 1 | algebra, geometry, calculus, vectors |
| `cooking/` | 2 | bread, soup, roasting |
| `trivia/` | 3 | general, science, and food facts |

The content is deliberately written to create **controlled overlaps** so the
router and any blending can be probed:

- `physics/energy-and-derivatives.md` and `math/calculus.md` share the language
  of rates, derivatives, and integrals.
- `math/vectors.md` and `physics/newtonian-mechanics.md` both describe vectors,
  velocity, and force.
- `physics/thermodynamics.md` and `cooking/roasting.md` both use heat,
  temperature, and browning.
- `trivia/science-trivia.md` and `trivia/food-trivia.md` overlap physics/math
  and cooking respectively, acting as the general "catch-all" domain.

Hold-out prompts used by `datasets_e2e_test.go` are built from these keywords to
check that the top-1 domain is right and that ambiguous prompts resolve to one
of the overlapping domains rather than a third, unrelated one.
