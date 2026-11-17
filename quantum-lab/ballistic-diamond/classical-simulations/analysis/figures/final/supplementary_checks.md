## Supplementary checks (random-entry cells, Sn 35 keV, ⟨110⟩)

### 1. Quench length: 200 ps against 50 ps

| quench | runs | mean offset from the 50 ps map relation [vacancies] |
|---|---|---|
| 200 ps | 45 | +4.6 |
| 50 ps | 25 | +5.5 |

Difference 200 ps − 50 ps: -0.9 vacancies per ion (95% interval -9.8 to +7.5), against 267 retained by a typical non-channeled ion.

### 2. Lattice vibration: 300 K against 1100 K at 0.5° tilt

| lattice | runs | 1D vibration amplitude [Å] | median depth [nm] | deeper than 100 nm | vacancies per ion | within 10 nm of Sn (median) | no vacancy within 10 nm | no divacancy or larger within 10 nm |
|---|---|---|---|---|---|---|---|---|
| 300 K | 40 | 0.0282 | 144 | 57% | 143 | 11 | 20% | 48% |
| 1100 K | 40 | 0.0546 | 77 | 30% | 186 | 48 | 0% | 12% |

Ions that stopped deeper than 150 nm:

| lattice | ions | vacancies along the track (median) | within 10 nm of Sn (median) | no divacancy or larger within 10 nm | Sn with no vacant site within 2 Å |
|---|---|---|---|---|---|
| 300 K | 20 | 20 | 1.5 | 95% | 85% |
| 1100 K | 8 | 47 | 6.5 | 50% | 50% |

Damage at a given depth, as the mean offset from the 300 K map relation (vacancies per ion at the same stop depth):

| lattice | along the whole track | within 10 nm of the Sn |
|---|---|---|
| 300 K | +5.3 (-0.0 to +11.0) | +2.7 (-3.5 to +9.8) |
| 1100 K | -8.9 (-15.3 to -2.5) | +3.7 (-4.4 to +11.4) |

### 3. Which classical temperature matches real diamond at 300 K

Measured in the simulations: ⟨u_x²⟩ = 7.96e-04 Å² at 300 K and 2.98e-03 Å² at 1100 K (classical lattice, so linear in temperature).

| target for real diamond at 300 K | ⟨u_x²⟩ [Å²] | 1D amplitude [Å] | classical lattice temperature that matches it |
|---|---|---|---|
| Debye model, θ_D = 2230 K | 1.52e-03 | 0.039 | 565 K |
| X-ray Debye–Waller, lower | 1.90e-03 | 0.044 | 704 K |
| X-ray Debye–Waller, upper | 2.30e-03 | 0.048 | 850 K |

The target values are quoted from memory of the literature and should be checked against a primary source before a temperature is fixed for publication.

### Registration quality of these runs

| lattice | residual [Å] | largest slice step [Å] | false vacancies in pristine control (max) |
|---|---|---|---|
| 300 K | 0.061 | 0.269 | 5 |
| 1100 K | 0.119 | 0.464 | 2 |
