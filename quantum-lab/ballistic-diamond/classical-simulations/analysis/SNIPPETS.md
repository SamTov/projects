# Damage snippets for lattice KMC

One file per simulated impact: the defects inside a sphere around the place
where the implanted ion came to rest.  Produced by `damage_ingest.py` from
the molecular-dynamics end states.

## What a file describes

- **Simulation**: one Sn ion, 35 keV, into a ⟨110⟩ diamond slab at 300 K
  (classical MD, Tersoff carbon + ZBL, Lindhard–Scharff electronic stopping).
- **State**: as quenched, 50 ps after the collision cascade.  The ballistic
  phase and prompt recombination are finished; thermally activated
  migration has not begun.  This is the starting point for annealing.
- **Region**: every bulk vacancy and interstitial within 10 nm (100 Å) of
  the final Sn position.
- **Defect definition**: Wigner–Seitz.  Each carbon atom is assigned to its
  nearest lattice site of a pristine reference crystal.  A site with no atom
  is a vacancy; each atom beyond the first on a site is an interstitial.

## File layout

`snippets/orient-110/energy-35/temperature-300/angle-<tilt>-<index>.xyz`,
extended XYZ:

```
<number of entries>
Properties=species:S:1:pos:R:3:lattice_n:I:3:kind:S:1 pbc="F F F" origin=... (metadata)
Sn   0.0000   0.0000   0.0000    0   0   0 ion
X    0.5963  -0.1533   0.1045    0   0   0 vacancy
C    1.2051   2.8810  -0.4402    2   2   0 interstitial
...
```

| column | meaning |
|---|---|
| species | `Sn` the ion, `X` a vacant lattice site, `C` an interstitial carbon atom |
| pos (3) | position in Å relative to the ion, in the simulation frame: x = [110], y = [00-1], z = [-110]; the beam travels along −z, so z > 0 is toward the surface |
| lattice_n (3) | the ideal diamond lattice site as integers in units of a/4 along the cubic axes [100], [010], [001], with a = 3.5678 Å; the origin is the lattice site nearest the ion |
| kind | `ion`, `vacancy`, `interstitial` |

For a vacancy, `lattice_n` is the empty site.  For an interstitial it is the
site the extra atom shares (its host site); `pos` is where the atom actually
sits.  For the ion it is always `0 0 0`, and the header field
`Sn_offset_from_site_A` gives the ion's displacement from that site.

With the origin on a lattice site, the diamond sites are exactly the integer
triples that are all even with a sum divisible by 4, plus one of the two
all-odd families (sum ≡ 1 or ≡ 3 mod 4, fixed within a file).  Cartesian
position of a site in the cubic frame: `(a/4) * lattice_n`.

## Header metadata

`E_keV`, `angle_deg` (beam tilt off the ⟨110⟩ axis), `T_K`, `ensemble` (grid
index of the entry point), `depth_nm` (ion depth below the surface),
`entry_x_A`, `entry_y_A` (where the ion crossed the surface), `n_vac`,
`n_int`, and the vacancy-cluster counts `V1` (isolated), `V2` (two adjacent
sites), `V3plus` — clusters are built by linking nearest-neighbour vacant
sites.

## Companion table

`summary.csv` has one row per impact with the same quantities plus the
vacancy counts along the whole ion track, the counts within 2 and 5 nm, the
number of vacant sites within 2 Å and 3 Å of the ion (`sn_nvac_2A`,
`sn_nvac_3A`), and quality indicators of the defect identification.

## Things to know before using them

- **Depth scale.**  A classical 300 K lattice vibrates less than real
  diamond (no zero-point motion), so ions channel roughly 25–40% too deep in
  absolute terms.  Which ions channel, and how little damage a channeled ion
  leaves, is set by geometry and is not affected.
- **One realisation per entry point.**  Each file is a single stochastic
  cascade; average over files of similar depth rather than reading one.
- **Deep-channeled ions end as interstitial Sn** with no vacancy next to
  them.  Forming an SnV centre from that state needs vacancies supplied
  from elsewhere, which is exactly what the annealing step has to answer.
- **No electronic effects on defects**: charge states and the SnV formation
  chemistry itself are outside a classical potential.
