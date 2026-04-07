# Subwavelength Triangular Ladder DMRG

DMRG study of bosonic quantum phases in subwavelength optical lattices (brick wall / triangular ladder geometry) using [ITensors.jl](https://github.com/ITensor/ITensors.jl). Accompanies the paper [*Chiral and pair superfluidity in triangular ladder produced by state-dependent Kronig-Penney lattice*](https://arxiv.org/abs/2603.04498) (Burba et al., 2026).

## What it does

Computes the ground state of interacting bosons via DMRG and maps out quantum phase diagrams as a function of hopping and interaction strengths. Key phases identified: **Superfluid (SF)**, **Mott Insulator (MI)**, **Density Wave (DW)**, and **Pair Superfluid (PSF)**.

## Hamiltonians

- **Brick wall** (subwavelength lattice, s- and p-band variants) — main model
- **Bose-Hubbard** — standard reference
- **Extended BH** with nearest-neighbor interactions (V, D, P terms)
- **XXZ** spin chain
- **Zhou PSF** (deep lattice pair hopping)
- **Huber–Creutz ladder**

## Observables computed

- Density profile and density-density correlations
- Single-particle correlation `g1(i,j)` and pair correlation `g2`
- Density wave order parameter `δN`
- Superfluid and pair-superfluid structure factors
- Entanglement entropy and central charge (CFT fit)
- Topological order parameters (`proj_Q`, `proj_P`)

## Scripts

| File | Purpose |
|------|---------|
| `ITensor_DMRG_General_v2.jl` | Main DMRG script — Hamiltonians, observables, phase diagram routines |
| `ITensor_DMRG_XXZ_Hamiltonian_v2.jl` | XXZ spin chain DMRG |
| `ITensor_VUMPS_General_v1.jl` | Infinite-system VUMPS |
| `Bethe_Ansatz_Energy_v1.jl` | Exact Bethe Ansatz energies for benchmarking |
| `QOPack/` | Custom Julia package with shared utilities (DMRG helpers, exact diagonalization, fitting, plotting) |

## Requirements

Julia with: `ITensors`, `ITensorMPS`, `NPZ`, `HDF5`, `PyPlot`
