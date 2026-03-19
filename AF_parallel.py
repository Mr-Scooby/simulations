#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
import logging
import os
import numpy as np

log = logging.getLogger(__name__)

# Cached once per worker process
_G_KNHAT = None


def _init_af_worker(n_hat_flat, k_out):
    """Cache the angular grid in each worker."""
    global _G_KNHAT
    _G_KNHAT = k_out * np.asarray(n_hat_flat, dtype=float)


def _af_chunk_worker(args):
    """Compute the AF contribution of one atom chunk."""
    r_chunk, w_chunk = args

    # phase for all directions and all atoms in this chunk
    dots = _G_KNHAT @ r_chunk.T

    # sum over atoms in the chunk
    return np.exp(1j * dots) @ w_chunk


def array_factor_general_parallel(
    n_hat_flat,
    grid_shape,
    k_out,
    r_xyz,
    w=None,
    chunk_atoms=10000,
    n_workers=None,
):
    """
    Static array factor parallelized over atom chunks.

    Each worker receives one chunk of atoms and returns its AF contribution.
    The parent process sums all chunk contributions.
    """
    nt, np_ = grid_shape
    n_atoms = r_xyz.shape[0]
    n_dirs = n_hat_flat.shape[0]

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    if w is None:
        w = np.ones(n_atoms, dtype=np.complex128)
        log.info("AF parallel: using uniform weights")
    else:
        w = np.asarray(w, dtype=np.complex128)
        if w.shape != (n_atoms,):
            raise ValueError(f"w must have shape ({n_atoms},), got {w.shape}")
        log.info("AF parallel: using provided weights")

    # Build the list of atom chunks
    jobs = []
    for a0 in range(0, n_atoms, chunk_atoms):
        a1 = min(a0 + chunk_atoms, n_atoms)
        jobs.append((
            np.asarray(r_xyz[a0:a1], dtype=float),
            np.asarray(w[a0:a1], dtype=np.complex128),
        ))

    n_chunks = len(jobs)

    log.info(
        "AF parallel: start | n_atoms=%d, n_dirs=%d, chunk_atoms=%d, n_chunks=%d, n_workers=%d",
        n_atoms, n_dirs, chunk_atoms, n_chunks, n_workers
    )

    AF_flat = np.zeros(n_dirs, dtype=np.complex128)

    # Cache n_hat_flat once per worker, then send only chunks
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_af_worker,
        initargs=(n_hat_flat, k_out),
    ) as ex:
        for ic, af_chunk in enumerate(ex.map(_af_chunk_worker, jobs), start=1):
            AF_flat += af_chunk

            if ic == 1 or ic == n_chunks or ic % 10 == 0:
                log.info("AF parallel: processed chunk %d / %d", ic, n_chunks)

    log.info("AF parallel: finished")

    return AF_flat.reshape(nt, np_)
