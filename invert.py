import numpy as np
import time
from parser import parse_res2dinv, geometric_factor
from mesh import build_mesh, electrode_node
from forward import solve_potentials_for_sources
from sensitivity import build_jacobian, node_mask_core

def prepare(path):
    d = parse_res2dinv(path)
    ex_all = [e[0] for rdg in d['readings'] for e in rdg['electrodes']]
    ez_all = [e[1] for rdg in d['readings'] for e in rdg['electrodes']]
    mesh = build_mesh(ex_all, ez_all)

    # eletrodos unicos (posicao x) -> indice
    uniq_x = sorted(set(ex_all))
    x_to_idx = {x: k for k, x in enumerate(uniq_x)}
    elec_node_ix = []
    for x in uniq_x:
        node, ix = electrode_node(mesh, x)
        elec_node_ix.append((0, ix))  # eletrodos sempre na superficie (linha j=0)

    for rdg in d['readings']:
        c1, c2, p1, p2 = rdg['electrodes']
        rdg['K'] = geometric_factor(c1, c2, p1, p2)
        rdg['enode'] = [x_to_idx[c1[0]], x_to_idx[c2[0]], x_to_idx[p1[0]], x_to_idx[p2[0]]]
        rdg['node_ix'] = [elec_node_ix[x_to_idx[e[0]]] for e in [c1, c2, p1, p2]]

    return d, mesh, uniq_x, elec_node_ix


def smoothness_matrix(mesh, core_mask):
    """matriz de rugosidade (diferencas 1a ordem horiz+vert) nos nos do core."""
    nx, nz = mesh['nx'], mesh['nz']
    idx_map = -np.ones((nz, nx), dtype=int)
    core_idx = np.where(core_mask.ravel())[0]
    idx_map.ravel()[core_idx] = np.arange(len(core_idx))

    rows = []
    n = len(core_idx)
    Lrows, Lcols, Lvals = [], [], []
    r = 0
    for j in range(nz):
        for i in range(nx):
            if idx_map[j, i] < 0:
                continue
            a = idx_map[j, i]
            if i+1 < nx and idx_map[j, i+1] >= 0:
                b = idx_map[j, i+1]
                Lrows += [r, r]; Lcols += [a, b]; Lvals += [1.0, -1.0]; r += 1
            if j+1 < nz and idx_map[j+1, i] >= 0:
                b = idx_map[j+1, i]
                Lrows += [r, r]; Lcols += [a, b]; Lvals += [1.0, -1.0]; r += 1
    import scipy.sparse as sp
    L = sp.csr_matrix((Lvals, (Lrows, Lcols)), shape=(r, n))
    return L


def run_inversion(path, n_iter=6, n_k=18, verbose=True, progress_cb=None):
    d, mesh, uniq_x, elec_node_ix = prepare(path)
    nx, nz = mesh['nx'], mesh['nz']
    readings = d['readings']

    obs_rhoa = np.array([r_['value'] for r_ in readings])
    Ks = np.array([r_['K'] for r_ in readings])
    obs_R = obs_rhoa / Ks  # transfer resistance observada

    span = max(uniq_x) - min(uniq_x)
    depth_max = mesh['depth_max']
    core_mask = node_mask_core(mesh, min(uniq_x)-mesh['spacing'], max(uniq_x)+mesh['spacing'], depth_max)
    n_core = core_mask.sum()
    if verbose:
        print(f"malha: nx={nx} nz={nz} total_nos={nx*nz}  core_nos={n_core}")

    # modelo inicial: semi-espaco homogeneo = mediana da resistividade aparente observada
    rho0 = float(np.median(obs_rhoa))
    log_rho = np.full((nz, nx), np.log(rho0))
    if verbose:
        print(f"rho0 inicial (mediana obs) = {rho0:.1f} ohm.m")

    L = smoothness_matrix(mesh, core_mask)
    core_idx = np.where(core_mask.ravel())[0]

    lam = None  # calibrado dinamicamente na 1a iteracao (escala de J varia MUITO
                # em relacao a escala da matriz de suavidade L; um lambda fixo
                # "generico" deixa a regularização dominando sempre e o modelo nao
                # se move em direção aos dados)
    history = []
    best_rms = np.inf
    best_log_rho = log_rho.copy()
    t_start = time.time()
    for it in range(n_iter):
        sigma_nodes = np.exp(-log_rho)
        # eletrodos unicos: os nos de superficie correspondentes a cada eletrodo
        src_nodes = [j*nx+i for (j, i) in elec_node_ix]
        pots = solve_potentials_for_sources(mesh, sigma_nodes, src_nodes, n_k=n_k)
        phi_fields = np.array([pots[s].reshape(nz, nx) for s in src_nodes])

        J, R_pred, _ = build_jacobian(readings, elec_node_ix, phi_fields, sigma_nodes, mesh, core_mask)

        resid = np.log(obs_R) - np.log(np.abs(R_pred))
        rms = 100*np.sqrt(np.mean(resid**2))
        history.append(rms)
        if rms < best_rms:
            best_rms = rms
            best_log_rho = log_rho.copy()
            best_R_pred = R_pred.copy()

        JTJ = J.T @ J
        JTr = J.T @ resid
        LTL = (L.T @ L).toarray()

        if lam is None:
            # calibra a escala relativa entre termo de dados e termo de suavidade
            ratio = np.trace(JTJ) / max(np.trace(LTL), 1e-30)
            lam = 40.0 * ratio
            lam_min = 0.3 * ratio

        if verbose:
            print(f"iter {it}: RMS(log)={rms:.2f}%   lambda={lam:.3e}   t={time.time()-t_start:.1f}s")
        if progress_cb:
            progress_cb(it, n_iter, rms)

        # amortecimento de norma minima (estabiliza a direção "nivel medio", que a
        # matriz de suavidade pura (L) nao controla - padrao em inversao tipo Occam)
        eps = 0.03 * max(np.diag(LTL).mean(), 1e-6) * lam
        A = JTJ + lam*LTL + eps*np.eye(len(core_idx))
        try:
            dm = np.linalg.solve(A, JTr)
        except np.linalg.LinAlgError:
            dm = np.linalg.lstsq(A, JTr, rcond=None)[0]

        # amortece reescalando (preserva direção) em vez de recortar componente a componente
        maxabs = np.max(np.abs(dm))
        step_limit = 0.6
        if maxabs > step_limit:
            dm = dm * (step_limit/maxabs)
        log_rho.ravel()[core_idx] += dm
        lam = max(lam*0.55, lam_min)

    return {
        'mesh': mesh, 'log_rho': best_log_rho, 'core_mask': core_mask,
        'history': history, 'best_rms': best_rms, 'obs_rhoa': obs_rhoa, 'R_pred': best_R_pred,
        'Ks': Ks, 'readings': readings, 'uniq_x': uniq_x
    }


if __name__ == '__main__':
    res = run_inversion('/home/claude/ert/linha01.dat', n_iter=6, n_k=14)
    print("RMS final:", res['history'][-1])
