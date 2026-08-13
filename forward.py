import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def harm(a, b):
    return 2*a*b/(a+b)

def build_operator(mesh, sigma_nodes, k):
    """Monta a matriz esparsa (divergencia(sigma*grad) - k^2*sigma) para um dado
    numero de onda k, discretizacao nodal tipo Patankar (harmonic mean nas faces),
    grade nao-uniforme, topo com fluxo nulo (Neumann natural), bordas laterais/inferior
    Dirichlet V=0."""
    xs, depths = mesh['xs'], mesh['depths']
    nx, nz = mesh['nx'], mesh['nz']
    N = nx*nz

    hx = np.diff(xs)             # nx-1
    hz = np.diff(depths)         # nz-1

    rows, cols, vals = [], [], []
    rhs_diag_boundary = np.zeros(N, dtype=bool)

    def idx(i, j):
        return j*nx + i

    for j in range(nz):
        for i in range(nx):
            n = idx(i, j)
            on_boundary = (i == 0 or i == nx-1 or j == nz-1)
            if on_boundary:
                rows.append(n); cols.append(n); vals.append(1.0)
                rhs_diag_boundary[n] = True
                continue

            hx_w, hx_e = hx[i-1], hx[i]
            hz_d = hz[j]  # abaixo
            hz_u = hz[j-1] if j > 0 else None

            Ax = (hx_w + hx_e)/2.0
            if j == 0:
                dz_local = hz_d/2.0
            else:
                dz_local = (hz_u + hz_d)/2.0

            s0 = sigma_nodes[j, i]
            diag = 0.0

            # oeste
            sw = harm(s0, sigma_nodes[j, i-1])
            Tw = sw * dz_local / hx_w
            rows.append(n); cols.append(idx(i-1, j)); vals.append(Tw)
            diag -= Tw

            # leste
            se = harm(s0, sigma_nodes[j, i+1])
            Te = se * dz_local / hx_e
            rows.append(n); cols.append(idx(i+1, j)); vals.append(Te)
            diag -= Te

            # sul (mais profundo)
            ss = harm(s0, sigma_nodes[j+1, i])
            Ts = ss * Ax / hz_d
            rows.append(n); cols.append(idx(i, j+1)); vals.append(Ts)
            diag -= Ts

            # norte (mais raso) - so existe se j>0
            if j > 0:
                sn = harm(s0, sigma_nodes[j-1, i])
                Tn = sn * Ax / hz_u
                rows.append(n); cols.append(idx(i, j-1)); vals.append(Tn)
                diag -= Tn

            CV = Ax * dz_local
            diag -= (k**2) * s0 * CV

            rows.append(n); cols.append(n); vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A


# --- quadratura em numero de onda (transformada cosseno, y=0) ---
def wavenumbers(n=25, kmin=1e-5, kmax=10.0):
    u = np.linspace(np.log(kmin), np.log(kmax), n)
    k = np.exp(u)
    # pesos trapezoidais em log(k), incluindo jacobiano dk = k du
    w = np.gradient(u) * k
    return k, w


def solve_potentials_for_sources(mesh, sigma_nodes, source_node_indices, n_k=25):
    """Retorna dict: source_node_idx -> vetor potencial real (N,) no plano y=0,
    para corrente unitaria injetada em cada no fonte."""
    xs, depths = mesh['xs'], mesh['depths']
    nx, nz = mesh['nx'], mesh['nz']
    N = nx*nz
    ks, ws = wavenumbers(n_k, 1e-5, 10.0)

    pot_accum = {s: np.zeros(N) for s in source_node_indices}

    for k, w in zip(ks, ws):
        A = build_operator(mesh, sigma_nodes, k)
        lu = spla.splu(A.tocsc())
        for s in source_node_indices:
            b = np.zeros(N)
            b[s] = -1.0  # corrente unitaria; RHS=-I na formulacao div(sigma grad V) - k^2 sigma V = -I delta
            v_hat = lu.solve(b)
            pot_accum[s] += (1.0/np.pi) * w * v_hat

    return pot_accum
