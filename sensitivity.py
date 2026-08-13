import numpy as np

def cell_gradients(phi, mesh):
    """phi: array (nz,nx) potencial no plano y=0 p/ corrente unitaria numa fonte.
    Retorna Ex,Ez em cada CELULA (nz-1,nx-1): derivadas medias nas bordas da celula."""
    xs, depths = mesh['xs'], mesh['depths']
    hx = np.diff(xs)      # (nx-1,)
    hz = np.diff(depths)  # (nz-1,)

    top = (phi[:-1, 1:] - phi[:-1, :-1]) / hx[None, :]
    bot = (phi[1:, 1:] - phi[1:, :-1]) / hx[None, :]
    Ex = 0.5*(top+bot)

    left = (phi[1:, :-1] - phi[:-1, :-1]) / hz[:, None]
    right = (phi[1:, 1:] - phi[:-1, 1:]) / hz[:, None]
    Ez = 0.5*(left+right)
    return Ex, Ez  # (nz-1, nx-1)


def cell_volumes(mesh):
    hx = np.diff(mesh['xs'])
    hz = np.diff(mesh['depths'])
    return np.outer(hz, hx)  # (nz-1, nx-1)


def node_mask_core(mesh, xmin, xmax, zmax):
    xs, depths = mesh['xs'], mesh['depths']
    xin = (xs >= xmin) & (xs <= xmax)
    zin = depths <= zmax
    mask = np.outer(zin, xin)  # (nz,nx) bool
    return mask


def lump_cell_to_nodes(cellmap, nz, nx):
    """Distribui um mapa por-celula (nz-1,nx-1) igualmente para os 4 nos de canto
    de cada celula, acumulando (mass lumping)."""
    node = np.zeros((nz, nx))
    node[:-1, :-1] += 0.25*cellmap
    node[:-1, 1:]  += 0.25*cellmap
    node[1:, :-1]  += 0.25*cellmap
    node[1:, 1:]   += 0.25*cellmap
    return node


def build_jacobian(readings, elec_node_ix, phi_fields, sigma_nodes, mesh, core_mask):
    """
    readings: lista de dicts com 'electrodes' (x,z) e chave auxiliar 'enode' (indices dos 4 eletrodos na lista unica)
    phi_fields: array (n_elec, nz, nx) potencial p/ corrente unitaria em cada eletrodo unico
    Retorna: J (n_data, n_core_nodes) em espaco log(rho_pred) / log(rho_node), e R_pred (n_data,) transfer resistance prevista
    """
    nx, nz = mesh['nx'], mesh['nz']
    n_data = len(readings)
    core_idx = np.where(core_mask.ravel())[0]
    n_core = len(core_idx)
    J = np.zeros((n_data, n_core))
    R_pred = np.zeros(n_data)

    Vol = cell_volumes(mesh)  # (nz-1,nx-1)
    Ex_list = []
    Ez_list = []
    for e in range(phi_fields.shape[0]):
        Ex, Ez = cell_gradients(phi_fields[e], mesh)
        Ex_list.append(Ex); Ez_list.append(Ez)
    Ex_arr = np.array(Ex_list)  # (n_elec, nz-1, nx-1)
    Ez_arr = np.array(Ez_list)

    for d, rdg in enumerate(readings):
        iC1, iC2, iP1, iP2 = rdg['enode']
        phiC1 = phi_fields[iC1]; phiC2 = phi_fields[iC2]
        phiP1 = phi_fields[iP1]; phiP2 = phi_fields[iP2]

        # transfer resistance prevista (I=1): V(P1)-V(P2) p/ dipolo de corrente (C1:+1,C2:-1)
        Vfield = phiC1 - phiC2
        jP1, iP1n = rdg['node_ix'][2]
        jP2, iP2n = rdg['node_ix'][3]
        R = Vfield[jP1, iP1n] - Vfield[jP2, iP2n]
        R_pred[d] = R

        ExC = Ex_arr[iC1]-Ex_arr[iC2]; EzC = Ez_arr[iC1]-Ez_arr[iC2]
        ExP = Ex_arr[iP1]-Ex_arr[iP2]; EzP = Ez_arr[iP1]-Ez_arr[iP2]
        Scell = -(ExC*ExP + EzC*EzP) * Vol   # (nz-1,nx-1) : dR/dsigma_cell

        Snode = lump_cell_to_nodes(Scell, mesh['nz'], mesh['nx'])  # (nz,nx)
        # dR/dln(rho_node) = dR/dsigma_node * dsigma/dlnrho = Snode * (-sigma_node)
        dRdlnrho = Snode * (-sigma_nodes)
        # d ln(rho_pred)/d ln(rho_node) = (1/R) * dR/dlnrho_node
        row = (dRdlnrho / R).ravel()[core_idx]
        J[d, :] = row

    return J, R_pred, core_idx
