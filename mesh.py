import numpy as np

def build_mesh(electrode_x, electrode_z, pad_frac=0.6, n_pad=10, nz=20, depth_max=None,
               growth=1.15, sub_per_gap=3):
    ex = np.array(sorted(set(electrode_x)))
    spacing = np.min(np.diff(ex))
    span = ex[-1] - ex[0]
    if depth_max is None:
        depth_max = 0.22 * span  # regra prática comum p/ dipolo-dipolo

    # subdivide o intervalo entre eletrodos p/ melhor resolucao horizontal
    fine_x = set(ex.tolist())
    for a, b in zip(ex[:-1], ex[1:]):
        for s in np.linspace(a, b, sub_per_gap+2)[1:-1]:
            fine_x.add(float(s))

    # nós x: eletrodos+subdivisoes + padding geometrico nas bordas. O contorno
    # precisa ficar longe em relacao ao ALCANCE HORIZONTAL da pesquisa (span),
    # nao apenas em relacao a profundidade de investigacao, senao a corrente
    # "vaza" para o contorno de Dirichlet e distorce o potencial calculado.
    target_pad = max(4.0*span, 10.0*spacing)
    pad_growth = (target_pad/spacing) ** (1.0/n_pad)
    left_pad = [ex[0] - spacing*(pad_growth**k) for k in range(1, n_pad+1)][::-1]
    right_pad = [ex[-1] + spacing*(pad_growth**k) for k in range(1, n_pad+1)]
    xs = np.array(sorted(set(list(left_pad) + list(fine_x) + list(right_pad))))

    # superficie interpolada linearmente nas posicoes de eletrodo, extrapolada plana nas bordas
    surf_x = np.array(sorted(set(electrode_x)))
    surf_z = np.array([np.mean([z for xx, z in zip(electrode_x, electrode_z) if xx == sx]) for sx in surf_x])
    surface = np.interp(xs, surf_x, surf_z, left=surf_z[0], right=surf_z[-1])

    # nos z (profundidade a partir da superficie local), espacamento geometrico crescente
    d0 = spacing/(3.0*(sub_per_gap+1))
    depths = [0.0]
    d = d0
    while depths[-1] < depth_max and len(depths) < nz:
        depths.append(depths[-1] + d)
        d *= growth
    # padding extra em profundidade para afastar o contorno de Dirichlet inferior:
    # deve alcancar distancia comparavel ao padding horizontal (target_pad),
    # senao o contorno raso "suga" o potencial e distorce os dados calculados
    while depths[-1] < target_pad:
        d *= growth
        depths.append(depths[-1] + d)
    depths = np.array(depths)

    return {
        'xs': xs, 'surface': surface, 'depths': depths,
        'nx': len(xs), 'nz': len(depths), 'spacing': spacing, 'depth_max': depth_max
    }


def node_index(mesh, ix, iz):
    return iz * mesh['nx'] + ix


def electrode_node(mesh, ex_):
    ix = int(np.argmin(np.abs(mesh['xs'] - ex_)))
    return node_index(mesh, ix, 0), ix


if __name__ == '__main__':
    from parser import parse_res2dinv
    d = parse_res2dinv('/home/claude/ert/linha01.dat')
    ex = [e[0] for rdg in d['readings'] for e in rdg['electrodes']]
    ez = [e[1] for rdg in d['readings'] for e in rdg['electrodes']]
    m = build_mesh(ex, ez)
    print('nx,nz=', m['nx'], m['nz'])
    print('xs=', m['xs'])
    print('depths=', m['depths'])
    print('depth_max=', m['depth_max'])
    print('surface=', m['surface'])
