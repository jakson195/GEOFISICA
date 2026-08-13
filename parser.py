import numpy as np

def parse_res2dinv(path):
    with open(path, 'r') as f:
        lines = [l.rstrip('\n').rstrip('\r') for l in f]

    title = lines[0].strip()
    a_spacing = float(lines[1].strip())
    array_type = int(float(lines[2].strip()))
    coord_flag = int(float(lines[3].strip()))  # 3 = general array, real x,z coords given
    # line[4] is a comment line: "Type of measurement..."
    meas_type = int(float(lines[5].strip()))   # 0 = apparent resistivity
    n_data = int(float(lines[6].strip()))
    # lines[7], lines[8] extra flags (topography/IP), not needed for our purposes

    data_start = 9
    readings = []
    for i in range(n_data):
        parts = lines[data_start + i].split()
        n_elec = int(parts[0])
        vals = list(map(float, parts[1:1 + 2 * n_elec]))
        value = float(parts[1 + 2 * n_elec])
        elecs = [(vals[2*k], vals[2*k+1]) for k in range(n_elec)]  # (x, elevation)
        readings.append({'electrodes': elecs, 'value': value})

    return {
        'title': title,
        'a_spacing': a_spacing,
        'array_type': array_type,
        'coord_flag': coord_flag,
        'meas_type': meas_type,
        'n_data': n_data,
        'readings': readings
    }


def geometric_factor(c1, c2, p1, p2):
    def r(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])
    term = 1.0/r(c1, p1) - 1.0/r(c1, p2) - 1.0/r(c2, p1) + 1.0/r(c2, p2)
    return 2*np.pi/term


if __name__ == '__main__':
    d = parse_res2dinv('/home/claude/ert/linha01.dat')
    print(d['title'], d['a_spacing'], d['array_type'], d['coord_flag'], d['meas_type'], d['n_data'])
    print('N leituras parseadas:', len(d['readings']))
    r0 = d['readings'][0]
    print(r0)
    c1, c2, p1, p2 = r0['electrodes']
    K = geometric_factor(c1, c2, p1, p2)
    print('K=', K, ' value(rho_a dado)=', r0['value'])

    all_x = sorted(set(e[0] for rdg in d['readings'] for e in rdg['electrodes']))
    print('Posicoes unicas de eletrodo (x):', all_x)
    print('n unicas:', len(all_x))
