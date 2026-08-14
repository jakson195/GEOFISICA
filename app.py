import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import tempfile, os, io, csv

from invert import run_inversion

st.set_page_config(page_title="Inversão 2D de Eletrorresistividade", layout="wide")

st.title("Inversão 2D de Eletrorresistividade (ERT)")
st.caption(
    "Ferramenta própria de inversão de dados de resistividade aparente (formato geral "
    "estilo Res2dinv). Modelagem direta 2.5D (domínio de número de onda) + inversão "
    "Gauss-Newton amortecida com regularização de suavidade. **Não é o software "
    "comercial RES2DINV** — é uma implementação independente, mais simples, feita para "
    "uso exploratório."
)


def dat_text_from_table(df, esp, title="Linha manual", elev_map=None):
    """Converte a tabela de entrada (A,B,M,N em piquetes + R ap) para o mesmo
    formato .dat (arranjo geral, tipo 11) que o parser já sabe ler.
    Convenção: posicao_x = (piquete - 1) * ESP.
    elev_map: dict {piquete: elevação(m)} opcional; sem isso assume topografia plana (z=0)."""
    elev_map = elev_map or {}
    def z_of(piquete):
        return float(elev_map.get(piquete, 0.0))

    rows = []
    for _, r in df.iterrows():
        try:
            a, b, m, n = float(r['A']), float(r['B']), float(r['M']), float(r['N'])
            rap = float(r['R ap'])
        except (ValueError, TypeError):
            continue
        if pd.isna(a) or pd.isna(b) or pd.isna(m) or pd.isna(n) or pd.isna(rap):
            continue
        xa, xb, xm, xn = (a-1)*esp, (b-1)*esp, (m-1)*esp, (n-1)*esp
        za, zb, zm, zn = z_of(a), z_of(b), z_of(m), z_of(n)
        rows.append(f"4  {xa:.3f} {za:.3f}  {xb:.3f} {zb:.3f}  {xm:.3f} {zm:.3f}  {xn:.3f} {zn:.3f}  {rap:.6f}")

    lines = [
        title, f"{esp:.3f}", "11", "3",
        "Type of measurement (0=app.resistivity,1=resistance)",
        "0", str(len(rows)), "2", "0",
    ] + rows
    return "\n".join(lines)


def geometric_calcs(df, esp):
    df = df.copy()
    n = pd.to_numeric(df['NÍVEL'], errors='coerce')
    a_ = pd.to_numeric(df['A'], errors='coerce'); b_ = pd.to_numeric(df['B'], errors='coerce')
    m_ = pd.to_numeric(df['M'], errors='coerce'); nn_ = pd.to_numeric(df['N'], errors='coerce')
    sp = pd.to_numeric(df['SP (mV)'], errors='coerce')
    v = pd.to_numeric(df['V (mV)'], errors='coerce')
    i_ = pd.to_numeric(df['i (mA)'], errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        g = 1.0/((1.0/n) - (2.0/(n+1)) + (1.0/(n+2)))
        k = 2*3.14*g*esp
        rap = ((v - sp)/i_ * k).abs()
    df['G'] = g
    df['K'] = k
    df['R ap'] = rap
    return df


def parse_num(s):
    s = s.strip()
    if s == "" or s.lower() in ("nan", "-"):
        return np.nan
    s = s.replace(" ", "")
    if "," in s and "." in s:
        # padrão BR: ponto = milhar, virgula = decimal
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_pasted_block(text):
    """Aceita bloco colado do Excel (separado por TAB), com ou sem cabecalho,
    em 3 layouts possiveis (colunas na mesma ordem da planilha original):
      11 col: MEDIDA PIQUETE ESP A B M N NIVEL SP V i
      10 col: MEDIDA PIQUETE A B M N NIVEL SP V i        (sem ESP, ja fixo no app)
       8 col: A B M N NIVEL SP V i                        (sem MEDIDA/PIQUETE)
    Retorna (dataframe, esp_detectado_ou_None, mensagem_erro_ou_None)."""
    lines = [l for l in text.strip().split("\n") if l.strip() != ""]
    if not lines:
        return None, None, "Nada para processar."

    rows = [l.split("\t") for l in lines]
    ncols = len(rows[0])

    # remove linha de cabecalho se a 1a celula nao for numerica
    first_cell = rows[0][0].strip().replace(",", ".")
    try:
        float(first_cell)
    except ValueError:
        rows = rows[1:]
    if not rows:
        return None, None, "Nenhuma linha de dado encontrada (só cabeçalho?)."

    esp_detectado = None
    out = []
    for r in rows:
        r = r + [""] * (ncols - len(r))  # completa se faltar celula
        vals = [parse_num(x) for x in r]
        if ncols >= 11:
            medida, piquete, esp, a, b, m, n, niv, sp, v, i_ = vals[:11]
            esp_detectado = esp
        elif ncols == 10:
            medida, piquete, a, b, m, n, niv, sp, v, i_ = vals[:10]
        elif ncols == 8:
            a, b, m, n, niv, sp, v, i_ = vals[:8]
            medida, piquete = np.nan, np.nan
        else:
            return None, None, (
                f"Colei {ncols} colunas — esperado 8 (A,B,M,N,NÍVEL,SP,V,i), "
                f"10 (+MEDIDA,PIQUETE) ou 11 (+ESP). Confira a seleção no Excel."
            )
        out.append({"MEDIDA": medida, "PIQUETE": piquete, "A": a, "B": b, "M": m, "N": n,
                     "NÍVEL": niv, "SP (mV)": sp, "V (mV)": v, "i (mA)": i_})

    df = pd.DataFrame(out)
    if df["MEDIDA"].isna().all():
        df["MEDIDA"] = range(1, len(df)+1)
    return df, esp_detectado, None


def dat_text_from_readings(readings, a_spacing, title="filtrado"):
    """Reconstroi o .dat (arranjo geral) a partir de uma lista de leituras ja
    parseadas (cada uma com 'electrodes':[(x,z)x4] e 'value'). Usado para re-rodar
    a inversão depois de excluir pontos ruidosos, sem depender do arquivo original."""
    lines = [title, f"{a_spacing:.3f}", "11", "3",
             "Type of measurement (0=app.resistivity,1=resistance)",
             "0", str(len(readings)), "2", "0"]
    for r in readings:
        parts = ["4"]
        for (x, z) in r['electrodes']:
            parts.append(f"{x:.3f}"); parts.append(f"{z:.3f}")
        parts.append(f"{r['value']:.6f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


with st.expander("Formato de arquivo esperado (upload .dat)", expanded=False):
    st.markdown("""
    Formato tipo **Res2dinv - arranjo geral (código 11)**, com coordenadas reais (x, elevação)
    de cada eletrodo por leitura:

    ```
    Título
    Espaçamento base entre eletrodos
    11
    3
    Type of measurement (0=app.resistivity,1=resistance)
    0
    Número de leituras
    2
    0
    4  x1 z1  x2 z2  x3 z3  x4 z4   valor
    4  x1 z1  x2 z2  x3 z3  x4 z4   valor
    ...
    ```
    Cada leitura: eletrodos na ordem **C1, C2, P1, P2** (corrente, corrente, potencial, potencial),
    seguidos do valor de resistividade aparente (ohm·m). Testado com dados em arranjo dipolo-dipolo.
    """)

tab_upload, tab_manual = st.tabs(["📁 Importar arquivo .dat", "✍️ Entrada manual de dados"])

tmp_path_to_run = None

with tab_upload:
    uploaded = st.file_uploader("Importar arquivo .dat", type=["dat", "txt"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
            content = uploaded.getvalue().decode("utf-8", errors="ignore")
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            tmp.write(content.encode("utf-8"))
            tmp_path_upload = tmp.name
    else:
        tmp_path_upload = None

with tab_manual:
    st.markdown(
        "**Copie o bloco de dados direto da sua planilha Excel** (selecione as células, "
        "Ctrl+C) **e cole aqui** (Ctrl+V). Aceita 3 formatos de colagem, na mesma ordem "
        "de colunas da sua planilha:"
    )
    st.markdown(
        "- **11 colunas**: MEDIDA, PIQUETE, ESP, A, B, M, N, NÍVEL, SP, V, i\n"
        "- **10 colunas**: MEDIDA, PIQUETE, A, B, M, N, NÍVEL, SP, V, i *(ESP definido abaixo)*\n"
        "- **8 colunas**: A, B, M, N, NÍVEL, SP, V, i *(mínimo necessário)*\n\n"
        "Pode incluir a linha de cabeçalho ou não — o app detecta sozinho. "
        "Números com vírgula decimal (padrão BR) são aceitos."
    )

    esp_manual = st.number_input("ESP do projeto (m)", value=st.session_state.get("esp_from_paste", 15.0),
                                  min_value=0.01, step=0.5,
                                  help="Usado se a colagem não incluir a coluna ESP. "
                                       "Se detectado na colagem, já vem preenchido aqui.")

    pasted = st.text_area("Colar dados aqui (Ctrl+V)", height=150, key="paste_area",
                           placeholder="Cole aqui o bloco copiado do Excel...")

    colp1, colp2 = st.columns([1, 3])
    with colp1:
        process_btn = st.button("Processar colagem", type="secondary")

    if "manual_df" not in st.session_state:
        st.session_state["manual_df"] = pd.DataFrame([
            {"MEDIDA": i+1, "PIQUETE": 1, "A": 1, "B": 2, "M": 3+i, "N": 4+i,
             "NÍVEL": i+1, "SP (mV)": 0.0, "V (mV)": 0.0, "i (mA)": 0.0}
            for i in range(13)
        ])

    if process_btn:
        if pasted.strip() == "":
            st.warning("Cole algum dado antes de processar.")
        else:
            df_new, esp_detectado, err = parse_pasted_block(pasted)
            if err:
                st.error(err)
            else:
                st.session_state["manual_df"] = df_new
                if esp_detectado is not None and not np.isnan(esp_detectado):
                    st.session_state["esp_from_paste"] = float(esp_detectado)
                st.success(f"{len(df_new)} linhas importadas da colagem.")

    st.caption("Depois de colar, você ainda pode ajustar célula a célula na tabela abaixo:")

    edited = st.data_editor(
        st.session_state["manual_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="manual_editor",
        column_config={
            "MEDIDA": st.column_config.NumberColumn(format="%d"),
            "PIQUETE": st.column_config.NumberColumn(format="%d"),
            "A": st.column_config.NumberColumn(format="%d"),
            "B": st.column_config.NumberColumn(format="%d"),
            "M": st.column_config.NumberColumn(format="%d"),
            "N": st.column_config.NumberColumn(format="%d"),
            "NÍVEL": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.session_state["manual_df"] = edited

    computed = geometric_calcs(edited, esp_manual)
    st.markdown("**Calculado automaticamente (G, K, R ap):**")
    st.dataframe(
        computed[["MEDIDA", "PIQUETE", "A", "B", "M", "N", "NÍVEL", "SP (mV)", "V (mV)", "i (mA)", "G", "K", "R ap"]],
        use_container_width=True, hide_index=True
    )

    valid = computed.dropna(subset=["A", "B", "M", "N", "R ap"])
    st.caption(f"{len(valid)} de {len(computed)} linhas prontas para inversão (com A,B,M,N,SP,V,i preenchidos).")

    with st.expander("📐 Topografia (opcional) — elevação por piquete", expanded=False):
        st.caption(
            "Por padrão o terreno é considerado plano. Se quiser inserir a topografia real "
            "(depois de já ter os dados calculados), preencha a elevação de cada piquete abaixo."
        )
        piquetes_usados = sorted(set(
            pd.concat([valid["A"], valid["B"], valid["M"], valid["N"]]).dropna().astype(int).tolist()
        ))
        if "topo_df" not in st.session_state or "Piquete" not in st.session_state["topo_df"].columns \
                or set(st.session_state["topo_df"]["Piquete"]) != set(piquetes_usados):
            prev = st.session_state.get("topo_df")
            prev_map = dict(zip(prev["Piquete"], prev["Elevação (m)"])) if prev is not None and "Piquete" in prev.columns else {}
            st.session_state["topo_df"] = pd.DataFrame(
                [{"Piquete": p, "Elevação (m)": prev_map.get(p, 0.0)} for p in piquetes_usados],
                columns=["Piquete", "Elevação (m)"]
            )
        topo_edited = st.data_editor(
            st.session_state["topo_df"], use_container_width=True, hide_index=True, key="topo_editor",
            column_config={"Piquete": st.column_config.NumberColumn(format="%d", disabled=True)}
        )
        st.session_state["topo_df"] = topo_edited
        usar_topo = st.checkbox("Aplicar esta topografia na inversão", value=False)

    elev_map = None
    if usar_topo:
        elev_map = dict(zip(st.session_state["topo_df"]["Piquete"], st.session_state["topo_df"]["Elevação (m)"]))

    if len(valid) >= 4:
        dat_text = dat_text_from_table(computed, esp_manual, title="Entrada manual", elev_map=elev_map)
        st.download_button("Baixar .dat gerado", dat_text, file_name="entrada_manual.dat", mime="text/plain")
        if st.button("Usar estes dados na inversão", key="use_manual"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dat", mode="w") as tmp:
                tmp.write(dat_text)
                tmp_path_to_run = tmp.name
            st.session_state["tmp_path_manual"] = tmp_path_to_run
    else:
        st.info("Preencha ao menos 4 leituras completas (A,B,M,N,SP,V,i) para habilitar a inversão.")

col1, col2, col3 = st.columns(3)
with col1:
    n_iter = st.slider("Iterações da inversão", 4, 20, 12)
with col2:
    n_k = st.slider("Resolução da quadratura (nº de números de onda)", 8, 30, 16,
                     help="Mais = mais preciso e mais lento.")
with col3:
    can_run_upload = tmp_path_upload is not None
    run_btn = st.button("Rodar inversão (arquivo importado)", type="primary", disabled=not can_run_upload)

col4, col5 = st.columns(2)
with col4:
    sub_per_gap = st.select_slider(
        "Resolução horizontal da malha", options=[1, 2, 3, 4, 5, 6], value=3,
        help="Subdivisões entre eletrodos vizinhos. Mais = imagem mais detalhada, porém mais lento."
    )
with col5:
    nz_layers = st.select_slider(
        "Resolução vertical da malha (nº de camadas)", options=[10, 14, 18, 20, 24, 28, 32], value=20,
        help="Mais camadas = mais detalhe em profundidade, porém mais lento."
    )

robust_weighting = st.checkbox(
    "Ponderação robusta (reduz automaticamente o peso de leituras ruidosas)", value=True,
    help="Técnica tipo Huber: pontos com erro muito acima do típico pesam menos na "
         "inversão a cada iteração, sem precisar excluí-los manualmente. Recomendado "
         "manter ligado; para os casos mais difíceis, combine com a exclusão manual "
         "de outliers mais abaixo, depois de rodar."
)
depth_weight_power = st.slider(
    "Peso de profundidade (relaxa a suavidade em zonas profundas/pouco sensíveis)",
    0.0, 2.0, 0.0, 0.25,
    help="0 = padrão (suavidade igual em toda a malha). Valores maiores permitem mais "
         "contraste em regiões profundas onde os dados têm pouca sensibilidade — em "
         "geral melhora o RMS, mas region\u00f5es de baixa cobertura de dados (perto das "
         "bordas, em profundidade) continuam sendo mal restringidas por natureza: nem "
         "o RES2DINV nem este app têm como 'adivinhar' o que os dados não enxergam."
)

if "result" not in st.session_state:
    st.session_state["result"] = None

run_now_path = None
if run_btn and tmp_path_upload is not None:
    run_now_path = tmp_path_upload
elif st.session_state.get("tmp_path_manual"):
    run_now_path = st.session_state.pop("tmp_path_manual")

if run_now_path is not None:
    tmp_path = run_now_path
    progress_bar = st.progress(0.0, text="Iniciando...")
    rms_log = []

    def cb(it, n_total, rms):
        progress_bar.progress((it+1)/n_total, text=f"Iteração {it+1}/{n_total} — RMS(log) = {rms:.1f}%")
        rms_log.append(rms)

    try:
        with st.spinner("Resolvendo modelagem direta e invertendo..."):
            res = run_inversion(tmp_path, n_iter=n_iter, n_k=n_k, verbose=False, progress_cb=cb,
                                 sub_per_gap=sub_per_gap, nz_layers=nz_layers, robust=robust_weighting,
                                 depth_weight_power=depth_weight_power)
        st.session_state["result"] = res
        progress_bar.progress(1.0, text="Concluído.")
    except Exception as e:
        st.error(f"Falha ao processar o arquivo: {e}")
        st.session_state["result"] = None
    finally:
        os.unlink(tmp_path)

res = st.session_state["result"]

if res is not None:
    mesh = res['mesh']
    log_rho = res['log_rho']
    core_mask = res['core_mask']
    xs, depths = mesh['xs'], mesh['depths']
    surface = mesh['surface']
    rho = np.exp(log_rho)
    rho_masked = np.where(core_mask, rho, np.nan)

    Xg, Dg = np.meshgrid(xs, depths)
    Zg = surface[None, :] - Dg

    uniq_x = res['uniq_x']
    xmin_core = min(uniq_x) - mesh['spacing']
    xmax_core = max(uniq_x) + mesh['spacing']
    x_ref = min(uniq_x)  # posicao do 1o eletrodo = referencia da estaca inicial

    # estado de exclusao de leituras: unico, compartilhado entre clique nos graficos
    # e a tabela de exclusao manual. Reinicia quando um novo resultado é carregado.
    if st.session_state.get("excl_for_id") != id(res):
        st.session_state["excluded_idx"] = set()
        st.session_state["excl_for_id"] = id(res)

    st.subheader("Nomenclatura de estacas")
    colest1, colest2 = st.columns(2)
    with colest1:
        estaca_espac = st.number_input("Espaçamento entre estacas (m)", value=20.0, min_value=0.1, step=1.0)
    with colest2:
        estaca_inicial = st.number_input("Nome da estaca inicial (nº)", value=0, step=1,
                                          help="Ex.: se o primeiro eletrodo deve se chamar EST 20, "
                                               "digite 20 aqui — a sequência (21, 22, ...) segue automática.")

    def estaca_num(x):
        return int(round(estaca_inicial + (x - x_ref) / estaca_espac))

    def add_estaca_axis(ax, xmin, xmax):
        ax_top = ax.secondary_xaxis('top')
        first_tick = x_ref + estaca_espac * np.floor((xmin - x_ref) / estaca_espac)
        ticks = np.arange(first_tick, xmax + estaca_espac, estaca_espac)
        ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]
        if len(ticks) == 0:
            return ax_top
        labels = [f"EST {estaca_num(t)}" for t in ticks]
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(labels, rotation=45, fontsize=7)
        return ax_top

    st.subheader("Escala de cores (resistividade)")
    obs_all_tmp = res['obs_rhoa']
    rho_data_min = float(min(np.nanpercentile(rho_masked, 2), np.percentile(obs_all_tmp, 2)))
    rho_data_max = float(max(np.nanpercentile(rho_masked, 98), np.percentile(obs_all_tmp, 98)))
    colsc1, colsc2, colsc3 = st.columns([1, 1, 1])
    with colsc1:
        auto_scale = st.checkbox("Escala automática (padrão)", value=True)
    with colsc2:
        vmin_user = st.number_input("Mínimo (ohm.m)", value=round(rho_data_min, 1),
                                     min_value=0.01, disabled=auto_scale)
    with colsc3:
        vmax_user = st.number_input("Máximo (ohm.m)", value=round(rho_data_max, 1),
                                     min_value=0.01, disabled=auto_scale)

    if auto_scale:
        vmin_plot = np.log10(max(rho_data_min, 0.01))
        vmax_plot = np.log10(max(rho_data_max, rho_data_min*1.01))
    else:
        vmin_plot = np.log10(max(vmin_user, 0.01))
        vmax_plot = np.log10(max(vmax_user, vmin_user*1.01))
    st.caption(
        "Escala automática usa o intervalo entre os percentis 2% e 98% dos dados "
        "(evita que 1 ou 2 células extremas, comuns em inversões deste tipo, "
        "\"esmaguem\" as cores do resto da seção). Ajuste manualmente se preferir outro corte."
    )

    def apply_colorbar_ticks(cbar, vmin_log, vmax_log):
        vmin, vmax = 10**vmin_log, 10**vmax_log
        exp_start = int(np.floor(np.log10(vmin)))
        exp_end = int(np.ceil(np.log10(vmax)))
        ticks = []
        for e in range(exp_start, exp_end + 1):
            for m in (1, 2, 5):
                v = m * 10.0**e
                if vmin * 0.98 <= v <= vmax * 1.02:
                    ticks.append(v)
        ticks = sorted(set(ticks))
        if len(ticks) < 2:
            ticks = list(np.logspace(vmin_log, vmax_log, 5))
        cbar.set_ticks([np.log10(t) for t in ticks])
        cbar.set_ticklabels([f"{t:,.0f}" if t >= 1 else f"{t:.2f}" for t in ticks])

    def pseudo_coords(readings):
        """Pseudo-posicao (x,z) de cada leitura p/ visualizar os dados brutos:
        x = ponto medio dos 4 eletrodos; z = -abertura total do arranjo * 0.35
        (aproximação genérica, valida pra qualquer arranjo, so para inspeção visual —
        não é usada nos cálculos da inversão)."""
        px, pz = [], []
        for r in readings:
            elec_x = [e[0] for e in r['electrodes']]
            px.append(np.mean(elec_x))
            pz.append(-(max(elec_x)-min(elec_x))*0.35)
        return np.array(px), np.array(pz)

    def interactive_pseudo_section(px, pz, color_vals, color_label, hover_extra, key, colorscale='turbo'):
        """Pseudo-secao interativa (Plotly): clique/selecione (retangulo ou laço, no
        menu da direita do gráfico) para marcar leituras a excluir. Pontos já marcados
        aparecem com X preto por cima."""
        excl = st.session_state["excluded_idx"]
        n = len(px)
        estacas_hover = [f"EST {estaca_num(x)}" for x in px]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=px, y=pz, mode='markers',
            marker=dict(size=13, color=color_vals, colorscale=colorscale,
                        colorbar=dict(title=color_label), line=dict(width=1, color='black')),
            customdata=np.stack([np.arange(n), estacas_hover, hover_extra], axis=1),
            hovertemplate="%{customdata[1]}<br>x=%{x:.1f} m<br>" + color_label + "=%{customdata[2]}<extra></extra>",
            name="leituras",
        ))
        if excl:
            idxs = [i for i in excl if i < n]
            if idxs:
                fig.add_trace(go.Scatter(
                    x=px[idxs], y=pz[idxs], mode='markers',
                    marker=dict(size=16, symbol='x-thin', line=dict(width=3, color='black')),
                    name="excluído", hoverinfo='skip'
                ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Distância ao longo da linha (m)", yaxis_title="Pseudo-profundidade",
            yaxis=dict(showticklabels=False), dragmode='select',
            showlegend=False,
        )
        event = st.plotly_chart(fig, on_select="rerun", selection_mode=("points", "box", "lasso"),
                                 key=key, use_container_width=True)
        try:
            points = event.selection.points if hasattr(event, "selection") else event.get("selection", {}).get("points", [])
        except Exception:
            points = []
        if points:
            for pt in points:
                pidx = pt.get("customdata", [None])[0] if isinstance(pt, dict) else None
                if pidx is None and isinstance(pt, dict):
                    pidx = pt.get("point_index")
                if pidx is not None:
                    st.session_state["excluded_idx"].add(int(pidx))

    st.subheader("Pseudo-seção dos dados observados")
    st.caption(
        "Visualização espacial dos pontos medidos em campo (posição × profundidade "
        "aproximada), coloridos pela resistividade aparente observada — antes da inversão. "
        "**Clique num ponto (ou use o laço/retângulo de seleção) para marcar como ruído/excluir.**"
    )
    px, pz = pseudo_coords(res['readings'])
    obs_all = res['obs_rhoa']
    interactive_pseudo_section(
        px, pz, obs_all, "Resistividade (ohm.m)",
        hover_extra=[f"{v:.1f} ohm.m" for v in obs_all], key="obs_pseudo_plotly"
    )
    n_excl_now = len(st.session_state["excluded_idx"])
    if n_excl_now:
        st.caption(f"🔴 {n_excl_now} leitura(s) marcada(s) para exclusão (veja/edite a lista completa "
                    "na seção 'Identificar e excluir pontos com ruído', mais abaixo).")

    st.subheader("Seção de resistividade invertida")
    colviz1, colviz2, colviz3 = st.columns([1, 1, 1])
    with colviz1:
        show_isolines = st.checkbox("Mostrar isolinhas", value=False)
        n_isolines = st.slider("Nº de isolinhas", 4, 20, 10, disabled=not show_isolines)
    with colviz2:
        chanfrar = st.checkbox("Chanfrar bordas (corta cantos de baixa cobertura)", value=True)
        chanfro_angulo = st.slider("Ângulo do corte (graus, 45° = padrão)", 20, 70, 45,
                                    disabled=not chanfrar,
                                    help="Ângulo a partir da vertical nas bordas. Elimina a área "
                                         "nos cantos que a malha extrapola/interpola sem cobertura "
                                         "real de dados dos eletrodos das pontas.")
    with colviz3:
        vert_exag = st.slider("Exagero vertical", 0.5, 4.0, 1.0, 0.1,
                               help="1.0 = escala real (sem distorção). Maior = estica a profundidade visualmente.")
        fig_width = st.slider("Escala horizontal (largura, pol.)", 6, 20, 11)

    st.markdown("**Faixa de elevação exibida** (o RES2DINV costuma mostrar uma faixa mais rasa que a malha calculada):")
    colel1, colel2 = st.columns(2)
    elev_default_top = float(surface.max()) + 3
    elev_default_bottom = float(surface.min() - mesh['depth_max'])
    with colel1:
        elev_top = st.number_input("Elevação máxima (topo, m)", value=round(elev_default_top, 1))
    with colel2:
        elev_bottom = st.number_input("Elevação mínima (base, m)", value=round(elev_default_bottom, 1))

    fig, ax = plt.subplots(figsize=(fig_width, 5))
    lograho_masked = np.log10(rho_masked)

    if chanfrar:
        slope = 1.0 / np.tan(np.radians(chanfro_angulo))  # profundidade permitida por metro de afastamento da borda
        dist_left = Xg - xmin_core
        dist_right = xmax_core - Xg
        dist_edge = np.minimum(dist_left, dist_right)
        depth_grid = surface[None, :] - Zg  # profundidade real de cada no da malha
        chanfro_mask = depth_grid <= np.maximum(dist_edge, 0) * slope
        lograho_masked = np.where(chanfro_mask, lograho_masked, np.nan)

    vmin_s = vmin_plot if vmin_plot is not None else np.nanmin(lograho_masked)
    vmax_s = vmax_plot if vmax_plot is not None else np.nanmax(lograho_masked)
    pc = ax.pcolormesh(Xg, Zg, lograho_masked, shading='auto', cmap='turbo', vmin=vmin_s, vmax=vmax_s)
    cbar = fig.colorbar(pc, ax=ax, label='Resistividade (ohm.m)')
    apply_colorbar_ticks(cbar, vmin_s, vmax_s)
    if show_isolines:
        masked = np.ma.masked_invalid(lograho_masked)
        cs = ax.contour(Xg, Zg, masked, levels=n_isolines, colors='black', linewidths=0.5, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"{10**v:.0f}")
    elev_e = [surface[np.argmin(np.abs(xs-x))] for x in uniq_x]
    ax.scatter(uniq_x, elev_e, marker='v', color='k', s=25, zorder=5, label='Eletrodos')
    ax.set_xlim(xmin_core, xmax_core)
    ax.set_ylim(elev_bottom, elev_top)
    ax.set_xlabel('Distância ao longo da linha (m)')
    ax.set_ylabel('Elevação (m)')
    ax.set_title(f"RMS de ajuste (log) = {res['best_rms']:.1f}%")
    ax.set_aspect(vert_exag)
    ax.legend(loc='lower right')
    add_estaca_axis(ax, xmin_core, xmax_core)
    plt.tight_layout()
    st.pyplot(fig)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', dpi=150)
    st.download_button("Baixar seção (PNG)", buf_png.getvalue(), file_name="secao_resistividade.png", mime="image/png")

    st.subheader("Classificação por faixas / Seção interpretativa geológica")
    st.caption(
        "Defina faixas de resistividade com cor e (opcionalmente) uma classificação "
        "geológica/geotécnica para cada uma. Usado para colorir a seção em faixas "
        "discretas (em vez de escala contínua) e para gerar uma seção interpretativa "
        "com hachuras, pronta para relatório."
    )

    HATCH_OPTIONS = {
        "nenhuma": "", "pontos": ".", "linhas diagonais": "/", "linhas diagonais (inv.)": "\\",
        "linhas cruzadas": "x", "linhas horizontais": "-", "linhas verticais": "|",
        "tijolo": "+", "círculos": "o", "estrelas": "*",
    }
    COLOR_OPTIONS = ["blue", "green", "red", "orange", "purple", "brown", "gray",
                      "cyan", "magenta", "gold", "darkgreen", "navy", "chocolate", "black"]

    if "class_df" not in st.session_state:
        st.session_state["class_df"] = pd.DataFrame([
            {"De (ohm.m)": 0, "Até (ohm.m)": 500, "Rótulo": "Argila saturada", "Cor": "blue", "Hachura": "nenhuma"},
            {"De (ohm.m)": 500, "Até (ohm.m)": 1500, "Rótulo": "Rocha alterada", "Cor": "green", "Hachura": "linhas diagonais"},
            {"De (ohm.m)": 1500, "Até (ohm.m)": 4500, "Rótulo": "Rocha sã", "Cor": "red", "Hachura": "linhas cruzadas"},
        ])

    class_edited = st.data_editor(
        st.session_state["class_df"], num_rows="dynamic", use_container_width=True, key="class_editor",
        column_config={
            "Cor": st.column_config.SelectboxColumn(options=COLOR_OPTIONS),
            "Hachura": st.column_config.SelectboxColumn(options=list(HATCH_OPTIONS.keys())),
        }
    )
    st.session_state["class_df"] = class_edited
    class_df_valid = class_edited.dropna(subset=["De (ohm.m)", "Até (ohm.m)", "Cor"]).sort_values("De (ohm.m)")

    colcl1, colcl2, colcl3 = st.columns(3)
    with colcl1:
        usar_classificado = st.checkbox("Colorir a seção principal com estas faixas (em vez de escala contínua)", value=False)
    with colcl2:
        gerar_interpretativa = st.checkbox("Gerar seção interpretativa com hachuras (para relatório)", value=False)
    with colcl3:
        suavizar = st.checkbox("Suavizar contornos (interpolação)", value=True,
                                help="Interpola a malha para uma grade mais fina só para desenho — "
                                     "deixa as bordas entre classes com curvas suaves em vez de "
                                     "seguir o contorno em degraus das células da malha de cálculo. "
                                     "Não altera o modelo, só a exibição.")

    def build_fine_grid(mesh, log_rho_full, xmin_core, xmax_core, n_x=320, n_z=160):
        from scipy.interpolate import RectBivariateSpline
        xs_, depths_ = mesh['xs'], mesh['depths']
        # interpolação BIlinear (kx=ky=1): spline cúbica "dispara" (overshoot) perto de
        # células isoladas com valor extremo, comuns nesse tipo de inversão — linear evita isso
        spline = RectBivariateSpline(depths_, xs_, log_rho_full, kx=1, ky=1)
        xs_fine = np.linspace(xmin_core, xmax_core, n_x)
        depth_max_plot = mesh['depth_max']
        depths_fine = np.linspace(0, depth_max_plot, n_z)
        logrho_fine = spline(depths_fine, xs_fine)
        # limite de seguranca: nunca extrapolar alem do range real do modelo
        logrho_fine = np.clip(logrho_fine, log_rho_full.min(), log_rho_full.max())
        surface_fine = np.interp(xs_fine, xs_, mesh['surface'])
        Dg_fine = np.meshgrid(xs_fine, depths_fine)[1]
        Xg_fine = np.meshgrid(xs_fine, depths_fine)[0]
        Zg_fine = surface_fine[None, :] - Dg_fine
        return Xg_fine, Zg_fine, Dg_fine, logrho_fine, xs_fine, depths_fine, surface_fine

    if len(class_df_valid) >= 1:
        bounds = sorted(set(class_df_valid["De (ohm.m)"].tolist() + class_df_valid["Até (ohm.m)"].tolist()))
        cores_lista = class_df_valid["Cor"].tolist()
        rotulos_lista = class_df_valid["Rótulo"].fillna("").tolist()
        hachuras_lista = [HATCH_OPTIONS.get(h, "") for h in class_df_valid["Hachura"].tolist()]

        if suavizar:
            Xg_plot, Zg_plot, Dg_plot, logrho_plot, xs_p, depths_p, surface_p = build_fine_grid(
                mesh, log_rho, xmin_core, xmax_core)
            rho_plot_base = np.exp(logrho_plot)  # log_rho é log NATURAL (ln), não log10
            if chanfrar:
                slope_p = 1.0 / np.tan(np.radians(chanfro_angulo))
                dist_edge_p = np.minimum(Xg_plot - xmin_core, xmax_core - Xg_plot)
                chanfro_mask_p = Dg_plot <= np.maximum(dist_edge_p, 0) * slope_p
                rho_plot_base = np.where(chanfro_mask_p, rho_plot_base, np.nan)
        else:
            Xg_plot, Zg_plot = Xg, Zg
            rho_plot_base = np.where(chanfro_mask, rho_masked, np.nan) if chanfrar else rho_masked

        if usar_classificado:
            import matplotlib.colors as mcolors
            cmap_classes = mcolors.ListedColormap(cores_lista)
            norm_classes = mcolors.BoundaryNorm(bounds, cmap_classes.N, clip=True)

            fig_c, ax_c = plt.subplots(figsize=(fig_width, 5))
            ax_c.pcolormesh(Xg_plot, Zg_plot, rho_plot_base, shading='gouraud' if suavizar else 'auto',
                             cmap=cmap_classes, norm=norm_classes)
            ax_c.scatter(uniq_x, elev_e, marker='v', color='k', s=25, zorder=5, label='Eletrodos')
            ax_c.set_xlim(xmin_core, xmax_core)
            ax_c.set_ylim(elev_bottom, elev_top)
            ax_c.set_xlabel('Distância ao longo da linha (m)')
            ax_c.set_ylabel('Elevação (m)')
            ax_c.set_aspect(vert_exag)
            legend_patches = [mpatches.Patch(
                facecolor=c, label=f"{r if r else ''} ({b0:.0f}–{b1:.0f} ohm.m)".strip())
                for c, r, b0, b1 in zip(cores_lista, rotulos_lista,
                                         class_df_valid["De (ohm.m)"], class_df_valid["Até (ohm.m)"])]
            ax_c.legend(handles=legend_patches, loc='lower right', fontsize=8)
            ax_c.set_title("Seção classificada por faixas")
            add_estaca_axis(ax_c, xmin_core, xmax_core)
            plt.tight_layout()
            st.pyplot(fig_c)
            buf_c = io.BytesIO()
            fig_c.savefig(buf_c, format='png', dpi=150)
            st.download_button("Baixar seção classificada (PNG)", buf_c.getvalue(),
                                file_name="secao_classificada.png", mime="image/png")

        if gerar_interpretativa:
            rho_clipped = np.clip(rho_plot_base, bounds[0], bounds[-1] * 0.999999)
            rho_clipped = np.ma.masked_invalid(rho_clipped)
            fig_i, ax_i = plt.subplots(figsize=(fig_width, 5))
            cs_i = ax_i.contourf(Xg_plot, Zg_plot, rho_clipped,
                                  levels=bounds, colors=cores_lista, hatches=hachuras_lista, extend='neither')
            for coll in cs_i.collections if hasattr(cs_i, 'collections') else []:
                coll.set_edgecolor('black')
                coll.set_linewidth(0.0)
            ax_i.scatter(uniq_x, elev_e, marker='v', color='k', s=25, zorder=5, label='Eletrodos')
            ax_i.set_xlim(xmin_core, xmax_core)
            ax_i.set_ylim(elev_bottom, elev_top)
            ax_i.set_xlabel('Distância ao longo da linha (m)')
            ax_i.set_ylabel('Elevação (m)')
            ax_i.set_aspect(vert_exag)
            legend_patches_i = [mpatches.Patch(
                facecolor=c, hatch=h, edgecolor='black',
                label=f"{r if r else ''} ({b0:.0f}–{b1:.0f} ohm.m)".strip())
                for c, h, r, b0, b1 in zip(cores_lista, hachuras_lista, rotulos_lista,
                                            class_df_valid["De (ohm.m)"], class_df_valid["Até (ohm.m)"])]
            ax_i.legend(handles=legend_patches_i, loc='lower right', fontsize=8)
            ax_i.set_title("Seção interpretativa (classificação geológica)")
            add_estaca_axis(ax_i, xmin_core, xmax_core)
            plt.tight_layout()
            st.pyplot(fig_i)
            buf_i = io.BytesIO()
            fig_i.savefig(buf_i, format='png', dpi=200)
            st.download_button("Baixar seção interpretativa (PNG, p/ relatório)", buf_i.getvalue(),
                                file_name="secao_interpretativa.png", mime="image/png")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Convergência do ajuste")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(res['history'], marker='o')
        ax2.axhline(res['best_rms'], color='green', ls='--', label='melhor RMS')
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('RMS (log) %')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    with colB:
        st.subheader("Observado vs. previsto")
        obs = res['obs_rhoa']
        pred = np.abs(res['R_pred'] * res['Ks'])
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.loglog(obs, pred, 'o', alpha=0.6)
        lims = [min(obs.min(), pred.min())*0.7, max(obs.max(), pred.max())*1.4]
        ax3.plot(lims, lims, 'k--', lw=1)
        ax3.set_xlabel('Observado (ohm.m)')
        ax3.set_ylabel('Previsto (ohm.m)')
        plt.tight_layout()
        st.pyplot(fig3)

    st.subheader("Exportar modelo")
    rows = []
    for j in range(mesh['nz']):
        for i in range(mesh['nx']):
            if core_mask[j, i]:
                rows.append((xs[i], surface[i]-depths[j], depths[j], rho[j, i]))
    csv_buf = io.StringIO()
    w = csv.writer(csv_buf)
    w.writerow(['x_m', 'elevacao_m', 'profundidade_m', 'resistividade_ohm_m'])
    w.writerows(rows)
    st.download_button("Baixar modelo invertido (CSV)", csv_buf.getvalue(),
                        file_name="modelo_invertido.csv", mime="text/csv")

    st.subheader("Identificar e excluir pontos com ruído")
    st.caption(
        "Compara cada leitura observada com a prevista pelo modelo atual. Erros grandes "
        "geralmente indicam mau contato de eletrodo ou ruído de campo. Marque o que quiser "
        "excluir e rode a inversão de novo — deve melhorar o RMS."
    )

    obs = res['obs_rhoa']
    pred = np.abs(res['R_pred'] * res['Ks'])
    log_err_pct = 100*np.abs(np.log10(pred/obs))
    diff_pct = 100*(pred-obs)/obs

    diag_rows = []
    for idx, r in enumerate(res['readings']):
        (c1, _), (c2, _), (p1, _), (p2, _) = r['electrodes']
        diag_rows.append({
            "Excluir": idx in st.session_state["excluded_idx"],
            "#": idx,
            "C1 (m)": c1, "C2 (m)": c2, "P1 (m)": p1, "P2 (m)": p2,
            "Observado (ohm.m)": round(obs[idx], 1),
            "Previsto (ohm.m)": round(pred[idx], 1),
            "Erro (%)": round(diff_pct[idx], 1),
            "Erro |log| (%)": round(log_err_pct[idx], 1),
        })
    diag_df = pd.DataFrame(diag_rows).sort_values("Erro |log| (%)", ascending=False).reset_index(drop=True)

    st.markdown("**Onde estão os pontos com mais ruído (pseudo-seção do erro):** "
                 "clique num ponto (ou use o laço/retângulo) para marcar como ruído.")
    px_d, pz_d = pseudo_coords(res['readings'])
    interactive_pseudo_section(
        px_d, pz_d, log_err_pct, "Erro |log| (%)",
        hover_extra=[f"{v:.1f}%" for v in log_err_pct], key="err_pseudo_plotly", colorscale='Reds'
    )
    st.caption("Pontos mais vermelhos/escuros = maior discrepância entre observado e previsto.")

    colx1, colx2 = st.columns([1, 2])
    with colx1:
        top_n = st.number_input("Marcar automaticamente os N piores", min_value=0,
                                 max_value=len(diag_df), value=0, step=1)
    if top_n > 0:
        worst_idx = set(diag_df.sort_values("Erro |log| (%)", ascending=False).head(top_n)["#"])
        st.session_state["excluded_idx"].update(worst_idx)
        diag_df["Excluir"] = diag_df["#"].isin(st.session_state["excluded_idx"])

    edited_diag = st.data_editor(
        diag_df, use_container_width=True, hide_index=True, key="diag_editor",
        disabled=["#", "C1 (m)", "C2 (m)", "P1 (m)", "P2 (m)", "Observado (ohm.m)",
                  "Previsto (ohm.m)", "Erro (%)", "Erro |log| (%)"],
    )
    # tabela é a fonte definitiva apos edicao manual (permite tambem DESmarcar)
    st.session_state["excluded_idx"] = set(edited_diag.loc[edited_diag["Excluir"], "#"])

    n_marked = len(st.session_state["excluded_idx"])
    st.caption(f"{n_marked} de {len(edited_diag)} leituras marcadas para exclusão.")
    if st.button("Limpar todas as marcações"):
        st.session_state["excluded_idx"] = set()
        st.rerun()

    if st.button("Re-rodar inversão sem os pontos marcados", disabled=n_marked == 0):
        keep_ids = set(edited_diag.loc[~edited_diag["Excluir"], "#"])
        filtered_readings = [r for i, r in enumerate(res['readings']) if i in keep_ids]
        if len(filtered_readings) < 4:
            st.error("Restariam menos de 4 leituras — exclua menos pontos.")
        else:
            dat_text2 = dat_text_from_readings(filtered_readings, mesh['spacing'], title="filtrado_sem_ruido")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dat", mode="w") as tmp:
                tmp.write(dat_text2)
                tmp_path2 = tmp.name
            progress_bar2 = st.progress(0.0, text="Iniciando...")

            def cb2(it, n_total, rms):
                progress_bar2.progress((it+1)/n_total, text=f"Iteração {it+1}/{n_total} — RMS(log) = {rms:.1f}%")

            try:
                with st.spinner(f"Reinvertendo com {len(filtered_readings)} leituras (excluídas {n_marked})..."):
                    res2 = run_inversion(tmp_path2, n_iter=n_iter, n_k=n_k, verbose=False, progress_cb=cb2,
                                          sub_per_gap=sub_per_gap, nz_layers=nz_layers, robust=robust_weighting,
                                          depth_weight_power=depth_weight_power)
                st.session_state["result"] = res2
                progress_bar2.progress(1.0, text="Concluído.")
                st.success(f"Novo RMS: {res2['best_rms']:.1f}%  (antes: {res['best_rms']:.1f}%)")
                st.rerun()
            except Exception as e:
                st.error(f"Falha ao reprocessar: {e}")
            finally:
                os.unlink(tmp_path2)

    with st.expander("Avisos sobre a qualidade do resultado"):
        st.markdown("""
        - Este é um solver **próprio e simplificado**, validado contra a solução analítica de
          semi-espaço homogêneo (erro típico ~7–14%), não uma reimplementação do RES2DINV comercial.
        - A inversão é fortemente **subdeterminada** (poucas leituras vs. muitas células do modelo),
          então a regularização de suavidade tende a atenuar contrastes muito extremos nos dados.
        - RMS de ajuste alto pode indicar dados ruidosos/outliers (eletrodo com mau contato, etc.)
          além das limitações do próprio solver.
        - Use como ferramenta exploratória, não como substituto de um software validado
          profissionalmente para laudos/relatórios técnicos.
        """)
else:
    st.info("Importe um arquivo .dat ou preencha a aba de entrada manual, depois rode a inversão.")
