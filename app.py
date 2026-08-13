import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

with st.expander("Formato de arquivo esperado", expanded=False):
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

uploaded = st.file_uploader("Importar arquivo .dat", type=["dat", "txt"])

col1, col2, col3 = st.columns(3)
with col1:
    n_iter = st.slider("Iterações da inversão", 4, 20, 12)
with col2:
    n_k = st.slider("Resolução da quadratura (nº de números de onda)", 8, 30, 16,
                     help="Mais = mais preciso e mais lento.")
with col3:
    run_btn = st.button("Rodar inversão", type="primary", disabled=uploaded is None)

if "result" not in st.session_state:
    st.session_state["result"] = None

if run_btn and uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
        content = uploaded.getvalue().decode("utf-8", errors="ignore")
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        tmp.write(content.encode("utf-8"))
        tmp_path = tmp.name

    progress_bar = st.progress(0.0, text="Iniciando...")
    rms_log = []

    def cb(it, n_total, rms):
        progress_bar.progress((it+1)/n_total, text=f"Iteração {it+1}/{n_total} — RMS(log) = {rms:.1f}%")
        rms_log.append(rms)

    try:
        with st.spinner("Resolvendo modelagem direta e invertendo..."):
            res = run_inversion(tmp_path, n_iter=n_iter, n_k=n_k, verbose=False, progress_cb=cb)
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

    st.subheader("Seção de resistividade invertida")
    fig, ax = plt.subplots(figsize=(11, 5))
    lograho_masked = np.log10(rho_masked)
    pc = ax.pcolormesh(Xg, Zg, lograho_masked, shading='auto', cmap='turbo')
    cbar = fig.colorbar(pc, ax=ax, label='Resistividade (ohm.m)')
    ticks = list(range(int(np.floor(np.nanmin(lograho_masked))), int(np.ceil(np.nanmax(lograho_masked)))+1))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{10**t:.0f}" for t in ticks])
    elev_e = [surface[np.argmin(np.abs(xs-x))] for x in uniq_x]
    ax.scatter(uniq_x, elev_e, marker='v', color='k', s=25, zorder=5, label='Eletrodos')
    ax.set_xlim(xmin_core, xmax_core)
    ax.set_ylim(surface.min()-mesh['depth_max'], surface.max()+3)
    ax.set_xlabel('Distância ao longo da linha (m)')
    ax.set_ylabel('Elevação (m)')
    ax.set_title(f"RMS de ajuste (log) = {res['best_rms']:.1f}%")
    ax.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', dpi=150)
    st.download_button("Baixar seção (PNG)", buf_png.getvalue(), file_name="secao_resistividade.png", mime="image/png")

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
    st.info("Envie um arquivo .dat e clique em 'Rodar inversão' para começar.")
