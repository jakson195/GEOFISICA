# Inversão 2D de Eletrorresistividade (ERT)

App web (Streamlit) para importar dados de resistividade aparente (formato tipo
Res2dinv, arranjo geral) e gerar a seção 2D invertida.

Implementação própria e independente: modelagem direta 2.5D (domínio de número
de onda, elementos finitos/diferenças finitas) + inversão Gauss-Newton amortecida
com regularização de suavidade (estilo Occam). Não é o software comercial
RES2DINV — é uma ferramenta mais simples, para uso exploratório.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre em `http://localhost:8501`.

## Publicar de graça na internet (Streamlit Community Cloud)

1. Crie uma conta gratuita em https://github.com (se ainda não tiver).
2. Crie um repositório novo (pode ser privado ou público) e suba estes arquivos:
   `app.py`, `invert.py`, `parser.py`, `mesh.py`, `forward.py`, `sensitivity.py`,
   `requirements.txt`.
   - Pelo próprio site do GitHub: botão **"Add file" → "Upload files"**, arraste
     todos os arquivos desta pasta, e clique em **"Commit changes"**.
3. Acesse https://share.streamlit.io e faça login com sua conta GitHub.
4. Clique em **"New app"**, selecione o repositório que você criou, o branch
   (`main`) e o arquivo principal (`app.py`).
5. Clique em **"Deploy"**. Em 1–2 minutos o app estará no ar, com uma URL pública
   tipo `https://seu-usuario-nome-do-app.streamlit.app`.

Qualquer atualização que você enviar ao repositório GitHub atualiza o app
publicado automaticamente.

## Formato do arquivo .dat esperado

Arranjo geral (código 11) estilo Res2dinv, com coordenadas reais (x, elevação)
por eletrodo em cada leitura — ver detalhes dentro do próprio app (seção
"Formato de arquivo esperado").

## Limitações conhecidas

- Solver próprio, validado contra a solução analítica de semi-espaço homogêneo
  com erro típico de ~7–14% (não tem o refinamento de décadas de um produto
  comercial).
- Inversão fortemente subdeterminada com poucos dados: contrastes muito
  extremos tendem a ser suavizados.
- Testado com arranjo dipolo-dipolo; outros arranjos no mesmo formato geral
  devem funcionar, mas não foram testados extensivamente.
