import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import io

# --- FUNÇÃO DE LIMPEZA CORRIGIDA ---
def clean_numeric_col(series):
    """
    Converte uma coluna para numérico de forma inteligente.
    Se a coluna já for numérica, não faz nada.
    Se for texto, verifica se usa vírgula como decimal (padrão BR) para converter.
    """
    # 1. Se a coluna já é um tipo numérico, retorna sem alterações.
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # 2. Garante que a coluna é do tipo string para manipulação.
    if not pd.api.types.is_string_dtype(series):
        # Tenta uma conversão direta para outros tipos (ex: mistos)
        return pd.to_numeric(series, errors='coerce')
        
    series = series.astype(str).str.strip()
        
    # 3. Lógica principal: só remove pontos se houver uma vírgula (indicando formato BR)
    # O método .any() verifica se existe pelo menos um valor com vírgula na coluna.
    if series.str.contains(',').any():
        st.write(f"Detectado formato brasileiro (com vírgula) na coluna '{series.name}'. Convertendo...")
        # Remove pontos (milhar) e substitui vírgula (decimal) por ponto
        series = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    
    # 4. Converte a série limpa para numérico. Se o formato já usava ponto (ex: 837.40),
    # a condição do 'if' acima será falsa e a conversão aqui funcionará diretamente.
    return pd.to_numeric(series, errors='coerce')

# --- Início da Aplicação Streamlit ---
st.set_page_config(page_title="Análise de Volatilidade", layout="wide")
st.title("Cálculo de Volatilidade Histórica e GARCH")

# (O código para criar o arquivo modelo permanece o mesmo)
# ...

# Upload de arquivo flexível (aceita .xlsx e .csv)
uploaded_file = st.file_uploader("Faça upload do seu arquivo de dados", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        # Tenta ler como Excel, se falhar, tenta como CSV
        file_name = uploaded_file.name
        if file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Por favor, use .xlsx ou .csv")
            st.stop()
        
        st.subheader("1. Dados Originais Carregados")
        st.dataframe(df.head())

        # --- Etapa de Limpeza e Conversão ---
        st.subheader("2. Processamento e Limpeza dos Dados")

        # Conversão de data flexível
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            st.write("✔️ Coluna 'Date' convertida para formato de data.")
        else:
            st.error("Erro: Coluna 'Date' não encontrada no arquivo!")
            st.stop()

        # Conversão de colunas numéricas com a nova função
        cols_to_process = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in cols_to_process:
            if col in df.columns:
                df[col] = clean_numeric_col(df[col])
        st.write("✔️ Colunas de preço convertidas para formato numérico.")

        # O restante do código continua como antes...
        original_rows = len(df)
        df.dropna(subset=['Date', 'Close'], inplace=True)
        st.write(f"✔️ Linhas com 'Date' ou 'Close' vazios foram removidas. Restaram {len(df)} de {original_rows} linhas.")
        
        st.write("Amostra dos dados após limpeza:")
        st.dataframe(df.head())

        if df.empty:
            st.error("Alerta: O DataFrame ficou vazio após a limpeza. Verifique se as colunas 'Date' e 'Close' contêm dados válidos.")
            st.stop()

        # --- Etapa de Cálculo ---
        st.subheader("3. Cálculo da Volatilidade")

        df = df.sort_values(by='Date', ascending=True).copy()
        df['Retornos_Log'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(subset=['Retornos_Log'], inplace=True)

        if df.empty:
            st.error("Não foi possível calcular os retornos. Verifique os dados da coluna 'Close'.")
            st.stop()

        volatilidade_historica = {}
        volatilidade_garch = {}
        
        periodos_dias = {
            1: 252, 1.5: 380, 2: 509, 2.5: 635, 3: 761, 3.5: 880, 4: 1000
        }
        
        with st.spinner('Calculando volatilidade...'):
            for years, dias in periodos_dias.items():
                if len(df) >= dias:
                    df_period = df.tail(dias).copy()
                    
                    vol_anualizada_hist = (df_period['Retornos_Log'].std(ddof=1) * np.sqrt(252))
                    volatilidade_historica[years] = vol_anualizada_hist
                    
                    try:
                        model = arch_model(df_period['Retornos_Log'] * 100, vol='Garch', p=1, q=1)
                        garch_result = model.fit(disp='off', show_warning=False)
                        forecast = garch_result.forecast(horizon=1)
                        vol_diaria_garch = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
                        vol_anualizada_garch = vol_diaria_garch * np.sqrt(252)
                        volatilidade_garch[years] = vol_anualizada_garch
                    except Exception:
                        volatilidade_garch[years] = np.nan
        
        st.success("Cálculos finalizados!")

        if not volatilidade_historica:
             st.warning("Não foi possível calcular a volatilidade. Verifique se o arquivo tem dados suficientes.")
             st.stop()

        df_volatilidade = pd.DataFrame({
            'Volatilidade Histórica': pd.Series(volatilidade_historica),
            'Volatilidade GARCH(1,1)': pd.Series(volatilidade_garch)
        })
        df_volatilidade.index.name = 'Período (Anos)'
        
        st.subheader("Tabela de Volatilidade")
        st.dataframe(df_volatilidade.style.format("{:.2%}"))
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_volatilidade.to_excel(writer, sheet_name='Volatilidade')
        output.seek(0)
        
        st.download_button(
            label="📥 Baixar Resultados em Excel",
            data=output,
            file_name="volatilidade.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet"
        )

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
