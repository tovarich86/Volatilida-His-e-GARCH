import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import io

# --- Funções Auxiliares para Limpeza ---
def clean_numeric_col(series):
    """Converte uma coluna para numérico, tratando vírgulas e pontos."""
    if series.dtype == 'object':
        # Converte para string para garantir que os métodos .str funcionem
        series = series.astype(str)
        # 1. Remove pontos de milhar
        series = series.str.replace('.', '', regex=False)
        # 2. Substitui vírgula decimal por ponto
        series = series.str.replace(',', '.', regex=False)
    # Converte para numérico, erros viram NaN (Not a Number)
    return pd.to_numeric(series, errors='coerce')

# --- Início da Aplicação Streamlit ---
st.set_page_config(page_title="Análise de Volatilidade", layout="wide")
st.title("Cálculo de Volatilidade Histórica e GARCH")

# (O código para criar o arquivo modelo permanece o mesmo)
# ...

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Faça upload do arquivo Excel contendo os dados", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        st.subheader("1. Dados Originais Carregados")
        st.write("Abaixo estão as 5 primeiras linhas do arquivo que você enviou. Verifique se as colunas 'Date' e 'Close' foram lidas corretamente.")
        st.dataframe(df.head())

        # --- Etapa de Limpeza e Conversão ---
        st.subheader("2. Processamento e Limpeza dos Dados")

        # Converte a coluna de Data
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            st.write("✔️ Coluna 'Date' convertida para formato de data.")
        else:
            st.error("Erro: Coluna 'Date' não encontrada no arquivo!")
            st.stop() # Interrompe a execução

        # Converte colunas numéricas
        cols_to_process = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in cols_to_process:
            if col in df.columns:
                df[col] = clean_numeric_col(df[col])
        st.write("✔️ Colunas de preço ('Open', 'High', 'Low', 'Close') convertidas para formato numérico.")

        # Remove linhas com valores nulos essenciais
        original_rows = len(df)
        df.dropna(subset=['Date', 'Close'], inplace=True)
        st.write(f"✔️ Linhas com 'Date' ou 'Close' vazios foram removidas. Restaram {len(df)} de {original_rows} linhas.")
        
        st.write("Amostra dos dados após limpeza:")
        st.dataframe(df.head())

        # Verifica se o DataFrame ficou vazio
        if df.empty:
            st.error("Alerta: O DataFrame ficou vazio após a limpeza. Verifique se o formato das datas (ex: 27/08/2025) e dos números (ex: 1.234,56) está correto no seu arquivo Excel.")
            st.stop()

        # --- Etapa de Cálculo ---
        st.subheader("3. Cálculo da Volatilidade")

        # Ordenar os dados: do mais antigo para o mais novo para os cálculos
        df = df.sort_values(by='Date', ascending=True).copy()
        
        # Calcular retornos
        df['Retornos_Log'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(subset=['Retornos_Log'], inplace=True)

        if df.empty:
            st.error("Não foi possível calcular os retornos. Verifique os dados da coluna 'Close'.")
            st.stop()

        volatilidade_historica = {}
        volatilidade_garch = {}
        
        periodos_dias = {
            1: 252, 1.5: 380, 2: 509, 2.5: 635, 3: 761, 3.5: 880, 4: 1000,
            4.5: 1125, 5: 1250, 5.5: 1375, 6: 1500, 6.5: 1625, 7: 1750,
            7.5: 1875, 8: 2000, 8.5: 2125, 9: 2250, 9.5: 2375, 10: 2500
        }
        
        with st.spinner('Calculando volatilidade para diferentes períodos...'):
            for years, dias in periodos_dias.items():
                if len(df) >= dias:
                    df_period = df.tail(dias).copy() # Pega os dados mais recentes
                    
                    if df_period.empty or len(df_period) < 30:
                        continue
                    
                    # Volatilidade histórica
                    vol_anualizada_hist = (df_period['Retornos_Log'].std(ddof=1) * np.sqrt(252))
                    volatilidade_historica[years] = vol_anualizada_hist
                    
                    # Volatilidade GARCH(1,1)
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

        # --- Etapa de Exibição e Download ---
        if not volatilidade_historica and not volatilidade_garch:
             st.warning("Não foi possível calcular a volatilidade para nenhum período. O arquivo pode ter menos dados do que o necessário (mínimo de 252 dias úteis).")
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
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante o processamento do arquivo: {e}")
        st.error("Por favor, verifique se o arquivo está no formato correto e não está corrompido.")
