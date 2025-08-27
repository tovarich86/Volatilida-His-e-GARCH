import streamlit as st
import pandas as pd
import numpy as np
# A biblioteca 'arch' é opcional. Se não estiver instalada, o GARCH não será calculado.
try:
    from arch import arch_model
    ARCH_INSTALLED = True
except ImportError:
    ARCH_INSTALLED = False

# --- FUNÇÃO DE LIMPEZA ---
def clean_numeric_col(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    if not pd.api.types.is_string_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    series = series.astype(str).str.strip()
    if series.str.contains(',').any():
        st.write(f"Detectado formato brasileiro (com vírgula) na coluna '{series.name}'. Convertendo...")
        series = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(series, errors='coerce')

# --- INÍCIO DA APLICAÇÃO ---
st.set_page_config(page_title="Análise de Volatilidade", layout="wide")
st.title("Comparativo de Modelos de Volatilidade")

uploaded_file = st.file_uploader("Faça upload do seu arquivo de dados", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else: # .csv
            df = pd.read_csv(uploaded_file)
        
        st.subheader("1. Processamento e Limpeza dos Dados")

        # Conversão de data flexível
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            st.error("Erro: Coluna 'Date' não encontrada no arquivo!")
            st.stop()

        # Conversão de colunas numéricas
        cols_to_process = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in cols_to_process:
            if col in df.columns:
                df[col] = clean_numeric_col(df[col])

        df.dropna(subset=['Date', 'Close'], inplace=True)
        st.write("✔️ Dados limpos e convertidos.")
        
        if df.empty:
            st.error("Alerta: O DataFrame ficou vazio após a limpeza.")
            st.stop()

        # --- Etapa de Cálculo ---
        st.subheader("2. Cálculo da Volatilidade")

        df = df.sort_values(by='Date', ascending=True).copy()
        df['Retornos_Log'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(subset=['Retornos_Log'], inplace=True)

        volatilidade_historica = {}
        volatilidade_ewma = {}
        volatilidade_garch = {}
        
        periodos_dias = {
            1: 252, 1.5: 380, 2: 509, 2.5: 635, 3: 761, 4: 1000, 5: 1250
        }
        
        lambda_ewma = st.sidebar.slider("Parâmetro Lambda (λ) para EWMA", 0.80, 0.99, 0.94, 0.01)

        with st.spinner('Calculando...'):
            for years, dias in periodos_dias.items():
                if len(df) >= dias:
                    df_period = df.tail(dias).copy()
                    
                    # Volatilidade Histórica
                    vol_hist = df_period['Retornos_Log'].std(ddof=1) * np.sqrt(252)
                    volatilidade_historica[years] = vol_hist

                    # Volatilidade EWMA
                    df_period['Retornos_Sq'] = df_period['Retornos_Log']**2
                    df_period['EWMA_Var'] = df_period['Retornos_Sq'].ewm(alpha=(1-lambda_ewma), adjust=False).mean()
                    vol_diaria_ewma = np.sqrt(df_period['EWMA_Var'].iloc[-1])
                    volatilidade_ewma[years] = vol_diaria_ewma * np.sqrt(252)

                    # Volatilidade GARCH (se a biblioteca estiver instalada)
                    if ARCH_INSTALLED:
                        try:
                            model = arch_model(df_period['Retornos_Log'] * 100, vol='Garch', p=1, q=1)
                            garch_result = model.fit(disp='off', show_warning=False)
                            forecast = garch_result.forecast(horizon=1)
                            vol_diaria_garch = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
                            volatilidade_garch[years] = vol_diaria_garch * np.sqrt(252)
                        except Exception:
                            volatilidade_garch[years] = np.nan
        
        st.success("Cálculos finalizados!")
        
        # --- Etapa de Exibição ---
        results = {
            'Volatilidade Histórica': pd.Series(volatilidade_historica),
            f'Volatilidade EWMA (λ={lambda_ewma})': pd.Series(volatilidade_ewma)
        }
        if ARCH_INSTALLED and volatilidade_garch:
            results['Volatilidade GARCH(1,1)'] = pd.Series(volatilidade_garch)

        df_volatilidade = pd.DataFrame(results)
        df_volatilidade.index.name = 'Período (Anos)'
        
        st.subheader("Tabela Comparativa de Volatilidade")
        st.dataframe(df_volatilidade.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
