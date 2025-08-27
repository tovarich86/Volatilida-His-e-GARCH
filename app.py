import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import io

# Configurar a página
st.set_page_config(page_title="Análise de Volatilidade", layout="wide")
st.title("Cálculo de Volatilidade Histórica e GARCH")

# Criar um layout de arquivo para importação
st.subheader("📥 Baixar Modelo de Arquivo para Importação (necessário somente DATE e CLOSE)")
# Modelo com padrão de vírgula para exemplificar
modelo_df = pd.DataFrame({
    'Date': ['04/01/2016', '05/01/2016'],
    'Adj Close': ['7,24812', '7,15102'],
    'Close': ['12,69', '12,52'],
    'High': ['12,97', '12,84'],
    'Low': ['12,47', '12,41'],
    'Open': ['12,48', '12,67'],
    'Volume': [4587900, 2693500],
    'Ticker': ['VALE3.SA', 'VALE3.SA']
})
output_model = io.BytesIO()
with pd.ExcelWriter(output_model, engine='xlsxwriter') as writer:
    modelo_df.to_excel(writer, sheet_name='Modelo', index=False)
    # O comando close() é chamado automaticamente ao sair do bloco 'with'
    # writer.close() # não é necessário

output_model.seek(0)

st.download_button(
    label="📥 Modelo de Arquivo (padrão pt-br)",
    data=output_model,
    file_name="modelo_importacao_ptbr.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Faça upload do arquivo Excel contendo os dados", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # --- INÍCIO DA MODIFICAÇÃO ---
    # Converte as colunas de preço para o formato numérico correto
    # Lista de colunas que podem conter números com vírgula decimal
    cols_to_process = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    
    for col in cols_to_process:
        # Verifica se a coluna existe no DataFrame e se é do tipo 'object' (texto)
        if col in df.columns and df[col].dtype == 'object':
            st.write(f"Convertendo coluna '{col}' para formato numérico...")
            # 1. Remove o separador de milhar (ponto)
            # 2. Substitui o separador decimal (vírgula) por ponto
            # 3. Converte a string para um número, tratando erros
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Alternativa mais simples se não houver separador de milhar:
    # df = pd.read_excel(uploaded_file, decimal=',')
    # A abordagem acima (com loop) é mais segura pois trata também os separadores de milhar.
    # --- FIM DA MODIFICAÇÃO ---
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Remover linhas onde a data ou o preço de fechamento são nulos após a conversão
    df.dropna(subset=['Date', 'Close'], inplace=True)
    
    # Ordenar os dados do mais novo para o mais antigo
    df = df.sort_values(by='Date', ascending=False).copy()
    
    # Criar dicionários para armazenar os resultados
    volatilidade_historica = {}
    volatilidade_garch = {}
    
    # Definir os períodos exatos em dias úteis, incluindo intervalos intermediários
    periodos_dias = {
        1: 252, 1.5: 380, 2: 509, 2.5: 635, 3: 761, 3.5: 880, 4: 1000,
        4.5: 1125, 5: 1250, 5.5: 1375, 6: 1500, 6.5: 1625, 7: 1750,
        7.5: 1875, 8: 2000, 8.5: 2125, 9: 2250, 9.5: 2375, 10: 2500
    }
    
    # Loop para calcular a volatilidade
    for years, dias in periodos_dias.items():
        if len(df) >= dias:
            df_period = df.head(dias).copy()
            # Inverter a ordem para o cálculo do retorno (do mais antigo para o mais novo)
            df_period = df_period.iloc[::-1]
            
            df_period['Retornos_Log'] = np.log(df_period['Close'] / df_period['Close'].shift(1))
            df_period = df_period.dropna()
            
            if df_period.empty or len(df_period) < 30:
                volatilidade_historica[years] = np.nan
                volatilidade_garch[years] = np.nan
                continue
            
            # Volatilidade histórica
            vol_anualizada_hist = (df_period['Retornos_Log'].std(ddof=1) * np.sqrt(252))
            volatilidade_historica[years] = vol_anualizada_hist
            
            # Volatilidade GARCH(1,1)
