import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import io

# Configurar a página
st.set_page_config(page_title="Análise de Volatilidade", layout="wide")
st.title("Cálculo de Volatilidade Histórica e GARCH")

# Criar um layout de arquivo para importação
st.subheader("📥 Baixar Modelo de Arquivo para Importação")
modelo_df = pd.DataFrame({
    'Date': ['04/01/2016', '05/01/2016'],
    'Adj Close': [7.24812, 7.15102],
    'Close': [12.69, 12.52],
    'High': [12.97, 12.84],
    'Low': [12.47, 12.41],
    'Open': [12.48, 12.67],
    'Volume': [4587900, 2693500],
    'Ticker': ['VALE3.SA', 'VALE3.SA']
})
output_model = io.BytesIO()
with pd.ExcelWriter(output_model, engine='xlsxwriter') as writer:
    modelo_df.to_excel(writer, sheet_name='Modelo', index=False)
    writer.close()
output_model.seek(0)

st.download_button(
    label="📥 Baixar Modelo de Arquivo",
    data=output_model,
    file_name="modelo_importacao.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Faça upload do arquivo Excel contendo os dados", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
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
        df_period = df.head(dias).copy()
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
        try:
            model = arch_model(df_period['Retornos_Log'] * 10, vol='Garch', p=1, q=1)
            garch_result = model.fit(disp='off')
            vol_diaria_media = (garch_result.conditional_volatility.mean() / 10) 
            vol_anualizada_garch = vol_diaria_media * np.sqrt(252)
            volatilidade_garch[years] = vol_anualizada_garch
        except:
            volatilidade_garch[years] = np.nan
    
    # Criar DataFrame de resultados
    df_volatilidade = pd.DataFrame({
        'Volatilidade Histórica': pd.Series(volatilidade_historica),
        'Volatilidade GARCH(1,1)': pd.Series(volatilidade_garch)
    })
    df_volatilidade.index.name = 'Período (Anos)'
    
    # Exibir os dados sem multiplicação por 100 (base percentual já aplicada)
    st.subheader("Tabela de Volatilidade")
    st.dataframe(df_volatilidade)
    
    # Criar um arquivo Excel para download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_volatilidade.to_excel(writer, sheet_name='Volatilidade')
        writer.close()
    output.seek(0)
    
    # Botão para download
    st.download_button(
        label="📥 Baixar Resultados em Excel",
        data=output,
        file_name="volatilidade.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
