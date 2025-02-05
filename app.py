import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import io

# Configurar a página
st.set_page_config(page_title="Análise de Volatilidade", layout="wide")
st.title("Cálculo de Volatilidade Histórica e GARCH")

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Faça upload do arquivo Excel contendo os dados", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, parse_dates=['Date'])
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
        vol_anualizada_hist = df_period['Retornos_Log'].std(ddof=1) * np.sqrt(252)
        volatilidade_historica[years] = vol_anualizada_hist
        
        # Volatilidade GARCH(1,1)
        try:
            model = arch_model(df_period['Retornos_Log'] * 10, vol='Garch', p=1, q=1)
            garch_result = model.fit(disp='off')
            vol_diaria_media = garch_result.conditional_volatility.mean() / 10
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
    
    # Formatar para o padrão brasileiro (vírgula para decimais)
    df_volatilidade = df_volatilidade.applymap(lambda x: f"{x:.2%}".replace('.', ',') if pd.notna(x) else '-')
    
    # Exibir os resultados no Streamlit
    st.subheader("Tabela de Volatilidade")
    st.dataframe(df_volatilidade.style.format("{:.2%}".replace('.', ',')))
    
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
