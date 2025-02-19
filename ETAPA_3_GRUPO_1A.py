import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import zipfile
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import chi2_contingency

# Configuração da página
st.set_page_config(layout="wide")

# Título do Dashboard
st.title("Análise de Acidentes de Trânsito (2017-2023)")

# Carregamento dos dados
@st.cache_data
def load_data():
    zip_file_path = "accidents_2017_to_2023_english.zip"
    csv_file_name = "accidents_2017_to_2023_english.csv"
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(csv_file_name) as f:
            df = pd.read_csv(f)
    return df

df = load_data()

# Preprocessamento dos dados
def preprocess_data(df):
    # Preenchimento de valores ausentes
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remoção de outliers
    def remove_outliers(df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df = remove_outliers(df, column)

    # Conversão de colunas textuais
    df['inverse_data'] = pd.to_datetime(df['inverse_data'], errors='coerce')
    df['hour'] = pd.to_datetime(df['hour'], format='%H:%M:%S', errors='coerce').dt.hour

    # One-Hot Encoding (apenas para colunas categóricas que não serão usadas diretamente)
    categorical_columns_to_encode = ['type_of_accident']  # Removi 'week_day' e 'cause_of_accident' para mantê-las intactas
    df = pd.get_dummies(df, columns=categorical_columns_to_encode, drop_first=True)

    # Normalização
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df

df = preprocess_data(df)

# Sidebar para seleção de hipóteses
st.sidebar.title("Selecione a Hipótese")
hypothesis = st.sidebar.radio("", ["Hipótese 1", "Hipótese 2", "Hipótese 3", "Hipótese 4", "Hipótese 5"])

# Hipótese 1
if hypothesis == "Hipótese 1":
    st.header("Hipótese 1: Acidentes graves são mais comuns em condições climáticas adversas")
    st.markdown("""
    **Gráfico de Barras Empilhadas:** Condição da Vítima por Condição Climática
    """)
    
    # Preparando os dados
    stacked_data = df.groupby(["wheather_condition", "victims_condition"]).size().unstack(fill_value=0)
    
    # Criando o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    colormap = plt.get_cmap("Accent")
    colors = [colormap(i) for i in np.linspace(0, 1, len(stacked_data.columns))]
    hatch_patterns = ["x", "o"]
    
    bottom = np.zeros(len(stacked_data))
    for i, (col, color) in enumerate(zip(stacked_data.columns, colors)):
        ax.bar(stacked_data.index, stacked_data[col], label=col, bottom=bottom, color=color, hatch=hatch_patterns[i % len(hatch_patterns)], edgecolor="black")
        bottom += stacked_data[col].values
    
    ax.set_title("Condição da Vítima por Condição Climática")
    ax.set_xlabel("Condição Climática")
    ax.set_ylabel("Contagem")
    ax.legend(title="Condição da Vítima")
    ax.set_xticklabels(stacked_data.index, rotation=0)
    
    st.pyplot(fig)
    
    st.markdown("""
    **Descrição do Gráfico:**  
    O gráfico de barras empilhadas mostra a distribuição da condição das vítimas (feridas ou não feridas) em diferentes condições climáticas.  
    - **Eixo X:** Condições climáticas (por exemplo, "Clear sky", "Rainy", "Fog").  
    - **Eixo Y:** Número de acidentes.  
    - **Cores:** Representam a condição das vítimas (feridas ou não feridas).  

    **Conclusão:** A hipótese é refutada. Acidentes graves são mais comuns em condições climáticas normais.
    """)

# Hipótese 2
elif hypothesis == "Hipótese 2":
    st.header("Hipótese 2: Estradas de pista simples têm mais acidentes graves comparadas a rodovias de pista dupla")
    st.markdown("""
    **Gráfico de Barras Empilhadas:** Condição da Vítima por Tipo de Rodovia
    """)
    
    # Preparando os dados
    stacked_data = df.groupby(["road_type", "victims_condition"]).size().unstack(fill_value=0)
    
    # Criando o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    colormap = plt.get_cmap("Accent")
    colors = [colormap(i) for i in np.linspace(0, 1, len(stacked_data.columns))]
    hatch_patterns = ["/", "o"]
    
    bottom = np.zeros(len(stacked_data))
    for i, (col, color) in enumerate(zip(stacked_data.columns, colors)):
        ax.bar(stacked_data.index, stacked_data[col], label=col, bottom=bottom, color=color, hatch=hatch_patterns[i % len(hatch_patterns)], edgecolor="black")
        bottom += stacked_data[col].values
    
    ax.set_title("Condição da Vítima por Tipo de Rodovia")
    ax.set_xlabel("Tipo de Rodovia")
    ax.set_ylabel("Contagem")
    ax.legend(title="Condição da Vítima")
    ax.set_xticklabels(stacked_data.index, rotation=0)
    
    st.pyplot(fig)
    
    st.markdown("""
    **Descrição do Gráfico:**  
    O gráfico de barras empilhadas mostra a distribuição da condição das vítimas (feridas ou não feridas) em diferentes tipos de rodovia.  
    - **Eixo X:** Tipo de rodovia (por exemplo, "Simple", "Double").  
    - **Eixo Y:** Número de acidentes.  
    - **Cores:** Representam a condição das vítimas (feridas ou não feridas).  

    **Conclusão:** A hipótese é refutada. Acidentes graves são proporcionalmente mais frequentes em rodovias de pista dupla.
    """)

# Hipótese 3
elif hypothesis == "Hipótese 3":
    st.header("Hipótese 3: Acidentes com ferimentos graves em horários noturnos estão associados a determinadas causas")
    st.markdown("""
    **Boxplot:** Distribuição de Ferimentos Graves por Causa em Acidentes Noturnos
    """)
    
    # Filtro de acidentes noturnos (18h às 6h)
    df['is_night'] = df['hour'].apply(lambda x: 'Noturno' if (x >= 18 or x < 6) else 'Diurno')
    df_night = df[df['is_night'] == 'Noturno']

    # Verifique se a coluna 'cause_of_accident' existe
    if 'cause_of_accident' in df_night.columns:
        # Boxplot
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df_night,
            x='cause_of_accident',
            y='severely_injured',
            order=['Alcohol consumption', 'Driver was sleeping'],  # Ajuste conforme necessário
        )
        plt.title("Distribuição de Ferimentos Graves por Causa em Acidentes Noturnos")
        plt.xlabel("Causa do Acidente")
        plt.ylabel("Número de Ferimentos Graves (Severely Injured)")
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
        
        st.markdown("""
        **Descrição do Gráfico:**  
        O boxplot mostra a distribuição do número de ferimentos graves em acidentes noturnos, categorizados por causa do acidente.  
        - **Eixo X:** Causa do acidente (por exemplo, "Alcohol consumption", "Driver was sleeping").  
        - **Eixo Y:** Número de ferimentos graves.  

        **Conclusão:** A hipótese é refutada. Não há evidências robustas de que o tipo de rodovia seja determinante para a gravidade dos acidentes.
        """)

        ## Gráfico 2: Número de Registros para Causas Selecionadas
        # Causas específicas
        filtered_causes = ['Alcohol consumption', 'Driver was sleeping']
        df_filtered = df_night[df_night['cause_of_accident'].isin(filtered_causes)]
        cause_counts = df_filtered['cause_of_accident'].value_counts()

        # Gráfico de barras
        fig = plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=cause_counts.index, y=cause_counts.values, order=filtered_causes, palette="viridis")

        # Adicionando texturas (hatch) nas barras
        hatches = ['/', 'o']  # Definindo texturas
        for i, bar in enumerate(ax.patches):
            bar.set_hatch(hatches[i])  # Atribuindo textura a cada barra

        # Títulos e rótulos
        plt.title("Número de Registros para Causas Selecionadas")
        plt.xlabel("Causa do Acidente")
        plt.ylabel("Número de Registros")
        plt.xticks(rotation=45, ha='right')

        st.pyplot(fig)
        
        st.markdown("""
        **Descrição do Gráfico:**  
        O gráfico de barras mostra o número de registros de acidentes noturnos para as causas selecionadas.  
        - **Eixo X:** Causa do acidente (por exemplo, "Alcohol consumption", "Driver was sleeping").  
        - **Eixo Y:** Número de registros.  

        **Conclusão:** A hipótese é refutada. Não há evidências robustas de que o tipo de rodovia seja determinante para a gravidade dos acidentes.
        """)
    else:
        st.error("A coluna 'cause_of_accident' não foi encontrada no DataFrame. Verifique o nome da coluna ou o pré-processamento.")

# Hipótese 4
elif hypothesis == "Hipótese 4":
    st.header("Hipótese 4: Os acidentes ocorrem com maior frequência nos finais de semana")
    st.markdown("""
    **Gráfico de Barras:** Contagem de Acidentes por Dia da Semana
    """)

    # Verifique se a coluna 'week_day' existe
    if 'week_day' in df.columns:
        # Contagem de acidentes por dia da semana
        accidents_per_day = df['week_day'].value_counts().sort_index()

        # Gráfico de barras
        fig = plt.figure(figsize=(10, 6))
        accidents_per_day.plot(kind='bar', color='skyblue')
        plt.title('Contagem de Acidentes por Dia da Semana')
        plt.xlabel('Dia da Semana')
        plt.ylabel('Número de Acidentes')
        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.markdown("""
        **Descrição do Gráfico:**  
        O gráfico de barras mostra a contagem de acidentes por dia da semana.  
        - **Eixo X:** Dia da semana (por exemplo, "Segunda", "Terça").  
        - **Eixo Y:** Número de acidentes.  

        **Conclusão:** A hipótese é refutada. A distribuição dos acidentes entre os dias da semana é uniforme.
        """)
    else:
        st.error("A coluna 'week_day' não foi encontrada no DataFrame. Verifique o nome da coluna ou o pré-processamento.")

# Hipótese 5
elif hypothesis == "Hipótese 5":
    st.header("Hipótese 5: Distribuição de Acidentes por Hora do Dia")
    st.markdown("""
    **Gráfico de Linha:** Distribuição de Acidentes por Hora do Dia
    """)
    
    # Contagem de acidentes por hora
    hour_counts = df['hour'].value_counts().sort_index()
    
    # Gráfico de linha
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hour_counts.index, hour_counts.values, color='blue', marker='o', markersize=8, linewidth=2, alpha=0.85, label='Acidentes por Hora')
    ax.scatter(hour_counts.index, hour_counts.values, color='blue', marker='o', s=70, edgecolors='black', alpha=0.9)
    ax.set_title('Distribuição de Acidentes por Hora do Dia', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hora do Dia', fontsize=14)
    ax.set_ylabel('Número de Acidentes', fontsize=14)
    ax.legend(title="Legenda", title_fontsize='13', fontsize='12', loc='upper left', frameon=True, shadow=True, borderpad=1)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xticks(range(hour_counts.index.min(), hour_counts.index.max() + 1))
    ax.tick_params(axis='both', labelsize=12)
    
    st.pyplot(fig)
    
    st.markdown("""
    **Descrição do Gráfico:**  
    O gráfico de linha mostra a distribuição de acidentes ao longo das horas do dia.  
    - **Eixo X:** Hora do dia (0 a 23).  
    - **Eixo Y:** Número de acidentes.  

    **Conclusão:** A análise revelou que os horários de pico (manhã e tarde) apresentam uma concentração significativamente maior de acidentes.
    """)

    # Mineração de dados: K-Means Clustering
    st.markdown("""
    **Mineração de Dados: K-Means Clustering**
    """)
    
    # Preparando os dados para clustering
    hour_data = pd.DataFrame({'Hour': hour_counts.index, 'Accidents': hour_counts.values})
    
    # Normalizando os dados
    scaler = StandardScaler()
    hour_data_scaled = scaler.fit_transform(hour_data[['Hour', 'Accidents']])
    
    # Método do cotovelo para determinar o número ideal de clusters
    inertia = []
    k_values = range(1, 10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(hour_data_scaled)
        inertia.append(kmeans.inertia_)
    
    # Gráfico do método do cotovelo
    fig = plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia, marker='o', color='blue')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    st.pyplot(fig)
    
    st.markdown("""
    **Descrição do Gráfico:**  
    O gráfico do método do cotovelo ajuda a identificar o número ideal de clusters para o algoritmo K-Means.  
    - **Eixo X:** Número de clusters.  
    - **Eixo Y:** Inércia (medida de quão distantes os pontos estão dos centróides).  

    **Conclusão:** O ponto de "cotovelo" no gráfico sugere o número ideal de clusters.
    """)
    
    # Aplicando K-Means com 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    hour_data['Cluster'] = kmeans.fit_predict(hour_data_scaled)
    
    # Mapeando diferentes formas para cada cluster
    markers = {0: 'o', 1: 's', 2: '*'}
    
    # Gráfico de dispersão com clusters
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=hour_data['Hour'], y=hour_data['Accidents'],
        hue=hour_data['Cluster'], palette='viridis',
        style=hour_data['Cluster'], markers=markers,
        s=100, edgecolor='black'
    )
    plt.title('Clusters de Acidentes por Hora do Dia', fontsize=16)
    plt.xlabel('Hora do Dia', fontsize=14)
    plt.ylabel('Número de Acidentes', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    st.markdown("""
    **Descrição do Gráfico:**  
    O gráfico de dispersão mostra os clusters gerados pelo algoritmo K-Means, agrupando as horas do dia com base no número de acidentes.  
    - **Eixo X:** Hora do dia (0 a 23).  
    - **Eixo Y:** Número de acidentes.  
    - **Cores e Formas:** Representam os clusters identificados.  

    **Conclusão:** Os clusters ajudam a identificar padrões de acidentes ao longo do dia.
    """)
    
    # Segmentação manual por períodos do dia
    def categorize_period(hour):
        if 0 <= hour < 10:
            return 'Manhã'
        elif 10 <= hour < 18:
            return 'Tarde'
        else:
            return 'Noite'
    
    hour_data['Period'] = hour_data['Hour'].apply(categorize_period)
    
    # Definindo cores e formas para cada período
    color_palette = {
        'Manhã': '#D9D96D',  # Amarelo suave
        'Tarde': '#A8A8A8',  # Cinza suave
        'Noite': '#3A3AC4'   # Azul intenso
    }
    markers = {
        'Manhã': 'o',   # Círculo
        'Tarde': 's',   # Quadrado
        'Noite': '*'    # Estrela
    }
    
    # Gráfico de dispersão com períodos do dia
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=hour_data['Hour'], y=hour_data['Accidents'],
        hue=hour_data['Period'], palette=color_palette,
        style=hour_data['Period'], markers=markers,
        s=100, edgecolor='black'
    )
    plt.title('Distribuição de Acidentes por Hora do Dia (Com Períodos)', fontsize=16)
    plt.xlabel('Hora do Dia', fontsize=14)
    plt.ylabel('Número de Acidentes', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Período', title_fontsize='13', fontsize='12', loc='upper left', frameon=True, shadow=True, borderpad=1)
    st.pyplot(fig)
    
    st.markdown("""
    **Descrição do Gráfico:**  
    O gráfico de dispersão mostra a distribuição de acidentes ao longo do dia, dividida em três períodos: Manhã, Tarde e Noite.  
    - **Eixo X:** Hora do dia (0 a 23).  
    - **Eixo Y:** Número de acidentes.  
    - **Cores e Formas:** Representam os períodos do dia.  

    **Conclusão:** Os horários de pico (manhã e tarde) apresentam uma concentração significativamente maior de acidentes.
    """)