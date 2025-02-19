import streamlit as st

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns

import matplotlib.pyplot as plt

import zipfile

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Processando
# Nome do arquivo ZIP
zip_file_path = "accidents_2017_to_2023_english.zip"
csv_file_name = "accidents_2017_to_2023_english.csv"  # Nome do arquivo dentro do ZIP

# Abrindo o arquivo ZIP e lendo o CSV
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        df = pd.read_csv(f)

# Preenchimento de valores ausentes
# medianas (numéricos) ou modos (categóricos)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
    elif df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])

# Remoção de outliers para colunas numéricas
# Usando intervalo interquartílico (IQR)
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

# conversão de colunas textuais para formatos apropriados
df['inverse_data'] = pd.to_datetime(df['inverse_data'], errors='coerce')  # converte para data
df['hour'] = pd.to_datetime(df['hour'], format='%H:%M:%S', errors='coerce').dt.hour  # extração da hora somente

# seleção de algumas colunas categóricas para one-hot encoding
categorical_columns_to_encode = ['week_day', 'cause_of_accident', 'type_of_accident']
data = pd.get_dummies(df, columns=categorical_columns_to_encode, drop_first=True)

# normalização de colunas numéricas (opcional)
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

#Hipótese 1
st.title('Hipótese 1')
st.markdown("""
            
Hipótese: Acidentes graves são mais comuns em condições climáticas adversas.

Tipo de Gráfico e atributos envolvidos: Gráfico de barras empilhadas com os atributos wheater_condition no eixo x em relação à victims_condition no eixo y.

Comportamento esperado do gráfico:

Validação: Há uma concentração maior de acidentes graves em condições como "chuva" ou "neblina".
Refutação: Acidentes graves são mais comuns em condições climáticas normais.            

            """)

# Preparando os dados para o gráfico de barras empilhadas
stacked_data = df.groupby(["wheather_condition", "victims_condition"]).size().unstack(fill_value=0)

# Criando o gráfico
fig, ax = plt.subplots(figsize=(8, 5))

colormap = plt.get_cmap("Accent")
colors = [colormap(i) for i in np.linspace(0, 1, len(stacked_data.columns))]


# Padrões de textura para acessibilidade
hatch_patterns = ["x","o"]

# Desenhar barras empilhadas com cores e texturas
bottom = np.zeros(len(stacked_data))
for i, (col, color) in enumerate(zip(stacked_data.columns, colors)):
    bars = ax.bar(
        stacked_data.index,
        stacked_data[col],
        label=col,
        bottom=bottom,
        color=color,
        hatch=hatch_patterns[i % len(hatch_patterns)],
        edgecolor="black",  # Para melhor contraste
    )
    bottom += stacked_data[col].values

# Personalizando o gráfico
plt.title("Condição da Vítima por Condição Climática")
plt.xlabel("Condição Climática")
plt.ylabel("Contagem")
plt.legend(title="Condição da Vítima")
plt.xticks(rotation=0)

# Exibir o gráfico
plt.tight_layout()
plt.show()

st.markdown("""
A hipótese é refutada.

Motivo principal:

Acidentes graves (com vítimas feridas) são mais comuns em condições climáticas normais, como "Clear sky" e "Cloudy", do que em condições adversas, como "Rainy" ou "Fog".
Observação adicional:

Em condições adversas, como "Rainy", o número de acidentes com vítimas feridas é relevante, mas a frequência total de acidentes é muito menor em comparação com condições normais.
            """)


st.markdown("""
**Tarefa de mineração: Análise de Associação (Algoritmo Apriori).**

Validação: Regras do tipo: "wheather_condition = Rainy → victims_condition = With injured victims" aparecem com alta confiança e suporte.

Refutação: Não há regras significativas que associem "Rainy" ou "Cloudy" a acidentes graves.
            """)



# Selecionar apenas as colunas necessárias para a análise
df_subset = df[['victims_condition', 'wheather_condition']].dropna()

# Transformar os dados em formato one-hot encoding (transacional)
df_trans = pd.get_dummies(df_subset)

# Aplicar o algoritmo Apriori com um suporte mínimo de 0.01 (1%)
frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)

# Gerar regras de associação com confiança mínima de 50%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filtrar regras onde "road_type_Simple" está nos antecedentes e condições das vítimas nos consequentes
filtered_rules = rules[
    (rules['antecedents'].apply(lambda x: 'wheather_condition_Rainy' in x)) &
    (rules['consequents'].apply(lambda x: 'victims_condition_With injured victims' in x or
                                          'victims_condition_With dead victims' in x))
]


# Exibir as regras filtradas
st.write("Regras filtradas:")
st.write(filtered_rules)

st.markdown("""
Esta regra filtrada sugere que, apesar de uma confiança moderada (65.4%) de que "chuva" leva a "vítimas feridas", a associação entre essas duas condições não é forte. O lift abaixo de 1, leverage negativa, e conviction baixa indicam que, na verdade, "chuva" não está fortemente associada a acidentes graves com vítimas feridas. Isso reflete uma associação fraca, o que refuta a hipótese de que condições climáticas adversas resultam em mais acidentes graves.
""")

#Hipótese 2

st.title('Hipótese 2')
st.markdown("""
Hipótese: Estradas de pista simples têm mais acidentes graves comparadas a rodovias de pista dupla.

Tipo de Gráfico e atributos envolvidos: Gráfico de barras empilhadas com victims_condition no eixo y em relação a road_type no eixo x.

Comportamento esperado do gráfico:

Validação: Maior proporção de acidentes graves ocorre em rodovias de pista simples.
Refutação: Não há diferença significativa ou rodovias de pista dupla têm mais acidentes graves.
            """)

# Preparando os dados para o gráfico de barras empilhadas
stacked_data = df.groupby(["road_type", "victims_condition"]).size().unstack(fill_value=0)

# Criando o gráfico de barras empilhadas
ax = stacked_data.plot(kind="bar", stacked=True, colormap="Accent", figsize=(8, 6))

# Adicionando os valores exatos às barras empilhadas
for i, bars in enumerate(ax.containers):
    for bar in bars:
        # Adiciona rótulos apenas para barras com altura maior que 0
        if bar.get_height() > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Centraliza o texto
                bar.get_y() + bar.get_height() / 2,  # Posiciona o texto no centro da barra
                int(bar.get_height()),  # Valor inteiro da altura da barra
                ha="center",  # Alinhamento horizontal
                va="center",  # Alinhamento vertical
                fontsize=9,   # Tamanho da fonte
                color="black" if i % 2 == 0 else "white"  # Ajusta a cor do texto para contraste
            )

# Personalizando o gráfico
plt.title("Condição da Vítima por Tipo de Rodovia", fontsize=14)
plt.xlabel("Tipo de Rodovia", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.legend(title="Condição da Vítima", fontsize=10)
plt.xticks(rotation=0)

# Exibindo o gráfico
fig = plt.tight_layout()
st.pyplot(fig)

# Calculate the proportion of accidents for single and double lane roads
stacked_data = df.groupby(["road_type", "victims_condition"]).size().unstack(fill_value=0)

# Calculate totals for each road type
total_simple = stacked_data.loc['Simple'].sum()
total_double = stacked_data.loc['Double'].sum()

# Calculate proportions of accidents with injured victims
proportion_simple = stacked_data.loc['Simple', 'With injured victims'] / total_simple
proportion_double = stacked_data.loc['Double', 'With injured victims'] / total_double

st.write(f"Proporção de acidentes com vítimas em pistas simples: {proportion_simple:.2%}")
st.write(f"Proporção de acidentes com vítimas em pistas duplas: {proportion_double:.2%}")

st.markdown("""
A hipótese é refutada.

Motivo principal:

Acidentes graves (com vítimas feridas) são proporcionalmente mais frequentes em rodovias de pista dupla ("Double") do que em rodovias de pista simples ("Simple"). Embora o número absoluto de acidentes graves em rodovias simples seja alto, isso ocorre devido à maior quantidade total de acidentes registrados nesse tipo de rodovia, e não por uma maior severidade.
Observação adicional:

Rodovias de pista múltipla ("Multiple") apresentam a menor frequência de acidentes, tanto graves quanto leves, indicando maior segurança geral nesse tipo de infraestrutura.
            """)

st.markdown("""
**Tarefa de mineração: Análise de Associação (Algoritmo Apriori).**

Validação: Regras do tipo: "road_type = Simple → victims_condition = With injured victims" aparecem com alta confiança e suporte.

Refutação: Não há regras significativas que associem "pista simples" a acidentes graves.
            """)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Selecionar apenas as colunas necessárias para a análise
df_subset = df[['victims_condition', 'road_type']].dropna()

# Transformar os dados em formato one-hot encoding (transacional)
df_trans = pd.get_dummies(df_subset)

# Aplicar o algoritmo Apriori com um suporte mínimo de 0.01 (1%)
frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)

# Gerar regras de associação com confiança mínima de 50%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filtrar regras onde "road_type_Simple" está nos antecedentes e condições das vítimas nos consequentes
filtered_rules = rules[
    (rules['antecedents'].apply(lambda x: 'road_type_Simple' in x)) &
    (rules['consequents'].apply(lambda x: 'victims_condition_With injured victims' in x or
                                          'victims_condition_With dead victims' in x))
]

# Exibir as regras filtradas
st.write("Regras filtradas:")
st.write(filtered_rules)

st.markdown("""
Embora haja uma regra com confiança relativamente alta (70,85%), o lift, leverage e zhang’s metric sugerem que a associação entre acidentes em rodovias de pista simples e vítimas feridas é fraca. Portanto, não há evidências robustas de que o tipo de rodovia seja determinante para a gravidade dos acidentes.
            """)

#Hipótese 3

st.title('Hipótese 3')

st.markdown("""
            **Tarefa de geração de gráfico: Boxplot**

Hipótese: Acidentes com ferimentos graves em horários noturnos estão associados a determinadas causas, como direção sob efeito de álcool ou sono.

Tipo de Gráfico e atributos envolvidos: Boxplot com o eixo X sendo o cause_of_accident (categorias de causa) e o eixo y sendo o severy_injured (número de vítimas com ferimentos graves).

Comportamento esperado do gráfico:

Validação: As causas "álcool" e "sono" devem apresentar uma distribuição de gravidade dos ferimentos (severely_injured) mais alta, refletindo a relação entre essas causas e acidentes graves.

Refutação: A gravidade dos acidentes é similar para todas as causas, sem relação específica com "álcool" ou "sono".
""")


# filtro de acidentes noturnos (18h às 6h)
df['is_night'] = df['hour'].apply(lambda x: 'Noturno' if (x >= 18 or x < 6) else 'Diurno')
df_night = df[df['is_night'] == 'Noturno']

# categorias de 'cause_of_accident' com pelo menos um valor válido em 'severely_injured'
valid_categories = (
    df_night.groupby('cause_of_accident')['severely_injured']
    .count()
    .loc[lambda x: x > 0]  # categorias com pelo menos 1 valor válido
    .index
)

df_night_filtered = df_night[df_night['cause_of_accident'].isin(valid_categories)]

# boxplot
fig = plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_night_filtered,
    x='cause_of_accident',
    y='severely_injured',
    order=['Alcohol consumption', 'Driver was sleeping'],  # Ajuste conforme necessário
)
plt.title("Distribuição de Ferimentos Graves por Causa em Acidentes Noturnos")
plt.xlabel("Causa do Acidente")
plt.ylabel("Número de Ferimentos Graves (Severely Injured)")
plt.xticks(rotation=45)

st.pyplot(fig)

##Gráfico 2
# causas específicas
filtered_causes = ['Alcohol consumption', 'Driver was sleeping']
df_filtered = df_night_filtered[df_night_filtered['cause_of_accident'].isin(filtered_causes)]
cause_counts = df_filtered['cause_of_accident'].value_counts()

# gráfico de barras
fig = plt.figure(figsize=(8, 6))
sns.barplot(x=cause_counts.index, y=cause_counts.values, order=filtered_causes, palette="viridis")
plt.title("Número de Registros para Causas Selecionadas")
plt.xlabel("Causa do Acidente")
plt.ylabel("Número de Registros")
plt.xticks(rotation=45, ha='right')
plt.show()

st.pyplot(fig)


st.markdown("""
            **Tarefa de mineração de dados: K-means ou HDBSCAN**

Acredita-se que com a clusterização pode identificar padrões ocultos nos dados, agrupando os acidentes com base em características como:

*   Hora (período noturno).
*   Causa do acidente.
*   Severidade (severely_injured).

O que se espera com essa tarefa de mineração:

*   Validação: Formam-se clusters em que acidentes no período noturno estão associados a causas específicas (álcool, sono) e apresentam maior gravidade.
*   Refutação: Não há clusters específicos; acidentes graves aparecem igualmente distribuídos nos períodos e causas.
""")

columns_to_cluster = ['hour', 'cause_of_accident', 'severely_injured']
df_cluster = df[columns_to_cluster].copy()

df_cluster['cause_of_accident'] = df_cluster['cause_of_accident'].fillna('Unknown')
df_cluster = df_cluster.dropna(subset=['hour', 'severely_injured'])

# One-Hot Encoding para 'cause_of_accident'
encoder = OneHotEncoder()
encoded_causes = encoder.fit_transform(df_cluster[['cause_of_accident']]).toarray()
cause_names = encoder.get_feature_names_out(['cause_of_accident'])

df_cluster = df_cluster.drop('cause_of_accident', axis=1)
df_cluster = pd.concat([df_cluster, pd.DataFrame(encoded_causes, columns=cause_names)], axis=1)

imputer = SimpleImputer(strategy='mean')
df_scaled_imputed = imputer.fit_transform(df_cluster)

# K-means (simplificado)
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['kmeans_cluster'] = kmeans.fit_predict(df_scaled_imputed)

# Análise dos clusters
# Agrupamento por cluster
cluster_summary = df_cluster.groupby('kmeans_cluster').mean()

# Mostrar os dados processados
st.subheader("Dados Agrupados por Cluster")
st.write(cluster_summary)

# Visualizar os clusters em um gráfico
st.subheader("Distribuição dos Clusters")
fig, ax = plt.subplots()
for cluster in df_cluster['kmeans_cluster'].unique():
    subset = df_cluster[df_cluster['kmeans_cluster'] == cluster]
    ax.scatter(subset['hour'], subset['severely_injured'], label=f'Cluster {cluster}')
ax.set_title("Distribuição dos Clusters")
ax.set_xlabel("Hour")
ax.set_ylabel("Severely Injured")
ax.legend()
st.pyplot(fig)

st.markdown("""
            
*   Cluster 0 tende a ter valores mais altos para algumas causas de acidente, como cause_of_accident_Alcohol consumption e cause_of_accident_Alcohol ingestion by the driver, sugerindo que esse cluster pode estar relacionado a acidentes mais graves ou frequentes envolvendo álcool.
*   Cluster 1 tem uma distribuição das causas de acidentes mais próxima do Cluster 0, mas com um valor ligeiramente menor para causas envolvendo álcool.
*   Cluster 2 parece ter uma frequência mais uniforme e valores mais baixos para muitas causas, o que sugere que os acidentes nesse cluster podem ser de naturezas mais variadas, com causas menos comuns ou menos graves.

Em resumo, pode-se dizer que a hipótese é refutada, uma vez que a clusterização só conseguiu ser validada em questão do consumo de álcool e não associada ao período/horário do acidente.

            """)





st.title('Hipótese 4')

st.markdown(
"""
**Tarefa de geração de gráfico: Barra**

Hipótese: Os acidentes ocorrem com maior frequência nos finais de semana.

Tipo de Gráfico e atributos envolvidos: Gráfico utilizado: Gráfico de barras.

Atributos envolvidos:

Eixo X: Dias da semana (variável categórica).

Eixo Y: Contagem de acidentes.

Comportamento esperado do gráfico:

Validação da hipótese: O gráfico de barras deve mostrar que alguns dias da semana têm contagens de acidentes significativamente maiores em relação aos outros.

Refutação da hipótese: Caso o gráfico mostre que as contagens de acidentes estão uniformemente distribuídas entre todos os dias, a hipótese é refutada.

"""
)

# Contagem de acidentes por dia da semana
accidents_per_day = df['week_day'].value_counts().sort_index()

# Visualizar a contagem de acidentes por dia da semana
accidents_per_day

# Gráfico de barras para a contagem de acidentes por dia da semana
fig = plt.figure(figsize=(10, 6))
accidents_per_day.plot(kind='bar', color='skyblue')
plt.title('Contagem de Acidentes por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Número de Acidentes')
plt.xticks(rotation=45)

st.pyplot(fig)


st.markdown(
"""
**Tarefa de mineração de dados: Teste Qui-Quadrado de Independência:**

Utilizado para avaliar se existe uma relação significativa entre as variáveis "dias da semana" e "frequência de acidentes".

O que se espera com essa tarefa de mineração:

Validação: Um p-valor abaixo de 0.05 indica que existe uma relação significativa entre os dias da semana e a frequência de acidentes, validando a hipótese.
Refutação: Um p-valor acima de 0.05 indica que não há relação significativa entre os dias da semana e a frequência de acidentes, refutando a hipótese.
"""
)
from scipy.stats import chi2_contingency

# tabela de contingência com a contagem de acidentes por dia da semana
contingency_table = pd.crosstab(index=df['week_day'], columns='acidentes', values=df['week_day'], aggfunc='count')

# teste Qui-Quadrado
chi2, p, dof, expected = chi2_contingency(contingency_table)

# p-valor
st.write(f"Valor de p: {p}")


st.markdown("""
            Um p-valor de 1.0 significa que não há evidência estatística para rejeitar a hipótese nula. Em outras palavras, o resultado do teste Qui-Quadrado indica que não existe uma relação significativa entre os dias da semana e a frequência de acidentes no seu conjunto de dados.

Isso sugere que a distribuição dos acidentes entre os dias da semana é uniforme, ou seja, não há evidência de que certos dias (como finais de semana) tenham mais acidentes do que outros de forma significativa, segundo esse teste.
            """)

st.title("Hipótese 5")

st.markdown("""
Gráfico utilizado: Gráfico de barras com agrupamento por clusters.

Atributos envolvidos:

Eixo X: Horas do dia (0-23).

Eixo Y: Contagem de acidentes.
            """)

##Refazer ------------------------------------------------

# Nome do arquivo ZIP
zip_file_path = "accidents_2017_to_2023_portugues.zip"
csv_file_name = "accidents_2017_to_2023_portugues.csv"  # Nome do arquivo dentro do ZIP

# Abrindo o arquivo ZIP e lendo o CSV
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        df = pd.read_csv(
            f,
            sep=",",         # Delimitador é vírgula
            quotechar='"',   # Aspas que encapsulam os valores
            skipinitialspace=True,  # Ignora espaços após a vírgula
        )

# Renomeando as colunas para inglês
df.columns = [
    "inverse_data", "week_day", "hour", "state", "road_id", "km", "city",
    "cause_of_accident", "type_of_accident", "victims_condition",
    "weather_timestamp", "road_direction", "wheather_condition",
    "road_type", "road_delineation", "people", "deaths", "slightly_injured",
    "severely_injured", "uninjured", "ignored", "total_injured",
    "vehicles_involved", "latitude", "longitude", "regional", "police_station"
]

# Preenchimento de valores ausentes
# medianas (numéricos) ou modos (categóricos)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
    elif df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])


# Remoção de outliers para colunas numéricas
# Usando intervalo interquartílico (IQR)
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

  # conversão de colunas textuais para formatos apropriados
df['inverse_data'] = pd.to_datetime(df['inverse_data'], errors='coerce')  # converte para data
df['hour'] = pd.to_datetime(df['hour'], format='%H:%M:%S', errors='coerce').dt.hour  # extração da hora somente

# seleção de algumas colunas categóricas para one-hot encoding
categorical_columns_to_encode = ['week_day', 'cause_of_accident', 'type_of_accident']
data = pd.get_dummies(df, columns=categorical_columns_to_encode, drop_first=True)

# normalização de colunas numéricas (opcional)
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])



#-------------------------------------------------------


hour_counts = df['hour'].value_counts().sort_index()

hour_counts.index = hour_counts.index.astype(int) 

# Ajustando o gráfico de acidentes por hora do dia
plt.figure(figsize=(10, 6))

# Configurações de estilo
cores = ['#4B4BC4']  # Azul similar ao gráfico de comparação
marcadores = ['o']  # Círculos
linhas = ['-']  # Linha contínua

print(hour_counts)

# Plotagem do gráfico
plt.plot(
    hour_counts.index.to_numpy(),  # Converte para NumPy
    hour_counts.values,           # Valores já são NumPy compatíveis
    color=cores[0], marker=marcadores[0], markersize=8,
    linewidth=2, alpha=0.85, linestyle=linhas[0], label='Acidentes por Hora'
)
plt.scatter(hour_counts.index, hour_counts.values, color=cores[0], marker=marcadores[0], s=70, edgecolors='black', alpha=0.9)

# Configurações de título, eixos e legenda
plt.title('Distribuição de Acidentes por Hora do Dia', fontsize=16, fontweight='bold')
plt.xlabel('Hora do Dia', fontsize=14)
plt.ylabel('Número de Acidentes', fontsize=14)

plt.legend(title="Legenda", title_fontsize='13', fontsize='12', loc='upper left', frameon=True, shadow=True, borderpad=1)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.xticks(range(hour_counts.index.min(), hour_counts.index.max() + 1), fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
#plt.show()

st.pyplot(fig)

st.markdown("""
Clusters: Agrupamentos das horas com base em padrões de frequência.
            """)

hour_data = pd.DataFrame({'Hour': hour_counts.index, 'Accidents': hour_counts.values})
from sklearn.preprocessing import StandardScaler

# Normalizando os dados
scaler = StandardScaler()
hour_data_scaled = scaler.fit_transform(hour_data[['Hour', 'Accidents']])

inertia = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(hour_data_scaled)
    inertia.append(kmeans.inertia_)

# Plotando o gráfico do método do cotovelo
fig = plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', color='blue')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
#plt.show()


st.write('Esse método ajuda a identificar o número de clusters de maneira eficiente e objetiva, fornecendo uma base sólida para a escolha do modelo adequado para segmentar os dados com K-Means.')
st.pyplot(fig)
st.write('Ele agrupa as horas do dia (0 a 23) em 3 clusters com base no número de acidentes. Ele tenta identificar, por exemplo, um cluster de manhã (onde os acidentes podem ser mais ou menos frequentes), um cluster à tarde e outro cluster à noite, dependendo dos padrões de acidentes ao longo do dia.')

kmeans = KMeans(n_clusters=3, random_state=42)
hour_data['Cluster'] = kmeans.fit_predict(hour_data_scaled)

fig = plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=hour_data['Hour'], y=hour_data['Accidents'],
    hue=hour_data['Cluster'], palette='viridis', s=100, edgecolor='black'
)
plt.title('Clusters de Acidentes por Hora do Dia', fontsize=16)
plt.xlabel('Hora do Dia', fontsize=14)
plt.ylabel('Número de Acidentes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
#plt.show()

st.pyplot(fig)

st.markdown("""
A análise revelou que os horários de pico (manhã e tarde) apresentam uma concentração significativamente maior de acidentes em comparação com o período da noite. Esses resultados estão alinhados com as expectativas, dado que os horários da manhã e da tarde coincidem com o fluxo intenso de veículos durante os deslocamentos diários para trabalho, escola e outras atividades. O algoritmo K-Means, aplicado aos dados escalados, destacou uma distribuição clara entre os períodos definidos, evidenciando o aumento da frequência de acidentes nas janelas de tempo que correspondem aos horários de pico.

O código foi modificado para definir previamente os períodos do dia e os grupos de clusters, sem a necessidade do uso do K-Means para segmentação dos dados. A segmentação foi feita com base nos horários, dividindo o dia em três períodos específicos: Manhã, Tarde e Noite. A divisão foi feita conforme os intervalos de horário:

Manhã: das 00:00 às 10:00 horas.

Tarde: das 10:00 às 18:00 horas.

Noite: das 18:00 às 24:00 horas.

Para representar visualmente os clusters, foram escolhidas cores distintas que garantem boa legibilidade e contrastes adequados:

Manhã: Usamos o tom de amarelo suave, #D9D96D. Essa cor foi escolhida para representar o início do dia de maneira clara e luminosa.

Tarde: Para a tarde, foi utilizado o tom de cinza #A8A8A8, uma cor mais neutra, que traz suavidade e equilíbrio ao gráfico.

Noite: O período da noite foi representado por um tom de azul mais intenso, #3A3AC4, para trazer uma sensação de profundidade e destaque para as horas noturnas.

Essas cores foram escolhidas para proporcionar uma distinção clara entre os períodos, mantendo uma paleta harmoniosa, que facilita a interpretação visual dos dados. Além disso, as cores foram selecionadas para garantir bom contraste, mesmo para usuários com deficiências de percepção de cores, como daltonismo.
            """)

# Definir uma função para categorizar as horas nos períodos desejados
def categorize_period(hour):
    if 0 <= hour < 10:
        return 'Manhã'
    elif 10 <= hour < 18:
        return 'Tarde'
    else:
        return 'Noite'


# Criar a coluna 'Period' com a categorização
hour_data['Period'] = hour_data['Hour'].apply(categorize_period)

# Definir as cores específicas para cada período
color_palette = {
    'Manhã': '#D9D96D',  # Cor mais suave de amarelo
    'Tarde': '#A8A8A8',  # Um cinza mais suave
    'Noite': '#3A3AC4'   # Azul mais intenso
}


# Plotando o gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=hour_data['Hour'], y=hour_data['Accidents'],
    hue=hour_data['Period'], palette=color_palette, s=100, edgecolor='black'
)

# Configurações do gráfico
plt.title('Distribuição de Acidentes por Hora do Dia (Com Períodos)', fontsize=16)
plt.xlabel('Hora do Dia', fontsize=14)
plt.ylabel('Número de Acidentes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Período', title_fontsize='13', fontsize='12', loc='upper left', frameon=True, shadow=True, borderpad=1)
fig = plt.tight_layout()
plt.show()

st.pyplot(fig)
