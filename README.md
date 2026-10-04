# atv-knn-pratica

![Python](https://img.shields.io/badge/python-3.14+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.5+-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Prática de classificação e recomendação com **K-Nearest Neighbors (KNN)** e **Perceptron**, implementados do zero com Python e NumPy, sem scikit-learn. O projeto reúne três desafios, cada um em seu próprio notebook, aplicados a cenários de negócio: detecção de fraude, previsão de churn e recomendação de servidores.

## Desafios

| # | Notebook | Algoritmo | Problema |
|---|----------|-----------|----------|
| 1 | [`desafio_1_perceptron_fraude.ipynb`](desafio_1_perceptron_fraude.ipynb) | Perceptron | Classificar transações como legítimas ou suspeitas |
| 2 | [`desafio_2_knn_churn.ipynb`](desafio_2_knn_churn.ipynb) | KNN | Classificar clientes em baixo ou alto risco de churn |
| 3 | [`desafio_3_recomendacao_servidores.ipynb`](desafio_3_recomendacao_servidores.ipynb) | Vizinhos mais próximos | Recomendar servidores a partir dos requisitos da aplicação |

### Desafio 1: Perceptron para detecção de fraude

Implementa um Perceptron de camada única treinado com a regra de atualização de Rosenblatt:

$$w \leftarrow w + \eta \cdot (y - \hat{y}) \cdot x \qquad b \leftarrow b + \eta \cdot (y - \hat{y})$$

- **Dados:** 8 transações com 2 atributos, rotuladas como `0` (legítima) ou `1` (suspeita).
- **Treino:** taxa de aprendizado `0.1`, até 100 épocas, com parada antecipada quando não há mais erros.
- **Parâmetros aprendidos:** pesos ≈ `[0, 0.4]` e viés = `-1.2`. Na prática, o modelo marca como suspeita toda transação cujo segundo atributo é ≥ 3.

| Transação | Atributos | Diagnóstico |
|-----------|-----------|-------------|
| A | `[5.0, 4.0]` | Suspeita (risco de fraude) |
| B | `[7.0, 6.5]` | Suspeita (risco de fraude) |

### Desafio 2: KNN para previsão de churn

Classificador KNN com distância calculada de forma vetorizada e duas métricas disponíveis:

- **Euclidiana:** $d(x, p) = \sqrt{\sum_i (x_i - p_i)^2}$
- **Manhattan:** $d(x, p) = \sum_i |x_i - p_i|$

A classe final é decidida por votação majoritária entre os `k` vizinhos mais próximos. A função retorna a classe prevista, os índices dos vizinhos e as distâncias até eles.

- **Atributos:** `[dias de inatividade, chamados críticos]`
- **Classes:** `0` = baixo risco, `1` = alto risco
- **Configuração:** `k = 3`, métrica euclidiana

| Cliente | Atributos | Vizinhos (índices) | Distâncias | Diagnóstico |
|---------|-----------|--------------------|------------|-------------|
| 1 | `[13.0, 10.0]` | 4, 5, 6 | 6.32, 7.07, 9.22 | **Alto risco** |
| 2 | `[7.0, 1.5]` | 3, 2, 1 | 1.12, 2.06, 4.27 | **Baixo risco** |

### Desafio 3: Recomendação de servidores

Usa a mesma ideia de vizinhança, mas sem votação: em vez de classificar, ranqueia os itens de um catálogo pela distância euclidiana até o perfil desejado e devolve os `k` mais próximos.

- **Catálogo:** 6 servidores descritos por `[vCPUs, RAM (GB), SSD (GB)]`, de "Micro Instância Web" a "Big Data & AI Training".
- **Perfil demandado:** 12 vCPUs, 28 GB de RAM e 850 GB de SSD, com `k = 2`.

| Posição | Servidor | vCPUs | RAM | SSD | Distância |
|---------|----------|-------|-----|-----|-----------|
| 1º | High Performance Computing | 32 | 64 GB | 1000 GB | 155.55 |
| 2º | Database Enterprise | 16 | 32 GB | 500 GB | 350.05 |

## Como executar

**Requisitos:** Python 3.14+ e, de preferência, o gerenciador [uv](https://docs.astral.sh/uv/) (o projeto já inclui `uv.lock`).

```bash
# 1. Clone o repositório
git clone https://github.com/LucasBParada/atv-knn-pratica.git
cd atv-knn-pratica

# 2. Instale as dependências
uv sync

# 3. Abra os notebooks
uv run jupyter notebook
```

<details>
<summary>Sem o uv (pip + venv)</summary>

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install notebook numpy
jupyter notebook
```

</details>

## Estrutura do projeto

```
atv-knn-pratica/
├── desafio_1_perceptron_fraude.ipynb
├── desafio_2_knn_churn.ipynb
├── desafio_3_recomendacao_servidores.ipynb
├── main.py              # ponto de entrada (placeholder)
├── pyproject.toml       # metadados e dependências
├── uv.lock              # versões travadas das dependências
└── LICENSE
```

## Tecnologias

- [Python](https://www.python.org/) 3.14
- [NumPy](https://numpy.org/) para cálculo vetorizado de distâncias e do Perceptron
- [Jupyter Notebook](https://jupyter.org/) para execução e documentação dos experimentos
- [uv](https://docs.astral.sh/uv/) para gerenciamento de dependências

## Pontos de atenção

- **Escala dos atributos (desafio 3):** os atributos não são normalizados, então o SSD (em GB, valores de centenas) domina o cálculo da distância e pesa muito mais que vCPUs e RAM. Aplicar normalização (min-max ou z-score) antes de calcular as distâncias tornaria o ranking mais equilibrado.
- **Base de dados pequena:** os conjuntos têm poucos exemplos e são didáticos. Os resultados servem para ilustrar os algoritmos, não para avaliar desempenho em dados reais.

## Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autor

**Lucas de Barros Parada**: [@LucasBParada](https://github.com/LucasBParada)
