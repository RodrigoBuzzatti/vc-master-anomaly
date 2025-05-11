# Projeto de Detecção de Anomalias em Imagens de Inspeção de Qualidade

Este projeto utiliza técnicas de aprendizado de máquina para detecção de anomalias em imagens de inspeção de qualidade, especificamente em linhas de produção. São utilizadas tanto técnicas clássicas como métodos baseados em redes neurais convolucionais (CNNs) pré-treinadas. O pipeline do projeto inclui a carga e pré-processamento das imagens, extração de atributos, redução de dimensionalidade e aplicação de diferentes modelos de detecção de anomalias.

## Descrição do Pipeline
Carregamento das Imagens: As imagens são carregadas a partir de diretórios especificados, contendo subpastas para imagens com defeito (def_front) e imagens sem defeito (ok_front).

Extração de Atributos: Dependendo da configuração, a extração de atributos pode ser feita simplesmente fazendo um flatten dos pixeis ou utilizando técnicas clássicas como Local Binary Patterns (LBP) e Histogram of Oriented Gradients (HOG), ou utilizando uma CNN pré-treinada como o VGG16.

Redução de Dimensionalidade: Após a extração dos atributos, é aplicado o PCA para reduzir a dimensionalidade dos dados, mantendo 99% da variância explicada.

Treinamento dos Modelos: Três modelos são treinados:

**Isolation Forest**: Para detecção de anomalias.

**One-Class SVM**: Para detecção de anomalias.

**Random Forest**: Modelo supervisionado para classificação das imagens.

Inferência e Avaliação: As inferências são realizadas nos dados de teste, e os resultados são avaliados e visualizados através de matrizes de confusão e relatórios de classificação.

## Resultados

Os resultados das inferências são exibidos diretamente no console e através de gráficos gerados pelo script `anomaly_detection.py`. As matrizes de confusão permitem avaliar o desempenho de cada modelo, indicando a proporção de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.

## Estrutura do Projeto

- `anomaly_detection.py`: Script principal para carregar as imagens, extrair atributos, treinar os modelos e realizar as inferências.
- `utils.py`: Funções utilitárias para carga de imagens, extração de atributos e visualização de resultados.
- `requirements.txt`: Lista de dependências necessárias para executar o projeto.
- `.gitignore`: Arquivo que define quais arquivos ou pastas devem ser ignorados pelo Git.

## Dependências

Para instalar as dependências do projeto, utilize o seguinte comando:

```bash
pip install -r requirements.txt
````

## Dataset

O dataset foi baixado do kaggle [aqui](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).
Foi usada a versão de resolução 300 x 300 e a mesma separação proposta na competição original.
