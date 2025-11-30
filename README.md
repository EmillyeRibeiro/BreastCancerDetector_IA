# BreastCancerDetector_IA

> Este repositório contém o código e a documentação para um sistema de detecção de câncer de mama em imagens, utilizando **Redes Neurais Convolucionais (CNN)** e **Transfer Learning / Fine-Tuning** com modelos pré-treinados, como a ResNet50. O projeto inclui diferentes arquiteturas testadas, experimentos de otimização, análise de resultados e estrutura completa para replicação do treinamento. O objetivo é classificar imagens mamárias entre “benignas” e “malignas”, demonstrando como modelos de Deep Learning podem auxiliar na identificação precoce de câncer de mama, ferecendo suporte a estudos acadêmicos e aplicações práticas em visão computacional e à medicina.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-navy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-yellow)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-purple)](https://matplotlib.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Sumário

- [Descrição](#descrição)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Scripts](#scripts)
- [Requisitos](#requisitos)
- [Resultados](#resultados)
- [Contribuições](#contribuições)
- [Referências](#referências)

## Artigo Completo

O artigo detalhado deste estudo — contendo **metodologia, estruturas de modelos, experimentos, resultados e discussões** — está disponível em PDF na pasta de documentação:

**BreastCancerDetector_IA/**  
└─ **docs/**  
&nbsp;&nbsp;&nbsp;&nbsp;└─ **Artigo.pdf**

### 📄 Baixar / Visualizar Artigo
👉 [Clique aqui para abrir o PDF](./docs/Artigo.pdf)

*Observação:* Navegadores modernos geralmente permitem visualizar PDFs diretamente. Caso não funcione, utilize o botão de download.

## Descrição do Projeto

O objetivo principal deste repositório é construir um sistema capaz de:

- Classificar imagens de mama entre **benignas** e **malignas**  
- Comparar diferentes arquiteturas de CNN  
- Avaliar ganhos obtidos através de **Fine-Tuning** com ResNet50  
- Explorar hiperparametrizações, camadas adicionais e regularização  
- Gerar gráficos, métricas e análises estatísticas

### A metodologia inclui:

**CNNs criadas do zero**
- Camadas Convolution, MaxPooling e Dense  
- Ajustes de Dropout e funções de ativação  
- Experimentos com diferentes profundidades da rede  

**Fine-Tuning com ResNet50**
- Congelamento de camadas iniciais  
- Treinamento das últimas camadas convolucionais  
- Ajuste fino da taxa de aprendizado  
- Data Augmentation avançado  

**Pré-processamento de imagens**
- Redimensionamento  
- Normalização  
- Leitura de caminho de imagens a partir da planilha  
- Divisão entre treino/validação/teste  

**Avaliação dos modelos**
- Acurácia final  
- Matriz de confusão  
- Gráficos de perda e acurácia  
- Comparação entre modelos CNN e Fine-Tuning  

---

## Estrutura do Repositório
**BreastCancerDetector_IA/**  
├─ **dataset/**  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ **exemplo_imagem.png**  
│&nbsp;&nbsp;&nbsp;&nbsp;└─ **Planilha.csv**  
│  
├─ **docs/**  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ **Artigo.pdf**  
│&nbsp;&nbsp;&nbsp;&nbsp;└─ **Códigos de IA para subir no GitHube.pdf**  
│  
├─ **src/**  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ **cnn_teste1.py**  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ **cnn_teste2_maior_acuracia.py**  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ **cnn_teste3.py**  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ **cnn_teste4_tuned.py**  
│&nbsp;&nbsp;&nbsp;&nbsp;├─ **finetuning_resnet50_v1.py**  
│&nbsp;&nbsp;&nbsp;&nbsp;└─ **finetuning_resnet50_v2.py**  
│  
├─ **results/**  
│&nbsp;&nbsp;&nbsp;&nbsp;└─ *(gerado automaticamente pelos scripts)*  
│  
├─ **LICENSE**  
├─ **README.md**  
└─ **requirements.txt**

## Scripts

- `cnn_teste1.py` — CNN simples (teste 1)
- `cnn_teste2_maior_acuracia.py` — variante com maior acurácia encontrada
- `cnn_teste3.py` — teste alternativo
- `cnn_teste4_tuned.py` — CNN com ajustes (dropout, lr, etc.)
- `finetuning_resnet50_v1.py` — fine-tuning usando ResNet50 (fase 1 + fine-tune)
- `finetuning_resnet50_v2.py` — outra versão de fine-tuning

## Resultados e Contribuições
O projeto gera métricas de acurácia, matrizes de confusão, gráficos de treino e validação, comparações entre diferentes arquiteturas de CNN e análises do impacto do Fine-Tuning, sendo que contribuições, sugestões e melhorias são bem-vindas por meio de issues ou pull requests.

## Reprodutibilidade
Os experimentos foram executados com random_state fixo, pré-processamento consistente, arquitetura modular e scripts independentes e versionados, o que garante total reprodutibilidade dos resultados e permite comparações justas entre diferentes modelos.

## Tecnologias Utilizadas

- Python 3.10+
- TensorFlow / Keras
- ResNet50 (Fine-Tuning)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

GPU opcional para acelerar o treinamento

## Referências

- Artigo e códigos originais (stored in `docs/`).
- Estudos citados no artigo: Shen et al., Hanis et al., etc.