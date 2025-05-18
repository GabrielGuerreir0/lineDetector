# 🚗 Detector de Linhas de Pavimento em Python

Este projeto implementa um sistema de **detecção de linhas de pavimento (faixas de estrada)** utilizando **redes neurais treinadas**. O objetivo é identificar as marcações das pistas em imagens de rodovias, com suporte a diferentes formatos de datasets como **CULane** e **Tusimple**.

---

## 🚀 Funcionalidades

- ✅ Detecção de linhas de pavimento em imagens estáticas e vídeos
- ✅ Modelos pré-treinados para os datasets **CULane** e **Tusimple**
- ✅ Suporte a diferentes configurações via argumentos de linha de comando

---

## 📦 Estrutura do Projeto

```
.
├── Detector/                  # Código-fonte principal
│   ├── backbone.py            # Definição da arquitetura de backbone
│   ├── detector.py            # Classe de detecção de linhas
│   └── model.py               # Carregamento e configuração de modelo
├── images/                    # Imagens de teste (exemplo)
├── models/                    # Modelos pré-treinados (vazio inicialmente)
│   ├── culane_18.pth          # Modelo CULane (adicionar manualmente)
│   └── tusimple_18.pth        # Modelo Tusimple (adicionar manualmente)
├── main.py                    # Script principal de execução
├── output.jpg                 # Exemplo de saída gerada
├── warped_output.jpg          # Saída com perspectiva corrigida
├── requirements.txt           # Dependências Python
├── .gitignore
└── README.md                  # Este arquivo
```

---

## ⚙️ Requisitos

- Python **3.7 ou superior**
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- `numpy`
- `torchvision`

Para instalar as dependências, execute:

```bash
pip install -r requirements.txt
```

---

## 📂 Modelos Pré-treinados

Antes de executar o projeto, você **deve criar a pasta `models/`** na raiz do repositório e adicionar manualmente os arquivos de modelo:

```
models/
├── culane_18.pth
└── tusimple_18.pth
```

- `culane_18.pth`: Modelo pré-treinado no dataset **CULane**
- `tusimple_18.pth`: Modelo pré-treinado no dataset **Tusimple**

> ⚠️ **Importante**: Sem esses arquivos, o detector não funcionará.

---

## 🏃 Como Executar

### Execução Básica

```bash
python main.py --model culane   # Usar modelo CULane
python main.py --model tusimple # Usar modelo Tusimple
```

### Argumentos Principais

- `--model {culane,tusimple}`: Seleciona qual modelo carregar
- `--input <caminho>`: Caminho para imagem ou vídeo de entrada (padrão: `images/`)
- `--output <caminho>`: Caminho para salvar a saída processada

### Exemplo completo

```bash
python main.py --model culane --input images/estrada.jpg --output results/saida.jpg
```

---

## 🎯 Testes e Exemplos

- As imagens de exemplo estão na pasta `images/`
- Arquivos como `output.jpg` e `warped_output.jpg` mostram os resultados da detecção e da correção de perspectiva

---

## 🤝 Contribuição

Contribuições são bem-vindas!  
Sinta-se à vontade para abrir **issues** ou enviar **pull requests** com melhorias, correções de bugs ou novas funcionalidades.

---

## 📄 Licença

Este projeto está licenciado sob a **licença MIT**.  
Consulte o arquivo `LICENSE` para mais detalhes.
