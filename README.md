# Instalação

Para executar o notebook é preciso que o python esteja instalado

Obs: Recomendado utilizar o VsCode ou semelhantes.

Iniciando um ambiente virtual (venv):

```
python -m venv venv 
```

Ativando o ambiente virtual:

Linux:

```
./venv/bin/activate
```

Windows:

```
./venv/Lib/Scripts/activate
```

Listando os pacotes:

```
pip list
```

Instalando os pacotes:

```
pip install -r requirements.txt
```

# Utilização

### Config.py
O arquivo config.py é responsável por armazenar configurações de funcionamento (não há necessidade de mexer nele), ele guarda:
- a seed e os algoritmos e suas configurações

### main.ipynb
O arquivo main.ipynb faz a avaliação do db.
