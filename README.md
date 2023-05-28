# word2vec-training
## Para Doc2Vec

### Modos de treino

Em geral, pode se fazer de duas maneiras:
* Treina o modelo direto com o corpus de interesse
* Treina o modelo com um corpus grande (e.g. Wikipedia), depois infere os vetores para o corpus de interesse

### doc2vec\_tokenize.py

Pre-processa o corpus, de forma que as sentenças são separadas por quebra de linha, e os tokens são separados por espaços. Usa o pacote NLTK e é necessário ter o módulo punkt do NLTK instalado. Para instalar: abrir um terminal do python, chamar nltk.download() e procurar o pacote punkt.

### doc2vec\_train.py

Treina um modelo a partir de um corpus pre-processado. Como saída, cria o modelo, um txt com as configurações usadas e opcionalmente um CSV com os embeddings, todos dentro de um diretorio.

Esse script faz as seguintes suposições:
* Cada arquivo é um documento
* Cada linha do arquivo é uma sentença
* As palavras/tokens estão delimitados por espaços

### doc2vec\_infer.py

Gera embeddings para um corpus pré-processado a partir de um modelo doc2vec.

## Para corpus Wikipedia

### wiki\_downloader.py

Baixa o dump no formato bz2, extrai para xml, e roda o utils/WikiExtractor.py para gerar os arquivos de texto. Cada arquivo contem mais de um documento, separados em blocos:

    <doc id="1" ... >
    conteudo do documento
    </doc>
    <doc id="2" ... >
    conteudo do proximo documento
    </doc>
    ...

Por isso, é necessário separar cada documento em um arquivo ou remover as tags XML dependendo do que se quer fazer.

### wiki\_remove\_tags.py

Remove as tags XML dos arquivos da Wikipédia. Use esse se for treinar o modelo, mas não for usar os embeddings dos artigos da Wikipédia em si.

### wiki\_split\_docs.py

Separa cada artigo em seu próprio arquivo.
