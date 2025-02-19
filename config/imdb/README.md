# IMDb dataset

This dataset, proposed in \citet{maas2011learning}, is an NLP 
binary classification task for sentiment analysis based on 50,000 movie reviews.  As in \citet{he2016effective}, we pad and cut the sentences to 500 words, and use a learnable word embedding vector of size 512 ($T=500$, $d_{in}=512$, $d_{out}=1$). The learning loss is the binary cross-entropy, and the model's performance is evaluated with accuracy.

This experiment requires a file `imdb_data.pkl` that can be dowloaded with `keras.datasets.imdb.load_data` and placed in `downloaded_dataset\imdb`