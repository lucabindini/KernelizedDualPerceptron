# KernelizedDualPerceptron
Implementazione dell'algoritmo *Perceptron*, nella sua forma duale, per problemi di classificazione binaria con utilizzo di funzioni kernel
(*lineare*, *polinomiale*, *RBF*) al posto del prodotto scalare.

## Datasets e riutilizzo del codice:
Per riutilizzare il codice è sufficiente scaricare i datasets dai link sottostanti ed inserirli, in formato *.csv*, nella cartella del progetto
*UCI_datasets* (facendo attenzione a rinominarli correttamente).
* **Banknote Authentication**: https://archive.ics.uci.edu/ml/datasets/banknote+authentication (rinomina il file "banknote.csv").
* **QSAR Biodegradation**: https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation (rinomina il file "biodeg.csv").
* **QSAR Androgen Receptor**: https://archive.ics.uci.edu/ml/datasets/QSAR+androgen+receptor (rinomina il file "androgen.csv").

## Breve descrizione dei files:
Qui di seguito una breve descrizione di ogni file del progetto:
* **datasets.py**: attraverso il metodo *load_single_dataset* carica il dataset e lo splitta in train, validation e test set.
* **KernelizedDualPerceptron.py**: classe che implementa l'algoritmo *Perceptron* in forma duale con la possibilità di utilizzare funzioni kernel 
al posto del prodotto scalare.    
* **kernels.py**: contiene i metodi per il calcolo delle 3 funzioni kernel (*lineare*, *polinomiale*, *RBF*).   
* **performances.py**: contiene i metodi per il calcolo dell'accuratezza ed il rate d'errore.
* **plots.py**: contiene i metodi che generano grafici per confrontare l'accuratezza al variare del datasets e del kernels.
* **main.py**: ha il compito di eseguire l'intero codice.


## Prerequisiti e librerie esterne:
L'intero codice è scritto in linguaggio *Python 3.8*.<br/> Sono state utilizzate alcune librerie esterne: 
* **pandas**: utilizzata per l'acquisizione dei dataset da file *.csv*.
* **numpy**: utilizzata per funzioni matematiche e gestione dei datasets in modo efficiente.
* **scikit-learn**: utilizzata per splittare il dataset.
* **matplotlib**: utilizzata per la creazione dei grafici.
* **termcolor**: utilizzata per la stampa a colori sulla Python Console.

