
# Panoramica

Questo progetto affronta il problema della **stima dell’età dell’Abalone** come **regressione supervisionata**, utilizzando una rete neurale artificiale (ANN) di tipo **Multi-Layer Perceptron (MLP) feed-forward** con **un singolo hidden layer** e **output lineare**.

L’obiettivo è stimare l’età a partire da misure morfometriche osservabili (lunghezze e pesi) evitando metodi invasivi (es. conteggio anelli su conchiglia).

# Dataset

Il dataset “Abalone” contiene:

- **Feature numeriche**: `Length`, `Diameter`, `Height`, `Whole weight`, `Shucked weight`, `Viscera weight`, `Shell weight`
- **Feature categoriale**: `Sex ∈ {M, F, I}`
- **Variabile**: `Rings` (conteggio anelli)

## Target (Age)

L’età è derivata dalla variabile `Rings` tramite:

$$
Age = Rings + 1.5
$$

**Nota importante:** `Rings` **non viene usata tra i predittori**, perché sarebbe un’informazione quasi direttamente equivalente al target (rischio di *leakage*). Il modello usa solo misure morfometriche + sesso.

# Struttura del problema

- **Input**: vettore di feature $X$ (misure + variabili dummy di `Sex`)
- **Output**: stima dell’età $\widehat{Age}$
- **Tipo**: regressione (output reale continuo)

# Pipeline del progetto

## 1) Preparazione dei dati

### 1.1 Derivazione del target

- Calcolo `Age = Rings + 1.5`
- Rimozione di `Rings` dall’insieme dei predittori

### 1.2 Encoding della variabile categoriale Sex

`Sex` è nominale, quindi viene trasformata con **one-hot encoding**:

- `SexF`, `SexM`, `SexI ∈ {0,1}`
- vincolo implicito: `SexF + SexM + SexI = 1`

### 1.3 Normalizzazione (scaling)

Per stabilità numerica e dinamica del gradiente:

- **Input scaling**: min–max su tutte le feature numeriche (e/o su tutte le feature finali incluse le dummy)
- **Target scaling**: min–max su `Age` in $[0,1]$

Formula min–max:

$$
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

**Valutazione finale:** le predizioni vengono **de-normalizzate** per riportare $\widehat{Age}$ in anni, così le metriche (MSE/MAE) risultano interpretabili.

## 2) Suddivisione Training/Test

Per stimare la capacità di generalizzazione:

- split **70% training / 30% test**
- **seed** fissato per riproducibilità

Il training set è usato per apprendere pesi e bias, il test set è usato solo per valutare prestazioni finali e confronto con benchmark.

# Modello: ANN (MLP feed-forward)

## Architettura

- **Input layer**: feature normalizzate + dummy `Sex`
- **Hidden layer**: un solo strato con $H$ neuroni
- **Output layer**: 1 neurone con **attivazione lineare** (regressione)

Forma generale:

$$
\widehat{Age}=f(X;W)
$$

dove $W$ include tutti i pesi e bias.

## Neurone hidden

Per il neurone $j$:

$$
z_j=\sum_i w_{ij}x_i+b_j
$$

$$
h_j=\varphi(z_j)
$$

## Output lineare (regressione)

$$
\widehat{Age}=\sum_j v_j h_j + b_0
$$

Scelta motivata dal fatto che l’età è una variabile continua e non deve essere vincolata a intervalli tipo $(0,1)$ o $(-1,1)$ (come accadrebbe con logistic/tanh anche in output).

# Funzioni di attivazione (Hidden layer)

Nel progetto le funzioni di attivazione sono trattate come **iperparametri** e confrontate:

## Logistic (sigmoide)

$$
\varphi(z)=\frac{1}{1+e^{-z}}
$$

Output in $(0,1)$.

## Tanh

$$
\varphi(z)=\tanh(z)
$$

Output in $(-1,1)$, centrata in 0.

## Collegamento con saturazione (concetto chiave)

Per valori grandi di $|z|$, logistic e tanh entrano in **saturazione** (tratti quasi piatti), con:

$$
\varphi'(z)\approx 0
$$

Questo riduce il gradiente e rallenta l’apprendimento. Da qui l’importanza di scaling e scelta di learning rate.

# Funzione obiettivo (Loss)

Si ottimizza una loss quadratica:

## SSE (Sum of Squared Errors)

$$
SSE=\sum_{k=1}^{n}(y_k-\hat{y}_k)^2
$$

## MSE (Mean Squared Error)

$$
MSE=\frac{1}{n}\sum_{k=1}^{n}(y_k-\hat{y}_k)^2
$$

Minimizzare SSE o MSE porta allo stesso optimum (differiscono per la costante $1/n$).  
MSE è più comodo per confronto tra modelli perché è “per osservazione”.

# Ottimizzazione e criteri di stop

L’addestramento avviene con ottimizzazione iterativa basata sul gradiente (backprop):

$$
W^{(t+1)}=W^{(t)}-\eta \nabla L\big(W^{(t)}\big)
$$

Dove:

- $\eta$ = **learning rate** (ampiezza del passo)
- **threshold** = criterio di arresto su miglioramento loss
- **stepmax** = massimo numero di iterazioni (vincolo di sicurezza)

# Iperparametri e Tuning (Grid Search)

Gli iperparametri sono scelte esterne al training dei pesi: non vengono “imparate” dal gradiente, ma selezionate confrontando prestazioni.

## Iperparametri esplorati

- Numero neuroni hidden: $H\in\{3,5,8\}$
- Attivazione hidden: `{logistic, tanh}`
- Learning rate: $\eta\in\{0.001,0.005,0.01\}$
- Threshold: `{0.01, 0.05}`

Totale configurazioni:

$$
3\times 2\times 3\times 2 = 36
$$

## Selezione del modello

Per ogni combinazione:

1. training sul training set  
2. predizione sul test  
3. de-normalizzazione della predizione  
4. calcolo metriche (MSE, MAE)  
5. selezione della configurazione con **MSE minimo** (e MAE come supporto interpretativo)

# Metriche di valutazione

Le prestazioni vengono misurate su **test set** in scala anni.

## MAE (Mean Absolute Error)

$$
MAE=\frac{1}{n}\sum_{k=1}^{n}\left|y_k-\hat{y}_k\right|
$$

Interpretabile direttamente in anni (errore medio assoluto).

## MSE (Mean Squared Error)

Sensibile agli errori grandi (quadrato), utile per evidenziare outlier/casi difficili.

# Risultati principali (modello migliore)

La configurazione selezionata tramite grid search ottiene sul test:

- **MSE = 4.1375**
- **MAE = 1.4397 anni**

Interpretazione operativa:

- MAE ≈ 1.44 → in media l’errore assoluto è ~1 anno e mezzo  
- MSE enfatizza gli errori elevati → segnala presenza di alcuni casi con scarto maggiore

# Diagnostica grafica (validazione qualitativa)

Nel progetto si analizzano grafici tipici di regressione:

## Osservato vs Predetto

- confronto rispetto alla bisettrice $y=\hat{y}$
- dispersione maggiore in alcune regioni indica difficoltà del modello su determinate fasce di età

## Residui vs Predetti

- residui centrati intorno a zero → assenza di bias forte
- eventuale pattern “a ventaglio” → possibile **eteroschedasticità** (varianza condizionale dei residui non costante)

# Benchmark e confronto modelli

Per contestualizzare il risultato, la MLP viene confrontata con:

- OLS (regressione lineare)
- Ridge
- Lasso
- Elastic Net

Il confronto evidenzia un errore inferiore per la rete neurale rispetto ai modelli lineari/regolarizzati, coerente con la presenza di componenti non lineari nella relazione $X \rightarrow Age$.

# Riproducibilità

Per riprodurre gli stessi risultati è fondamentale:

- fissare il **seed** prima dello split train/test e dell’addestramento
- applicare lo stesso scaling calcolato sul training anche al test (stessi min/max)
- usare gli stessi set di iperparametri del grid search

# Output atteso

Alla fine della pipeline si ottengono:

- modello MLP selezionato (miglior combinazione di iperparametri)
- metriche test (MSE, MAE) in anni
- grafici diagnostici (osservato vs predetto, residui)
- tabella completa risultati grid search

# Glossario rapido

- **Parametro**: peso/bias appreso dal training
- **Iperparametro**: scelta esterna (H, attivazione, learning rate, threshold)
- **z (pre-attivazione)**: somma pesata degli input + bias
- **Saturazione**: regione piatta di logistic/tanh con derivata quasi zero
- **Generalizzazione**: prestazione su dati non visti (test)

