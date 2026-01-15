##########################################################################
###########################################################################
###                                                                     ###
###                  PROGETTO: RETE NEURALE SU ABALONE                  ###
###                          VERSIONE ROBUSTA                          ###
###                  PREVISIONE DELL'ETÀ DEL MOLLUSCO                   ###
###                                                                     ###
###########################################################################
###########################################################################

#### PACCHETTI ####
library(doParallel)
library(fastDummies)
library(ggplot2)
library(Metrics)
library(neuralnet)
library(NeuralNetTools)
library(tidyverse)
library(reshape2)
library(car)
library(glmnet)
library(lmtest)
#IMPORTANTE
#selezionare directory in cui andare a trovare il dataset abalone.data

# il file UCI non ha header
cols <- c("Sex","Length","Diameter","Height",
          "Whole_weight","Shucked_weight",
          "Viscera_weight","Shell_weight","Rings")

abalone <- read.csv("abalone.data",
                    header = FALSE,
                    col.names = cols)

# variabile dipendente: età = anelli + 1.5
abalone$Age <- abalone$Rings + 1.5

# creiamo le dummies per Sex (M,F,I)
abalone <- dummy_columns(abalone, select_columns = "Sex")
abalone$Sex <- NULL   # togliamo la categorica originale
summary(abalone)
# IMPORTANTE:
# Salviamo 'age_min' e 'age_max' prima della trasformazione.
# Ci serviranno alla fine per "denormalizzare" le previsioni e calcolare l'errore
# in anni reali, non in valori scalati.
# per comodità teniamo una copia "non scalata" per i min/max della risposta
age_min <- min(abalone$Age)
age_max <- max(abalone$Age)

#### 2. ANALISI DESCRITTIVA MINIMA ####
# distribuzione dell'età
summary(abalone$Age)

ggplot(abalone, aes(x = Age)) +
  geom_histogram(bins = 30, fill = "grey40") +
  theme_minimal() +
  labs(title = "Distribuzione dell'età (anni)", x = "Età", y = "Frequenza")

# correlazioni tra variabili quantitative principali
num_vars <- c("Length","Diameter","Height",
              "Whole_weight","Shucked_weight",
              "Viscera_weight","Shell_weight","Age")

matcor <- cor(abalone[, num_vars])
matcor[upper.tri(matcor)] <- NA
correlation <- data.frame(melt(matcor, na.rm = TRUE))

ggplot(correlation, aes(Var1, Var2, fill = value)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  theme_minimal() +
  labs(title = "Correlazione tra le variabili quantitative", x = "", y = "") +
  scale_fill_gradient2(low = "#b9781d", mid = "white", high = "#1db954")

#### 3. NORMALIZZAZIONE  ####
# individuiamo le colonne da scalare con min–max:
# Age = risposta, poi tutte le variabili fisiche
# NOTA METODOLOGICA:
# Le reti neurali sono molto sensibili alla scala dei dati. Se una variabile ha valori
# tra 0-1 e un'altra tra 100-1000, i pesi si aggiorneranno in modo sbilanciato,
# rendendo la convergenza lenta o impossibile.
# Utilizziamo il Min-Max Scaling per portare tutto nel range [0, 1].


scale_cols <- c("Age",
                "Length","Diameter","Height",
                "Whole_weight","Shucked_weight",
                "Viscera_weight","Shell_weight")

abalone[, scale_cols] <- scale(
  abalone[, scale_cols],
  center = apply(abalone[, scale_cols], 2, min),
  scale  = apply(abalone[, scale_cols], 2, max) - apply(abalone[, scale_cols], 2, min)
)

# le dummies restano 0/1, quindi non le tocchiamo

#### 4. COSTRUZIONE TRAIN / TEST ####
set.seed(123)        #fissiamo il seme per la ripetibilità dei risultati
n <- nrow(abalone)
id <- sample(1:n, n * 0.7)
train <- abalone[id, ]
test  <- abalone[-id, ]

#### 5. COSTRUZIONE DELLA FORMULA PER LA RETE ####
# risposta: Age
# tutte le altre colonne: predittori
ind <- paste(names(train)[!names(train) %in% c("Age","Rings")], collapse = "+")
f   <- as.formula(paste("Age ~", ind))

#### 6. RICERCA DELLA RETE NEURALE MIGLIORE (1 STRATO NASCOSTO) – VERSIONE ROBUSTA ####
cat("\n========================================\n")
cat("INIZIO ADDESTRAMENTO RETE NEURALE\n")
cat("========================================\n\n")

# 1) cluster
numCores <- parallel::detectCores()
cl <- parallel::makeCluster(numCores - 1, outfile = "debug_abalone_s1.txt")

# 2) griglia STABILE
# STRATEGIA DI TUNING:
# Non sapendo a priori quale configurazione sia la migliore, testiamo diverse combinazioni
# (Grid Search) di 4 iperparametri fondamentali:
# 1. Nodi hidden: complessità della rete (pochi = underfitting, troppi = overfitting).
# 2. Threshold: soglia di arresto del gradiente (più bassa = più precisione ma più tempo).
# 3. Learning Rate: velocità di discesa del gradiente (troppo alto non converge, troppo basso è lento).
# 4. Funzione di attivazione: logistica vs tangente iperbolica.

# PARALLELIZZAZIONE:
# Poiché addestrare decine di reti richiede tempo, distribuiamo il carico su tutti
# i core della CPU meno uno (per non bloccare il PC).
lRate <- c(0.01, 0.005, 0.001)
tresh <- c(0.05, 0.01)
nodi  <- c(3, 5, 8)
func  <- c("logistic", "tanh")
par_s1 <- expand.grid(nodi, tresh, lRate, func)
par_s1 <- par_s1[sample(1:nrow(par_s1), nrow(par_s1)), ]

cat("Numero di configurazioni da testare:", nrow(par_s1), "\n\n")

# 3) esportazione oggetti
parallel::clusterExport(
  cl,
  varlist = c("train","test","f","par_s1","age_min","age_max"),
  envir = environment()
)

# 4) addestramento in parallelo
neuralnets_s1 <- parallel::parLapply(cl, 1:nrow(par_s1), function(i) {
  library(neuralnet)
  library(Metrics)
  set.seed(123)
  
  nn <- neuralnet(
    f,
    data          = train,
    hidden        = c(par_s1[i, 1]),
    threshold     = par_s1[i, 2],
    learningrate  = par_s1[i, 3],
    act.fct       = par_s1[i, 4],
    linear.output = TRUE,
    err.fct       = "sse",
    lifesign      = "none",
    stepmax       = 1e+06
  )
  
  # se non ha pesi, restituisco NULL
  if (!("weights" %in% names(nn))) {
    return(NULL)
  }
  
  # previsioni
  pred <- compute(nn, test[, names(train)[!names(train) %in% c("Age","Rings")]])$net.result
  
  # riportiamo alla scala originale
  pred_no_scale       <- pred * (age_max - age_min) + age_min
  test_age_no_scale   <- test$Age * (age_max - age_min) + age_min
  
  data.frame(
    nodi_hidden = par_s1[i, 1],
    threshold   = par_s1[i, 2],
    learningrate= par_s1[i, 3],
    act_fct     = par_s1[i, 4],
    mse         = mse(test_age_no_scale, pred_no_scale),
    mae         = mae(test_age_no_scale, pred_no_scale)
  )
})

# 6) FILTRA i NULL in modo sicuro
prova <- Filter(Negate(is.null), neuralnets_s1)

if (length(prova) == 0) {
  stop("ERRORE: Nessuna rete è riuscita a convergere con i parametri scelti. Aumenta stepmax o riduci ulteriormente i learning rate.")
}

# 7) unisci e ordina
result_s1 <- dplyr::bind_rows(prova) %>% dplyr::arrange(mse)

cat("\nRETI NEURALI CONVERGEITE:", length(prova), "/", nrow(par_s1), "\n\n")
cat("TOP 10 MIGLIORI CONFIGURAZIONI:\n")
print(head(result_s1, 10))

#### 7. RISTIMA DELLA MIGLIORE RETE NEURALE ####
best1 <- result_s1[1, ]

cat("\n========================================\n")
cat("MIGLIORE CONFIGURAZIONE TROVATA\n")
cat("========================================\n")
cat("Nodi hidden:", best1$nodi_hidden, "\n")
cat("Threshold:", best1$threshold, "\n")
cat("Learning rate:", best1$learningrate, "\n")
cat("Funzione attivazione:", best1$act_fct, "\n")
cat("MSE:", best1$mse, "\n")
cat("MAE:", best1$mae, "\n\n")

set.seed(123)
nn_s1 <- neuralnet(
  f,
  data          = train,
  hidden        = best1$nodi_hidden,
  threshold     = best1$threshold,
  learningrate  = best1$learningrate,
  act.fct       = as.character(best1$act_fct),
  linear.output = TRUE,
  err.fct       = "sse",
  lifesign      = "full",
  stepmax       = 1e+06
)
# NOTA SULL'OUTPUT:
# Impostiamo 'linear.output = TRUE' perché stiamo facendo REGRESSIONE (prevediamo un numero continuo).
# Se fosse stata una classificazione, avremmo messo FALSE per avere un output probabilistico.
# act.fct definisce come i neuroni elaborano il segnale nello strato nascosto.
# previsione finale con il modello definitivo
x_test <- test[, names(train)[!names(train) %in% c("Age","Rings")]]
pred_s1 <- neuralnet::compute(nn_s1, x_test)$net.result

pred_s1_no_scale  <- pred_s1 * (age_max - age_min) + age_min
test_no_scale     <- test$Age * (age_max - age_min) + age_min

nn_s1_result <- data.frame(
  Nodi_hidden   = best1$nodi_hidden,
  threshold     = best1$threshold,
  learning_rate = best1$learningrate,
  act_function  = best1$act_fct,
  mse           = Metrics::mse(test_no_scale, pred_s1_no_scale),
  mae           = Metrics::mae(test_no_scale, pred_s1_no_scale),
  rmse          = sqrt(Metrics::mse(test_no_scale, pred_s1_no_scale))
)

cat("\nRISULTATI FINALI RETE NEURALE:\n")
print(nn_s1_result)

#### 8. VISUALIZZAZIONE DELLA RETE ####
plot(nn_s1)

plotnet(
  nn_s1,
  circle_cex = 2,
  node_labs  = FALSE,
  rel_rs     = c(1, 5),
  circle_col = "#1db954",
  bord_col   = "black",
  max_sp     = TRUE,
  cex_val    = 1.2
)

#### 9. GRAFICI DI VALUTAZIONE ####
# vettori già riportati alla scala originale
osservato  <- test_no_scale           # età vera
predetto   <- pred_s1_no_scale        # età stimata dalla rete

ggplot(data.frame(osservato, predetto),
       aes(x = osservato, y = predetto)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Osservato vs Predetto - Rete neurale",
    x = "Età osservata",
    y = "Età predetta"
  )

residui <- osservato - predetto
ggplot(data.frame(predetto, residui),
       aes(x = predetto, y = residui)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Residui della rete neurale",
    x = "Età predetta",
    y = "Errore (osservato - predetto)"
  )

###########################################################
#### 10. CONFRONTO CON MODELLO DI REGRESSIONE LINEARE  ####
###########################################################
# VERIFICA DELLA ROBUSTEZZA (VIF):
# Le variabili fisiche dell'Abalone (lunghezza, diametro, peso) sono fortemente correlate tra loro.
# Questo crea "multicollinearità", che rende instabili i coefficienti della regressione lineare.
# Il VIF (Variance Inflation Factor) ci dice quanto è grave: se VIF > 5 o 10, c'è un problema.
# Le reti neurali e la regressione Ridge gestiscono questo problema meglio della OLS standard.
cat("\n========================================\n")
cat("REGRESSIONE LINEARE\n")
cat("========================================\n\n")

ind_lin <- paste(names(train)[!names(train) %in% c("Age","Rings")], collapse = "+")
f_lin   <- as.formula(paste("Age ~", ind_lin))

# Rimuoviamo la dummy ridondante (Sex_M come categoria base)
pred_cols <- names(train)[!names(train) %in% c("Age", "Rings", "Sex_M")]
f_lin <- as.formula(paste("Age ~", paste(pred_cols, collapse = " + ")))

lm_mod <- lm(f_lin, data = train)
summary(lm_mod)

# VIF per multicollinearità
cat("\nVariance Inflation Factors (VIF):\n")
print(car::vif(lm_mod))

# Previsioni
prev_lin     <- predict(lm_mod, newdata = test)
prev_lin_no_scale <- prev_lin * (age_max - age_min) + age_min

mse_lin <- Metrics::mse(test_no_scale, prev_lin_no_scale)
mae_lin <- Metrics::mae(test_no_scale, prev_lin_no_scale)
rmse_lin <- sqrt(mse_lin)

cat("\nRISULTATI REGRESSIONE LINEARE:\n")
cat("MSE:", mse_lin, "\n")
cat("MAE:", mae_lin, "\n")
cat("RMSE:", rmse_lin, "\n\n")

##########################################################################
##      REGRESSIONE RIDGE, LASSO, ELASTIC NET SU DATASET ABALONE       ##
##########################################################################
# PERCHÉ USIAMO LA REGOLARIZZAZIONE?
# Per contrastare la multicollinearità vista sopra ed evitare l'overfitting.
# - RIDGE (Alpha=0): Riduce i coefficienti verso lo zero ma non li annulla (buono per variabili correlate).
# - LASSO (Alpha=1): Azzera completamente i coefficienti inutili (fa selezione delle variabili).
# - ELASTIC NET (0 < Alpha < 1): Un mix tra i due.
#
# Utilizziamo la Cross-Validation (cv.glmnet) per trovare il 'lambda' ottimale,
# ovvero quanto penalizzare il modello per trovare il miglior compromesso bias-varianza.
cat("========================================\n")
cat("RIDGE, LASSO, ELASTIC NET\n")
cat("========================================\n\n")

## Matrici x, togliendo variabili che non devono entrare
x_train_df <- train[, !names(train) %in% c("Age", "Rings")]
x_test_df  <- test[,  !names(test)  %in% c("Age", "Rings")]

# glmnet vuole l'utilizzo matrici
x_train <- as.matrix(x_train_df)
x_test  <- as.matrix(x_test_df)

y_train <- train$Age
y_test  <- test$Age

##########################################################################
## RIDGE REGRESSION (alpha = 0) ########################################
##########################################################################
cat("Addestramento RIDGE...\n")

set.seed(123)
cv_ridge <- cv.glmnet(
  x_train, y_train,
  alpha = 0,           # ridge
  nfolds = 10
)

lambda_ridge <- cv_ridge$lambda.min
ridge_mod <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_ridge)

# Riporta alla scala originale
# DENORMALIZZAZIONE DEI RISULTATI:
# La rete ci restituisce un valore tra 0 e 1. Per capire quanto stiamo sbagliando,
# dobbiamo invertire la formula del Min-Max Scaling:
# ValoreReale = ValoreScalato * (Max - Min) + Min
#
# Solo ora possiamo calcolare MSE e MAE significativi (es. un MAE di 1.5 significa
# che sbagliamo in media di 1 anno e mezzo).
pred_ridge <- predict(ridge_mod, s = lambda_ridge, newx = x_test)
pred_ridge_no_scale <- pred_ridge * (age_max - age_min) + age_min

mse_ridge <- Metrics::mse(test_no_scale, pred_ridge_no_scale)
mae_ridge <- Metrics::mae(test_no_scale, pred_ridge_no_scale)
rmse_ridge <- sqrt(mse_ridge)

cat("Lambda ottimo:", lambda_ridge, "\n")
cat("MSE:", mse_ridge, "\n")
cat("MAE:", mae_ridge, "\n")
cat("RMSE:", rmse_ridge, "\n\n")

##########################################################################
## LASSO (alpha = 1) ###################################################
##########################################################################
cat("Addestramento LASSO...\n")

set.seed(123)
cv_lasso <- cv.glmnet(
  x_train, y_train,
  alpha = 1,           # lasso
  nfolds = 10
)

lambda_lasso <- cv_lasso$lambda.min
lasso_mod <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_lasso)

pred_lasso <- predict(lasso_mod, s = lambda_lasso, newx = x_test)
pred_lasso_no_scale <- pred_lasso * (age_max - age_min) + age_min

mse_lasso <- Metrics::mse(test_no_scale, pred_lasso_no_scale)
mae_lasso <- Metrics::mae(test_no_scale, pred_lasso_no_scale)
rmse_lasso <- sqrt(mse_lasso)

cat("Lambda ottimo:", lambda_lasso, "\n")
cat("MSE:", mse_lasso, "\n")
cat("MAE:", mae_lasso, "\n")
cat("RMSE:", rmse_lasso, "\n\n")

##########################################################################
## ELASTIC NET (alpha tra 0 e 1) #######################################
##########################################################################
cat("Addestramento ELASTIC NET...\n")

alphas <- c(0.25, 0.5, 0.75)
results_en <- data.frame(
  alpha = numeric(),
  lambda = numeric(),
  mse = numeric(),
  mae = numeric(),
  rmse = numeric()
)

set.seed(123)
for (a in alphas) {
  cv_en <- cv.glmnet(
    x_train, y_train,
    alpha = a,
    nfolds = 10
  )
  lam <- cv_en$lambda.min
  en_mod <- glmnet(x_train, y_train, alpha = a, lambda = lam)
  
  pred_en <- predict(en_mod, s = lam, newx = x_test)
  pred_en_no_scale <- pred_en * (age_max - age_min) + age_min
  
  mse_en <- Metrics::mse(test_no_scale, pred_en_no_scale)
  mae_en <- Metrics::mae(test_no_scale, pred_en_no_scale)
  rmse_en <- sqrt(mse_en)
  
  results_en <- rbind(
    results_en,
    data.frame(
      alpha = a,
      lambda = lam,
      mse = mse_en,
      mae = mae_en,
      rmse = rmse_en
    )
  )
}

cat("\nRISULTATI ELASTIC NET:\n")
print(results_en)

# scegliamo l'elastic net migliore
best_en_idx <- which.min(results_en$mse)
best_en     <- results_en[best_en_idx, ]

##########################################################################
## 11. CONFRONTO FINALE ################################################
##########################################################################
cat("\n========================================\n")
cat("CONFRONTO FINALE DI TUTTI I MODELLI\n")
cat("========================================\n\n")
# INTERPRETAZIONE DELLA CLASSIFICA:
# - MSE (Mean Squared Error): Penalizza molto gli errori grossi. Utile se sbagliare di tanto è grave.
# - MAE (Mean Absolute Error): È l'errore medio assoluto. Più interpretabile (è in anni).
# - RMSE: La radice dell'MSE, riporta l'errore sulla scala originale (anni).
#
# Se la Rete Neurale ha l'errore più basso, significa che la relazione tra le misure
# del mollusco e l'età non è perfettamente lineare e la rete ha catturato questa complessità.

resumo <- data.frame(
  Modello = c("Rete Neurale", "Regressione Lineare", "Ridge", "Lasso",
              paste0("Elastic Net (alpha=", best_en$alpha, ")")),
  MSE     = c(nn_s1_result$mse,
              mse_lin,
              mse_ridge,
              mse_lasso,
              best_en$mse),
  MAE     = c(nn_s1_result$mae,
              mae_lin,
              mae_ridge,
              mae_lasso,
              best_en$mae),
  RMSE    = c(nn_s1_result$rmse,
              rmse_lin,
              rmse_ridge,
              rmse_lasso,
              best_en$rmse)
)

print(resumo)

cat("\n\nRANKING PER MSE:\n")
print(resumo[order(resumo$MSE), ])

cat("\n\nRANKING PER MAE:\n")
print(resumo[order(resumo$MAE), ])

# Salva i risultati in CSV
write.csv(resumo, "confronto_modelli.csv", row.names = FALSE)
cat("\n✓ Risultati salvati in 'confronto_modelli.csv'\n")