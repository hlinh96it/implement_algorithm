data(meats)

meats_train <- meats[1:130,]
meats_val <- meats[131:175,]
meats_test <- meats[176:215,]

# Split the column names in X and Y
X_colnames <- colnames(meats)[1:100]
Y_colnames <- colnames(meats)[101:103]

# Split each train, val, test into two matrices
X_train_matrix <- as.matrix(meats_train[X_colnames])
Y_train_matrix <- as.matrix(meats_train[Y_colnames])

X_val_matrix <- as.matrix(meats_val[X_colnames])
Y_val_matrix <- as.matrix(meats_val[Y_colnames])

X_test_matrix <- as.matrix(meats_test[X_colnames])
Y_test_matrix <- as.matrix(meats_test[Y_colnames])

my_plsr <- plsr(Y_train_matrix ~ X_train_matrix, ncomp=100, scale = TRUE, validation='LOO')
