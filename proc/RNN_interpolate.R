library(keras)
library(ggplot2)
library(tibble)
library(dplyr)
library(tidyr)

install_keras(tensorflow = "gpu")

source("~/R/myFunctions/func_stripna.R")
source("~/R/myFunctions/func_interpna.R")

# NORMALIZE returns matrix
normalit<-function(m, i=length(m)){
  (m - min(m[1:i], na.rm=T)+0.01)/(max(m[1:i], na.rm=T)-0.01-min(m[1:i], na.rm=T)+0.01)
}

# GENERATOR FUNCTION
sequence_generator <- function(start) {
  value <- start - 1
  function() {
    value <<- value + 1
    value
  }
}

# GENERATOR ---------------------------------------------------------------
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = batch_size, step = step) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+(batch_size-1), max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1, 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay, ind.stat]
    }     
    samples <- samples[,,-ind.stat]
    list(samples, targets)
  }
}

# load daily data
load("~/R/Data/SACTN_daily_v4.2.Rdata")
df.data <- SACTN_daily_v4.2

# load daily cluster data
load("~/Desktop/StationCP_final/ClusterGroups.Rdata")

# create a data.frame for each station in the first column
data.clust.stat <- site_clust %>% 
  dplyr::filter(results.5 == 1) %>% 
  dplyr::select(index)

# spread the data and create week and month of the year
data.clust <- df.data %>% 
  dplyr::filter(index %in% as.matrix(data.clust.stat)) %>% 
  spread(index, as.numeric(temp))# %>% 
  #mutate(month = as.integer(format(date, "%m")))
  #mutate(week = as.integer(format(date,"%V")),
  #       month = as.integer(format(date, "%m")))

glimpse(data.clust)
ggplot(data.clust, aes(x = 1:nrow(data.clust), y = data.clust$`Plettenberg Bay/SAWS`)) + geom_line()

# get index to first and last non-NA value per station and subset
ind.st <- which(!is.na(data.clust$`Plettenberg Bay/SAWS`)==T)[1]
ind.en <- tail(which(!is.na(data.clust$`Plettenberg Bay/SAWS`)==T),n=1)

data.sub <- data.clust[ind.st:ind.en,]

ggplot(data.sub, aes(x = date, y = data.sub$`Plettenberg Bay/SAWS`)) +
  geom_line() +
  ylab("Station Temp") + xlab("Date") +
  ggtitle("Station Data")

# Retain largest section without big NA breaks
ind.x <- strip.na(data.sub$`Plettenberg Bay/SAWS`)

x <- data.frame(date = data.sub$date[ind.x],
                temp = data.sub$`Plettenberg Bay/SAWS`[ind.x],
                stat01 = data.sub$`Knysna/SAWS`[ind.x],
                stat02 = data.sub$`Tsitsikamma/SAWS`[ind.x],
                stat03 = data.sub$`Tsitsikamma/DEA`[ind.x])

ggplot(data = x) +
  geom_line(aes(x = date, y = temp), size = 1) +
  ggtitle("Training Data")
  #geom_line(aes(x = date, y = stat01), color="blue", size = .5) +
  #geom_line(aes(x = date, y = stat02), color="darkorange", size = .5)

# Interpolate small breaks
data.train <- interp.na(x)
x$interp <- data.train

ggplot(data = x) +
  geom_line(aes(x = date, y = interp), size = 1, colour = "red") +
  geom_line(aes(x = date, y = temp), size = 1) +
  ggtitle("Interpolated Data")
  
data <- data.sub[ind.x,]
data$`Plettenberg Bay/SAWS` <- data.train

# remove 'date' column
data <- data[,-1]

# set up the model procedure
lookback = 10     # go back 30 days
step = 1          # 
delay = 1         # predict x days ahead
batch_size = 90   # size of batches

# get a 50%, 25%, 25% proportion for training, validating, testing
t1 <- round((nrow(data)/8) * 5)
t2a <- t1+1
t2b <- t1 + round((nrow(data)/8) * 2)
t3 <- t2b +1

# add row of '0' for normalit function (e.g. temp range now {0, 8:27})
# so that normalized 8 won't be close to missing value of 0
#data <- rbind(rep(0, length.out=ncol(data)),data)

# get the min and max of training set to normalize each series
# (see https://blogs.rstudio.com/tensorflow/posts/
# 2017-12-20-time-series-forecasting-with-recurrent-neural-networks/
# for reference)
# train_data <- data[1:t1,]
# mean <- apply(train_data, 2, mean, na.rm=T)
# std <- apply(train_data, 2, sd, na.rm=T)
# data <- scale(data, center = mean, scale = std)

# data <- data %>%
#   apply(., 2, normalit, i=t1)

#data <- data[-1,]
# data.sub <- data.sub %>%
#   apply(., 2, normalit)

# where target is 0 set training to 0
data[is.na(data)] <- 0

# get index to station of interest
ind.stat <- which(colnames(data) == "Plettenberg Bay/SAWS")
obs <- scale(x$temp, center =  mean[ind.stat], scale = std[ind.stat])
ind.0 <- which(is.na(data[ind.stat]))
data[ind.0, ] <- 0

# delay ind.stat by 1 day in new column
data <- as.data.frame(data) %>%
  mutate(self = .[[ind.stat]])
data[,ind.stat] <- c(0,data$self[1:(nrow(data)-1)])

data.sub <- data

rec.minmax <- c(min(data[1:t1,ind.stat], na.rm=T),
                max(data[1:t1,ind.stat], na.rm=T))

#data.2 <- data.matrix(data)   # use this for training

data <- data.matrix(data)
data[which(is.na(data))] <- 0
data.2 <- data.matrix(data.sub)
                         
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = t1,
  shuffle = F,             # shuffle = T,
  step = step, 
  batch_size = batch_size
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  #delay = 1,
  min_index = t2a,
  max_index = t2b,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = t3,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps <- (t2b - t2a - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data) - t3 - lookback) / batch_size

# a Naive method that assumes the next temp is the same as current
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]], ind.stat]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method()

# Basic ANN (run this with val_gen(delay = 1) i.e. using day before to predict
# to compare with the naive model above)
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)

# RNN
model <- keras_model_sequential() %>% 
  layer_masking(mask_value = 0, input_shape = c(lookback, dim(data)[[-1]]-1)) %>% 
  layer_simple_rnn(units = 100, 
            #input_shape = list(NULL, dim(data)[[-1]]-1),
            activation = "relu",
            dropout = 0.01, 
            recurrent_dropout = 0.01,
            return_sequences = FALSE,
            use_bias = TRUE,
            kernel_initializer = "normal",
            batch_input_shape = c(batch_size, lookback, dim(data)[[-1]]-1)) %>% 
  #layer_flatten() %>% 
  layer_dense(units = 1, activation = "relu")


model %>% compile(
  optimizer = "adam",
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = (nrow(data) / (batch_size*step)),
  epochs = 100,
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)

# Stacked RNN
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            #dropout = 0.2,
            activation = "relu",
            recurrent_dropout = 0.05,
            return_sequences = TRUE,
            batch_size = batch_size,
            #stateful = TRUE,
            #batch_input_shape = c(batch_size, lookback, dim(data)[[-1]])) %>% 
            input_shape = c(lookback, dim(data)[[-1]]-1)) %>% # less one for removing ind.stat
  layer_gru(units = 32,# activation = "relu",     # activation = "sigmoid",
            #dropout = 0.1,
            recurrent_dropout = 0.05,
            return_sequences = FALSE) %>%       # if FALSE then uses only last output
  #layer_flatten() %>%                          # use this if previous layer ret_seq = T
  layer_dense(units = 1, activation = "relu")

#model %>% time_distributed(layer_dense(units = 1))

model %>% compile(
  optimizer = optimizer_rmsprop(),    # optimizer = "adam"
  loss = "mae"                        # loss = "
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = (nrow(data) / (batch_size*step)),               # n samples / batch_size
  epochs = 100,
  validation_data = val_gen,
  validation_steps = val_steps
)

#reset_states(model)

plot(history)

# to predict time series
pred.st <- lookback+1
#pred.st <- t3-150
pred.en <- nrow(data.2) - delay
rows <- pred.st:pred.en
obs.plot <- obs[rows]

# if doing Stateful need predict length to be multiple of batch == 91
rows <- rows[1:((batch_size)*floor((length(rows)/(batch_size))))]

samples <- array(0, dim = c(length(rows), 
                            lookback / step,
                            dim(data)[[-1]]))
targets <- array(0, dim = c(length(rows)))

# fill data.2 with original station data i.e. not na.interp
data.2[, dim(data.2)[2]] <- x$temp

for (j in 1:length(rows)) {
  indices <- seq(rows[[j]] - lookback, rows[[j]]-1, 
                 length.out = dim(samples)[[2]])
  samples[j,,] <- data.2[indices,]
  targets[[j]] <- data.2[rows[[j]] + delay, ind.stat]
}         
samples <- samples[,,-ind.stat]
targets[which(is.na(targets))] <- 0
samples[which(is.na(samples))] <- 0

# set samples for station (except first lookback)
#samples[2:nrow(samples), , dim(data)[2]-1] <- 0
#samples[c(4:8,100:150,320:335,400:nrow(samples)), , dim(data)[2]-1] <- 0
#r <- round(nrow(samples)/batch_size)*batch_size
#samples <- samples[1:r,,]
#targets <- targets[1:r]
#prediction <- predict(model, samples, batch_size = batch_size)

# predict using predictions
self_loc <- dim(data)[2]-1       # location of 'self' feature
target_pred <- vector()
for (i in 1:length(targets)){
  # if there is a missing value in 'self', predict it
  if (samples[i,lookback,self_loc] == 0){
    p <- predict(model, samples[(i-1),,,drop = F], batch_size = 1)
    samples[i,lookback,self_loc]  <- p
    for (ii in 1:lookback-1){
      samples[(i+ii),(lookback-ii),self_loc] <- p  # fill in diagonally
    }
    print(p)
  }
  target_pred[i] <- predict(model, samples[i,,,drop = F], batch_size = 1)
} 

# test various features
sample1 <- samples
sample1[,,c(1:15,17)] <- 0
predict1 <- predict(model, sample1, batch_size = batch_size)
plot(predict1, type = "l", col = "blue", lwd = 3, ylim = c(0,1), main = "month Variable")   
lines(targets, col="red", lwd=3)

sample2 <- samples
sample2[,,c(1:14,17)] <- 0
predict2 <- predict(model, sample2, batch_size = batch_size)
plot(predict2, type = "l", col = "blue", lwd = 3, ylim = c(0,1), main = "month + week Variable")   
lines(targets, col="red", lwd=3)

sample3 <- samples
sample3[,,2:17] <- 0
predict3 <- predict(model, sample3, batch_size = batch_size)
plot(predict3, type = "l", col = "blue", lwd = 3, ylim = c(0,1), main = "knysna Variable")   
lines(targets, col="red", lwd=3)

sample4 <- samples
sample4[,,c(2:14,17)] <- 0
predict4 <- predict(model, sample4, batch_size = batch_size)
plot(predict4, type = "l", col = "blue", lwd = 3, ylim = c(0,1), main = "knysna+month+week Variable")   
lines(targets, col="red", lwd=3)

sample5 <- samples
sample5[,,15:16] <- 0
predict5 <- predict(model, sample5, batch_size = batch_size)
plot(predict5, type = "l", col = "blue", lwd = 3, ylim = c(0,1), main = "all, no month+week Variable")   
lines(targets, col="red", lwd=3)

sample6 <- samples
sample6[,,1:16] <- 0
predict6 <- predict(model, sample6, batch_size = batch_size)
plot(predict6, type = "l", col = "blue", lwd = 3, ylim = c(0,1), main = "previous day only Variable")   
lines(targets, col="red", lwd=3)

#reset_states(model)
samples[which(samples==0)] <- NA
targets[which(targets==0)] <- NA

plot(samples[1:(nrow(samples)-lookback),lookback,1], type="l", col="gray80", lwd=3, ylim = c(8,27),
     main = "Layers 1028 +Mask; Stateful=F; Dropout=0.01,0.01;
     Lookback=10; Epoch=100; Simple; Batch=90; Return Seq=F;
     Activation=relu,relu; No Month,Week; Kern_opt=Normal; Opt=Adam")
lines(samples[1200:1500,lookback,2], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,4], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,5], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,6], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,7], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,8], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,9], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,10], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,11], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,12], col="gray80", lwd=3)
lines(samples[1200:1500,lookback,13], col="gray80", lwd=3)
lines(rowMeans(samples[1:(nrow(samples)-lookback),lookback,1:13], na.rm = T), col="gray50", lwd=3)
#lines(samples[,lookback,14], col="chartreuse", lwd=3, lty=3)
lines(prediction, col="blue", lwd=3)
lines(target_pred[1:(nrow(samples)-lookback)], col="blue", lwd=3)
lines(targets, col="red", lwd=3)
lines(x$temp[lookback:nrow(samples)], col="red", lwd=3)
lines(samples[,lookback,14], col="orange", lwd=3)

pred.scale <- prediction * (rec.minmax[2] - rec.minmax[1]) + rec.minmax[1]
