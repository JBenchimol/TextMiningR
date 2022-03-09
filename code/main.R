# Replication Code for
# "Text Mining Methodologies with R: An Application to Central Bank Texts"
# Jonathan Benchimol, Sophia Kazinnik and Yossi Saadon
# Machine Learning with Applications

# Loading required R packages
library(xts)
# install.packages("igraph", type = "binary")
library(igraph)
library(plyr)
library(tm) # Key library, documentation: https://cran.r-project.org/web/packages/tm/tm.pdf
library(NLP)
library(devtools)
library(SnowballC)
library(ggplot2) 
library(cluster)  
library(wordcloud)
# library(qdap)
library(quanteda)
library(topicmodels)
# library(XLConnect)
library(lattice)
library(gplots)
library(data.table)
# library(xlsx)
library(stringi)
library(pheatmap)
library(readtext)
library(quanteda.textmodels)
library(stringr)    
library(dplyr)    
library(tidyr) 
library(rowr) 
library(ggthemes)
install.packages("austin", repos="http://R-Forge.R-project.org")
library(austin)

# install.packages("BiocManager")
# BiocManager::install("Rgraphviz")
library(Rgraphviz)

##############################
# Part 1: Reading Text Files #
##############################

# Establishing Working Directory
# Please set to folder with text files
setwd("../data") 
getwd()

# Creating corpus 
file.path <- file.path("../data")
corpus <- Corpus(DirSource(file.path)) # Creates a framework that holds a collection of documents (corpus)
 
inspect(corpus[1]) # Checking that the document - here, its document #1 - was correctly read

# Extracting document names, dates, and creating time series
list.my.files <- list.files(file.path, full.names = FALSE)
list.my.files
# names(list.my.files) <- basename(list.my.files)
# document.corpus.names <- ldply(names(list.my.files))
# document.corpus.names <- gsub("_", "", document.corpus.names)
document.corpus.names <- gsub(".txt", "", list.my.files)

document.corpus.names.df <- data.frame(document.corpus.names, row.names = document.corpus.names)

day <- substr(document.corpus.names, 1, 2)
month <- substr(document.corpus.names, 3, 4)
year <- substr(document.corpus.names, 5, 8)


#########################
# Part 2: Text Cleaning #
#########################

# Using this function to remove idiosyncratic characters, numbers/punctuation, stop-words
toSpace <- content_transformer(function(x, pattern){return (gsub(pattern, " ", x))})
corpus <- tm_map(corpus, toSpace, "-")
corpus <- tm_map(corpus, toSpace, ")")
corpus <- tm_map(corpus, toSpace, ":")
corpus <- tm_map(corpus, toSpace, "%")
corpus <- tm_map(corpus, toSpace, "@")
corpus <- tm_map(corpus, toSpace, " - ")
corpus <- tm_map(corpus, toSpace, "\n")
corpus <- tm_map(corpus, toSpace, ",")
#corpus <- tm_map(corpus, toSpace, ".")
corpus <- tm_map(corpus, function(x) iconv(x, to='latin1', sub='byte'))
corpus <- tm_map(corpus, removeNumbers) 
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)

inspect(corpus[1])

corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

corpus <- tm_map(corpus, removeWords, c("twelv", "end", "also", "age",
                                        "analysis", "number", "two", "three",
                                        "minut", "third", "fourth", "spokesperson",
                                        "staff", "like", "five", "four", 
                                        "topf", "nathan", "six",
                                        "wwwbankisraelgovil",
                                        "wwwbankisraelorgil", "moreov", "seven"))
corpus <- tm_map(corpus, stripWhitespace)

# Converting the corpsus into Document Term Matrix
# Choosing words with length between 3 and 12 characters
dtm <- DocumentTermMatrix(corpus, control=list(wordLengths=c(3, 12))) # https://en.wikipedia.org/wiki/Document-term_matrix

Terms(dtm)

# Creating a matrix with term frequencies
termFreq <- colSums(as.matrix(dtm))
head(termFreq)
tail(termFreq)

# Removing some of the most sparse terms
dtm.sparse <- removeSparseTerms(dtm,0.05)
dim(dtm.sparse)

term.frequencies <- colSums(as.matrix(dtm.sparse))
order.frequencies <- order(term.frequencies)

term.frequencies[head(order.frequencies)] # View DTM by word frequency using head/tail functions
term.frequencies[tail(order.frequencies)]

find.frequency.terms.100 <- findFreqTerms(dtm.sparse,lowfreq=100)
find.frequency.terms.100

find.frequency.terms.500 <- findFreqTerms(dtm.sparse,lowfreq=500)
find.frequency.terms.500

find.frequency.terms.1000 <- findFreqTerms(dtm.sparse,lowfreq=1000)
find.frequency.terms.1000

find.frequency.terms.1500 <- findFreqTerms(dtm.sparse,lowfreq=1500)
find.frequency.terms.1500

find.frequency.terms.2000 <- findFreqTerms(dtm.sparse,lowfreq=2000)
find.frequency.terms.2000


############################
# Part 3: Plotting Figures #
############################

corlimit <- 0.6
title <- ""
freq.term.tdm <- findFreqTerms(dtm,lowfreq=1650)  
pdf('../results/freq_term_corr.pdf')
plot(dtm,main=title,cex.main = 3, term=freq.term.tdm, corThreshold=corlimit,
     attrs=list(node=list(width=15,fontsize=40,fontcolor=129,color="red")))
dev.off()

# Creating Corpus Histogram
sorted.frequencies <- sort(colSums(as.matrix(dtm.sparse)), decreasing=TRUE)   
head(sorted.frequencies, 20)   

word.frequencies.frame <- data.frame(word=names(sorted.frequencies), freq=sorted.frequencies)   
head(word.frequencies.frame) 
word.frequencies.frame <- word.frequencies.frame[order(-sorted.frequencies),]

# Plotting Frequencies
# Term appears at least 1600 times in the corpus (can be customized)
pdf('../results/plotted_frequencies.pdf')
plotted.frequencies <- ggplot(subset(word.frequencies.frame, freq>1600), aes(reorder(word, -freq), freq))    
plotted.frequencies <- plotted.frequencies + geom_bar(stat="identity")   
plotted.frequencies <- plotted.frequencies + theme(axis.text.x=element_text(angle=45, hjust=1, size=18)) 
plotted.frequencies <- plotted.frequencies + theme(axis.text=element_text(size=17), axis.title=element_text(size=16,face="bold"))
# plotted.frequencies <- plotted.frequencies + theme(panel.background = element_rect(fill = 'white'))
plotted.frequencies <- plotted.frequencies + xlab("Corpus Terms") 
plotted.frequencies <- plotted.frequencies + ylab("Frequencies") 
plotted.frequencies  # Printing word frequencies
dev.off()

# Plotting Frequencies (term appears at least 1800 times in the corpus)
pdf('../results/plotted_frequencies_1800.pdf')
plotted.frequencies <- ggplot(subset(word.frequencies.frame, freq>1800), aes(reorder(word, -freq), freq))    
plotted.frequencies <- plotted.frequencies + geom_bar(stat="identity")   
plotted.frequencies <- plotted.frequencies + theme(axis.text.x=element_text(angle=45, hjust=1, size=18)) 
plotted.frequencies <- plotted.frequencies + theme(axis.text=element_text(size=17), axis.title=element_text(size=16,face="bold"))
#plotted.frequencies <- plotted.frequencies + theme(panel.background = element_rect(fill = 'white'))
plotted.frequencies <- plotted.frequencies + xlab("Corpus Terms") 
plotted.frequencies <- plotted.frequencies + ylab("Frequencies") 
plotted.frequencies #printing word frequencies
dev.off()

# Dendogram Figure
pdf('../results/dendogram.pdf')
dendogram <- dist(t(dtm.sparse), method="euclidian")   
dendogram.fit <- hclust(d=dendogram, method="ward.D")   
plot(dendogram.fit, cex=1.4, main="", cex.main=6)
dev.off()

# Adjacency Figure
dtm.sparse.matrix <- as.matrix(dtm.sparse)
tdm.sparse.matrix <- t(dtm.sparse.matrix)
tdm.sparse.matrix <- tdm.sparse.matrix %*% dtm.sparse.matrix

graph.tdm.sparse <- graph.adjacency(tdm.sparse.matrix, weighted=T, mode="undirected")
graph.tdm.sparse <- simplify(graph.tdm.sparse)

# Visualizing the Adjacency Figure
pdf('../results/igraph.pdf')
plot.igraph(graph.tdm.sparse, layout=layout.fruchterman.reingold(graph.tdm.sparse, niter=10, area=120*vcount(graph.tdm.sparse)^2),
            vertex.color = 169)
dev.off()

# Wordclouds
set.seed(142) # This is just the design of the wordcloud picture, can be changed (use same seed)
pal2 <- brewer.pal(8,"Dark2")

pdf('../results/wordclouds.pdf')
wordcloud(names(term.frequencies), term.frequencies, min.freq=100, random.order=FALSE, colors=pal2, scale=c(5.2, .7)) # Can be changed depending on the desired term frequency
wordcloud(names(term.frequencies), term.frequencies, min.freq=400, random.order=FALSE, colors=pal2, scale=c(5.2, .7)) # Can be changed depending on the desired term frequency
wordcloud(names(term.frequencies), term.frequencies, min.freq=700, random.order=FALSE, colors=pal2, scale=c(5.2, .7)) # Can be changed depending on the desired term frequency
wordcloud(names(term.frequencies), term.frequencies, min.freq=1000, random.order=FALSE, colors=pal2, scale=c(5.2, .7))
wordcloud(names(term.frequencies), term.frequencies, min.freq=1500, random.order=FALSE, colors=pal2, scale=c(5.2, .7))
wordcloud(names(term.frequencies), term.frequencies, min.freq=2000, random.order=FALSE, colors=pal2, scale=c(5.2, .7))
dev.off()

# Another weighting scheme - term frequency/inverse document frequency

# Wordclouds w. tf-idf
# Creating a new dtm with tf-idf weighting instead of term frequency weighting
dtm.tf.idf <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf, wordLengths=c(3, 12)))
dtm.tf.idf.sparse<-removeSparseTerms(dtm.tf.idf,0.95) 
dim(dtm.tf.idf.sparse) 

term.frequencies.tf.idf <- colSums(as.matrix(dtm.tf.idf.sparse))

sorted.frequencies.tf.idf <- sort(colSums(as.matrix(dtm.tf.idf.sparse)), decreasing=TRUE)   
head(sorted.frequencies.tf.idf, 20)   

word.frequencies.frame.tf.idf <- data.frame(word=names(sorted.frequencies.tf.idf), freq=sorted.frequencies.tf.idf)   
head(word.frequencies.frame.tf.idf) 

# Barplot of tf-idf terms
pdf('../results/barplots.pdf')
plotted.frequencies.tf.idf <- ggplot(subset(word.frequencies.frame.tf.idf, freq>0.5), aes(reorder(word, -freq), freq))    
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + geom_bar(stat="identity")   
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + theme(axis.text.x=element_text(angle=45, hjust=1, size=18)) 
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + theme(axis.text=element_text(size=17), axis.title=element_text(size=17))
# plotted.frequencies <- plotted.frequencies + theme(panel.background = element_rect(fill = 'white'))
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + xlab("Corpus Terms") 
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + ylab("Frequencies") 
plotted.frequencies.tf.idf  #printing word frequencies
dev.off()

# Wordclouds with tf-idf
set.seed(142) # This is just the design of the wordcloud picture, can be changed (use same seed)
pal2 <- brewer.pal(8,"Dark2")

pdf('../results/wordclouds_tf_idf.pdf')
wordcloud(names(term.frequencies.tf.idf), term.frequencies.tf.idf, min.freq=0.09, random.order=FALSE, colors=pal2, scale=c(6, .4))
wordcloud(names(term.frequencies.tf.idf), term.frequencies.tf.idf, min.freq=0.19, random.order=FALSE, colors=pal2, scale=c(6, .4))
wordcloud(names(term.frequencies.tf.idf), term.frequencies.tf.idf, min.freq=0.39, random.order=FALSE, colors=pal2, scale=c(5, .6))
wordcloud(names(term.frequencies.tf.idf), term.frequencies.tf.idf, min.freq=0.49, random.order=FALSE, colors=pal2, scale=c(5, .6))
wordcloud(names(term.frequencies.tf.idf), term.frequencies.tf.idf, min.freq=0.59, random.order=FALSE, colors=pal2, scale=c(5, .6))
dev.off()

# Creating Corpus Histogram w. Tf-Idf Weighting
word.frequencies.frame.tf.idf <- data.frame(word=names(sorted.frequencies.tf.idf), freq=sorted.frequencies.tf.idf)   
head(word.frequencies.frame.tf.idf) 
word.frequencies.frame.tf.idf <- word.frequencies.frame.tf.idf[order(-sorted.frequencies.tf.idf),]

# Plotting Frequencies
pdf('../results/barplot_copurs_tf_idf.pdf')
plotted.frequencies.tf.idf <- ggplot(subset(word.frequencies.frame.tf.idf, freq>0.49), aes(reorder(word, -freq), freq))    
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + geom_bar(stat="identity")   
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + theme(axis.text.x=element_text(angle=45, hjust=1, size=18)) 
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + theme(axis.text=element_text(size=17), axis.title=element_text(size=17))
# plotted.frequencies <- plotted.frequencies + theme(panel.background = element_rect(fill = 'white'))
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + xlab("Corpus Terms") 
plotted.frequencies.tf.idf <- plotted.frequencies.tf.idf + ylab("Frequencies w. tf-idf Weighting") 
plotted.frequencies.tf.idf  # printing word frequencies
dev.off()

# Dendogram Figure
tdm.tf.idf.sparse <- as.matrix(t(dtm.tf.idf.sparse))
tdm.tf.idf.sparse <- tdm.tf.idf.sparse %*% t(tdm.tf.idf.sparse)

dtm.tf.idf.sparse.095 <- removeSparseTerms(dtm.tf.idf, 0.05)
dendogram <- dist(t(dtm.tf.idf.sparse.095), method="euclidian")   
dendogram.fit <- hclust(d=dendogram, method="ward.D")    

pdf('../results/dendogram_tf_idf.pdf')
plot(dendogram.fit, cex=1.4, main="", cex.main=4)
dev.off()

# HeatMap w. Tf-Idf (change format)
dtm.tf.idf.sparse.matrix <- as.matrix(dtm.tf.idf.sparse)
# rownames(dtm.tf.idf.sparse.matrix) <- document.corpus.names.df$date

# arrange(document.corpus.names.df, date)
date.dtm.tf.idf.sparse.matrix <- cbind(dtm.tf.idf.sparse.matrix,document.corpus.names.df$date)

my_palette <- colorRampPalette(c("white", "pink", "red"))(n = 299)

# Heatmaps
pdf('../results/heatmaps.pdf')
heatmap.2(date.dtm.tf.idf.sparse.matrix[1:12,1:18],
          main = "", # heat map title
          dendrogram = "none",
          keysize = 1,
          margins =c(9,7),
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          col=my_palette,       # use on color palette defined earlier
          srtCol=45,
          cexCol=1.4,
          Colv="NA")            # turn off column clustering

heatmap.2(date.dtm.tf.idf.sparse.matrix[13:24,1:18],
          main = "Frequencies", # heat map title
          dendrogram = "none",
          keysize = 1,
          margins =c(9,7),
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          col=my_palette,       # use on color palette defined earlier
          srtCol=45,
          cexCol=1.4,
          Colv="NA")            # turn off column clustering

heatmap.2(date.dtm.tf.idf.sparse.matrix[25:36,1:18],
          main = "Frequencies", # heat map title
          dendrogram = "none",
          keysize = 1,
          margins =c(9,7),
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          col=my_palette,       # use on color palette defined earlier
          srtCol=45,
          cexCol=1.4,
          Colv="NA")            # turn off column clustering
dev.off()


##########################
# Part 4: Topic Modeling #
##########################

#Gibbs Sampling Calibration
burnin <- 4000
iter <- 1500
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

# Number of topics
# This is arbitrary, need to make educated guess/play around with data
k <- 4

# Run LDA using Gibbs sampling
lda.results <-LDA(dtm.sparse, k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

# Write out results
# Docs to topics
lda.topics <- as.matrix(topics(lda.results))
# write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))
lda.topics
# Top 6 terms in each topic
lda.results.terms <- as.matrix(terms(lda.results,11))
# write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))
lda.results.terms
# Probabilities associated with each topic assignment
topic.probabilities <- as.data.frame(lda.results@gamma)
write.csv(topic.probabilities,file=paste("LDAGibbs", k ,"TopicProbabilities.csv"))
topic.probabilities <- as.matrix(topic.probabilities)

# Probabilities for each term in each topic.
posterior.terms <- t(posterior(lda.results)$terms)

# Heatmaps with topic modeling

# This is set manually by considering the words appearing in each list
colnames(topic.probabilities) <- c("Key Rate", "Inflation", "Monetary Policy", "Housing Market")
# rownames(topic.probabilities) <- document.corpus.names.df$date

date.topic.probabilities <- cbind(document.corpus.names.df$date, topic.probabilities)

pdf('../results/heatmaps_lda.pdf')
heatmap.2(date.topic.probabilities[100:112,],
          main = "", # heat map title
          dendrogram = "none",
          keysize = 1,
          margins =c(10,8),
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          col=my_palette,       # use on color palette defined earlier
          srtCol=45,
          cexCol=1.4,
          Colv="NA")            # turn off column clustering
dev.off()

####################
# Part 5: Wordfish #
####################

# Converting our corpus into another format
quanteda.corpus <- corpus(corpus)
dfm.corpus <- dfm(quanteda.corpus)
rownames(dfm.corpus) <- rownames(document.corpus.names.df)

# Defining which documents in the corpus represent the most dovish and the most hawkish positions
dovish <- which(rownames(dfm.corpus) %in% "28012002" | rownames(dfm.corpus) %in% "11112008")
hawkish <- which(rownames(dfm.corpus) %in% "09062002" | rownames(dfm.corpus) %in% "24062002")

# Running the wordfish algorithm
wordfish <- textmodel_wordfish(dfm.corpus, dir = c(dovish, hawkish))
summary(wordfish, n = 10)
coef(wordfish)
str(wordfish) 

# Extracting estimated parameters
documents <- wordfish$docs
theta <- wordfish$theta
se.theta <- wordfish$se.theta

predicted.wordfish <- predict(wordfish, interval = "confidence")

# Extracting sentiment score based on the algorithm
wordfish.score <- as.data.frame(predicted.wordfish$fit)

# Plotting Wordfish Score
wordfish.score$day <- substr(document.corpus.names, 1, 2)
wordfish.score$month <- substr(document.corpus.names, 3, 4)
wordfish.score$year <- substr(document.corpus.names, 5, 8)

wordfish.score$date <- paste(wordfish.score$month, wordfish.score$day, wordfish.score$year, sep="/")
wordfish.score$date <- as.Date(wordfish.score$date, "%m/%d/%Y")

wordfish.score <- wordfish.score[ order(wordfish.score$date), ]

wordfish.score$lag <- lag(wordfish.score$fit)
wordfish.score$change <- 100*(wordfish.score$fit - wordfish.score$lag)/wordfish.score$fit

# Plot Wordfish Score over time
pdf('../results/wordfish.pdf')
ggplot(wordfish.score, aes(date, fit, group = 1)) +
  geom_line(aes(x = wordfish.score$date, y = fit)) + 
  ggtitle("") +
  theme_hc() +
  scale_colour_hc() +
  xlab("Date") + 
  ylab("Sentiment") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=15)) 

# Plot Change in Wordfish Score over time
ggplot(wordfish.score, aes(date, change, group = 1)) +
  geom_line(aes(x = date, y = change)) + 
  ggtitle("") +
  theme_hc() +
  scale_colour_hc() +
  xlab("Date") + 
  ylab("Sentiment") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=15)) 
dev.off()

######################
# Part 5: Wordscores #
######################

# Assiging scores to documents that we think represent most dovish/hawkish positions
reference.scores <- rep(NA, nrow(dfm.corpus))
reference.scores[str_detect(rownames(dfm.corpus), "28012002")] <- -1  
reference.scores[str_detect(rownames(dfm.corpus), "11112008")] <- -1  
 
reference.scores[str_detect(rownames(dfm.corpus), "24062002")] <- 1
reference.scores[str_detect(rownames(dfm.corpus), "09062002")] <- 1

# Running the Wordscores algorithm
wordscores <- textmodel_wordscores(dfm.corpus, reference.scores, scale="linear", smooth=1) 

summary(wordscores, n = 10)
coef(wordscores)
str(wordscores)

# Extracting predicted wordscores
predicted.wordscores <- predict(wordscores)

wordscores.score <- as.data.frame(predicted.wordscores)
wordscores.score$document.corpus.names <- NULL

# Plots
# wordscores.score <- wordscores.score[ order(wordscores.score$date), ]

wordscores.score$lag <- lag(wordscores.score$predicted.wordscores)
wordscores.score$change <- 100*(wordscores.score$predicted.wordscores - wordscores.score$lag)/wordscores.score$predicted.wordscores
wordscores.score$date <- rownames(wordscores.score)
wordscores.score$date <- as.Date(wordscores.score$date, format = "%d%m%Y")

# Plot Wordscores Score over time
pdf('../results/wordscore.pdf')
ggplot(wordscores.score, aes(date, predicted.wordscores)) +
  geom_line(aes(x = date, y = predicted.wordscores)) + 
  ggtitle("") +
  theme_hc() +
  scale_colour_hc() +
  xlab("Date") + 
  ylab("Sentiment") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=15)) 

# Plot Change in Wordscores Score over time
ggplot(wordscores.score, aes(date, change)) +
  geom_line(aes(x = date, y = change)) + 
  ggtitle("") +
  theme_hc() +
  scale_colour_hc() +
  xlab("Date") + 
  ylab("Sentiment") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=15)) 

dev.off()

