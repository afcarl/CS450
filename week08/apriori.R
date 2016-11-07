# Apriori explorations
# install.packages('arules');
library(arules);

data(Groceries);

groceryrules <- apriori(Groceries, parameter = list(support = 0.0001, confidence = 0.1, minlen = 5))

inspect(sort(groceryrules, by = "support"))

