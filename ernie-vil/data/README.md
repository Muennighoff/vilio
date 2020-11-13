The hm folder contains data for HM:

Two specifications due to some weird behavior of PaddlePaddle:
- "long" refers to that data has simply been copied to the end. Sometimes the model does not consider all the data in a file, hence we just copy data back to the end, so it considers at least all the unique datapoints and perhaps some duplicates.
- "dummy" refers to arbitrary labels of zero and some ones that have been added to the test sets, as the model needs labels even at test time
