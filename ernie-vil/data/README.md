The hm folder contains data for HM:

Two specifications due to some mistake in the model I cannot fathom:
- It does not consider all the data in a file, hence "long" means data has been cp'd to the end, so it doesn't matter if it skips some of that duplicate data. In the case of our training data, we just add on the dev set again, as it is data of high quality.
- The model needs labels even at test-time, hence "dummy" refers to arbitrary labels of zero and some ones that have been added, but are not real
