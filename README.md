# R_vs_Python_ML_Benchmarking

There was an article by Joos Korstanje in Towards Data Science that benchmarked Python against
R: https://towardsdatascience.com/is-python-faster-than-r-db06c5be5ce8

I found this intrieging and wanted to se how it would if I run these same tests on my own hardware and setup.

Using PyCharm and R on Windows Python was 5 times faster, to my surprice. Especially, since python used only 30% of my
CPU and not on full speed, and R used 95% (using all CPU cores). None of these used GPU.

The R script is using makePSOCKcluster(), but as far as I understand Intel MKL should automatically use all CPU cores.

I did some small tweeking to get these files running in my environments.

TODO: See if I can get these even faster by using GPU, which will be difficult since it is Intel GPU, without CODA.
