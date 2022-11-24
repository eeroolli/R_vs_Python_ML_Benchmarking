# R_vs_Python_ML_Benchmarking

There was an article by Joos Korstanje in Towards Data Science that benchmarked Python against
R: https://towardsdatascience.com/is-python-faster-than-r-db06c5be5ce8

I found this intriguing and wanted to se how it would if I run these same tests on my own hardware and setup. I did some
small tweaking to get these files running in my environments.

Using PyCharm and R on Windows Python was almost 5 times faster for each iteration, than the R Cluster solution, to my
surprice. Especially, since python used only 30% of my CPU and not on full speed, and R used 95% (using all CPU cores).
None of these used GPU. The R script is using makePSOCKcluster(), but as far as I understand Intel MKL should
automatically use all CPU cores. Therefore, I reinstalled Intel OneApi toolkit and Intel HTC to be certain that R is
using it.

## Results

Python was 30% faster than R with Intel MKL and 243% faster than R cluster solution on Intel MKL. So it looks like that
Intel OneApi is optimizing the CPU usage much better than the R package doParallel.

#### Total run time

* Python sklearn 1:38 min
* Python sklearn with Intel MKL:   1:38 min
* R Cluster solution:                 6 Min
* R Cluster solution with Intel MKL:    5:30 Min
* R with Intel MKL:                   2:05 min

TODO: See if I can get these even faster by using GPU, which will be difficult since it is Intel GPU, without CODA.
