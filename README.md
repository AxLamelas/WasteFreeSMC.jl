# WasteFreeSMC.jl

Implementation of waste-free sequential monte carlo as described by [Dau and Chopin](https://doi.org/10.1111/rssb.12475).
The MCMC steps uses [kernel metropolis hastings](https://proceedings.mlr.press/v32/sejdinovic14.html) calibrated on the current particle population in conjunction with [delayed rejection](https://doi.org/10.1093/biomet/88.4.1035).
