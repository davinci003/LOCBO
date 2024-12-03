# Robust Bayesian Optimization via Localized Online Conformal Prediction

Dongwon Kim, Matteo Zecchin, Sangwoo Park, Joonhyuk Kang, Osvaldo Simeone

Usage
=====

## Run
For synthetic objective function (Ackley function)

python LOCBO_Ackley.py 1 OCEI 5 64 5 2

For radio resource management problem

python LOCBO_Cuav.py 4 10 40 1 0.7 0 1


## Notes
To run the code, the following packages are necessary:

```torch``` for running Bayesian Optimization.

```Botorch 0.9.5``` for running Bayesian optimization.
