# Franck-Hertz Experiment

## Visualize Data
Plot using 

```bash
python3 plot.py
```

<!-- Should see that some of the files have a Franck-Hertz response. -->

Some of the data have two response curves. These files need to be split into two different files for analysis. (e.g. extract all events > V_ave and export to separate file)

## Some initial measurements using a spline

```bash
python3 interpolate_many.py
```


Run Number | Mean | STD
-----------|------|-----
 2 | 5.261882478016136 | 0.09231950981423502
 6 | 4.833048183719448 | 0.21530869148559717
 7 | 5.012456985251546 | 0.15140340270315178
 8 | 4.886712433315185 | 0.24433350266963855
 9 | 4.738020099832852 | 0.3459939469783181
10 | 5.221947297309619 | 0.0
23 | 5.037023842208447 | 0.3469290648357681
24 | 4.701701943223213 | 0.0
25 | 4.850574143585021 | 0.3687923148186916

Need to redo this and find the best spline for each set of data instead of trying one accuracy value.

<!-- !The hot electrons from the cathode should be a fermi-dirac distribution (need citation just guessing)

[fermi](https://latex.codecogs.com/svg.latex?\large&space;N=\frac{1}{e^{(\varepsilon-\mu)/\tau}+1}) 

The franck-hertz behavior should be a sawtooth wave with a drop at n*4.9V, n=1,2,3,... (need citation)

![sawtooth](https://latex.codecogs.com/svg.latex?\large&space;ST=\sum_{n=0}^N(ax+b-n\cdot~E)\cdot(u(x-n\cdot~E)-u(x-(n+1)\cdot~E))) 


We can combine these response characteristics using a convolution

![v1a](https://latex.codecogs.com/svg.latex?\large&space;V_1(v)=N*ST=\int_0^\infty\frac{1}{e^{(v-\nu-\mu)/\tau}+1}\sum_{n=0}^N\frac{a\nu-n\cdot~E}{E}\cdot(u(\nu-n\cdot~E)-u(\nu-(n+1)\cdot~E))d\nu) 

![v1b](https://latex.codecogs.com/svg.latex?\large&space;V_1(v)=N*ST=\sum_{n=0}^N\int_{n\cdot~E}^{(n+1)\cdot~E}\frac{\frac{a}{E}\nu-n}{e^{(v-\nu-\mu)/\tau}+1}d\nu)

We also know the plasma current is exponential (need citation)

![current](https://latex.codecogs.com/svg.latex?\large&space;I=ce^{k(x-d)}) 


We can combine this response with our fermi distribution and franck-hertz using a second convolution

![v2](https://latex.codecogs.com/svg.latex?\large&space;V_2(v)=V_1*I=\sum_{n=0}^N\int_0^\infty\int_{n\cdot~E}^{(n+1)\cdot~E}\frac{\frac{a}{E}\nu-n}{e^{(v-\eta-\nu-\mu)/\tau}+1}ce^{k(\eta-d)}d\nu~d\eta)

*I probably made a typo need to double check these equtions

Finally we can add the background current in

![v](https://latex.codecogs.com/svg.latex?\large&space;V(v)=A\cdot~N(v;\mu,\tau)*ST(v;a,E)*I_1(v;c_1,k_1,d_1)+I_2(v;c_2,k_2,d_2))

with the parameters

![params](https://latex.codecogs.com/svg.latex?\large&space;A,\mu,\tau,a,E,c_1,k_1,d_1,c_2,k_2,d_2)

The sawtooth parameters can be expanded

![STparams](https://latex.codecogs.com/svg.latex?\large&space;a_0,a_1,a_2,\hdots,a_N,E_0,E_1,E_2,\hdots,E_N)

The point of parameter _A_ is because the convolution will be renormalized at each step to avoid blow-up

The exponential parameters might(probably?) be equal

![Iparams](https://latex.codecogs.com/svg.latex?\large&space;c_1=c_2,k_1=k_2,d_1=d_2?)


Use model.py to visualize the model

```bash
python3 model.py
``` -->
<!-- 
## Data Cleaning

Cut data at V<0. some files need to be split into two responses
## Data Fitting

Use scipy

```bash
fit.py
```
 -->
