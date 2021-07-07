561\_notes
================
Shivam Verma
15/12/2019

-----

F statistic is also called (Coefficient of partial determination)  
\- [see:](https://en.wikipedia.org/wiki/Coefficient_of_determination)  
\- [See
also](http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm)  
\- The formula:
`((SS_res_reduced-SS_res_full)/k)/(SS_res_full/(n-p-1))`.  
\* `k` is number of parameters which is different  
\* `p` is number of parameters in the full model  
\* `n` is number of rows in the
dataset

``` r
gpa_data <- read_csv("C:/MyDisk/MDS/DSCI_561/DSCI_561_lab3_sverma92/gpa_data.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   high_gpa = col_double(),
    ##   math_sat = col_double(),
    ##   verb_sat = col_double(),
    ##   univ_gpa = col_double()
    ## )

``` r
model_1 <- lm(univ_gpa~high_gpa, gpa_data)
model_2 <- lm(univ_gpa~high_gpa+math_sat, gpa_data)

####
# Comparing model1 with null intercept only model
SS_res_reduced <- sum((gpa_data$univ_gpa - mean(gpa_data$univ_gpa))^2)
SS_res_full <- sum((broom::augment(model_1)$.resid)^2)
k <- 1
n <- nrow(gpa_data)
p <- 1

F_stas <- ((SS_res_reduced - SS_res_full)/k)/(SS_res_full/(n-p-1))
anova(lm(univ_gpa~1, gpa_data), model_1)
```

    ## Analysis of Variance Table
    ## 
    ## Model 1: univ_gpa ~ 1
    ## Model 2: univ_gpa ~ high_gpa
    ##   Res.Df     RSS Df Sum of Sq      F    Pr(>F)    
    ## 1    104 20.7981                                  
    ## 2    103  8.1587  1    12.639 159.57 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# Comparing model2 with null intercept only model
SS_res_full <- sum((broom::augment(model_2)$.resid)^2)
k <- 2
n <- nrow(gpa_data)
p <- 2

F_stas <- ((SS_res_reduced - SS_res_full)/k)/(SS_res_full/(n-p-1))
anova(lm(univ_gpa~1, gpa_data), model_2)
```

    ## Analysis of Variance Table
    ## 
    ## Model 1: univ_gpa ~ 1
    ## Model 2: univ_gpa ~ high_gpa + math_sat
    ##   Res.Df     RSS Df Sum of Sq      F    Pr(>F)    
    ## 1    104 20.7981                                  
    ## 2    102  7.9511  2    12.847 82.403 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# Comparing model2 with model1
SS_res_reduced <- sum((broom::augment(model_1)$.resid)^2)
k <- 1
n <- nrow(gpa_data)
p <- 2

F_stas <- ((SS_res_reduced - SS_res_full)/k)/(SS_res_full/(n-p-1))
anova(model_1, model_2)
```

    ## Analysis of Variance Table
    ## 
    ## Model 1: univ_gpa ~ high_gpa
    ## Model 2: univ_gpa ~ high_gpa + math_sat
    ##   Res.Df    RSS Df Sum of Sq      F Pr(>F)
    ## 1    103 8.1587                           
    ## 2    102 7.9511  1   0.20759 2.6631 0.1058

``` r
####

print(paste("The t-statistic from tidy is:", round(tidy(model_2)$statistic[3], digits=3), "square of which is:", round(tidy(model_2)$statistic[3]^2, digits=3)))
```

    ## [1] "The t-statistic from tidy is: 1.632 square of which is: 2.663"

``` r
print(paste("The F-statistic from anova is:", round(tidy(anova(model_1, model_2))$statistic[2], digits=3)))
```

    ## Warning: Unknown or uninitialised column: 'term'.

    ## [1] "The F-statistic from anova is: 2.663"

> In the special case of comparing the two exact same hypotheses in a
> least squares linear model, it can be shown that the F-statistic is
> equal to the T-statistic squared, and that the p-value of the F-test
> and T-test are equal (see
> [here](https://canovasjm.netlify.com/2018/10/29/when-does-the-f-test-reduce-to-t-test/)
> for more). For example, when you use glance() on the model\_1 object,
> you are performing an F-test comparing the null model to the model\_1
> object

When you call `anova(model_1, model_2)` then hypothesis that is getting
tested is as follows:  
\[H_0: \beta_2=0\]  
\[H_A: \beta_2\neq0\]  
We observe equivalent t-test in the `tidy(model_2)` table where we test
the coefficient of the second variable to be equal to
0.

-----

### Association is not causation ([Read this awesome book](https://rafalab.github.io/dsbook/association-is-not-causation.html))

  - This is perhaps the most important lesson one learns in a statistics
    class.

#### Spurious correlation ([Cool examples of spurious correlation](http://tylervigen.com/spurious-correlations))

  - The cases presented in the spurious correlation site are all
    instances of what is generally called data dredging, data fishing,
    or data snooping. It’s basically a form of what in the US they call
    cherry picking. An example of data dredging would be if you look
    through many results produced by a random process and pick the one
    that shows a relationship that supports a theory you want to
    defend.  
  - A Monte Carlo simulation can be used to show how data dredging can
    result in finding high correlations among uncorrelated variables.

<!-- end list -->

``` r
N <- 25
g <- 100000
sim_data <- tibble(group = rep(1:g, each=N), 
                   x = rnorm(N * g), 
                   y = rnorm(N * g))

# Next, we compute the correlation between X and Y for each group and look at the max:
res <- sim_data %>% 
  group_by(group) %>% 
  summarize(r = cor(x, y)) %>% 
  arrange(desc(r))

# plot the data from the group achieving max correlation
sim_data %>% filter(group == res$group[which.max(res$r)]) %>%
  ggplot(aes(x, y)) +
  geom_point() + 
  geom_smooth(method = "lm")
```

![](561_Notes_files/figure-gfm/Spurious%20Correlation-1.png)<!-- -->

``` r
# Distribution of all the correlations
res %>% ggplot(aes(x=r)) + geom_histogram(binwidth = 0.1, color = "black")
```

![](561_Notes_files/figure-gfm/Spurious%20Correlation-2.png)<!-- --> \>
If we performed regression on the highly correlated group and
interpreted the p-value, we would incorrectly claim this was a
statistically significant relation. This particular form of data
dredging is referred to as p-hacking.

#### Outliers

> Suppose we take measurements from two independent outcomes, X and Y,
> and we standardize the measurements. However, imagine we make a
> mistake and forget to standardize entry 23. We can simulate such data
> using:

``` r
set.seed(1985)
x <- rnorm(100,100,1)
y <- rnorm(100,84,1)
x[-23] <- scale(x[-23])
y[-23] <- scale(y[-23])

qplot(x, y)
```

![](561_Notes_files/figure-gfm/Outliers-1.png)<!-- -->

``` r
cor(x,y)
```

    ## [1] 0.9878382

``` r
# But this is driven by the one outlier. If we remove this outlier, the correlation is greatly reduced to almost 0, which is what it should be:

cor(x[-23], y[-23])
```

    ## [1] -0.04419032

``` r
# an alternative to the sample correlation for estimating the population correlation that is robust to outliers. It is called Spearman correlation. The idea is simple: compute the correlation on the ranks of the values. Here is a plot of the ranks plotted against each other:

qplot(rank(x), rank(y))
```

![](561_Notes_files/figure-gfm/Outliers-2.png)<!-- -->

``` r
# correlation
cor(rank(x), rank(y))
```

    ## [1] 0.002508251

``` r
cor(x, y, method = "spearman")
```

    ## [1] 0.002508251

#### Reversing cause and effect

  - Another way association is confused with causation is when the cause
    and effect are reversed. An example of this is claiming that
    tutoring makes students perform worse because they test lower than
    peers that are not tutored. In this case, the tutoring is not
    causing the low test scores, but the other way around.
  - We can easily construct an example of cause and effect reversal
    using the father and son height data.

<!-- end list -->

``` r
library(HistData)
data("GaltonFamilies")
GaltonFamilies %>%
  filter(childNum == 1 & gender == "male") %>%
  select(father, childHeight) %>%
  rename(son = childHeight) %>% 
  do(tidy(lm(father ~ son, data = .)))
```

    ## # A tibble: 2 x 5
    ##   term        estimate std.error statistic  p.value
    ##   <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    ## 1 (Intercept)   34.0      4.57        7.44 4.31e-12
    ## 2 son            0.499    0.0648      7.70 9.47e-13

> The model fits the data very well. If we look at the mathematical
> formulation of the model above, it could easily be incorrectly
> interpreted so as to suggest that the son being tall caused the father
> to be tall. The model is technically correct. The estimates and
> p-values were obtained correctly as well. What is wrong here is the
> interpretation.

#### Confounding

  - Confounders are perhaps the most common reason that leads to
    associations begin misinterpreted. If X and Y are correlated, we
    call Z a confounder if changes in Z causes changes in both X and Y.

> Admission data from six U.C. Berkeley majors, from 1973, showed that
> more men were being admitted than women: 44% men were admitted
> compared to 30% women.

``` r
library(HistData)
data(admissions)
admissions %>% group_by(gender) %>% 
  summarize(total_admitted = round(sum(admitted / 100 * applicants)), 
            not_admitted = sum(applicants) - sum(total_admitted)) %>%
  select(-gender) %>% 
  do(tidy(chisq.test(.))) %>% .$p.value
```

But closer inspection shows a paradoxical result. Here are the percent
admissions by major:

``` r
admissions %>% select(major, gender, admitted) %>%
  spread(gender, admitted) %>%
  mutate(women_minus_men = women - men)
```

> The paradox is that analyzing the totals suggests a dependence between
> admission and gender, but when the data is grouped by major, this
> dependence seems to disappear. This actually can happen if an
> uncounted confounder is driving most of the variability.

> Plot the total percent admitted to a major versus the percent of women
> that made up the applicants:

``` r
admissions %>% 
  group_by(major) %>% 
  summarize(major_selectivity = sum(admitted * applicants)/sum(applicants),
            percent_women_applicants = sum(applicants * (gender=="women")) /
                                             sum(applicants) * 100) %>%
  ggplot(aes(major_selectivity, percent_women_applicants, label = major)) +
  geom_text()
```

  - The plot suggests that women were much more likely to apply to the
    two “hard” majors. Gender and major’s selectivity are confounded.
    [create a facet plot as shown in the
    book 19.4.2](https://rafalab.github.io/dsbook/association-is-not-causation.html)
  - The majority of accepted men came from two majors: A and B. Few
    women applied to these majors.

> Controlling the confounder

``` r
admissions %>% 
  ggplot(aes(major, admitted, col = gender, size = applicants)) +
  geom_point()
admissions %>%  group_by(gender) %>% summarize(average = mean(admitted))
```

If we average the difference by major, we find that the percent is
actually 3.5% higher for women.

> Confounding is addressed by adding the counfounding variable in the
> model

##### Example of Confounding Simpson’s Paradox

You can see that X and Y are negatively correlated. However, once we
stratify by Z (shown in different colors below) another pattern emerges:

![simpsons\_paradox](C:/MyDisk/MDS/simpsons_paradox.PNG)

> It is really Z that is negatively correlated with X. If we stratify by
> Z, the X and Y are actually positively correlated as seen in the plot
> above.

-----

  - \(R^2=\frac{ESS}{TSS}\) is always positive
  - \(R^2=1-\frac{RSS}{TSS}\) is not always positive, RSS can be larger
    the TSS when the model does not have intercept or not LS estimates
  - \((cor(y,\hat{y}))^2=R^2\), For a SLR model estimated by LS this is
    true, but it’s *not* true in general
  - Although the \(R^2\) compares the RSS of the *full model* vs those
    of the *null model*, it does not really tell us how good our model
    *predicts*  
  - The mean square error of prediction (PMSE) measures the distance
    between the predicted and the observed response
      - Can estimate both in-sample & out-sample  
      - PMSE should be used to compare predicted with actual and not
        correlation as it can be confounded.  
  - Both the \(R^2\) and the *F*-test are based on *in-sample*
    predictions and does not give a measure of accuracy in new test
    samples

We compute Bootstrapping or Permutation to HT for the LS estimates: they
can be used if we believe the sample size is too small to trust
asymptotic results. These tests can be used for other estimators as
well.

#### Permutation test for regression

  - We need to generate samples from the null hypothesis\!\!
    \(H_0: \beta_g=0\)
  - Steps:
    1.  Shuffle the rows of \(X\) and combine them with the observed
        \(y\)
    2.  Compute the estimate in the permuted sample
    3.  Repeat 1-2 many times (\(B\))
    4.  Compute the
        \(\text{p-value}=2 \times \frac{\#[\hat{\beta}^*> \hat{\beta}]}{B}\)

#### Bootstrapping tests for regression

  - There are 2 types of resampling: (\(x\)-random) & (\(x\)-fix). Only
    (\(x\)-random) is discussed  
  - In classical tests the statistic is:
    \(T=\frac{\hat{\beta}-\beta_{H_0}}{\text{SE}[\hat{\beta}]}=\frac{\hat{\beta}-0}{\text{SE}[\hat{\beta}]}\)  
  - The bootstrap statistic:
    \(T^*=\frac{\beta^*-\hat{\beta}}{\text{SE}[\beta^*]}\), get \(B\) of
    these\!  
  - pval: \(\frac{1+\#[|T^*| > |T|]}{B+1}\) to test \(H_0:\beta_1=0\) vs
    \(H_A: \beta_1 \neq 0\)  
  - pval: \(\frac{1+\#[T^* > T]}{B+1}\) to test \(H_0:\beta_1=0\) vs
    \(H_A: \beta_1 > 0\)

<!-- end list -->

``` r
# generate sample
n <- 200 # small n will result in skewed sampling distribution, high value will give more normal distribution (by CLT), but note that residuals are always non-normal, regardless of n
x <-  seq(1, n)
y <- x + rlnorm(n)*x
df <- tibble(x = x, y = y)
model <- lm(y ~ x, data = df)
df <- df %>% 
  mutate(resid = predict(model) - y)
# generate bootstrap sampling distribution for beta_1
N <- 1000
boot_fits <- df %>% 
    rsample::bootstraps(times = N) %>% 
    mutate(
        lm   = map(splits, ~ lm(y ~ x, data = analysis(.x))),
        tidy = map(lm, broom::tidy) 
    ) %>% 
    select(-splits, -lm) %>% 
    unnest(tidy) %>% 
    filter(term == "x") %>% 
    select(-term)

# plot bootstrap sampling distribution for beta_1
ggplot(boot_fits, aes(estimate)) +
    geom_histogram(bins = 30, color="black", fill="white") +
  ggtitle('Bootstrap sampling distribution') +
  labs(x = expression(Estimate~of~beta[1])) 
```

![](561_Notes_files/figure-gfm/models-1.png)<!-- -->

``` r
b_obs <- tidy(model)  %>% filter(term == "x") %>% select(estimate, statistic,p.value) 

t_star <- data.frame(t_star=(boot_fits$estimate - b_obs$estimate)/boot_fits$std.error)

#pval using t-distribution
pval_t<-b_obs$p.value
print(paste("pval using t-distribution", pval_t))
```

    ## [1] "pval using t-distribution 1.94285241567769e-21"

``` r
#pval using bootstrapping and pivot method
pval_boot <- (1+sum(abs(t_star) > abs(b_obs$statistic)))/(N+1)
print(paste("pval using bootstrapping and pivot method", pval_boot))
```

    ## [1] "pval using bootstrapping and pivot method 0.000999000999000999"

-----

  - The conditional expectation is the best predictor of 𝑌 given a
    vector of variables 𝑋
  - A LM can be used as a predictor but it may not be the best
  - If the data is jointly normal, the best predictor is linear\!\!
  - The least squares regression is the best among other linear
    predictors

-----

  - Let’s only assume that the errors are \(iid\) (independent and
    identically distributed) and independent of \(X_j\)
      - with \(E[\varepsilon_i]=0\) and \(Var(\varepsilon_i)=\sigma\)  
  - Note that this assumption implies that
    \(E[Y|\mathbf{X}]=\beta_0 + \beta_1 X_{i1} + \ldots + \beta_p X_{ip}\)
      - this is just an assumption\!\! it’s true under normality but we
        are not assuming that (yet)  
  - Note that we are ***not assuming*** that
    \(\varepsilon_i \sim \mathcal{N}(0,\sigma)\)

### LS estimator

  - Find \(\beta_0, \beta_1, \ldots, \beta_p\) that minimizes the sum of
    squared errors:
    \(\sum_i^n(Y_i - \beta_0 - \beta_1 X_{i1} - \ldots - \beta_p X_{ip})^2\)  
  - The minimizer is the <font color="blue"> LS estimator </font>:
    \(\hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_p\)

#### Properties of the LS estimator:

  - Unbiased estimator meaning,
    \(E[\hat{\mathbf{\beta}}]=\mathbf{\beta}\)  
  - Best (lowest variance) Linear Unbiased
    Estimator
  - \(Var(\hat{\mathbf{\beta}})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}, \; \hat{\sigma}=\sqrt{\frac{\sum_{i=1}^n\hat{e}_i^2}{n-p-1}}\)
    (\(p\) slopes + 1 intercept)
      - `sig_hat <- augment(lm_s, dat_s) %>% select(.resid) %>%
        summarize(sig_hat= sqrt(sum(.resid^2)/(length(.resid)-2)))`
  - This is the estimate given by `lm`\!\!
  - The mean squared error for the estimator, not the
        prediction:
      - \(\text{MSE}(\hat{\beta})=E[(\hat{\beta}-\beta)^2]= Var(\hat{\beta})+\text{Bias}(\hat{\beta})^2= Var(\hat{\beta})+(E(\hat{\beta})-\beta)^2\)

#### Matrix Notation of the lienar model:

#### We want to minimize squared error:

  - \(S(\beta_0,\beta_1,\beta_2)=\mathbf{\varepsilon}^T\mathbf{\varepsilon}=(\mathbf{Y}- \mathbf{X} \mathbf{\beta})^T (\mathbf{Y} - \mathbf{X} \mathbf{\beta})\)  
  - \(\frac{\partial{S}}{\partial{\mathbf{\beta}}}=\mathbf{0}= -2\mathbf{X}^T(\mathbf{Y}-\mathbf{X}\mathbf{\beta}) \iff \mathbf{X}^T\mathbf{X}\hat{\mathbf{\beta}}=\mathbf{X}^T\mathbf{Y}\)  
  - \(\hat{\mathbf{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}\)  
  - Note that I *never* assume a distribution for \(\varepsilon\). the
    errors does not have to be Normal\!\!
  - Given \(\mathbf{X}\), \(\hat{\mathbf{\beta}}\) is a linear function
    of \(\mathbf{Y}\), and \(\mathbf{Y}\) has the same distribution as
    \(\varepsilon_i\)
  - If
    \(\varepsilon_i \sim \mathcal{N}(0,\sigma^2) \implies \hat{\mathbf{\beta}}\)
    is also Normal\!\! and we can construct an exact tests for the model
    coefficients
  - \(\hat{\beta}_1 \sim \mathcal{N}\left(\beta_1,\frac{\sigma^2}{(n-1)s_x^2}\right) \implies \quad z=\frac{\hat{\beta}_1-\beta_1}{SE(\hat{\beta}_1)}=\frac{\hat{\beta}_1-\beta_1}{\frac{\sigma}{\sqrt{(n-p-1)s^2_x}}} \sim \mathcal{N}(0,1)\)
  - Since \(\sigma\) is usually *unknown*:
    \(t=\frac{\hat{\beta}_1-\beta_1}{SE(\hat{\beta}_1)}=\frac{\hat{\beta}_1-\beta_1}{\frac{\hat{\sigma}}{\sqrt{(n-1)s^2_x}}} \sim \mathcal{t}_{n-p-1}\)

#### What if \(\varepsilon_i\) are ***not*** Normal??

  - According to Central Limit Theorem, It can be proved that under
    certain conditions, the *asymptotic* sampling distribution of the LS
    estimators is *normal*, even when the errors are not\!\!. However,
    this result requires a large sample size (and “large” depends on
    characteristics of the sample).  
  - `lm` uses this result and constructs a *t*-statistic under the null
    hypothesis \(H_0: \beta_j=0\) with
    \(t=\frac{\hat{\beta}_j-0}{SE(\hat{\beta}_j)}\sim \mathcal{t}_{n-p-1}\)

#### Confidence Intervals for Prediction (CIP)

  - Example: “Are you predicting the average value of a house with the
    given dimensions?”
  - The fitted value, \(\hat{Y}\) (random variable) predicts the true
    value of conditional expectation of Y given X  
  - There is only 1 sources of variation for this prediction: the
    uncertainty of the estimated
    coefficients  
  - \(\hat{Y}(x^*) \pm t_{n-2,0.975} \times SE_{\hat{\mu}_{Y|x^*}}; \; SE_{\hat{\mu}_{Y|x^*}}=\hat{\sigma} \sqrt{\frac{1}{n}+\frac{(x^*-\bar{x})^2}{(n-1)s_x^2}}\)
  - The *t*-distribution here is derived from the conditional
    distribution of \(\varepsilon\) given \(\mathbf{X}\). Thus, there is
    no CLT that we can rely on\!\! This result is assuming normality of
    the error terms\!\!

#### Prediction Intervals (PI)

  - Example: “Are you predicting the value of this given house that has
    these dimensions?”
  - The fitted value, \(\hat{Y}\) predicts the *actual black point*.  
  - There are 2 sources of variation: the uncertainty of the estimated
    coefficients *and* that of the error that generates the data.  
  - PI are wider than CI for prediction.  
  - \(\hat{Y}(x^*) \pm t_{n-2,0.975} \times SE(\hat{Y}(x^*))\);
    \(SE(\hat{Y}(x^*))=\hat{\sigma} \sqrt{1+\frac{1}{n}+\frac{(x^*-\bar{x})^2}{(n-1)s_x^2}}\)

#### Mulitcollinearity

  - The least squares estimates satisfy:
    \(\mathbf{X}^T\mathbf{X}\hat{\mathbf{\beta}}=\mathbf{X}^T\mathbf{Y}\).
    If \(\mathbf{X}^T\mathbf{X}\) is non-singular (analogous to
    \(\neq 0\)), then
    \(\hat{\mathbf{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}\)  
  - However, \(\mathbf{X}^T\mathbf{X}\) becomes nearly singular or
    singular when explanatory variables are collinear or multicollinear,
    aka *multicollinearity problem*.
      - the solution
        \((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}\) becomes
        very unstable\!\! (e.g., values and sign of some coefficients
        change as variables are
        added)  
      - \(Var(\hat{\mathbf{\beta}})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\)
        so the SEs of \(\hat{\mathbf{\beta}}\) can be large under
        multicollinearity  
  - Measured through the variance inflation factors (VIF):
    \(\text{VIF}_j=\frac{1}{1-R^2_{X_j,\boldsymbol{X}_{-j}}}, \; j=(1,\ldots,p)\)
    and \(R^2_{X_j,\boldsymbol{X}_{-j}}\) measures how much of the
    observed variation of \(X_j\) can be explained by other variables
