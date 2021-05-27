<meta http-equiv="content-type" content="text/html;charset=utf-8"/>
<span style="font-family: Menlo;">

# Master Notes

## Table of Contents

1. [Data Science Immersive Pacific](#Data-Science-Immersive-Pacific)
1. [Global](#Global)
1. [CLI / UNIX](#CLI-/-UNIX)
    1. [Basic Commands](#Basic-CLI-Commands)
1. [Git & Github](#Git-&-Github)
    1. [Basic Commands](#Basic-Git-Commands)
    1. [Advanced Commands](#Advanced-Git-Commands)
1. [Maths](#Maths)
    1. [Stats](###Stats)
        1. [Permutations](####Permutations)
        1. [Combinations](####Combinations)
        1. [Conditional probability](####Conditional-probability)
        1. [Bayes' Theorem](####-Bayes'-Theorem)
        1. [Probability mass function](####Probability-mass-function)
        1. [Cumulative distribution function](####Cumulative-distribution-function)
        1. [Expected Value](####Expected-value)
        1. [Mean](####Mean)
        1. [Covariance](####Covariance)
        1. [Variance](####Variance)
        1. [Correlation Coefficient](####Correlation-coefficient-(r))
        1. [Discrete Distributions](####Discrete-Distributions)
        1. [Continuous Distributions](####Continuous-distributions)
        1. [Hypothesis Testing](####Hypothesis-testing)
        1. [Bootstrapping](####Bootstrapping)
        1. [Central Limit Theorem](####Central-limit-theorem-(CLT))
        1. [Confidence Intervals](####Confidence-intervals)
        1. [Hypothesis Testing](####Hypothesis-Testing)
        1. [Statistical Power](####Statistical-Power)
    1. [Linear Algebra](###Linear-algebra)
        1. [Matrix Multiplication](####Matrixmultiplication)
        1. [Identity Matrix](####Identity-matrix-(I<sub>m</sub>))
        1. [Matrix Rank & Independence](####Matrix-rank-&-independence)
        1. [Invertible Matrix](####Invertible-matrix)
        1. [Matrix Rotation](####Matrix-rotation)
        1. [Matrix Reflection](####Matrix-reflection)
        1. [Systems of Equations](####Systems-of-equations-with-linear-algebra)
        1. [Vector Similarity](####Vector-similiarity)
            1. [Eigenvectors & Eigenvalues](#####Eigenvectors-and-Eigenvalues)
    1. [Differential Calculus](###Differntial-calculus)
        1. [Limits](####Limits)
        1. [Derivatives](####Derivatives)
        1. [Differential Power Rule](####Differential-power-rule)
        1. [Differential Chain Rule](####Differential-chain-rule)
    1. [Integral Calculus](###Integral-calculus)
        1. [Antiderivative](####Antiderivative)
        1. [Integral Power Rule](####Integral-power-rule)
        1. [Indefinite Integral by Substitution](####Indefinite-integrals-by-substitution)
        1. [Reimann Summation](####Reimann-summation)
1. [Plotting](##Plotting)
1. [SQL](##SQL-Databases)
1. [Python Libraries](#Python-Libraries)
    1. [Numpy](##Numpy)
    1. [SciPy](##SciPy)
        1. [Distributions](###Distributions)
    1. [Pandas](##Pandas)
        1. [Groupby](###Groupby-object)
1. [Vernacular](##Vernacular)
    1. [Inputs](###Inputs)
    1. [Outputs](###Output)
1. [Algorithms](##Algorithms)
1. [Python](##Python)
    1. [Type Hint](###Type-hint)
    1. [Classes](###Classes)
        1. [Class Attribute, Inheritance, Method](####Class-attribute,-inheritance,-method)
1. [References](##References)
    1. [Machine Learning](###Machine-Learning)
    1. [Statistics](###Statistics)
    1. [Computer Science / Programming](###Computer-Science/Programming)
    1. [Numpy / SciPy](###Numpy/Scipy)
    1. [SQL](###SQL)
    1. [Scikit-Learn](###scikit-learn)
    1. [Extra](###Extra)

---

## Data Science Immersive Pacific

* [Student Resource Document](https://docs.google.com/document/d/1-O00uVHoBe7b7SFS6GpN50emo2Wuz9jG9fPsHNUKtbA/edit)
* [Galvanize GitHub](https://github.com/GalvanizeDataScience/course-outline/tree/21-05-DS-RFP_RFP1)

---

## Global

* [Master Cheat Sheets](https://www.datacamp.com/community/data-science-cheatsheets)
* [Visualizations of Probability and Statistics](https://seeing-theory.brown.edu/)
* [Python Functions Glossary](https://docs.python.org/3/library/functions.html)

---

## CLI / UNIX

* [CLI Cheat Sheet](https://github.com/GalvanizeDataScience/lectures/blob/RFP/unix/unix.pdf)

### Basic CLI Commands

* make directory: `mkdir [folder name]`
* create file: `touch [file name]`
* move or rename: `mv [old] [new]`
* copy: `cp [old] [new]`
* recursive copy: `cp -r [old] [new]`
* remove: `rm [file name]`
* recursive remove: `rm -r [file name]`
* clear terminal: `cmd + k`
* find files in directory: `find . -name file_name`
* add text: `echo [text] >> [file name]`
* overwrite text: `echo [text] > [file name]`
* view content one page at a time: `more [file name]`
* more and backward movement: `less [file name]`
* view first N lines: `head -N [file name]`
* view last N lines: `tail -N [file name]`
* find in text: `grep [text] [file name]`
* view lines: `cat [file name]`
* replacing text via regex: `sed [regex] [file name]]`
* find and perform action via regex: `awk [regex] [file name]]`
* help manual: `man [command name]`
* sort lines: `cat [file name] sort`
* grab unique lines: `cat [file name] uniq`
* command combinations (pipe): `|`
* alter permissions: `chmod [numeric code] [file name]`
* full path of shell command: `which [command]`
* open and edit zsh profile: `vim ~/.zshrc`
* define new command: `alias`

---

## Git & Github

* [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
* [Git Online Learning Game](https://ohmygit.org/)

### Basic Git Commands

* initize repo: `git init`
* clone repo: `git clone [url]`
* status: `git status`
* add to stage: `git add [file name]`
* commit: `git commit -m "[message]"`
* push: `git push`
* pull: `git pull`
* see all branches: `git branch -a`
* create branch: `git branch [new branch name]`
* switch branches: `git checkout [branch name]`
* create and switch brnaches: `git checkout -b [new branch name]`
* roll to previous commit: `git reset`
* revert to previous commit: `git checkout [short hash] [file name]`

#### Advanced Git Commands

* create a new repository on the command line

```zsh
git init
git add .
git commit -m "init commit"
git branch -M main
git remote add origin https://github.com/josiah-d/REPO_NAME.git
git push -u origin main
```

* push an existing repository from the command line

```zsh
git remote add origin https://github.com/josiah-d/gal_sdi.git
git branch -M main
git push -u origin main
```

---

## Maths

### Stats

* [Stats Cheat Sheet](https://web.cs.elte.hu/~mesti/valszam/kepletek.pdf)

#### Permutations

* nPk = n! / (n - k)!

#### Combinations

* nCk = n! / ((n - k)!  * k!)

* Complex version
    * (KCk * (N - K)C(n - k)) / NCn
    * Where:
        * K is `number of successes available`
        * k is `number of successes needed`
        * N is `members available`
        * n is `members chosen`

#### Conditional probability

* P(A|B) = P(AB) / P(B)
* P(A&#8745;B) = P(A) * P(A|B) = P(B) * P(B|A) = P(B&#8745;A)
* Law of Total Probability
    * P(Z) = P(A) * P(A|Z) + P(b) * P(B|Z) + ... + P(n) * P(n|Z)

#### Bayes' Theorem

* P(A|B) = (P(B|A) * P(A)) / P(B)

```python
def cookie_jar(a, b):
    '''
    There are two jars of cookies.
    Each has chocolate and peanut butter cookies.
    Input 'a' is the fraction of cookies in Jar A which are chocolate.
    Input 'b' is the fraction of cookies in Jar B which are chocolate.
    A jar is chosen at random and a cookie is drawn.
    The cookie is chocolate.
    Return the probability that the cookie came from Jar A.

    Parameters
    ----------
    a: float
        Probability of drawing a chocolate cooking from Jar A
    b: float
        Probability of drawing a chocolate cooking from Jar B

    Returns
    -------
    float
        Conditional probability that cookie was drawn from Jar A given
        that a chocolate cookie was drawn.

    Calculation
    -----------

    P(Jar A | Choc Cookie) = [P(Choc Cookie) * P(Jar A)] /
                             [P(Choc Cookie) * P(Jar A) + [P(Choc Cookie) * P(Jar B)]
    '''
    return (a * 0.5) / (a * 0.5 + b * 0.5)
```

#### Probability mass function

* f<sub>X</sub>(t) = P(X = t)
    * Probability of returning a value for *X* equal to *t*

#### Probability density function

* f<sub>𝑋</sub>(𝑡)=𝑃(𝑋≤𝑡)
    * **Cumulative density** or **distribution** function (or sometimes **CDF** for short).  In a sense we could make precise, it says everything we can hope to say about the random variable.

If the distribution function is differentiable, then it's derivative is called the **probability density function**. 

$$ f_X(t) = F_X'(t) $$

The density function satisfies the following property:

  
$$ P(t_0 < X \leq t_1) = \int_{t_0}^{t_1} f(t) dt $$P(t_0 < X \leq t_1) = \int_{t_0}^{t_1} f(t) dt 
   * Probability of returning a value for *X* equal to *t*



#### Cumulative distribution function

* [CDF resources](https://learn-2.galvanize.com/cohorts/868/blocks/248/content_files/07_Continuous_Prob_Dists/00_unit_overview.md)
* F<sub>X</sub>(t) = P(X <= t)
    * Probability of returning a value for *X* less or equal to *t*
* CDF = &Sigma;<sub>lower</sub><sup>upper</sup> p(X = n)
* Normal distribution Z-test: `norm.cdf(1.5, loc=1.25, scale=.46)`

#### Expected value

* Discrete
    * E(X) = &Sigma;<sub>i</sub> (x<sub>i</sub> * p(x<sub>i</sub>))
* Continuous
    * E(X) = &int;<sub>-&infin;</sub><sup>&infin;</sup> (x * f(x))dx

#### Mean

* &mu; = n * p

#### Covariance

* Estimates amount and direction Y moves as X changes
* COV(X, Y) = 1/n * &Sigma;<sub>i=1</sub><sup>n</sup> (x<sub>i</sub> - <span STYLE="text-decoration:overline">x</span>)(y<sub>i</sub> - <span STYLE="text-decoration:overline">y</span>)

#### Variance

* VAR(X) = COV(X, X) = 1/n * &Sigma;<sub>i=1</sub><sup>n</sup> (x<sub>i</sub> - <span STYLE="text-decoration:overline">x</span>) ** 2

```python
n = total
p = mean / total
print((n*p) * (1-p))
```

* VAR = n * p * (1 - p)

#### Correlation coefficient (r)

* corr(X, Y) = [&Sigma;<sub>i=1</sub><sup>n</sup> (x<sub>i</sub> - <span STYLE="text-decoration:overline">x</span>)(y<sub>i</sub> - yBar)] / SQRT[(&Sigma;<sub>i=1</sub><sup>n</sup> (x<sub>i</sub> - <span STYLE="text-decoration:overline">x</span>) ** 2) * (&Sigma;<sub>i=1</sub><sup>n</sup> (y<sub>i</sub> - yBar) ** 2)
* corr(X, Y) = COV(X, Y) / sqrt(VAR(X) * VAR(Y))

#### Discrete Distributions

* Quick sheet sheet on choosing distributions:
    * Q: what's the probability of flipping 10 heads in 25 coin flips?
        * A: binomial
    * Q: what's the probability that the first heads will be on the third flip?
        * A: geometric
    * Q: what's the probability of flipping 20 heads over the course of 13 minutes if on average, you flip 10 heads per minute?
        * A: poisson
    * Q: what's the probability you wait more than a minute before flipping a heads, given that you flip an average of 10 heads per minute?
        * A: exponential
    * Q: the chances of drawing four hearts from a standard pack of cards, drawing without replacement.
        * A: Hyper-geometric

* Bernoulli Trials
    * Must be a binary trial
    * PMF
        * f(k) = 1 - p if k = 0, p if k = 1
    * CDF
        * F(k) = 0 for k < 0, 1 - p for 0 <= k < 1, 1 for k >= 1
    * &mu; = p or np
    * &sigma;<su>2</sup> = p(1-p) or np(1-p())

```python
def pmf(p, x, n):
    choose = math.factorial(n) / (math.factorial(n-x) * math.factorial(x))
    return choose * p ** x * (1 - p) ** (n - x)
```

* Binomial Distribution
    * Distribution of the number of successes in N Bernoulli Trials
    * PMF 
        * P(X=x) = p<sup>x</sup>(1-p)<sup>1-x</sup> = nCx * p<sup>x</sup>(1-p)<sup>n-x</sup>
    * Number of trials must be fixed
    * Trials must be indepenedent
    * Each trial must be Bernoulli Trials

![Normal](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/c548abe948b204550481654e32b9d4d4.png)

* Poisson Distribution
    * Has to contain a rate and a particular observation time 
    * PMF
        * f(k) = (&lambda;<sup>k</sup> * e<sup>-&lambda;</sup>) / k!
            * k is the number of items we want to find probability of 
            * &lambda; is the rate parameter

* Geometric Distribution
    * Distribution of the number of trials before first success

![Geometric](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/365d2527ed510a5c5bf3d220887aef13.png)

* Negative Binomial Distribution
    * Distribution of the number of trials before the ith success

![Empirical Rule](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/7e9688d48ba2b2e304969d13d8e4343d.png)

![Power Law](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/b2e012bf689efa1483f296698db3c0f0.png)

![Population and Sampling Distribution](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/3cd2651378842a8f25f7f70b26ab65a0.png)

#### Continuous distributions

* PDF
    * F<sub>X</sub>(t) = P(x <= t) = &int;<sub>-&infin;</sub><sup>t</sup> f<sub>X</sub>(t) dt

#### Hypothesis testing

* &mu; == `True` population mean 
* &mu;<sub>0</sub> == `hypothesized` population mean 

* Null hypothesis (H<sub>0</sub>)
    * the initial claim, assumption, presumption, or assertion being made
    * H<sub>0</sub> : &mu; == &mu;<sub>0</sub>
* Alternative hypothesis (H<sub>A</sub>)
    * An alternative that can be accepted if there is statistically significant evidence found to refute the null hypothesis
    * Upper tailed
        * H<sub>A</sub> : &mu; > &mu;<sub>0</sub>
    * Lower tailed
        * H<sub>A</sub> : &mu; < &mu;<sub>0</sub>
    * Two tailed
        * H<sub>A</sub> : &mu; != &mu;<sub>0</sub>
    * Errors
        * Type I
            * Reject H<sub>0</sub> when it is `True`
        * Type II
            * Accept H<sub>0</sub> when it is `False`
* t-statistic
    * t = (<span STYLE="text-decoration:overline">x</span> - &mu;<sub>0</sub>) / (s / sqrt(n)) * t(n - 1)


When assuming equal variance or not:
If one sample has a greater sample size than the other, we **cannot assume** equal variance. 


#### Bootstrapping

* Random sampling with replacement of a fixed sample or population
* assigns measures of accuracy to sample estimates
    * e.g. bias, variance, confidence intervals, prediction error

#### Central limit theorem (CLT)

* &mu;<sub><span STYLE="text-decoration:overline">X</span></sub> = &mu;
* &sigma;<sub><span STYLE="text-decoration:overline">X</span></sub> = &sigma; / sqrt(n)

When we are concerned with a **sample mean**, the central limit theorem lets us derive the **actual distribution of the sample mean**.  This allows us to calculate probabilities about the sample mean.

We are going to make good use of this next lesson when we design statistical hypothesis tests.

#### Confidence intervals

```
alpha = distribution_of_sample_minus_population_mean.ppf(0.025)
print("Sample Mean: {:2.2}".format(sample_mean))
print("95% confidence interval for the population mean: [{:2.2}, {:2.2}]".format(
    sample_mean + alpha, sample_mean - alpha)
    )
```    
```
def compute_confidence_interval(data, confidence_width):
    sample_mean = np.mean(data)
    sample_varaince = np.var(data)
    distribution_of_sample_minus_population_mean = stats.norm(0, np.sqrt(sample_varaince / len(data)))
    alpha = distribution_of_sample_minus_population_mean.ppf(0.5 - (confidence_width / 2.0))
    # Alpha is negative
    return sample_mean + alpha, sample_mean - alpha
```




* Used when there are no preconceived notions and desire to use sampling to learn about the population
![95% CI](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/21125c3404bf0e14de6dcc164f71e5b1.png)

* If we ***draw samples from the population*** and compute this confidence interval many, many times, then ***the computed interval should envelop*** the true population parameter ***approximately 95% of the time***.


##### Welch's T-Test
 **Welch's t-test**, or unequal variances t-test, is a two-sample location test which is used to test the hypothesis that two populations have equal means. It is named for its creator, Bernard Lewis Welch, and is an adaptation of Student's t-test, and is more reliable when the two samples have unequal variances and/or unequal sample sizes. These tests are often referred to as "unpaired" or "independent samples" t-tests, as they are typically applied when the statistical units underlying the two samples being compared are non-overlapping. 



1. Take a Skeptical Stance, and Clearly State This Hypothesis
   1. Example: There is **no difference** between the means of two samples.
2. Create a Probablistic Model of the Stiuation Assuming the Null Hypothesis is True
   1. Our measurements are **sample averages**, which, from the central limit theorem, we know are approximately normally distributed given the population average
3. Convert to a sample statistic whose variance is expected to be $1$
``` 
def welch_test_statistic(sample_1, sample_2):
    numerator = np.mean(sample_1) - np.mean(sample_2)
    denominator_sq = (np.var(sample_1) / len(sample_1)) + (np.var(sample_2) / len(sample_2))
    return numerator / np.sqrt(denominator_sq) 

test_statistic = welch_test_statistic(sample_1, sample_2)
print("Welch Test Statistic: {:2.2f}".format(test_statistic))

```

$$ \frac{\text{Difference in sample averages}}{\sqrt{\frac{\sigma^2_M}{25} + \frac{\sigma^2_N}{25}}} $$

$$ \frac{\text{Difference in sample averages}}{\sqrt{\frac{\sigma^2_M}{25} + \frac{\sigma^2_N}{25}}} $$


[Welche-Satterthwaite](https://en.wikipedia.org/wiki/Welch%E2%80%93Satterthwaite_equation) to compensate for SMALL sample sizes
 ```
 def welch_satterhwaithe_df(sample_1, sample_2):
    ss1 = len(sample_1)
    ss2 = len(sample_2)
    df = (
        ((np.var(sample_1)/ss1 + np.var(sample_2)/ss2)**(2.0)) / 
        ((np.var(sample_1)/ss1)**(2.0)/(ss1 - 1) + (np.var(sample_2)/ss2)**(2.0)/(ss2 - 1))
    )
    return df

df = welch_satterhwaithe_df(sample_1, sample_2)
print("Degrees of Freedom for Welch's Test: {:2.2f}".format(df))

x = np.linspace(-3, 3, num=250)

fig, ax = plt.subplots(1, figsize=(16, 3))
students = stats.t(df)
ax.plot(x, students.pdf(x), linewidth=2, label="Degree of Freedom: {:2.2f}".format(df))
ax.legend()
ax.set_title("Distribution of Welsh's Test Statistic Under the Null Hypothesis")
```

4. Decide how Suprised You Need to Be to Reject Your Skeptical Assumption
   1. To be reasonably skeptical,  take 𝛼 = 0.05

```
p_value = students.cdf(test_statistic) + (1 - students.cdf(-test_statistic))
print("p-value for different average kickflip height: {:2.2f}".format(p_value))

```
5. Calculate the Probability of Finding a Result Equally or More Extreme than Actually Observed Assuming the Probabilistic Model You Created.



##### Student's T Test
The t-distribution always has mean $0$ and varaince $1$, and has one parameter, the **degrees of freedom**.  Smaller degrees of freedom have heavyer tails, with the distribution beoming more normal as the degrees of freedom gets larger.

The $T$ statistic only has a t-distribution **under the assumption that the population distributions are Normal**!  We did *not* have to assume this for *any* other test, but when we need to estimate the variance of the population, we need more structure!

If the population is very non-normal, the properties of the t-test **will fail**.  You must have some legitimate a-priori reason to believe the populations are approximately normal to use a t-test!

For this reason, many statisticians advise **AGAINST** t-tests these days, preferring non-parametric tests like the signed rank test.

#### Non-Parametrics: Mann-Whitney Signed Rank Test
The [Mann-Whitney U-test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) is a modern alternative to the classical Student's and Welch's t-test that makes good use of modern computing power.  It makes **no** distributional assumptions (unlike the t-test, which assumes the populations are normal), and can always be used instead.


The idea of the Mann-Whitney test is to view this as a competition. We let each of Sample 1's outputs compete against all of Sample 2's outputs, and see how many times it wins (i.e. how many of Sample 1's outputs it beats). We then add these number of wins up over all of Sample 1's outputs.

```

def count_winning_pairs(sample_1, sample_2):
    sample_1, sample_2 = np.array(sample_1), np.array(sample_2)
    n_total_wins = 0
    for x in sample_1:
        n_wins = np.sum(x > sample_2) + 0.5*np.sum(x == sample_2)
        n_total_wins += n_wins
    return n_total_wins


nick_wins = count_winning_pairs(nick_heights, matt_heights)
matt_wins = count_winning_pairs(matt_heights, nick_heights)
print("Number of Nick Wins: {}".format(nick_wins))
print("Number of Matt Wins: {}".format(matt_wins))

```

##### The U-Test



To test the hypothesis that Nick is better than Matt, we need to adopt a Null hypothesis. The Null for the Mann-Whitney test is directly related to which competitor is better.

    $H_0$: Matt's Kickflips are equaliy likely to be higher than Nicks as the other way around. I.e.

$$P(\text{Height Matt Kickflip} &gt; \text{Height Nick Kickflip}) = 0.5$$

As is usual, assuming this null hypothesis is true, the rank-sum statistic assumes a known (but complicated) distribution. This time we can't write down the distribution in any explicit way, but python can calculate p-values using it.

#### Statistical Power

The relation between the power and the rejection level alpha:
-  As we decrease alpha we decrease the false positive rate, increase the false negative rate, and thus decrease the power. We can observe the effect of this by varying the rejection level in our pictures.

The relationship between effect size and power is straightforward: 
- it is easier to detect larger effects. That is, as we increase the effect size we would like to detect, our power to detect that effect increases. Said differently, it is much harder to detect minute effects than large ones.

The statistical power of a test is **affected by a few different things:**
1. The rejection level alpha.
2. The effect size we wish to detect.
3. The size of the sample we collect. 

As we **collect more data**, we tighten our estimate of the sample mean (or whatever sample statistic we happen to be studying) as the standard error of the estimate decreases. This tends to better separate the null and alternate distributions, which increases the power.

##### Calculating the Power for an Experiment

We first need to calculate the rejection threshold by using the percentile function of the null distribution, then compute the area under the alternate distribution to the right of the threshold.

```
def compute_power(n, sigma, alpha, mu0, mua):
    standard_error = sigma / n**0.5
    h0 = scs.norm(mu0, standard_error)
    ha = scs.norm(mua, standard_error)
    critical_value = h0.ppf(1 - alpha)
    power = 1 - ha.cdf(critical_value)
    return power
```

##### Calculating the Sample Size Needed to Obtain a Given Power


```
def sample_size_needed_for_power(alpha, power, mu0, mua, sigma):
    standard_normal = scs.norm(0, 1)
    beta = 1 - power
    numerator = sigma * (standard_normal.ppf(1 - alpha) - standard_normal.ppf(beta))
    denominator = mua - mu0
    return math.ceil((numerator / denominator) ** 2)

alpha, mu0, mua, sigma = 0.05, 0.0, 0.1, 0.5
powers = [0.5, 0.75, 0.9, 0.99, 0.999]

for power in powers:
    print("Sample Size Needed to Achive Power {:2.3f}: {}".format(
        power, 
        sample_size_needed_for_power(alpha, power, mu0, mua, sigma)))
```





### Linear algebra

* Grab length of the vector: `np.linalg.norm(x)`

#### Matrix multiplication

* (AB)<sub>ij</sub> = &Sigma;<sub>k=1</sub><sup>m</sup> a<sub>ik</sub> * b<sub>kj</sub>
* Dot product: A.dot(b) = A * B<sup>T</sup>

#### Identity matrix (I<sub>m</sub>)

* They are square, same number of rows and columns
* They are diagonal, only non-zero entries have the same row and column index
* All non-zero entries are 1

![Identity Matrix](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.kVdcPRjTiGUVjok2dRNhKAHaBx%26pid%3DApi&f=1)

#### Matrix rank & independence

* Linear dependence
    * Column can be constructed from other columns' data
* Linear independence
    * Unique data that cannot be constructed from other rows
* Rank
    * Number of linear independent columns (or rows) in a matrix

#### Invertible matrix

* `np.linalg.inv(MATRIX)`
* A * A<sup>-1</sup> = A<sup>-1</sup> * A = I
* It must be square, i.e. of size `n X n`
* It must be full-rank, i.e. the number of linearly independent rows/columns is equal to `n`
* Square matrices that stretch and/or rotate vectors without any collapsing of dimensions are invertible

#### Matrix rotation

* R = [[cos(&theta;), -sin(&theta;)], [sin(&theta;), cos(&theta;)]]
* 90&#176; counterclockwise: [[0 -1] [1 0]]
* 90&#176; clockwise: [[0 1] [-1 0]]
* Testing

R * [[1], [0]] = [[cos(&theta;)], [sin(&theta;)]]

**Radians**: &theta; = arcos(x) = arcsin(y) = arctan(y/x)

**Degrees**: [&theta; = arcos(x) = arcsin(y) = arctan(y/x)] * (180/&pi;)

![Unit Circle](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fwww.shelovesmath.com%2Fwp-content%2Fuploads%2F2012%2F11%2FUnit-Circle.png&f=1&nofb=1)

#### Matrix reflection

* x axis: [[1 0] [0 -1]]
* y axis: [[-1 0] [0 1]]
* x & y axes: [[-1 0] [0 -1]]

#### Systems of equations with linear algebra

Ax<sup>&rarr;</sup> = b<sup>&rarr;</sup>
A<sup>-1</sup>Ax<sup>&rarr;</sup> = A<sup>-1</sup>b<sup>&rarr;</sup>
Ix<sup>&rarr;</sup> = x<sup>&rarr;</sup> = A<sup>-1</sup>b<sup>&rarr;</sup>

* Where
    * ["A] is a matrix of coeff
    * x<sup>&rarr;</sup> is an unknown vector
    * b<sup>&rarr;</sup> is there product
    * A<sup>-1</sup> is an inverse matrix
    * I is an identity matrix

* Sample code

```python
A = np.array(matrix)
b = np.array(vector)
A_1 = np.linalg.inv(A)
I = A @ np.linalg.inv(A)
x = A_1.dot(b)

print(x)
```

#### Vector similiarity

* Euclidean distance
    * Distance between p<sup>&rarr;</sup> and q<sup>&rarr;</sup>
    * Displays relative magnitudes

d(p<sup>&rarr;</sup>,q<sup>&rarr;</sup>) = sqrt((q<sub>1</sub>-p<sub>1</sub>)<sup>2</sup>+(q<sub>2</sub>-p<sub>2</sub>)<sup>2</sup>+...+(q<sub>n</sub>-p<sub>n</sub>)<sup>2</sup>) = sqrt((q<sup>&rarr;</sup>-p<sup>&rarr;</sup>)<sup>2</sup>)

```python
def euc_dist(arr0, arr1):
    return np.linalg.norm(arr0 - arr1)
```

```python
import pandas as pd
import numpy as np

# Don't alter the code below
np.random.seed(1)
vec1 = pd.Series(np.random.randint(0, 10, 35))
vec2 = pd.Series(np.random.randint(10, 20, 35))

# Your code below
euc_dist = sum((vec1 - vec2)**2)**.5
```

* L2- Norm

||p<sup>&rarr;</sup>|| = sqrt(p<sub>1</sub><sup>2</sup>+p<sub>2</sub><sup>2</sup>+...+p<sub>n</sub><sup>2</sup>) = sqrt(p<sup>&rarr;</sup><sup>2</sup>)

* Normalize a data set

```python
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()['data']

min_ = iris_data.min(axis=0)
ptp = iris_data.ptp(axis=0)

iris_data_normalized = np.round((iris_data - min_) / ptp , 2)
```

* Cosine similarity
    * Displays direction

p<sup>&rarr;</sup>*q<sup>&rarr;</sup> = ||p<sup>&rarr;</sup>|| * ||q<sup>&rarr;</sup>|| * cos&Theta;

* Where &Theta; is the angle between the vectors
* 1 = aligned, -1 = opposite, 0 = orthogonal

sim(p<sup>&rarr;</sup>,q<sup>&rarr;</sup>) = cos&Theta; = p<sup>&rarr;</sup> * q<sup>&rarr;</sup> / ||p<sup>&rarr;</sup>|| * ||q<sup>&rarr;</sup>||

```python
def cosine_sim(arr0, arr1):
    return (arr0 @ arr1) / (np.linalg.norm(arr0) * np.linalg.norm(arr1))
```

##### Eigenvectors and Eigenvalues

* Eigenvectors must be square matrices and satisfy:

Av<sup>&rarr;</sup> = &lambda;v<sup>&rarr;</sup>

* Where &lambda; is a scalar, e.g. the eigenvalue

### Differntial calculus

#### Limits

lim<sub>x&rarr;a</sub> f(x) = L

![Limit](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/7f7dedfbccb1f194dfc0b1d484244008.png)

#### Derivatives

(f(x+h) - f(x)) / h = &Delta;f / &Delta;x

f'(x) = lim<sub>h&rarr;0</sub> (f(x+h) - f(x)) / h = lim<sub>&Delta;x&rarr;0</sub> &Delta;f / &Delta;x = d/dx f(x)

![Derivative](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/25aa43c1b0c93527738ab55c6ac65c6d.png)

#### Differential power rule

d/dx x<sup>n</sup> = nx<sup>n-1</sup>

#### Differential chain rule

f(x) = h(x) * g(x) = h(g(x))

f'(x) = h'(g(x)) * g'(x) = du/dx = (du/dv) * (dv/dx)

### Integral calculus

#### Antiderivative

F(x) = &int;f(x) dx

#### Integral power rule

&int;x<sup>n</sup> dx = x<sup>n+1</sup> / (n+1) + C

#### Indefinite integrals by substitution

f(g(x)) = &int;f'(g(x))g'(x) dx

#### Reimann summation

area = &Sigma;<sub>i=0</sub><sup>n-1</sup> f(x<sub>i</sub>)&Delta;x

* Sample code

```python
import numpy as np


# Define the function you're integrating
def f(x):
    return np.sqrt(x**3) / (x ** 3 + 6)


# Define an integration function
def riemann_sum(f, a, b, n):
    """Approximate the area under the curve 'f' from 'a' to 'b' via
    a Riemann summation of 'n' rectangles using the midpoint rule.

    Parameters
    ----------
        f : function
            The function over which to integrate
        a : float
            The lower bound of the integral
        b : float
            The upper bound of the integral
        n : int
            The number of rectangles to use

    Returns
    -------
        float
            An approximation of the definite integral
            of 'f' over the interval ['a', 'b'].
    """
    # The width of the rectangles, delta x
    dx = (b - a) / n

    # A length-n vector of x-values starting at a + dx/2 and ending at b - dx/2
    xi = np.linspace(a + 0.5*dx, b - 0.5*dx, n)

    # A length-n vector of just dx ([dx, dx, ...])
    dx_vec = np.array([dx]*n)

    # Evaluate f(x) for all xi values
    fxi = f(xi)

    # Return the dot product of f(xi) the vector of [dx, dx, ...]
    # This is just a quick and easy way to take the sum of the
    # products f(x_i)*dx
    return np.dot(fxi, dx_vec)


# Call riemann_sum() with the correct arguments
approx_area = riemann_sum(f, 0, 10, 100)
print(approx_area)
```

* Limit

dx = &Delta;x&rarr;0 = lim<sub>n&rarr;&infin; (b-a) / n

* Probability density function

f(x) = (2 * &pi;)<sup>-1/2</sup> * e <sup>(-1/2) * x<sup>2</sup></sup>

* y-axis symmetry

&int;<sub>-a</sub><sup>a</sup> f(x) dx = 2 &int;<sub>0</sub><sup>a</sup> f(x) dx

* Monte Carlo integrator

```python
import numpy as np


# Define the standard normal distribution pdf function
def std_nrm_pdf(x):
    return np.exp(-x**2 / 2) / (np.sqrt(2 * np.pi))


# Define the Monte Carlo integrator for an even function
def monte_carlo(f: callable, x_bound: float, y_bound: float, n: int) -> float:
    """A basic Monte Carlo integrator.

    Note: this integrator assumes that f is an even function,
    that the bounds of integration are -x_bound to x_bound,
    and that f(x) >= 0 for all x between 0 and x_bound.

    Parameters
    ----------
    f : callable
        The function object whose definite integral MC approximates.
    x_bound : float
        The horizontal bound of the integrating domain.
    y_bound : float
        The vertical bound of the integrating domain.
    n : int
        The number of random points to generate in the domain.

    Returns
    -------
    float
        The approximate area under f between -x_bound and x_bound.
    """
    area_dom = x_bound * y_bound

    # Generate vectors of random x- and y-values, scaled by
    # their respective bounds
    x_vals = x_bound * np.random.rand(n)
    y_vals = y_bound * np.random.rand(n)

    # Use a boolean mask y <= f(x) to get a new vector of only
    # the values where that condition is true
    ratio = len(y_vals[y_vals <= f(x_vals)]) / n

    return 2 * area_dom * ratio


# x_bound is 2 because that's the upper bound of integration
# y_bound is std_nrm_pdf(0) which for the standard normal
# distribution is the max value of std_nrm_pdf.
x_b = 2.0
y_b = std_nrm_pdf(0)

print(monte_carlo(f=std_nrm_pdf, x_bound=x_b, y_bound=y_b, n=10000))
```

* Integral between two points

&int;<sub>a</sub><sup>b</sup> f(x) - g(x) dx

---

## Plotting

* [Plotting Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)
* [Pyplot Docs](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
* Key parts
    * Title
    * Axes
    * Data
    * Legend
    * Annotations
    * Facets

![PyPlot Parts](https://s3.us-west-2.amazonaws.com/forge-production.galvanize.com/content/cdbb5b1b140a97b8ea83e687457cab1a.png)

---

## SQL Databases

* Read sql file

`.read filename.sql`

* Table Summary

`PRAGMA table_info(tablename);`

* Between

```sql
SELECT * FROM table_name
WHERE column_name BETWEEN low AND high;
```

* Group By

```sql
SELECT column_name FROM table_name
GROUP BY column_name;
```

* Order By

```sql
SELECT column_name, other_column_name FROM table_name
ORDER BY other_column_name;
```

* Select Distinct
    * Removes **dulicated** query results

```sql
SELECT DISTINCT column_name, other_column_name FROM table_name
ORDER BY other_column_name;
```

* Aggregate functions
    * `COUNT`

    ```sql
    SELECT other_column_name, COUNT(column_name) FROM table_name
    GROUP BY other_column_name;
    ```

    * `SUM`

    ```sql
    SELECT SUM(column_name) FROM table_name
    GROUP BY other_column_name;
    ```

    * `AVERAGE`

    ```sql
    SELECT AVG(column_name), other_column_name FROM table_name
    GROUP BY other_column_name;
    ```

    * `MIN`

    ```sql
    SELECT MIN(column_name), other_column_name FROM table_name
    GROUP BY other_column_name;
    ```

    * `MAX`

    ```sql
    SELECT MAX(column_name), other_column_name FROM table_name
    GROUP BY other_column_name;
    ```

* Join functions
    * `JOIN`
        * Inner Join

    ```sql
    SELECT t1.col1, t1.col2, t2.col1, t2.col2
    FROM table1 as t1
    JOIN table2 as t2
    ON t1.id = t2.col_id;
    ```

    * `LEFT JOIN`

    ```sql
    SELECT t1.col1, t1.col2, t2.col1, t2.col2
    FROM table1 as t1
    LEFT JOIN table2 as t2
    ON t1.id = t2.col_id;
    ```

    * `RIGHT JOIN`

    ```sql
    SELECT t1.col1, t1.col2, t2.col1, t2.col2
    FROM table1 as t1
    RIGHT JOIN table2 as t2
    ON t1.id = t2.col_id;
    ```

    * `FULL OUTER JOIN`

    ```sql
    SELECT t1.col1, t1.col2, t2.col1, t2.col2
    FROM table1 as t1
    FULL OUTER JOIN table2 as t2
    ON t1.id = t2.col_id;
    ```

    * **Multiple Joins**

    ```sql
    SELECT t1.col1, t1.col2, t2.col1, t2.col2, t3.col1, t3.col2
    FROM table1 as t1
    JOIN table2 as t2
    ON t1.id = t2.col_id
    JOIN table3 as t3
    ON t2.col2_id = t2.id
    ```

* Pick MAX count of GROUP BY table

    ```sql
    /*
    COUNT is given a variable name to allow it to be called later.
    */
    SELECT group_column, COUNT(*) as cnt
    FROM table
    GROUP BY group_column
    ORDER BY cnt DESC
    LIMIT 1;
    ```

---
# Python Libraries

## Numpy

* [Numpy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
* Remove rows w/ np.nan

```python
# Imports
from sklearn.datasets import load_diabetes
from random import choice
from random import seed
import numpy as np

# Do not change the code below
seed(1)
diabetes = load_diabetes()['data']
x_miss_idx = [choice(range(diabetes.shape[0])) for _ in range(10)]
y_miss_idx = [choice(range(diabetes.shape[1])) for _ in range(10)]

for x, y in zip(x_miss_idx, y_miss_idx):
    diabetes[x,y] = np.nan

# Your code below, the data is in the variable 'diabetes'
mask = np.any(np.isnan(diabetes) | np.equal(diabetes, 0), axis=1)
dbts_rmv_nan = diabetes[~mask]
```

* creates an equally spaced grid of numbers between two endpoints.`np.linspace`
* find index of the minimum: `np.argmin()`

### Useful Functions

**Multiplication**
  |Function | Description |
  | ---- | --- |
  |  A @ D | Multiplication operator
  | np.multiply(D,A) |	Multiplication
  | np.dot(A,D)| Dot product
  | np.vdot(A,D)|	Vector dot product
  | np.inner(A,D)|Inner product
  |np.outer(A,D)|	Outer product
  |np.tensordot(A,D) |Tensor dot product
  |np.kron(A,D)	| Kronecker product

** Exponential **  
|Function | Description |
| ---- | --- |
| linalg.expm(A) |Matrix exponential

---

## SciPy
```
import scipy.stats as sc
```

### Distributions

#### Normal
A normal continuous random variable.
The location (``loc``) keyword specifies the mean.
The scale (``scale``) keyword specifies the standard deviation.

As an instance of the `rv_continuous` class, `norm` object inherits from it
a collection of generic methods (see below for the full list),
and completes them with details specific for this particular distribution.
```
sc.norm
mean, var = norm.stats(x)

Methods
-------
rvs(loc=0, scale=1, size=1, random_state=None)
    Random variates.
pdf(x, loc=0, scale=1)
    Probability density function.
logpdf(x, loc=0, scale=1)
    Log of the probability density function.
cdf(x, loc=0, scale=1)
    Cumulative distribution function.
logcdf(x, loc=0, scale=1)
    Log of the cumulative distribution function.
sf(x, loc=0, scale=1)
    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
logsf(x, loc=0, scale=1)
    Log of the survival function.
ppf(q, loc=0, scale=1)
    Percent point function (inverse of ``cdf`` --- percentiles).
isf(q, loc=0, scale=1)
    Inverse survival function (inverse of ``sf``).
moment(n, loc=0, scale=1)
    Non-central moment of order n
stats(loc=0, scale=1, moments='mv')
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
entropy(loc=0, scale=1)
    (Differential) entropy of the RV.
fit(data, loc=0, scale=1)
    Parameter estimates for generic data.
expect(func, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
    Expected value of a function (of one argument) with respect to the distribution.
median(loc=0, scale=1)
    Median of the distribution.
mean(loc=0, scale=1)
    Mean of the distribution.
var(loc=0, scale=1)
    Variance of the distribution.
std(loc=0, scale=1)
    Standard deviation of the distribution.
interval(alpha, loc=0, scale=1)
    Endpoints of the range that contains alpha percent of the distribution
```
### Discrete

Display the probability mass function (pmf):
```
x = np.arange(distribution_name.ppf(0.01, p),
              distribution_name.ppf(0.99, p))
ax.plot(x, distribution_name.pmf(x, p), 'bo', ms=8, label='distribution_name pmf')
ax.vlines(x, 0, distribution_name.pmf(x, p), colors='b', lw=5, alpha=0.5)
```

#### Bernoulli

```python
mean, var, skew, kurt = bernoulli.stats(p, moments='mvsk')
```

#### Binomial

```python
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
```

#### Geometric

The probability mass function for geom is:

for , 

geom takes  as shape parameter, where  is the probability of a single success and  is the probability of a single failure.

* [Distribution Docs](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

## Pandas

* [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

### Groupby object

* `.aggregate`: Iterates through groups, operates as a whole, e.g. mean, max
* `.transform`: Iterates through groups, operates on each value, e.g. subtracting mean
* `.filter`: Check each group, removing groups that fall below a certain threshold
* `.apply`: Can do all of the above, but it less efficient
    * apply function to rows, specify columns and `axis=1`

```python
golf_df.apply(lambda x: x['Temperature'] + x['Humidity'], axis=1)
```

* cross tabulation frequency counts

```python
pd.crosstab(golf_df['Outlook'], golf_df['Result'])
```

* perform actions by DateTime

```python
date_df = golf_df.set_index('DateTime')  # Set index to DateTime object
date_df.resample('W').mean()  # Action
```

---

## Vernacular

### Inputs

* Explanatory Variable
* Independent Variable
* Controlled Variable
* Manipulated Variable
* Exposure Variable
* Predicated Variable
* Treatment Variable
* Input Variable
* Regressor (common in Statistics)
* Predictor (common in Data Science)
* Covariate
* Exogenous (from econometrics)

### Output

* Response Variable
* Dependent Variable
* Regressand (common in Statistics)
* Criterion
* Predicted Variable
* Measured Variable
* Explained Variable
* Experimental Variable
* Responding Variable
* Outcome Variable
* Output Variable
* Target (very common in Data Science)
* Label (very common in Data Science)

---

## Algorithms

![Machine Learning Algorithms](https://cdn.discordapp.com/attachments/822941807385509888/822941867792269352/image4.png)

---

# Python

### Type hint

python

%timeit adds a timer for the computing time when running the code
```
def factorial(n: int) -> int:
    prod = 1
    for num in range(1, n + 1):
        prod *= num
    return prod

>>> help(factorial)
Help on function factorial in module __main__:

factorial(n: int) -> int
```
## Object Oriented Programming ( OOP )

### Classes

[Magic Methods Cheat Sheet](https://www.tutorialsteacher.com/python/magic-methods-in-python)

#### Class attribute, inheritance, method

```python
class Flights:
    num_flights = 0

    def __init__(self, flight_num, arrival_time):
        self.flight_num = flight_num
        self.arrival_time = arrival_time
        Flights.num_flights += 1

    def flight_info(self):
        return f'flight number: {self.flight_num}\narrival time: {self.arrival_time}'

    @classmethod
    def print_num_flights(cls):
        # Class attribute example
        return f'total flights: {cls.num_flights}'

    def __str__(self):
        # Magic method example
        return f'flight number: {self.flight_num}\narrival time: {self.arrival_time}'


class PassengerFlights(Flights):
    def __init__(self, flight_num, arrival_time, passenger_num):
        # Inheritance example
        super().__init__(flight_num, arrival_time)
        self.passenger_num = passenger_num

    def __add__(self, other):
        # Magic method example
        return self.passenger_num + other.passenger_num


class CargoFlights(Flights):
    def __init__(self, flight_num, arrival_time, cargo_weight):
        # Inheritance example
        super().__init__(flight_num, arrival_time)
        self.cargo_weight = cargo_weight
```

---

## References

### Machine Learning

* [Machine Learning in Action](http://www.manning.com/pharrington/)
* [Programming Collective Intelligence](http://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325)
* [Machine Learning for Hackers](http://shop.oreilly.com/product/0636920018483.do)
* [An Introduction to Machine Learning](http://alex.smola.org/drafts/thebook.pdf)

### Statistics

* [Probabilistic Programming and Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
* [Think Stats](https://greenteapress.com/wp/think-stats-2e/)
* [Think Bayes](http://www.greenteapress.com/thinkbayes/)
* [All of Statistics](http://www.stat.cmu.edu/~larry/all-of-statistics/)
* [Mostly Harmless Econometrics](http://www.amazon.com/Mostly-Harmless-Econometrics-Empiricists-Companion/dp/0691120358)

### Computer Science/Programming

* [Think Python](https://greenteapress.com/wp/think-python-2e/)
* [Think Complexity: Analysis of Algorithms](http://greenteapress.com/complexity2/html/thinkcomplexity2003.html#sec20)

### Numpy/Scipy

* [scipy Lectures](https://scipy-lectures.github.io/intro/numpy/index.html)
* [Crash Course in Python for Scientist](http://nbviewer.ipython.org/gist/rpmuller/5920182)
* [Scientific Python Lectures](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-2-Numpy.ipynb)
* [Python Bootcamp Lectures](https://nbviewer.jupyter.org/github/profjsb/python-bootcamp/blob/master/Lectures/05_NumpyPandasMatplotlib/IntroNumPy.ipynb)
* [scipy Lectures](https://scipy-lectures.github.io)

### SQL

* [http://sqlfiddle.com/](http://sqlfiddle.com/)
* [http://use-the-index-luke.com/](http://use-the-index-luke.com/)
* [SQL School](http://sqlschool.modeanalytics.com/)

### scikit-learn

* [Introduction to Machine Learning with sklearn](http://researchcomputing.github.io/meetup_spring_2014/python/sklearn.html)
* [scikit-learn workshop](https://github.com/jakevdp/sklearn_pycon2014)
* [Machine Learning Tutorial](https://github.com/amueller/tutorial_ml_gkbionics)
* [Introduction to scikit-learn](http://nbviewer.ipython.org/github/tdhopper/Research-Triangle-Analysts--Intro-to-scikit-learn/blob/master/Intro%20to%20Scikit-Learn.ipynb)
* [Data analysis with scikit-learn](http://sebastianraschka.com/Articles/2014_scikit_dataprocessing.html)
* [Advanced Machine Learning with scikit-learn](https://us.pycon.org/2013/community/tutorials/23/)

### Extra

* [University of Colorado Computational Science workshops](http://researchcomputing.github.io/meetup_spring_2014/)
* [Networkx tutorial](http://snap.stanford.edu/class/cs224w-2012/nx_tutorial.pdf)

</span>
