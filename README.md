# KSG1 Transfer Entropy Calculator
Here, we present an estimation of the transfer entropy based on the nearest neighbor (KNN) method proposed by Kraskov, Stogbauer, and Grassberger (KSG). This method (algorithm 1) estimates the transfer entropy by determining the hyper-radius about each point in the full joint space enclosing $k$ other points. The algorithm used here utilizes the Chebyshev (maxnorm) distance about each point to create "hyper-squares". Algorithm 2 squeezes each side length in the hyper-square to more precisely fit each point to form a "hyper-rectangle" in order to reduce bias at the cost of variance. Consequently, algorithm 2 is preferred for obtaining accurate values, while algorithm 1 is preferred for hypothesis testing. Then, the corresponding radii are conserved to count the $k$ points in the respective marginal spaces. Prior to this, the time-series undergo a time-delay embedding in an attempt to reconstruct the dynamics of the original state space. The transfer entropy can written as follows

$$ TE_{Y \to X} = \langle \psi(k) + \psi(n_{{X^-}, i} + 1) - \psi(n_{(X^-, Y^-), i} + 1) - \psi(n_{(X^p, X^-), i} + 1)\rangle,$$

where $\psi(\cdot)$ represents the di-gamma function such that $\psi(x)=\frac{1}{\Gamma(x)} \frac{d \Gamma(x)}{dx}$, $k$ is the number of neighboring points selected for the full joint space, and $X^-$, $Y^-$, $X^p$ represent the embedded destination and source time-series and the predicted destination time-series, respectively.

It is important to note, the notation for the embedded and predicted time-series is simplified for clarity. More precisely, 
\begin{equation}
    \begin{split}
        X^- &= X_n^{(K, \tau_X)} &= \{ X_{n-(K-1)\tau_X},\ldots, X_{n-\tau_X}, X_n\}, \\
        Y^- &= Y_{n+1-u}^{(L, \tau_Y)} &= \{ Y_{n+1-u-(L-1)\tau_Y}, \ldots, Y_{n+1-u-\tau_Y}, Y_{n+1-u}\}, \\
        X^p &= X_{n+1} &= \{ X_1, \ldots, X_{n+1} \},
    \end{split}
\end{equation}

where $K$ and $L$ and $\tau_X$ and $\tau_Y$ represent the embedding dimension and time-delay for the destination and source time-series $X$ and $Y$, respectively. The parameter $u$ is optional, representing the source-destination delay. The transfer entropy itself originates from (Schreiber, 2000) in an attempt to quantify the statistical coherence between two time series by removing extraneous information by conditioning on the their past. It is often written as a conditional mutual information such that the chain rule for conditional entropy can be used to decompose the term into a sum of 4 joint entropies that serve as the basis for the above estimation of the transfer entropy. That is,

\begin{equation}
    \begin{split}
        TE_{Y\to X} &= I(X^p ; X^- \mid Y^-)\\
        &= H(X^p \mid X^-) - H(X^p \mid X^-, Y^-)\\
        &= H(X^p, X^-) + H(X^-, Y^-) - H(X^p, X^-, Y^-) - H(X^-).
    \end{split}
\end{equation}
