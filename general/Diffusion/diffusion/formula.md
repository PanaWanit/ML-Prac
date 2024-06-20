# Formula references


$$
 \begin{equation}
    \alpha_t = 1-\beta_t
 \end{equation}
$$

$$
 \begin{equation}
    \bar{\alpha}_t = \prod_{t=1}^T\alpha^{'}_t
 \end{equation}
$$

$\textbf{Reparameterization trick}$

$$
    \text{from \quad} q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}}x_{t-1}, (1-\bar{\alpha}_t)\bold{I})
$$

$$
\begin{equation}
    \therefore q(x_t | x_0)  = \sqrt{\bar{\alpha}}x_{t-1} + \sqrt{1-\bar{\alpha}_t}\epsilon_t;\quad \epsilon \sim \mathcal{N}(0, \bold{I})
\end{equation}
$$