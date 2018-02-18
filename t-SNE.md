<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: {
  	equationNumbers: { autoNumber: "AMS" },
    Macros: {
       b: ["\\mathbf{#1}", 1],
       bracket: ["\\left \\{ #1 \\right \\}", 1]
    }
  }
});
</script>
## t-SNE とは

主成分分析のような低次元への写像をうまい具合にやってくれる
高次元データのビジュアライゼーションなどに使われる

### t-SNE のメリット

- SNE の場合と比較して余分なパラメーターが存在しない
	- 手動で変更加える必要がない
- SNE と比較して近い要素を近くづけるだけでなく、確率分布的に遠いものは遠く配置してくれる
- ... 他いくつか

### t-SNE 導出

高次元データの同時確率分布を以下で表せると仮定する（ガウシアンカーネル）

$$
\newcommand{\norm}[1]{\|#1\|}
\newcommand{\expDist}[2]{\exp \bracket{- \norm{\b{x^{(#1)}} - \b{x^{(#2)}}} \,/\, 2 \sigma^2}}
\begin{align}
	p_{ij} = \frac{\expDist{i}{j}}{\sum_{k \neq l} \expDist{k}{l}}
\end{align}
$$

イメージとしては動径方向のデータ点同士の距離を確率分布として定義してあげた形になる。
また、あくまでデータ点同士の距離関係だけを問題にしているので、$ p_{ii} = 0 $ とする。

低次元データを $ \b{y_i} $ として、こちらのデータ点を 1 次元のスチューデントの t 分布で表すことにする。

$$
\newcommand{\tDist}[2]{\bracket{1 + \norm{\b{y^{(#1)}} - \b{y^{(#2)}}}^2}^{-1}}
\begin{align}
	q_{ij} = \frac{\tDist{i}{j}}{\sum_{k \neq l} \tDist{i}{j}}
\end{align}
$$

このようにして $ q_{ij} $ と $ p_{ij} $ の近似度を KL 距離で測り、これを最小とするような $ y_{i} $ の座標を求めてあげればよい。

したがって目的関数は

$$
\begin{align}
	C = \sum_{i, j} p_{ij} \ln \frac{p_{ij}}{q_{ij}} \label{eq:1.0}
\end{align}
$$

ここで以下のように $ b_{ij} $ と $ a_{ij} $ を定義する

$$
\begin{align}
	a_{ij} & = \expDist{i}{j} \\
	b_{ij} & = \tDist{i}{j}
\end{align}
$$

$ \eqref{eq:1.0} $ を $ y_d^{(l)} $ について微分すると

$$
\newcommand{ydl}[0]{y_d^{(l)}}
\newcommand{pij}[0]{p_{ij}}
\newcommand{qij}[0]{q_{ij}}
\newcommand{qst}[0]{q_{st}}
\newcommand{dij}[0]{d_{ij}}
\newcommand{bij}[0]{b_{ij}}
\newcommand{bst}[0]{b_{st}}
\begin{align}
	\frac{\partial C}{\partial \ydl}
		& = \sum_{i, j} \bracket{
			- \frac{\partial}{\partial \ydl} \pij \ln \qij
		} \\
		& = \sum_{i, j} \bracket{
			- \frac{\qij}{\pij} \frac{\partial \qij}{\partial \ydl}
		} \label{eq:1.1}
\end{align}
$$

ここで

$$
\newcommand{sumbnm}[0]{\sum_{n, m} b_{nm}}
\newcommand{difydl}[1]{\frac{\partial #1}{\partial \ydl}}
\begin{align}
	\frac{\partial \qij}{\partial \ydl} 
		& = \difydl{\bij} \frac{1}{\sumbnm} - \frac{\bij}{(\sumbnm)^2} \bracket{
			\sum_{s, t} \difydl{\bst}
		} \label{eq:1.2}
\end{align}
$$

また $ \difydl{\bst} $ の微分は以下のように与えられる。

$$
d_{ij} = \norm{\b{y^{(i)}} - \b{y^{(j)}}}^2
$$

とおいて

$$
\newcommand{Gijdl}[0]{G_{ijd}^{(l)}}
\newcommand{Gstdl}[0]{G_{std}^{(l)}}
\begin{align}
	\difydl{\bst} 
		& = \frac{\partial}{\partial \dij} \bracket{1 + (\dij)^2}^{-1}
			\difydl{} \bracket{\norm{\b{y^{(i)}} - \b{y^{(j)}}}^2} \\
		& = - (1 + (\dij)^2)^{-2} \sum_{k} \difydl{} \bracket{y_k^{(i)} - y_k^{(j)}}^2 \\
		& = - 2 \bij^2 (y_d^{(i)} - y_d^{(j)}) (\delta_{il} - \delta_{jl}) \\
		& = - 2 \bij^2 \, \Gijdl  \hspace{15pt} (\Gijdl = (y_d^{(i)} - y_d^{(j)}) (\delta_{il} - \delta_{jl}) とおいた) \label{eq:1.3}
\end{align}
$$

$ \eqref{eq:1.3} $ を $ \eqref{eq:1.2} $ に代入する

$$
\begin{align}
	\frac{\partial \qij}{\partial \ydl} 
		& = - 2 \bij^2 \Gijdl \frac{1}{\sumbnm} - \frac{\bij}{(\sumbnm)^2} \bracket{
			\sum_{s, t} - 2 \bij^2 \Gstdl
		} \\
		& = -2 \qij \bij \Gijdl + 2 \qij \sum_{s, t} \qst \bst \Gstdl \label{eq:1.4}
\end{align}
$$

最後に $ \eqref{eq:1.4} $ を $ \eqref{eq:1.1} $ に代入すると

$$
\begin{align}
	\frac{\partial C}{\partial \ydl}
		& = \sum_{i, j} - \frac{\qij}{\pij} \bracket{
			-2 \qij \bij \Gijdl + 2 \qij \sum_{s, t} \qst \bst \Gstdl
		} \\
		& = \sum_{i, j} \pij \bij \Gijdl - (\sum_{i, j} \pij) \bracket{
			\sum_{s, t} \qst \bst \Gstdl
		} \\
		& = \sum_{i, j} (\pij - \qij) \bij \Gijdl \\
		& = \sum_{i, j} (\pij - \qij) \bij (y_d^{(i)} - y_d^{(j)}) (\delta_{il} - \delta_{jl}) \label{eq:1.5}
\end{align}
$$

ここで一般的に $ A_{ij} = - A_{ij} $ のときに

$$
\begin{aligned}
	\sum_{i, j} A_{ij} (\delta_{il} - \delta{jl})
		& = \sum_{j} (\sum_{i} A_{ij} \delta_{il}) - \sum_{i} (\sum_{j} A_{ij} \delta_{jl}) \\
		& = \sum_{j} A_{lj} - \sum_{i} A_{il} \\
		& = 2 \sum_{j} A_{lj}
\end{aligned}
$$

が成り立つので

$$
\begin{align}
	\eqref{eq:1.5}
		& = \sum_{j} 4 (p_{lj} - q_{lj}) (y_d^{(i)} - y_d^{(l)}) b_{il}
\end{align}
$$

これより微分は以下で与えられることがわかった

$$
\begin{align}
	\bbox[10pt, border: 2px dotted black]{
		\frac{\partial C}{\partial \ydl} =
			\sum_{j} 4 (p_{lj} - q_{lj}) (y_d^{(i)} - y_d^{(l)}) \tDist{i}{l}
	}
\end{align}
$$