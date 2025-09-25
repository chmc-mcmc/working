using LinearAlgebra, DelimitedFiles, Random, Distributions, ForwardDiff
using Revise
push!(LOAD_PATH, @__DIR__)
using bugs,sampler

# ---------- 数据 --------------------------------------------------
using Random, Distributions, LinearAlgebra

# 参数
n  = 500
α  = 1.0
τ  = 0.25
nt = 200

# 构造 w（0/1 分组向量）
w = shuffle!(MersenneTwister(123), vcat(zeros(n - nt), ones(nt)))  # 等价于 RandomSample[Join[...]]

# 均值与方差
μc = α + 0 * τ
σc = 1.0
μt = α + 1 * τ
σt = 1.0
ρ  = 0.0
Σ  = [σc^2  ρ*σc*σt;
       ρ*σc*σt  σt^2]      # 2×2 协方差矩阵

# 生成潜在结果 (y0, y1)
science = rand(MvNormal([μc, μt], Σ), n)'   # n×2 矩阵，转置后方便取列
y0 = science[:, 1]
y1 = science[:, 2]

# 观测结果
yobs = @. y0 * (1 - w) + y1 * w

function U(q)
    α, τ, σt, σc = q
    logprior = dnorm(α, 0, 1/25) +
        dnorm(τ,  0, 1/25) +
        dnorm(σc, 0, 1/25) +
        dnorm(σt, 0, 1/25)

    loglik = 0.0
    @inbounds for i in 1:n
        μi    = α + τ * w[i]
        σi    = σt * w[i] + σc * (1 - w[i])
        loglik += dnorm(yobs[i], μi, σi)
    end
    return logprior + loglik
end

Uq(q)  = ForwardDiff.gradient(U, q)
Uqq(q) = ForwardDiff.hessian(U, q)

# ---------- 运行 HMC ----------------------------------------------
const Dim        = 4
const BURNIN     = 5000
const ITERATIONS = 10000
const PARTICLES  = 3


outbnd(q) = q[3] < 0 || q[4] <0

# 初始值
Random.seed!(123)   # 可复现
qinit = rand(PARTICLES, Dim)

QS = hmc(U, Uq, Uqq, Dim, BURNIN, ITERATIONS, PARTICLES, 0.5, outbnd, qinit)

# ---------- 保存结果 ----------------------------------------------
writedlm("causal.csv", QS, ',')

println("采样完成，结果已写入 causal.csv")
