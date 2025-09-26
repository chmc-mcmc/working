using LinearAlgebra, DelimitedFiles, Random, Distributions, ForwardDiff, Printf
push!(LOAD_PATH, @__DIR__)
@printf "%s" LOAD_PATH
using bugs, sampler

# ---------- 数据 --------------------------------------------------
const x = [1.0, 1.5, 1.5, 1.5, 2.5, 4.0, 5.0, 5.0, 7.0,
           8.0, 8.5, 9.0, 9.5, 9.5, 10.0, 12.0, 12.0, 13.0,
           13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5]
const Y = [1.80, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
           2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.32, 2.43,
           2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57]
const n = length(x)


function U(q)
    a, b, r, t = q
    ll=0
    for i in eachindex(x)
        μ =  a - b * r^x[i]
        ll=ll+dnorm(Y[i], μ, t)
    end
    lp = dnorm(a, 0, 1e-6) + dnorm(b, 0, 1e-6) + dgamma(t, 0.001, 0.001)
    return ll + lp
end

Uq(q)  = ForwardDiff.gradient(U, q)
Uqq(q) = ForwardDiff.hessian(U, q)

# ---------- 运行 HMC ----------------------------------------------
const Dim        = 4
const BURNIN     = 2000
const ITERATIONS = 4000
const PARTICLES  = 3

# 边界约束：r ∈ [0.5, 1], t > 0
outbnd(q) = q[3] < 0.5 || q[3] > 1.0 || q[4] <= 0.0

# 初始值
Random.seed!(123)   # 可复现
qinit = rand(PARTICLES, Dim) .* 0.5 .+ 0.5   # Uniform(0.5, 1)

QS = hmc(U, Uq, Uqq, Dim, BURNIN, ITERATIONS, PARTICLES, 0.5, outbnd, qinit)

# ---------- 保存结果 ----------------------------------------------
writedlm("dugongs.csv", QS, ',')

println("采样完成，结果已写入 dugongs.csv")
