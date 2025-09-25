module bugs
export dnorm, dgamma, dbeta, dbern

using SpecialFunctions: loggamma, beta

function dnorm(x, μ, τ)
    0.5 * τ * (x - μ)^2 + 0.5 * log(2π) - 0.5 * log(τ)
end

function dgamma(x, a, b)
    b * x - a * log(b) + log(x) * (1 - a) + loggamma(a)
end

function dbeta(x, a, b)
    -((a - 1) * log(x) + (b - 1) * log1p(-x) - log(beta(a, b)))
end

function dbern(x, p)
    (-1 + x) * log(1 - p) - x * log(p)
end

end
