module sampler
export hmc

using LinearAlgebra, Distributions, Statistics, Printf

const RATIOENERGY = 0.1
const RATIODT     = 0.1
const HIGHLEVEL   = 0.9
const LOWLEVEL    = 0.1
const INTERVAL    = 1001
const STEPS       = 3

function getW(q, r, Uqq)
    H = Uqq(q)
    F = eigen(H)
    ve = F.vectors
    e  = F.values
    # L_i = sign(e_i) * |e_i|^{-r}  (0 if e_i==0)
    L = [ei ≈ 0.0 ? 0.0 : sign(ei) * abs(ei)^(-r) for ei in e]
    return ve * Diagonal(L) * ve'
end

function K(p, q, r, Uqq)
    W = getW(q, r, Uqq)
    return 0.5 * dot(p, W * p)
end

function Kp(p, q, r, Uqq)
    W = getW(q, r, Uqq)
    return W * p
end

"""
Hamiltonian Monte Carlo sampler.
Returns (PARTICLES*(ITERATIONS-BURNIN)) × Dim array of samples.
"""
function hmc(U, Uq, Uqq, Dim, BURNIN, ITERATIONS, PARTICLES, r, outbnd, qinit)
    #r = 0.
    # initialise positions
    if !isnothing(qinit)
        qAll = copy(qinit)
    else
        qAll = rand(PARTICLES, Dim)
    end

    # initial total potential energy
    Utotal = sum(U(qAll[i,:]) for i in 1:PARTICLES)
    dt = 1e-9
    Htotal = Utotal ≈ 0.0 ? 1.0 : 2.0 * Utotal

    # storage for samples
    QS = zeros(PARTICLES * (ITERATIONS - BURNIN), Dim)

    for j in 0:(ITERATIONS-1)
        # --- momentum refresh -----------------------------------------
        pAll = rand(Normal(), PARTICLES, Dim)
        KtotalNew = sum(K(pAll[i,:], qAll[i,:], r, Uqq) for i in 1:PARTICLES)
        Utotal    = sum(U(qAll[i,:]) for i in 1:PARTICLES)
        Ktotal    = Htotal - Utotal
        # rescale momenta
        pAll .= pAll .* sqrt(abs(Ktotal / KtotalNew))

        AS = Float64[]    # acceptance probabilities
        ES = []           # energy trajectories (burn-in only)
        anybad = false

        # --- leapfrog / simplified dynamics ---------------------------
        for i in 1:PARTICLES
            bad = false
            p0 = pAll[i,:]
            q0 = qAll[i,:]
            q  = copy(q0)
            p  = copy(p0)

            # burn-in: full leapfrog trajectory
            if j < BURNIN
                UE = Float64[U(q)]
                for step in 1:STEPS
                    bad && break
                    dq = Kp(p, q, r, Uqq)
                    dp = -Uq(q)
                    q .= q .+ dt .* dq
                    p .= p .+ dt .* dp
                    if outbnd(q)
                        bad = true
                        q .= q0
                    else
                        push!(UE, U(q))
                    end
                end
                push!(ES, UE)
                anybad |= bad
            end

            # --- single drift step (always performed) -----------------
            q .= q0
            p .= p0
            dq = Kp(p, q, r, Uqq)
            q .= q .+ sqrt(STEPS) .* dt .* dq
            if outbnd(q)
                bad = true
                q .= q0
            end

            # Metropolis accept/reject
            α = bad ? 0.0 : exp(clamp(U(q0) - U(q), -30.0, 0.0))
            if α < rand()   # always reject if α==0
                q .= q0
            end
            qAll[i,:] .= q
            push!(AS, α)

            # store samples after burn-in
            if j >= BURNIN
                QS[(j - BURNIN)*PARTICLES + i, :] .= q
            end
        end

        # --- burn-in adaptations --------------------------------------
        if j < BURNIN && !anybad
            # unique argmin/argmax along trajectories
            s = unique(argmin.(ES))
            S = unique(argmax.(ES))
            if s == [1, STEPS+1] && S == [1, STEPS+1]
                dt *= (1 + RATIODT)
            elseif s == [1] && S == [STEPS+1]
                dt /= (1 + RATIODT)
            end

            αbar = mean(AS)
            if αbar > HIGHLEVEL
                Htotal = (Htotal - Utotal)*(1 + RATIOENERGY) + Utotal
            elseif αbar < LOWLEVEL
                Htotal = (Htotal - Utotal)/(1 + RATIOENERGY) + Utotal
            end

            if j % INTERVAL == 0
                @printf "%d %.6g %.6g %.6g %.6g %.4g %.4g %s %s\n" j Utotal Ktotal Htotal dt αbar std(AS) s S
            end
        else
            if j % INTERVAL == 0
                @printf "%d %.6g %.6g %.6g %.6g\n" j Utotal Ktotal Htotal dt
            end
        end
    end

    return QS
end

end
