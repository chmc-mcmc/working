import numpy as np
from scipy.stats import norm, uniform

RATIOENERGY=0.1
RATIODT=0.1
HIGHLEVEL=0.9
LOWLEVEL=0.1
INTERVAL=1001
STEPS=3

def K(p, q, r, Uqq):
    W = getW(q, r, Uqq)
    return 0.5 * np.dot(p, np.dot(W, p))

def Kp(p, q, r, Uqq):
    W = getW(q, r, Uqq)
    return np.dot(W, p)

def getW(q, r, Uqq):
    eig = np.linalg.eig(Uqq(q))
    ve = eig[1]
    e = eig[0]
    L = np.array([np.sign(e[i]) * np.abs(e[i])**(-r) if e[i] !=0 else 0 for i in range(len(e))])
    return np.dot(np.dot(ve, np.diag(L)), ve.T)

def hmc(U, Uq, Uqq, Dim, BURNIN, ITERATIONS, PARTICLES, outbnd=lambda q:False, qinit=None):
    r = 0.5
    if qinit is not None:
        qAll = qinit
    else:
        #qAll = norm.rvs(0, 1, size=(PARTICLES, Dim))
        qAll = np.random.uniform(0, 1, (PARTICLES, Dim))
    Utotal = np.sum([U(qAll[i]) for i in range(PARTICLES)])
    #dt = np.min([1 / np.sqrt(np.abs(np.linalg.eigvals(Uqq(qAll[i])))) for i in range(PARTICLES)])
    dt =1e-9
    Htotal = 2 * Utotal if Utotal != 0 else 1

    QS = np.zeros((PARTICLES * (ITERATIONS - BURNIN), Dim))

    for j in range(ITERATIONS):
        pAll = norm.rvs(0, 1, size=(PARTICLES, Dim))
        KtotalNew = np.sum([K(pAll[i], qAll[i], r, Uqq) for i in range(PARTICLES)])
        Utotal = np.sum([U(qAll[i]) for i in range(PARTICLES)])
        Ktotal = Htotal - Utotal
        pAll = pAll * np.sqrt(np.abs(Ktotal / KtotalNew))

        AS = []
        ES = []
        anybad = False

        for i in range(PARTICLES):
            bad = False
            p0 = pAll[i]
            q0 = qAll[i]
            q = q0.copy()
            p = q0.copy()
            if j < BURNIN:
                UE = [U(q)]
                for step in range(1, STEPS + 1):
                    if bad:
                        break
                    dq = Kp(p, q, r, Uqq)
                    dp = -Uq(q)
                    q = q + dt * dq
                    p = p + dt * dp
                    if outbnd(q):
                        bad = True
                        q = q0.copy()
                    else:
                        UE.append(U(q))
                ES.append(UE)
                anybad |= bad

            q=q0.copy()
            p=p0.copy()
            dq = Kp(p, q, r, Uqq)
            if False:
                q1 = q + np.sqrt(STEPS) * dt * dq
                dq1 = Kp(p, q1, r, Uqq)
                q = q + np.sqrt(STEPS) * dt * 0.5 * (dq + dq1)
            else:
                q = q + np.sqrt(STEPS) * dt * dq
            if outbnd(q):
                bad = True
                q = q0.copy()

            alpha = 0 if bad else np.exp(np.clip(U(q0) - U(q), -30, 0))
            if alpha < uniform.rvs():
                q = q0
            qAll[i] = q.copy()
            AS.append(alpha)
            if j >= BURNIN:
                QS[(j - BURNIN) * PARTICLES + i] = q.copy()

        if j < BURNIN and not anybad:
            s = np.unique([np.argmin(ES[i]) for i in range(PARTICLES)])
            S = np.unique([np.argmax(ES[i]) for i in range(PARTICLES)])
            if np.array_equal(s, [0, STEPS])and np.array_equal(S, [0, STEPS]):
                dt *= (1 + RATIODT)
            if np.array_equal(s, [0]) and np.array_equal(S, [STEPS]):
                dt /= (1 + RATIODT)

            hi = np.mean(AS) > HIGHLEVEL
            lo = np.mean(AS) < LOWLEVEL
            if hi:
                Htotal = (Htotal - Utotal) * (1 + RATIOENERGY) + Utotal
            if lo:
                Htotal = (Htotal - Utotal) / (1 + RATIOENERGY) + Utotal

            if j % INTERVAL == 0:
                print(f"{j} {Utotal} {Ktotal} {Htotal} {dt} {np.mean(AS)} {np.std(AS)} {s} {S}")
        else:
            if j % INTERVAL == 0:
                print(f"{j} {Utotal} {Ktotal} {Htotal} {dt}")

    return QS
