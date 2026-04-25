"""
=============================================================================
OPTIMIZACIÓN MULTIOBJETIVO DE CONTROLADOR PID MEDIANTE NSGA-II
=============================================================================
Objetivo: Encontrar los parámetros óptimos (Kp, Ki, Kd) de un controlador
PID para un horno industrial, minimizando simultáneamente el consumo
energético y un índice de rendimiento de control.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import datetime

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.indicators.hv import Hypervolume

# =============================================================================
# SECCIÓN 1: PARÁMETROS GLOBALES DE SIMULACIÓN Y PLANTA
# =============================================================================

dt = 0.1                              
tiempo_simulacion = 150               
t = np.arange(0, tiempo_simulacion, dt)
n = len(t)

setpoint = 200.0                      
t_init   = 25.0                       
potencia = 9000                       

K_planta    = 2.754                   
tau_planta  = 10.0                    
L_planta    = 5.0                     
pasos_retardo = int(L_planta / dt)    

w1, w2, w3 = 1, 2, 1.5

punto_referencia = np.array([2.0, 150])  
calculador_hv = Hypervolume(ref_point=punto_referencia)
lista_hv = []  

# =============================================================================
# SECCIÓN 2: SIMULACIÓN DEL LAZO CERRADO (PID + PLANTA)
# =============================================================================

def simular_lazo_cerrado(Kp: float, Ki: float, Kd: float):
    y        = np.ones(n) * t_init   
    pid      = np.zeros(n)           
    e        = np.zeros(n)           
    integral = 0.0                   

    for i in range(1, n - 1):
        e[i] = setpoint - y[i]

        integral   += e[i] * dt
        derivada    = (e[i] - e[i - 1]) / dt
        pid[i]      = (Kp * e[i]) + (Ki * integral) + (Kd * derivada)

        pid[i] = np.clip(pid[i], 0.0, 100.0)

        u_retrasada = pid[i - pasos_retardo] if i >= pasos_retardo else 0.0

        dy_dt   = (K_planta * u_retrasada - (y[i] - t_init)) / tau_planta
        y[i + 1] = y[i] + dy_dt * dt

    return y, pid

def calcular_objetivos(Kp: float, Ki: float, Kd: float):
    y, pid = simular_lazo_cerrado(Kp, Ki, Kd)

    potencia_instantanea = (pid / 100.0) * potencia          
    consumo = (np.sum(potencia_instantanea) * dt) / 3_600_000  

    max_temp  = np.max(y)
    OSh        = max(0.0, max_temp - setpoint)                 
    ess       = abs(setpoint - y[-1])                         
    banda     = 0.02 * setpoint                               
    fuera     = np.where(np.abs(y - setpoint) > banda)[0]
    ts        = t[fuera[-1]] if len(fuera) > 0 else t[-1]    

    rendimiento = (w1 * ts) + (w2 * OSh) + (w3 * ess)

    # Si el error estacionario es mayor a 5°C, penalizamos brutalmente para que el algoritmo descarte esta solución como "basura".
    if ess > 5.0:
        rendimiento += 10000.0

    return rendimiento, consumo

# =============================================================================
# SECCIÓN 3: DEFINICIÓN DEL PROBLEMA DE OPTIMIZACIÓN MULTIOBJETIVO
# =============================================================================

class OptimizacionPID(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var   = 3,
            n_obj   = 2,
            n_constr= 0,
            xl      = np.array([0.0, 0.0, 0.0]),   
            xu      = np.array([10.0, 5.0, 5.0])   
        )

    def _evaluate(self, x, out, *args, **kwargs):
        Kp, Ki, Kd = x
        rendimiento, consumo = calcular_objetivos(Kp, Ki, Kd)
        out["F"] = [rendimiento, consumo]

# =============================================================================
# SECCIÓN 4: CONFIGURACIÓN Y EJECUCIÓN DE LA CORRIDA NSGA-II
# =============================================================================

nombre_archivo_hv = f"registro_hv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Solo dejamos una configuración
cfg = {"nombre": "Corrida Única (η=20)", "color": "steelblue", "eta": 20}

POP_SIZE  = 100
N_GEN     = 20
SEED      = 1

print("=" * 60)
print("  OPTIMIZACIÓN NSGA-II — PID HORNO INDUSTRIAL")
print("=" * 60)

with open(nombre_archivo_hv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Corrida', 'Configuración', 'Semilla', 'Hypervolume'])

print(f"\n  Ejecutando {cfg['nombre']}...")

alg = NSGA2(
    pop_size             = POP_SIZE,
    eliminate_duplicates = True,
    crossover            = SBX(prob=0.9, eta=cfg["eta"]),
    mutation             = PolynomialMutation(prob=0.03, eta=cfg["eta"]),
)

res = minimize(
    OptimizacionPID(),
    alg,
    ("n_gen", N_GEN),
    seed         = SEED,
    save_history = True,
)

frente_pareto = res.F
hv = calculador_hv.do(frente_pareto)

with open(nombre_archivo_hv, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([1, cfg['nombre'], SEED, hv])

print("\n  ✓ Optimización finalizada")
print(f"  Hypervolume final: {hv}")
print(f"  Resultados guardados en: {nombre_archivo_hv}")

# =============================================================================
# SECCIÓN 5: ANÁLISIS DE RESULTADOS — EVOLUCIÓN POR GENERACIÓN
# =============================================================================
 
sorting = NonDominatedSorting()
gen_data = []
 
for gen_idx, estado in enumerate(res.history):
    F = estado.pop.get("F")   
    X = estado.pop.get("X")   

    nd_indices = sorting.do(F)[0]

    if len(nd_indices) > 0:
        mejor = nd_indices[np.argmin(F[nd_indices, 1])]   
        Kp_v, Ki_v, Kd_v = X[mejor]
        rend_v, cons_v    = F[mejor]

        gen_data.append({
            "Generación":  gen_idx + 1,
            "Kp":          Kp_v,
            "Ki":          Ki_v,
            "Kd":          Kd_v,
            "Consumo":     cons_v,
            "Rendimiento": rend_v,
        })
 
df = pd.DataFrame(gen_data)
 
print(f"\n--- Mejor candidato por generación ---")
print(df.to_string(index=False))
 
last_kp = df["Kp"].iloc[-1]
last_ki = df["Ki"].iloc[-1]
last_kd = df["Kd"].iloc[-1]
 
print(f"\n  Kp final = {last_kp:.4f}")
print(f"  Ki final = {last_ki:.4f}")
print(f"  Kd final = {last_kd:.4f}")

# =============================================================================
# SECCIÓN 6: SIMULACIÓN FINAL CON LOS PARÁMETROS ÓPTIMOS
# =============================================================================

y_sim, pid_sim = simular_lazo_cerrado(last_kp, last_ki, last_kd)

# =============================================================================
# SECCIÓN 7: VISUALIZACIÓN UNIFICADA (UNA SOLA CORRIDA)
# =============================================================================

ax1 = fig1.add_subplot(gs1[0, 0])
res_azul = resultados[0]
cfg_azul = NSGA2_CONFIGS[0]
ax1.scatter(res_azul.F[:, 1], res_azul.F[:, 0], color=cfg_azul["color"], s=15, label=cfg_azul["nombre"], alpha=0.8)

ax1.set_title(f"Frente: {cfg_azul['nombre']}", fontsize=11)
ax1.set_xlabel("Consumo [kWh]")
ax1.set_ylabel("Rendimiento (menor = mejor)")
ax1.grid(True, linestyle="--", alpha=0.5)

# ── 2. Evolución de Parámetros ──────────────────
gens = df["Generación"]
ax2.plot(gens, df["Kp"], color="royalblue",  lw=1.5, marker="o", ms=3, label="Kp")
ax2.plot(gens, df["Ki"], color="darkorange", lw=1.5, marker="s", ms=3, label="Ki")
ax2.plot(gens, df["Kd"], color="seagreen",   lw=1.5, marker="^", ms=3, label="Kd")
ax2.set_title("Evolución de Kp, Ki, Kd", fontsize=11)
ax2.set_xlabel("Generación")
ax2.set_ylabel("Valor del parámetro")
ax2.legend(fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.5)

# ── 3. Respuesta Temporal ───────────────────────
ax3.plot(t, np.full_like(t, setpoint), "r--", lw=1.2, label="Setpoint")
ax3.plot(t, y_sim, color=cfg["color"], lw=1.5, label="Temp (°C)")
ax3.plot(t, pid_sim, color="darkorange", lw=1.0, ls="--", label="u(t) [%]")

ax3.set_title(f"Respuesta del Sistema (Kp={last_kp:.2f}, Ki={last_ki:.2f}, Kd={last_kd:.2f})", fontsize=11)

ax3.set_xlabel("Tiempo (s)")
ax3.set_ylabel("Temp [°C] / Control [%]")
ax3.set_ylim(0, 350)
ax3.legend(fontsize=9, loc="lower right")
ax3.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("resultados_nsga2_pid.png", dpi=150, bbox_inches="tight")
plt.show()