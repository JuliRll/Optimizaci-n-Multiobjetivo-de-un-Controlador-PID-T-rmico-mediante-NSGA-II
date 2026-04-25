# Optimizaci-n-Multiobjetivo-de-un-Controlador-PID-T-rmico-mediante-NSGA-II
Este proyecto aborda la sintonización de un controlador PID (Proporcional, Integrativo, Derivativo) para procesos de control térmico industrial. 

A diferencia de los métodos clásicos uni-objetivo que solo priorizan la estabilidad y la velocidad de respuesta, este trabajo se centra en resolver el conflicto entre el alto rendimiento térmico (respuesta rápida y precisa) y la eficiencia energética (bajo consumo).

Se desarrolló un sistema de optimización multiobjetivo simulado sobre un modelo matemático de un horno industrial. El objetivo principal fue encontrar el conjunto de parámetros óptimos del controlador (Kp, Ki, Kd) que minimice simultáneamente el consumo energético y la penalización de rendimiento térmico, utilizando el algoritmo genético **NSGA-II**.

**Algoritmo Central:** Implementación y aplicación del **Non-Dominated Sorting Genetic Algorithm II (NSGA-II)**, un algoritmo evolutivo multiobjetivo, utilizando la librería `pymoo` de Python.

**Doble Objetivo de Optimización (Minimización):

• Consumo Energético:** Energía eléctrica total consumida por el actuador, expresada en Kilovatios-hora (kWh).

• **Rendimiento Térmico:** Función de costo combinada que penaliza el tiempo de establecimiento (ts), el sobreimpulso (OS) y el error en estado estable (ess).

**Modelado del Sistema:** El comportamiento dinámico se simuló a través de una Función de Transferencia de Primer Orden con Tiempo Muerto (PID térmico para un horno industrial), con parámetros de ganancia (K=2.754 [°C/%P]), tiempo muerto (L=5 [s]) y constante de tiempo (tau=10 [s]).

**Resultado Clave (Frente de Pareto):** El proyecto entregó un **Frente de Pareto**, un catálogo de soluciones no dominadas que representan los mejores *trade-offs* posibles entre el alto desempeño (bajo índice de rendimiento) y la máxima eficiencia (bajo consumo).
